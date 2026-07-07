from __future__ import annotations

"""Token-dispatcher data contracts for the MSCCL++ EP all-to-all backend.

This module defines the dispatch-output / combine-input containers that bridge
the MSCCL++ Expert-Parallel runtime (``mscclpp.ext.ep``) to the MoE runner
backends. The layout mirrors DeepEP's *normal* path because MSCCL++ EP is a port
of DeepEP: ``intranode_dispatch`` returns ``(recv_x, recv_x_scales,
recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert, ...)`` which maps
field-for-field onto :class:`MSCCLPPDispatchOutput`.

Only the format containers live here so the runner-side pre/post-permute
functions have a concrete, typed contract to bind to. The dispatcher that
actually drives ``mscclpp.ext.ep`` (holding the per-dispatch handle used by
combine, analogous to DeepEP's ``self.handle``) is a separate component.
"""

from typing import TYPE_CHECKING, List, NamedTuple, Optional

import torch

from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKOutput


class MSCCLPPDispatchOutput(NamedTuple):
    """MSCCL++ EP dispatch output (normal / intranode layout).

    Fields mirror ``ExpertParallelRuntime.intranode_dispatch``:

    * ``hidden_states``      -> ``recv_x``                       [num_recv_tokens, hidden]
    * ``hidden_states_scale``-> ``recv_x_scales``               (optional, fp8 path)
    * ``topk_ids``           -> ``recv_topk_idx``               [num_recv_tokens, top_k]
                                local expert ids, ``-1`` for experts not on this rank
    * ``topk_weights``       -> ``recv_topk_weights``           [num_recv_tokens, top_k]
    * ``num_recv_tokens_per_expert`` -> per-local-expert recv counts
    """

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    num_recv_tokens_per_expert: List[int]

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.MSCCLPP


assert isinstance(MSCCLPPDispatchOutput, DispatchOutput)


class MSCCLPPCombineInput(NamedTuple):
    """MSCCL++ EP combine input.

    ``hidden_states`` carries the per-recv-token expert output produced by the
    runner. ``topk_ids`` / ``topk_weights`` are forwarded so the dispatcher's
    combine can reduce contributions back to each source token across ranks.
    """

    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.MSCCLPP


assert isinstance(MSCCLPPCombineInput, CombineInput)


class MSCCLPPLLDispatchOutput(NamedTuple):
    """MSCCL++ EP low-latency (masked, expert-major) dispatch output.

    Mirrors :class:`DeepEPLLDispatchOutput`. The MSCCL++ LL kernels return a
    *padded expert-major* buffer instead of the token-major layout used by the
    normal path:

    * ``hidden_states``      -> ``DispatchOutput.tokens``
                                [num_local_experts, slots_per_expert, hidden],
                                ``slots_per_expert = world_size * max_tokens_per_rank``
    * ``hidden_states_scale``-> per-slot fp8 scales (optional, fp8 dispatch)
    * ``topk_ids``           -> original ``[num_tokens, top_k]`` routing ids,
                                forwarded for the weighted LL combine
    * ``topk_weights``       -> original ``[num_tokens, top_k]`` routing weights
    * ``masked_m``           -> ``DispatchOutput.num_tokens_per_expert``
                                [num_local_experts], valid slot count per expert
    * ``expected_m``         -> average tokens per expert (GEMM size hint)
    """

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    masked_m: torch.Tensor
    expected_m: int

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.MSCCLPP_LL


assert isinstance(MSCCLPPLLDispatchOutput, DispatchOutput)


class MSCCLPPLLCombineInput(NamedTuple):
    """MSCCL++ EP low-latency combine input.

    ``hidden_states`` is the per-slot expert output in the same masked
    expert-major layout as the dispatch buffer
    ([num_local_experts, slots_per_expert, hidden]). ``topk_ids`` /
    ``topk_weights`` are forwarded for parity with the DeepEP-LL contract; the
    MSCCL++ LL combine itself reads the routing weights from the dispatch handle
    held by the dispatcher.
    """

    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.MSCCLPP_LL


assert isinstance(MSCCLPPLLCombineInput, CombineInput)


class MSCCLPPDispatcher(BaseDispatcher):
    """MSCCL++ EP intranode (NVLink) all-to-all dispatcher, HT / normal mode.

    Thin wrapper around ``mscclpp.ext.ep.ExpertParallelRuntime``'s intranode
    dispatch/combine, mirroring the DeepEP-normal dispatcher:

    * :meth:`dispatch` runs ``get_dispatch_layout`` + ``intranode_dispatch`` to
      scatter each token to the ranks that own its top-k experts, and returns an
      :class:`MSCCLPPDispatchOutput`. ``recv_topk_idx`` comes back in *local*
      expert-id space (``idx - rank * num_local_experts``, ``-1`` for experts not
      on this rank), which is exactly the ``filter_expert`` layout the Triton
      runner consumes.
    * :meth:`combine` runs ``intranode_combine`` to reduce the per-rank expert
      outputs back to each source token.

    Combine is an **unweighted** cross-rank sum: ``intranode_combine`` plain-sums
    the hidden states and only reduces ``topk_weights`` into a separate (here
    unused) tensor, so it never re-applies the routing weights. The weights must
    already be folded in upstream -- the Triton runner does this during its
    within-rank top-k reduction -- so dispatch -> Triton -> combine applies each
    weight exactly once.

    Only the intranode (``num_rdma_bytes=0``, ``low_latency_mode=False``) path is
    wired here; it matches the validated 8-GPU single-node EP test.
    """

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
        num_sms: int = 20,
        nvl_chunk_send: int = 8,
        nvl_chunk_recv: int = 256,
    ):
        super().__init__()

        try:
            from mscclpp import CommGroup
            from mscclpp.ext import ep
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "MSCCL++ EP is not available. Build mscclpp with "
                "-DMSCCLPP_BUILD_EXT_EP=ON so that `from mscclpp.ext import ep` "
                "works, then select the `mscclpp` MoE a2a backend."
            ) from exc

        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype

        self._ep_group = CommGroup(torch_group=group)
        self.num_ranks = self._ep_group.nranks

        self.config = ep.Config(num_sms, nvl_chunk_send, nvl_chunk_recv)
        elem_size = torch.empty((), dtype=params_dtype).element_size()
        num_nvl_bytes = self.config.get_nvl_buffer_size_hint(
            hidden_size * elem_size, self.num_ranks
        )
        self.runtime = ep.ExpertParallelRuntime(
            self._ep_group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=0,
            low_latency_mode=False,
        )

        # Per-dispatch state required by the paired combine (the prefix
        # matrices / source-index / send-head layout produced by dispatch),
        # analogous to DeepEP's ``self.handle``. Reset after each combine.
        self._combine_handle: Optional[tuple] = None

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> MSCCLPPDispatchOutput:
        topk_ids = topk_output.topk_ids.to(torch.int64)
        topk_weights = topk_output.topk_weights.to(torch.float32)

        (
            num_tokens_per_rank,
            _num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            _layout_event,
        ) = self.runtime.get_dispatch_layout(
            topk_ids, self.num_experts, None, False, False
        )

        (
            recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert,
            rank_prefix_matrix,
            _channel_prefix_matrix,
            recv_channel_prefix_matrix,
            recv_src_idx,
            send_head,
            _dispatch_event,
        ) = self.runtime.intranode_dispatch(
            hidden_states,
            None,  # x_scales: unquantized (bf16) dispatch
            topk_ids,
            topk_weights,
            num_tokens_per_rank,
            is_token_in_rank,
            num_tokens_per_expert,
            0,  # num_recv_tokens == 0 -> non-cached (run notify_dispatch)
            None,  # rank_prefix_matrix (cached mode only)
            None,  # channel_prefix_matrix (cached mode only)
            1,  # expert_alignment
            self.config,
            None,  # previous_event
            False,  # async_finish
            False,  # allocate_on_comm_stream
        )

        # Stash exactly the handle fields intranode_combine needs.
        self._combine_handle = (
            recv_src_idx,
            rank_prefix_matrix,
            recv_channel_prefix_matrix,
            send_head,
            recv_topk_weights,
        )

        return MSCCLPPDispatchOutput(
            hidden_states=recv_x,
            hidden_states_scale=recv_x_scales,
            topk_ids=recv_topk_idx,
            topk_weights=recv_topk_weights,
            num_recv_tokens_per_expert=num_recv_tokens_per_expert,
        )

    def combine(self, combine_input: MSCCLPPCombineInput) -> torch.Tensor:
        hidden_states, _topk_ids, _topk_weights = combine_input
        assert (
            self._combine_handle is not None
        ), "MSCCLPPDispatcher.combine called before dispatch"
        (
            recv_src_idx,
            rank_prefix_matrix,
            recv_channel_prefix_matrix,
            send_head,
            recv_topk_weights,
        ) = self._combine_handle

        combined_x, _combined_topk_weights, _combine_event = (
            self.runtime.intranode_combine(
                hidden_states,
                recv_topk_weights,
                recv_src_idx,
                rank_prefix_matrix,
                recv_channel_prefix_matrix,
                send_head,
                self.config,
                None,  # previous_event
                False,  # async_finish
                False,  # allocate_on_comm_stream
            )
        )

        self._combine_handle = None
        return combined_x


class _SharedLLRuntime(NamedTuple):
    """Heavy LL state shared by every MoE layer (see _get_shared_ll_runtime)."""

    ep_group: object
    moe_comm: object
    dispatch_output_buffer: torch.Tensor
    num_ranks: int


# One LL runtime per (group, geometry, capacity) reused across all MoE layers.
# Each transformer layer constructs its own MSCCLPPLLDispatcher, but the LL RDMA
# buffers inside ``MoECommunicator`` and the
# ``(num_local_experts, world_size * max_tokens_per_rank, hidden)`` dispatch
# output buffer scale with ``max_tokens_per_rank`` and MUST NOT be allocated per
# layer -- a 48-layer model at a realistic capacity (e.g. 4096) would OOM.
# Mirrors DeepEP's global ``Buffer`` singleton. Sharing is safe because MoE
# layers execute sequentially within a forward pass (dispatch -> GEMM -> combine
# completes before the next layer dispatches), and the resulting static buffer
# addresses are exactly what CUDA graph capture requires.
_SHARED_LL_RUNTIME: dict = {}


def _get_shared_ll_runtime(
    group: torch.distributed.ProcessGroup,
    num_experts: int,
    num_local_experts: int,
    hidden_size: int,
    router_topk: int,
    num_max_dispatch_tokens_per_rank: int,
) -> _SharedLLRuntime:
    key = (
        id(group),
        num_experts,
        num_local_experts,
        hidden_size,
        router_topk,
        num_max_dispatch_tokens_per_rank,
    )
    cached = _SHARED_LL_RUNTIME.get(key)
    if cached is not None:
        return cached

    try:
        from mscclpp import CommGroup
        from mscclpp.ext import ep
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "MSCCL++ EP is not available. Build mscclpp with "
            "-DMSCCLPP_BUILD_EXT_EP=ON so that `from mscclpp.ext import ep` "
            "works, then select the `mscclpp` MoE a2a backend with "
            "`--mscclpp-mode low_latency`."
        ) from exc

    ep_group = CommGroup(torch_group=group)
    num_ranks = ep_group.nranks

    # MoECommunicator validates: num_experts % world_size == 0, even contiguous
    # expert placement, BF16 (or fp8_e4m3) input, and builds the low-latency
    # runtime (low_latency_mode=True, num_nvl_bytes=0).
    moe_comm = ep.MoECommunicator(
        comm=ep_group,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        topk=router_topk,
        max_tokens_per_rank=num_max_dispatch_tokens_per_rank,
        mode="ll",
        num_rdma_qps_per_rank=max(1, num_local_experts),
    )

    # Low-latency dispatch is a cross-rank all-to-all scatter that writes into a
    # caller-owned, expert-major buffer of shape
    # ``(num_local_experts, world_size * max_tokens_per_rank, hidden_size)``
    # (~num_experts x larger than the per-rank input, so it cannot alias
    # ``hidden_states``). Allocate it once and reuse it on every forward and on
    # every layer: the Triton grouped GEMM consumes this same tensor as its
    # input, and a static address is required for CUDA graph capture.
    dispatch_output_buffer = torch.empty(
        (
            num_local_experts,
            num_ranks * num_max_dispatch_tokens_per_rank,
            hidden_size,
        ),
        dtype=torch.bfloat16,
        device=torch.device("cuda", torch.cuda.current_device()),
    )

    runtime = _SharedLLRuntime(
        ep_group=ep_group,
        moe_comm=moe_comm,
        dispatch_output_buffer=dispatch_output_buffer,
        num_ranks=num_ranks,
    )
    _SHARED_LL_RUNTIME[key] = runtime
    return runtime


class MSCCLPPLLDispatcher(BaseDispatcher):
    """MSCCL++ EP low-latency (LL) all-to-all dispatcher.

    Wraps ``mscclpp.ext.ep.MoECommunicator`` (``mode="ll"``), the high-level LL
    API, mirroring the DeepEP low-latency dispatcher:

    * :meth:`dispatch` runs ``MoECommunicator.dispatch`` to scatter each token to
      the ranks owning its top-k experts and returns an
      :class:`MSCCLPPLLDispatchOutput`. The payload is a *padded expert-major*
      buffer ``[num_local_experts, slots_per_expert, hidden]`` plus a per-expert
      valid-slot count (``masked_m``); the per-dispatch ``DispatchHandle`` (which
      carries ``topk_ids`` / ``topk_weights`` / scatter metadata) is stashed for
      the paired combine.
    * :meth:`combine` runs ``MoECommunicator.combine`` to reduce the per-slot
      expert outputs back to each source token.

    Combine is a **weighted** cross-rank sum: ``low_latency_combine`` multiplies
    each expert contribution by the routing weight stored in the handle before
    summing. The Triton runner must therefore *not* re-apply the weights (the
    ``mscclpp_ll`` pre-permute drives the GEMM with unit weights), so
    dispatch -> Triton -> combine applies each weight exactly once -- the mirror
    image of the HT path, where the GEMM is weighted and the combine is plain.

    NOTE: single-node intranode LL currently hangs in the MSCCL++ LL kernels (IB
    loopback between two HCAs on the same host); cross-node LL with one GPU per
    node works as designed. See ``mscclpp/src/ext/ep/README.md``.
    """

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
        num_max_dispatch_tokens_per_rank: int,
    ):
        super().__init__()

        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank

        # Reuse one communicator + dispatch output buffer across every MoE layer
        # (see _get_shared_ll_runtime). Allocating these per layer OOMs at
        # realistic ``max_tokens_per_rank`` because they scale with capacity and
        # the model has many MoE layers.
        runtime = _get_shared_ll_runtime(
            group,
            num_experts,
            num_local_experts,
            hidden_size,
            router_topk,
            num_max_dispatch_tokens_per_rank,
        )
        self._ep_group = runtime.ep_group
        self.num_ranks = runtime.num_ranks
        self._moe_comm = runtime.moe_comm
        self._dispatch_output_buffer = runtime.dispatch_output_buffer

        # The DispatchHandle produced by dispatch and consumed by combine
        # (carries topk_ids / topk_weights / scatter metadata). Reset after each
        # combine, analogous to DeepEP's ``self.handle``. Kept per-instance: MoE
        # layers run sequentially, so each layer's dispatch->combine pair owns
        # the shared communicator for the duration of its forward.
        self._combine_handle = None

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> MSCCLPPLLDispatchOutput:
        topk_ids = topk_output.topk_ids.to(torch.int64)
        topk_weights = topk_output.topk_weights.to(torch.float32)

        dispatch_out, handle = self._moe_comm.dispatch(
            hidden_states,
            topk_ids,
            topk_weights,
            output_buffer=self._dispatch_output_buffer,
        )
        self._combine_handle = handle

        hidden_states_scale = None
        if dispatch_out.scales is not None:
            hidden_states_scale = dispatch_out.scales.local

        # Average tokens per expert (same hint DeepEP-LL passes to the masked
        # GEMM); ``world_size`` copies of each token are scattered across the
        # ``num_experts`` experts. Unused by the Triton runner but kept for
        # parity with the DeepEP-LL contract.
        expected_m = (
            hidden_states.shape[0] * self.num_ranks * self.router_topk
            + self.num_experts
        ) // self.num_experts

        return MSCCLPPLLDispatchOutput(
            hidden_states=dispatch_out.tokens,
            hidden_states_scale=hidden_states_scale,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            masked_m=dispatch_out.num_tokens_per_expert,
            expected_m=expected_m,
        )

    def combine(self, combine_input: MSCCLPPLLCombineInput) -> torch.Tensor:
        hidden_states, _topk_ids, _topk_weights = combine_input
        assert (
            self._combine_handle is not None
        ), "MSCCLPPLLDispatcher.combine called before dispatch"

        # low_latency_combine reads topk_ids / topk_weights from the handle and
        # applies the routing weights while reducing back to the source tokens.
        combined_x = self._moe_comm.combine(hidden_states, self._combine_handle)

        self._combine_handle = None
        return combined_x
