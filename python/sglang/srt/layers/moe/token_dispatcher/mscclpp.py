"""MSCCL++ expert-parallel dispatchers and MoE runner data contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, NamedTuple, Optional

import torch

from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInputFormat,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKOutput


@dataclass(frozen=True)
class MSCCLPPDispatchOutputBase(ABC):
    """Fields shared by all MSCCL++ dispatch layouts."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]

    @property
    @abstractmethod
    def format(self) -> DispatchOutputFormat:
        pass


@dataclass(frozen=True)
class MSCCLPPDispatchOutput(MSCCLPPDispatchOutputBase):
    """MSCCL++ high-throughput token-major dispatch output.

    Fields map the public ``MoECommunicator.dispatch`` result:

    * ``hidden_states``      -> ``recv_x``                       [num_recv_tokens, hidden]
    * ``hidden_states_scale``-> ``recv_x_scales``               (optional, fp8 path)
    * ``topk_output``        -> dispatched routing with global expert ids
    * ``num_recv_tokens_per_expert`` -> per-local-expert recv counts

    ``local_expert_start`` lets local-expert runners derive their id space from
    the canonical global ids without storing a second routing tensor.
    """

    topk_output: StandardTopKOutput
    num_recv_tokens_per_expert: List[int]
    local_expert_start: int

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.MSCCLPP


@dataclass(frozen=True)
class MSCCLPPCombineInputBase(ABC):
    """Fields shared by all MSCCL++ combine layouts."""

    hidden_states: torch.Tensor

    @property
    @abstractmethod
    def format(self) -> CombineInputFormat:
        pass


@dataclass(frozen=True)
class MSCCLPPCombineInput(MSCCLPPCombineInputBase):
    """High-throughput expert output consumed by handle-driven combine."""

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.MSCCLPP


@dataclass(frozen=True)
class MSCCLPPLLDispatchOutput(MSCCLPPDispatchOutputBase, ABC):
    """Base for MSCCL++ low-latency physical layouts."""


@dataclass(frozen=True)
class MSCCLPPExpertMajorLLDispatchOutput(MSCCLPPLLDispatchOutput):
    """Padded expert-major output consumed by the Triton runner.

    * ``hidden_states``      -> ``[num_local_experts, slots_per_expert, hidden]``
    * ``masked_m``           -> valid counts per local expert
    * ``expected_m``         -> average tokens per expert (GEMM size hint)
    """

    masked_m: torch.Tensor
    expected_m: int

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.MSCCLPP_LL_EXPERT_MAJOR


@dataclass(frozen=True)
class MSCCLPPRankMajorLLDispatchOutput(MSCCLPPLLDispatchOutput):
    """Fixed-capacity rank-major output consumed by FlashInfer CUTLASS."""

    topk_output: StandardTopKOutput
    expert_output_buffer: torch.Tensor

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.MSCCLPP_LL_RANK_MAJOR


@dataclass(frozen=True)
class MSCCLPPLLCombineInput(MSCCLPPCombineInputBase, ABC):
    """Base for MSCCL++ low-latency combine layouts."""


@dataclass(frozen=True)
class MSCCLPPExpertMajorLLCombineInput(MSCCLPPLLCombineInput):
    """Expert-major output consumed by handle-driven combine."""

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.MSCCLPP_LL_EXPERT_MAJOR


@dataclass(frozen=True)
class MSCCLPPRankMajorLLCombineInput(MSCCLPPLLCombineInput):
    """Rank-major registered output consumed by handle-driven combine."""

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.MSCCLPP_LL_RANK_MAJOR


class MSCCLPPDispatcher(BaseDispatcher):
    """MSCCL++ high-throughput token-major all-to-all dispatcher.

    ``MoECommunicator`` returns one row per source-token/destination-rank pair,
    with local expert ids and zeroed weights for non-local top-k slots.

    Combine is an **unweighted** cross-rank sum: ``intranode_combine`` plain-sums
    the hidden states and only reduces ``topk_weights`` into a separate (here
    unused) tensor, so it never re-applies the routing weights. The weights must
    already be folded in upstream -- the Triton runner does this during its
    within-rank top-k reduction -- so dispatch -> Triton -> combine applies each
    weight exactly once.

    Routing changes every MoE invocation, so this wrapper intentionally does
    not request the communicator's cached-layout path.
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
        num_sms: int = 20,
    ):
        super().__init__()

        try:
            from mscclpp import CommGroup
            from mscclpp.ep import (
                DispatchLayout,
                MoECommunicator,
                MoECommunicatorConfig,
                MoEMode,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "MSCCL++ EP is not available. Install an MSCCL++ build that "
                "provides `mscclpp.ep`, then select the `mscclpp` MoE a2a backend."
            ) from exc

        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank

        self._ep_group = CommGroup(torch_group=group)
        self.num_ranks = self._ep_group.nranks
        self._moe_comm = MoECommunicator(
            MoECommunicatorConfig(
                comm=self._ep_group,
                device=torch.cuda.current_device(),
                num_experts=num_experts,
                num_local_experts=num_local_experts,
                local_expert_start=self._ep_group.my_rank * num_local_experts,
                hidden_size=hidden_size,
                topk=router_topk,
                max_tokens_per_rank=num_max_dispatch_tokens_per_rank,
                mode=MoEMode.HIGH_THROUGHPUT,
                output_layout=DispatchLayout.TOKEN_MAJOR,
                num_sms=num_sms,
            )
        )
        if not self._moe_comm.is_available():
            raise RuntimeError("MSCCL++ EP high-throughput runtime is unavailable")

        self._combine_handle = None

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> MSCCLPPDispatchOutput:
        topk_ids = topk_output.topk_ids.to(torch.int64).contiguous()
        topk_weights = topk_output.topk_weights.to(torch.float32).contiguous()

        dispatch_out, handle = self._moe_comm.dispatch(
            hidden_states,
            topk_ids,
            topk_weights,
        )
        self._combine_handle = handle

        assert dispatch_out.topk_ids is not None
        assert dispatch_out.weights is not None
        num_tokens_per_expert = dispatch_out.layout.num_tokens_per_expert
        assert isinstance(num_tokens_per_expert, list)
        hidden_states_scale = (
            None if dispatch_out.quant is None else dispatch_out.quant.block_scales
        )
        local_topk_ids = dispatch_out.topk_ids
        global_topk_ids = torch.where(
            local_topk_ids >= 0,
            local_topk_ids + self._ep_group.my_rank * self.num_local_experts,
            local_topk_ids,
        )

        return MSCCLPPDispatchOutput(
            hidden_states=dispatch_out.tokens,
            hidden_states_scale=hidden_states_scale,
            topk_output=StandardTopKOutput(
                dispatch_out.weights,
                global_topk_ids,
                topk_output.router_logits,
            ),
            num_recv_tokens_per_expert=num_tokens_per_expert,
            local_expert_start=self._ep_group.my_rank * self.num_local_experts,
        )

    def combine(self, combine_input: MSCCLPPCombineInput) -> torch.Tensor:
        assert (
            self._combine_handle is not None
        ), "MSCCLPPDispatcher.combine called before dispatch"
        combined_x = self._moe_comm.combine(
            combine_input.hidden_states, self._combine_handle
        )

        self._combine_handle = None
        return combined_x


class _SharedLLRuntime(NamedTuple):
    """Heavy LL state shared by every MoE layer (see _get_shared_ll_runtime)."""

    ep_group: object
    moe_comm: object
    dispatch_output_buffer: Optional[torch.Tensor]
    expert_output_buffer: Optional[torch.Tensor]
    num_ranks: int


# One LL runtime per (group, geometry, capacity, layout) reused across all MoE layers.
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
    rank_major: bool,
) -> _SharedLLRuntime:
    key = (
        group,
        num_experts,
        num_local_experts,
        hidden_size,
        router_topk,
        num_max_dispatch_tokens_per_rank,
        rank_major,
    )
    cached = _SHARED_LL_RUNTIME.get(key)
    if cached is not None:
        return cached

    try:
        from mscclpp import CommGroup
        from mscclpp.ep import (
            CombineMode,
            DispatchLayout,
            MoECommunicator,
            MoECommunicatorConfig,
            MoEMode,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "MSCCL++ EP is not available. Install an MSCCL++ build that "
            "provides `mscclpp.ep`, then select the `mscclpp` MoE a2a backend."
        ) from exc

    ep_group = CommGroup(torch_group=group)
    num_ranks = ep_group.nranks

    if rank_major and not hasattr(DispatchLayout, "RANK_MAJOR"):
        raise RuntimeError(
            "MSCCL++ rank-major dispatch is unavailable; install the new "
            "mscclpp.ep API build before using FlashInfer CUTLASS."
        )
    output_layout = (
        DispatchLayout.RANK_MAJOR if rank_major else DispatchLayout.EXPERT_MAJOR
    )
    moe_comm = MoECommunicator(
        MoECommunicatorConfig(
            comm=ep_group,
            device=torch.cuda.current_device(),
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            local_expert_start=ep_group.my_rank * num_local_experts,
            hidden_size=hidden_size,
            topk=router_topk,
            max_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            mode=MoEMode.LOW_LATENCY,
            output_layout=output_layout,
            invalid_token_expert_id=num_experts,
            low_latency_combine_mode=CombineMode.RANK_LOCAL_REDUCE,
        )
    )
    if not moe_comm.is_available():
        raise RuntimeError("MSCCL++ EP low-latency runtime is unavailable")

    dispatch_output_buffer = None
    expert_output_buffer = None
    if rank_major:
        expert_output_buffer = moe_comm.get_expert_output_buffer()
    else:
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
        expert_output_buffer=expert_output_buffer,
        num_ranks=num_ranks,
    )
    _SHARED_LL_RUNTIME[key] = runtime
    return runtime


class MSCCLPPLLDispatcher(BaseDispatcher):
    """MSCCL++ EP low-latency (LL) all-to-all dispatcher.

    Uses the public ``mscclpp.ep.MoECommunicator`` low-latency API in one of two
    layouts:

    * :meth:`dispatch` runs ``MoECommunicator.dispatch`` to scatter each token to
      the ranks owning its top-k experts and returns an
    :class:`MSCCLPPLLDispatchOutput`. Triton uses padded expert-major output;
    FlashInfer CUTLASS uses the communicator's fixed rank-major buffers.
    * :meth:`combine` runs ``MoECommunicator.combine`` to reduce the per-slot
      expert outputs back to each source token.

    Expert-major combine applies routing weights from the handle, so Triton runs
    with unit weights. Rank-major CUTLASS applies weights while producing one
    rank-local partial per row; rank-local combine transports and reduces those
    partials without applying weights again.
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
        rank_major: bool = False,
    ):
        super().__init__()

        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        self.rank_major = rank_major

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
            rank_major,
        )
        self._ep_group = runtime.ep_group
        self.num_ranks = runtime.num_ranks
        self._moe_comm = runtime.moe_comm
        self._dispatch_output_buffer = runtime.dispatch_output_buffer
        self._expert_output_buffer = runtime.expert_output_buffer

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
        topk_ids = topk_output.topk_ids.to(torch.int64).contiguous()
        topk_weights = topk_output.topk_weights.to(torch.float32).contiguous()

        dispatch_out, handle = self._moe_comm.dispatch(
            hidden_states,
            topk_ids,
            topk_weights,
            output_buffer=self._dispatch_output_buffer,
        )
        self._combine_handle = handle

        hidden_states_scale = (
            None if dispatch_out.quant is None else dispatch_out.quant.block_scales
        )

        if self.rank_major:
            assert dispatch_out.topk_ids is not None
            assert dispatch_out.weights is not None
            assert dispatch_out.layout.num_tokens_per_rank is not None
            assert self._expert_output_buffer is not None
            return MSCCLPPRankMajorLLDispatchOutput(
                hidden_states=dispatch_out.tokens,
                hidden_states_scale=hidden_states_scale,
                topk_output=StandardTopKOutput(
                    dispatch_out.weights,
                    dispatch_out.topk_ids,
                    topk_output.router_logits,
                ),
                expert_output_buffer=self._expert_output_buffer,
            )

        masked_m = dispatch_out.layout.num_tokens_per_expert
        assert isinstance(masked_m, torch.Tensor)

        # Average tokens per expert (same hint DeepEP-LL passes to the masked
        # GEMM); ``world_size`` copies of each token are scattered across the
        # ``num_experts`` experts. Unused by the Triton runner but kept for
        # parity with the DeepEP-LL contract.
        expected_m = (
            hidden_states.shape[0] * self.num_ranks * self.router_topk
            + self.num_experts
            - 1
        ) // self.num_experts

        return MSCCLPPExpertMajorLLDispatchOutput(
            hidden_states=dispatch_out.tokens,
            hidden_states_scale=hidden_states_scale,
            masked_m=masked_m,
            expected_m=expected_m,
        )

    def combine(self, combine_input: MSCCLPPLLCombineInput) -> torch.Tensor:
        assert (
            self._combine_handle is not None
        ), "MSCCLPPLLDispatcher.combine called before dispatch"

        # The handle carries the layout-specific routing and scatter metadata.
        combined_x = self._moe_comm.combine(
            combine_input.hidden_states, self._combine_handle
        )

        self._combine_handle = None
        return combined_x
