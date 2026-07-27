# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Benchmark one MoE EP backend with FlashInfer CUTLASS MoE.

Run the script once per backend so only one MNNVL fabric workspace is resident:

1. ``--backend flashinfer``: FlashInfer ``MoeAlltoAll`` dispatch/combine.
2. ``--backend mscclpp``: MSCCL++ EP low-latency dispatch/combine.

Both paths use BF16 communication and BF16 expert compute with the same input,
routing decisions, routing weights, and rank-local expert weights for a fixed
seed. Both dispatchers produce fixed-capacity token buffers consumed through
SGLang's production FlashInfer CUTLASS fused-runner entry; only
dispatch/combine communication differs.

Example (DeepSeek-like shape on one 8-GPU node with MNNVL):

    torchrun --standalone --nproc-per-node=8 \
      benchmark/kernels/benchmark_flashinfer_vs_mscclpp_ep.py \
      --backend flashinfer --mnnvl
    torchrun --standalone --nproc-per-node=8 \
      benchmark/kernels/benchmark_flashinfer_vs_mscclpp_ep.py \
      --backend mscclpp --mnnvl

Small smoke run:

    torchrun --standalone --nproc-per-node=2 \
      benchmark/kernels/benchmark_flashinfer_vs_mscclpp_ep.py \
      --backend mscclpp \
      --tokens-per-rank 8 --hidden-size 4096 --intermediate-size 512 \
      --num-experts 8 --top-k 2 --warmup 2 --iters 5

By default, the FlashInfer workspace uses single-node CUDA virtual memory and
POSIX FD exchange over Unix sockets. Pass ``--mnnvl`` to use FlashInfer's
standard MNNVL fabric workspace. Multi-node runs require ``--mnnvl`` and all
participating GPUs must belong to the same NVIDIA Fabric cluster.
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

DTYPE = torch.bfloat16
MSCCLPP_LL_HIDDEN_SIZES = (4096, 6656, 7168, 8192, 8704, 9216)


@dataclass
class Inputs:
    hidden_states: torch.Tensor
    topk_ids_i64: torch.Tensor
    topk_ids_i32: torch.Tensor
    topk_weights: torch.Tensor


@dataclass
class SglangTopKOutput:
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor


@dataclass
class SglangDispatchOutput:
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor | None
    topk_output: SglangTopKOutput
    moe_output: torch.Tensor


@dataclass
class Timing:
    dispatch_us: float
    moe_us: float
    combine_us: float
    e2e_us: float


@dataclass
class TimelineTiming:
    dispatch_span_us: float
    dispatch_moe_gap_us: float
    moe_span_us: float
    moe_combine_gap_us: float
    combine_span_us: float
    e2e_span_us: float
    device_idle_us: float


@dataclass
class ExpertWeights:
    w13: torch.Tensor
    w2: torch.Tensor


@dataclass
class CapturedPipeline:
    graph: torch.cuda.CUDAGraph
    output: torch.Tensor
    operations_per_replay: int = 1
    start: torch.cuda.Event | None = None
    dispatch_end: torch.cuda.Event | None = None
    moe_end: torch.cuda.Event | None = None
    end: torch.cuda.Event | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("flashinfer", "mscclpp"),
        required=True,
        help="Benchmark exactly one backend so MNNVL fabric workspaces do not interfere.",
    )
    parser.add_argument(
        "--tokens-per-rank",
        type=str,
        default="1,4,16,32,64,128,256",
        help="Comma-separated local token counts to benchmark.",
    )
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Number of production-style single-step CUDA Graph replays measured.",
    )
    parser.add_argument(
        "--disable-torch-profiler",
        action="store_true",
        help="Report wall-clock E2E time per graph replay instead of per-stage CUDA kernel time.",
    )
    parser.add_argument(
        "--wall-clock-repeats",
        type=int,
        default=1,
        help="Number of wall-clock measurement rounds; median is reported.",
    )
    parser.add_argument(
        "--routing-samples",
        type=int,
        default=1,
        help="Number of independently routed single-step graphs cycled during wall-clock timing.",
    )
    parser.add_argument(
        "--graph-group-size",
        type=int,
        default=1,
        help=(
            "Number of independently routed full pipeline iterations captured in one "
            "CUDA Graph replay; wall-clock results are divided by this value."
        ),
    )
    parser.add_argument(
        "--report-stage-events",
        action="store_true",
        help="Record external CUDA events inside each full graph and report stage intervals.",
    )
    parser.add_argument(
        "--skip-moe-autotune",
        action="store_true",
        help="Use FlashInfer CUTLASS fallback tactics instead of exact-shape autotuning.",
    )
    parser.add_argument(
        "--mnnvl",
        action="store_true",
        help="Use FlashInfer's MNNVL fabric workspace; required for multi-node.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--weight-std",
        type=float,
        default=0.02,
        help="Standard deviation used for random BF16 expert weights.",
    )
    parser.add_argument(
        "--input-std",
        type=float,
        default=0.2,
        help="Standard deviation used for random BF16 input tokens.",
    )
    parser.add_argument(
        "--low-latency-num-blocks",
        type=int,
        default=130,
        help="MSCCL++ EP LL communication blocks (must be world_size + 2 .. 130).",
    )
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    return parser.parse_args()


def parse_token_counts(value: str) -> list[int]:
    token_counts = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not token_counts or any(count <= 0 for count in token_counts):
        raise ValueError("--tokens-per-rank must contain positive integers")
    return token_counts


def initialize_distributed() -> tuple[int, int, int, dist.ProcessGroup]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device("cuda", local_rank),
        )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size < 2:
        raise ValueError("This EP benchmark requires at least two ranks")

    cpu_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    return rank, world_size, local_rank, cpu_group


def validate_args(args: argparse.Namespace, world_size: int) -> None:
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    if world_size % local_world_size != 0:
        raise ValueError(
            f"world_size={world_size} must be divisible by "
            f"local_world_size={local_world_size}"
        )
    if world_size != local_world_size and not args.mnnvl:
        raise ValueError(
            "Multi-node runs require --mnnvl: "
            f"world_size={world_size}, local_world_size={local_world_size}"
        )
    if args.mnnvl:
        from flashinfer.comm.mnnvl import is_mnnvl_fabric_supported

        if not is_mnnvl_fabric_supported(torch.cuda.current_device()):
            raise RuntimeError(
                "--mnnvl requires NVIDIA Fabric support on every participating GPU"
            )
    if args.num_experts % world_size != 0:
        raise ValueError(
            f"num_experts={args.num_experts} must be divisible by world_size={world_size}"
        )
    if not 0 < args.top_k <= min(9, args.num_experts):
        raise ValueError("--top-k must be in [1, min(9, num_experts)]")
    if args.backend == "mscclpp" and args.hidden_size not in MSCCLPP_LL_HIDDEN_SIZES:
        raise ValueError(
            f"--hidden-size must be one of {MSCCLPP_LL_HIDDEN_SIZES} "
            "for MSCCL++ EP low-latency"
        )
    if args.intermediate_size <= 0:
        raise ValueError("--intermediate-size must be positive")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be non-negative and --iters must be positive")
    if args.wall_clock_repeats <= 0:
        raise ValueError("--wall-clock-repeats must be positive")
    if args.graph_group_size <= 0:
        raise ValueError("--graph-group-size must be positive")
    if not 0 < args.routing_samples <= args.iters:
        raise ValueError("--routing-samples must be in [1, iters]")
    if not args.disable_torch_profiler and args.wall_clock_repeats != 1:
        raise ValueError("--wall-clock-repeats requires --disable-torch-profiler")
    if not args.disable_torch_profiler and args.routing_samples != 1:
        raise ValueError("--routing-samples requires --disable-torch-profiler")
    if args.report_stage_events and not args.disable_torch_profiler:
        raise ValueError("--report-stage-events requires --disable-torch-profiler")
    if args.report_stage_events and args.iters % args.routing_samples != 0:
        raise ValueError(
            "--report-stage-events requires iters divisible by routing-samples"
        )
    if args.graph_group_size > 1:
        if not args.disable_torch_profiler:
            raise ValueError("--graph-group-size > 1 requires --disable-torch-profiler")
        if args.routing_samples != 1:
            raise ValueError(
                "--graph-group-size generates its own independent routing samples; "
                "leave --routing-samples at 1"
            )
        if args.report_stage_events:
            raise ValueError(
                "--report-stage-events is not supported with --graph-group-size > 1"
            )
    if args.backend == "mscclpp":
        min_blocks = world_size + 2
        if not min_blocks <= args.low_latency_num_blocks <= 130:
            raise ValueError(
                "--low-latency-num-blocks must be between "
                f"{min_blocks} and 130 for world_size={world_size}"
            )


def make_local_weights(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
) -> ExpertWeights:
    num_local_experts = args.num_experts // world_size
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 100_003 * rank)

    w13 = torch.empty(
        (
            num_local_experts,
            2 * args.intermediate_size,
            args.hidden_size,
        ),
        dtype=DTYPE,
        device=device,
    )
    w13.normal_(mean=0.0, std=args.weight_std, generator=generator)
    w2 = torch.empty(
        (
            num_local_experts,
            args.hidden_size,
            args.intermediate_size,
        ),
        dtype=DTYPE,
        device=device,
    )
    w2.normal_(mean=0.0, std=args.weight_std, generator=generator)
    return ExpertWeights(w13=w13, w2=w2)


def make_inputs(
    args: argparse.Namespace,
    rank: int,
    num_tokens: int,
    device: torch.device,
    sample_index: int = 0,
) -> Inputs:
    generator = torch.Generator(device=device)
    generator.manual_seed(
        args.seed + 1_000_003 * rank + 10_000_019 * sample_index + num_tokens
    )

    hidden_states = torch.empty(
        (num_tokens, args.hidden_size), dtype=DTYPE, device=device
    )
    hidden_states.normal_(mean=0.0, std=args.input_std, generator=generator)

    router_logits = torch.randn(
        (num_tokens, args.num_experts),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    selected_logits, topk_ids_i64 = torch.topk(
        router_logits, k=args.top_k, dim=-1, sorted=True
    )
    topk_weights = torch.softmax(selected_logits, dim=-1, dtype=torch.float32)
    topk_ids_i64 = topk_ids_i64.contiguous()
    return Inputs(
        hidden_states=hidden_states,
        topk_ids_i64=topk_ids_i64,
        topk_ids_i32=topk_ids_i64.to(torch.int32),
        topk_weights=topk_weights.contiguous(),
    )


class SglangCutlassMoe:
    def __init__(
        self,
        weights: ExpertWeights,
        rank: int,
        world_size: int,
        max_dispatched_tokens: int,
        top_k: int,
        enable_alltoall: bool,
    ) -> None:
        # Production model loading initializes the quantization registry before the MoE runner.
        import sglang.srt.layers.quantization  # noqa: F401

        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
            FlashInferCutlassMoeQuantInfo,
            _run_flashinfer_cutlass,
        )

        self.run_fused_experts = _run_flashinfer_cutlass
        if weights.w13.dtype != DTYPE or weights.w2.dtype != DTYPE:
            raise TypeError("FlashInfer CUTLASS BF16 MoE requires BF16 expert weights")
        if not enable_alltoall:
            raise ValueError("SGLang FlashInfer EP workflow requires enable_alltoall")
        self.quant_info = FlashInferCutlassMoeQuantInfo(
            quant_type="bf16",
            w13_weight=weights.w13,
            w2_weight=weights.w2,
            output_dtype=DTYPE,
            moe_tp_size=1,
            moe_tp_rank=0,
            moe_ep_size=world_size,
            moe_ep_rank=rank,
            apply_routed_scaling_factor=True,
        )
        self.runner_config = MoeRunnerConfig(
            num_experts=weights.w13.shape[0] * world_size,
            num_local_experts=weights.w13.shape[0],
            hidden_size=weights.w13.shape[-1],
            intermediate_size_per_partition=weights.w2.shape[-1],
            top_k=top_k,
            params_dtype=DTYPE,
            activation="silu",
            is_gated=True,
            apply_router_weight_on_input=False,
            inplace=True,
        )
        self.max_dispatched_tokens = max_dispatched_tokens
        self.router_logits = torch.empty(
            0, dtype=torch.float32, device=weights.w13.device
        )

    def __call__(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.dtype != DTYPE or output.dtype != DTYPE:
            raise TypeError("FlashInfer CUTLASS BF16 MoE requires BF16 input/output")
        if hidden_states.shape[0] > self.max_dispatched_tokens:
            raise ValueError(
                "dispatched rows exceed the configured SGLang MoE capacity"
            )
        dispatch_output = SglangDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=SglangTopKOutput(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                router_logits=self.router_logits,
            ),
            moe_output=output,
        )
        return self.run_fused_experts(
            dispatch_output=dispatch_output,
            quant_info=self.quant_info,
            runner_config=self.runner_config,
            output=output,
            enable_alltoall=True,
        )


def make_single_node_moe_alltoall(
    *,
    mapping: Any,
    max_num_tokens: int,
    top_k: int,
    num_experts: int,
    workspace_size_per_rank: int,
    comm_backend: Any,
) -> Any:
    from flashinfer.comm import MoeAlltoAll, moe_a2a_initialize
    from flashinfer.comm.dlpack_utils import pack_strided_memory
    from flashinfer.comm.mnnvl import SymmDeviceMemory

    class SingleNodeMoeAlltoAll(MoeAlltoAll):
        _WORKSPACE_CACHE: dict[tuple[int, int, int, int], dict] = {}

        @classmethod
        def get_workspace(
            cls,
            workspace_size_per_rank: int,
            ep_rank: int,
            ep_size: int,
            max_num_tokens: int,
            mapping: Any,
        ) -> dict:
            del mapping
            key = (workspace_size_per_rank, ep_rank, ep_size, max_num_tokens)
            if key not in cls._WORKSPACE_CACHE:
                shared_memory = SymmDeviceMemory(
                    buf_size=workspace_size_per_rank,
                    group_size=ep_size,
                    group_rank=ep_rank,
                    device_idx=torch.cuda.current_device(),
                    comm_backend_for_handle_transfer=comm_backend,
                    enable_multicast=False,
                    allocate_signal_pads=False,
                )
                workspace = pack_strided_memory(
                    ptr=int(shared_memory.uc_base_ptr),
                    segment_size=workspace_size_per_rank,
                    segment_stride=shared_memory.allocation_size,
                    num_segments=ep_size,
                    dtype=torch.uint8,
                    dev_id=torch.cuda.current_device(),
                )
                metainfo = moe_a2a_initialize(
                    workspace,
                    ep_rank,
                    ep_size,
                    max_num_tokens,
                )
                cls._WORKSPACE_CACHE[key] = {
                    "workspace_size_per_rank": workspace_size_per_rank,
                    "max_num_tokens": max_num_tokens,
                    "ep_rank": ep_rank,
                    "ep_size": ep_size,
                    "mnnvl_mem": shared_memory,
                    "workspace": workspace,
                    "metainfo": metainfo,
                }
            return cls._WORKSPACE_CACHE[key]

    return SingleNodeMoeAlltoAll(
        mapping=mapping,
        max_num_tokens=max_num_tokens,
        top_k=top_k,
        num_experts=num_experts,
        workspace_size_per_rank=workspace_size_per_rank,
    )


def make_torch_distributed_comm_backend(group: dist.ProcessGroup) -> Any:
    from flashinfer.comm.mnnvl import CommBackend

    class TorchDistributedCommBackend(CommBackend):
        def Get_rank(self) -> int:
            return group.rank()

        def Get_size(self) -> int:
            return group.size()

        def allgather(self, data: int) -> list[Any]:
            gathered = [None] * self.Get_size()
            dist.all_gather_object(gathered, data, group=group)
            return gathered

        def bcast(self, data: Any, root: int = 0) -> Any:
            objects = [data]
            dist.broadcast_object_list(objects, src=root, group=group)
            return objects[0]

        def Split(self, color: int, key: int) -> TorchDistributedCommBackend:
            del color, key
            return self

        def barrier(self) -> None:
            dist.barrier(group=group)

    return TorchDistributedCommBackend()


class FlashInferPipeline:
    name = "flashinfer_a2a"

    def __init__(
        self,
        args: argparse.Namespace,
        rank: int,
        world_size: int,
        num_tokens: int,
        weights: ExpertWeights,
        group: dist.ProcessGroup,
    ) -> None:
        from flashinfer.comm import MoeAlltoAll, moe_a2a_get_workspace_size_per_rank
        from flashinfer.comm.mapping import Mapping
        from flashinfer.comm.mnnvl import MnnvlConfig

        dispatch_bytes_per_token = (
            args.hidden_size * DTYPE.itemsize
            + args.top_k * torch.int32.itemsize
            + args.top_k * torch.float32.itemsize
        )
        combine_bytes_per_token = args.hidden_size * DTYPE.itemsize
        workspace_size = moe_a2a_get_workspace_size_per_rank(
            ep_size=world_size,
            max_num_tokens=num_tokens,
            total_dispatch_payload_size_per_token=dispatch_bytes_per_token,
            combine_payload_size_per_token=combine_bytes_per_token,
        )
        mapping = Mapping(
            rank=rank,
            tp_size=world_size,
            moe_ep_size=world_size,
            world_size=world_size,
            gpus_per_node=torch.cuda.device_count(),
            pp_size=1,
            cp_size=1,
        )
        comm_backend = make_torch_distributed_comm_backend(group)
        if args.mnnvl:
            self.name = "flashinfer_a2a_mnnvl"
            self.a2a = MoeAlltoAll(
                mapping=mapping,
                max_num_tokens=num_tokens,
                top_k=args.top_k,
                num_experts=args.num_experts,
                workspace_size_per_rank=workspace_size,
                mnnvl_config=MnnvlConfig(comm_backend=comm_backend),
            )
        else:
            self.a2a = make_single_node_moe_alltoall(
                mapping=mapping,
                max_num_tokens=num_tokens,
                top_k=args.top_k,
                num_experts=args.num_experts,
                workspace_size_per_rank=workspace_size,
                comm_backend=comm_backend,
            )
        self.num_tokens = num_tokens
        self.hidden_size = args.hidden_size
        self.num_experts = args.num_experts
        self.world_size = world_size
        self.moe = SglangCutlassMoe(
            weights=weights,
            rank=rank,
            world_size=world_size,
            max_dispatched_tokens=world_size * num_tokens,
            top_k=args.top_k,
            # FlashInfer's dispatcher uses the runner's fixed rank-major A2A mode.
            enable_alltoall=True,
        )

    def dispatch(self, inputs: Inputs) -> tuple[torch.Tensor, ...]:
        recv_tensors = self.a2a.dispatch(
            inputs.topk_ids_i32,
            [
                inputs.hidden_states,
                inputs.topk_ids_i32,
                inputs.topk_weights,
            ],
            self.num_tokens,
            invalid_token_expert_id=self.num_experts,
            expert_id_payload_index=1,
        )
        return tuple(recv_tensors)

    def run_moe(self, state: tuple[torch.Tensor, ...]) -> torch.Tensor:
        recv_hidden, recv_topk_ids, recv_topk_weights = state
        output = self.a2a.get_combine_payload_tensor_in_workspace(
            self.num_tokens, self.hidden_size, DTYPE
        ).view(-1, self.hidden_size)
        return self.moe(
            hidden_states=recv_hidden.view(-1, self.hidden_size),
            topk_ids=recv_topk_ids.view(-1, recv_topk_ids.shape[-1]),
            topk_weights=recv_topk_weights.view(-1, recv_topk_weights.shape[-1]),
            output=output,
        )

    def combine(
        self, state: tuple[torch.Tensor, ...], expert_output: torch.Tensor
    ) -> torch.Tensor:
        del state
        return self.a2a.combine(
            expert_output.view(self.world_size, self.num_tokens, self.hidden_size),
            self.num_tokens,
            payload_in_workspace=True,
        )

    def close(self) -> None:
        self.a2a._reset_workspace()


class MscclppPipeline:
    name = "mscclpp_ep_ll"

    def __init__(
        self,
        args: argparse.Namespace,
        rank: int,
        world_size: int,
        num_tokens: int,
        weights: ExpertWeights,
        comm_group: Any,
        device: torch.device,
    ) -> None:
        from mscclpp.ep import (
            CombineMode,
            DispatchLayout,
            MoECommunicator,
            MoECommunicatorConfig,
            MoEMode,
        )

        self.world_size = world_size
        self.hidden_size = args.hidden_size
        self.num_experts = args.num_experts
        self.num_tokens = num_tokens
        self.name = "mscclpp_ep_ll_rank_major"
        num_local_experts = args.num_experts // world_size
        self.communicator = MoECommunicator(
            MoECommunicatorConfig(
                comm=comm_group,
                device=device,
                num_experts=args.num_experts,
                num_local_experts=num_local_experts,
                local_expert_start=rank * num_local_experts,
                hidden_size=args.hidden_size,
                topk=args.top_k,
                max_tokens_per_rank=num_tokens,
                mode=MoEMode.LOW_LATENCY,
                output_layout=DispatchLayout.RANK_MAJOR,
                invalid_token_expert_id=args.num_experts,
                low_latency_num_blocks=args.low_latency_num_blocks,
                low_latency_combine_mode=CombineMode.RANK_LOCAL_REDUCE,
            )
        )
        if not self.communicator.is_available():
            raise RuntimeError("MSCCL++ EP low-latency runtime is unavailable")

        capacity = world_size * num_tokens
        self.dispatch_output = None
        self.expert_output = self.communicator.get_expert_output_buffer()
        self.combine_output = torch.empty(
            (num_tokens, args.hidden_size), dtype=DTYPE, device=device
        )
        self.moe = SglangCutlassMoe(
            weights=weights,
            rank=rank,
            world_size=world_size,
            max_dispatched_tokens=capacity,
            top_k=args.top_k,
            enable_alltoall=True,
        )

    def dispatch(self, inputs: Inputs) -> tuple[Any, Any]:
        return self.communicator.dispatch(
            inputs.hidden_states,
            inputs.topk_ids_i64,
            inputs.topk_weights,
            output_buffer=self.dispatch_output,
        )

    def run_moe(self, state: tuple[Any, Any]) -> torch.Tensor:
        dispatch_output, _ = state
        if dispatch_output.topk_ids is None or dispatch_output.weights is None:
            raise RuntimeError("MSCCL++ RANK_MAJOR dispatch metadata is missing")
        return self.moe(
            hidden_states=dispatch_output.tokens,
            topk_ids=dispatch_output.topk_ids,
            topk_weights=dispatch_output.weights,
            output=self.expert_output,
        )

    def combine(
        self, state: tuple[Any, Any], expert_output: torch.Tensor
    ) -> torch.Tensor:
        _, handle = state
        return self.communicator.combine(
            expert_output,
            handle,
            out=self.combine_output,
        )


def run_once(pipeline: Any, inputs: Inputs) -> torch.Tensor:
    state = pipeline.dispatch(inputs)
    expert_output = pipeline.run_moe(state)
    return pipeline.combine(state, expert_output)


def autotune_moe(
    pipeline: Any,
    inputs: Inputs,
    dispatched_tokens: int,
    sync_group: dist.ProcessGroup,
) -> None:
    from flashinfer.autotuner import autotune

    torch.cuda.synchronize()
    dist.barrier(group=sync_group)
    with autotune(True, tuning_buckets=(dispatched_tokens,)):
        run_once(pipeline, inputs)
    torch.cuda.synchronize()
    dist.barrier(group=sync_group)


def reduce_max(values: list[float], device: torch.device) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.cpu().tolist()


def reduce_mean(values: list[float], device: torch.device) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor.cpu().tolist()


def synchronize_stream_and_ranks(
    mscclpp_comm_group: Any | None,
    cpu_group: dist.ProcessGroup,
) -> None:
    torch.cuda.current_stream().synchronize()
    if mscclpp_comm_group is not None:
        mscclpp_comm_group.barrier()
    else:
        dist.barrier(group=cpu_group)


def assert_outputs_close(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    rtol: float,
    atol: float,
    device: torch.device,
    label: str,
) -> tuple[float, float]:
    diff = (reference.float() - candidate.float()).abs()
    max_abs = diff.max()
    max_rel = (diff / reference.float().abs().clamp_min(1e-6)).max()
    is_close = torch.tensor(
        [
            int(
                torch.allclose(
                    reference,
                    candidate,
                    rtol=rtol,
                    atol=atol,
                )
            )
        ],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(is_close, op=dist.ReduceOp.MIN)
    max_abs_value, max_rel_value = reduce_max([max_abs.item(), max_rel.item()], device)
    if not bool(is_close.item()):
        raise AssertionError(
            f"{label} outputs differ: "
            f"max_abs={max_abs_value:.6g}, max_rel={max_rel_value:.6g}, "
            f"rtol={rtol}, atol={atol}"
        )
    return max_abs_value, max_rel_value


def capture_pipeline(
    pipeline: Any,
    inputs: Inputs,
    warmup: int,
    sync_group: dist.ProcessGroup,
    record_stage_events: bool = False,
) -> CapturedPipeline:
    for _ in range(warmup):
        run_once(pipeline, inputs)
    torch.cuda.synchronize()
    dist.barrier(group=sync_group)

    events = (
        [torch.cuda.Event(enable_timing=True, external=True) for _ in range(4)]
        if record_stage_events
        else [None] * 4
    )
    start, dispatch_end, moe_end, end = events
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        if start is not None:
            start.record()
        state = pipeline.dispatch(inputs)
        if dispatch_end is not None:
            dispatch_end.record()
        expert_output = pipeline.run_moe(state)
        if moe_end is not None:
            moe_end.record()
        output = pipeline.combine(state, expert_output)
        if end is not None:
            end.record()

    torch.cuda.synchronize()
    dist.barrier(group=sync_group)
    return CapturedPipeline(
        graph=graph,
        output=output,
        start=start,
        dispatch_end=dispatch_end,
        moe_end=moe_end,
        end=end,
    )


def capture_grouped_pipeline(
    pipeline: Any,
    input_samples: list[Inputs],
    warmup: int,
    sync_group: dist.ProcessGroup,
) -> CapturedPipeline:
    if len(input_samples) <= 1:
        raise ValueError("grouped pipeline capture requires at least two input samples")
    for index in range(warmup):
        run_once(pipeline, input_samples[index % len(input_samples)])
    torch.cuda.synchronize()
    dist.barrier(group=sync_group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for inputs in input_samples:
            output = run_once(pipeline, inputs)

    torch.cuda.synchronize()
    dist.barrier(group=sync_group)
    return CapturedPipeline(
        graph=graph,
        output=output,
        operations_per_replay=len(input_samples),
    )


def classify_profiled_kernel(name: str) -> str | None:
    normalized = name.lower()
    if "moea2a" in normalized:
        if "dispatch" in normalized or "sanitizeexpertids" in normalized:
            return "dispatch"
        if "combine" in normalized:
            return "combine"
    if "mscclpp::ep::low_latency" in normalized:
        if "dispatchkernel" in normalized:
            return "dispatch"
        if "combinekernel" in normalized or "combinetmaloadkernel" in normalized:
            return "combine"
    return None


def summarize_profiled_replays(
    profiler: torch.profiler.profile,
    expected_replays: int,
) -> tuple[Timing, TimelineTiming]:
    events = profiler.events()
    launch_ids = [event.id for event in events if event.name == "cudaGraphLaunch"]
    if len(launch_ids) != expected_replays:
        raise RuntimeError(
            f"Torch Profiler observed {len(launch_ids)} CUDA Graph launches; "
            f"expected {expected_replays}"
        )

    launch_id_set = set(launch_ids)
    kernels_by_launch: dict[int, list[Any]] = {
        launch_id: [] for launch_id in launch_ids
    }
    for event in events:
        if (
            event.device_type == torch.autograd.DeviceType.CUDA
            and event.id in launch_id_set
        ):
            kernels_by_launch[event.id].append(event)

    replay_times: list[Timing] = []
    replay_timelines: list[TimelineTiming] = []
    for launch_id in launch_ids:
        stage_times = {"dispatch": 0.0, "moe": 0.0, "combine": 0.0}
        kernels = sorted(
            kernels_by_launch[launch_id],
            key=lambda event: (event.time_range.start, event.time_range.end),
        )
        markers = [classify_profiled_kernel(event.name) for event in kernels]
        dispatch_indices = [
            index for index, marker in enumerate(markers) if marker == "dispatch"
        ]
        combine_indices = [
            index for index, marker in enumerate(markers) if marker == "combine"
        ]
        if not dispatch_indices or not combine_indices:
            kernel_names = "\n".join(f"  {event.name}" for event in kernels)
            raise RuntimeError(
                "Torch Profiler could not identify dispatch/combine boundaries "
                f"for a graph replay:\n{kernel_names}"
            )
        dispatch_end = max(dispatch_indices)
        combine_begin = min(combine_indices)
        if dispatch_end + 1 >= combine_begin:
            raise RuntimeError("profiled dispatch, MoE, and combine kernels overlap")
        stage_kernels = {
            "dispatch": kernels[: dispatch_end + 1],
            "moe": kernels[dispatch_end + 1 : combine_begin],
            "combine": kernels[combine_begin:],
        }
        if not stage_kernels["moe"]:
            raise RuntimeError("profiled graph contains no MoE kernels")
        for index, event in enumerate(kernels):
            stage = (
                "dispatch"
                if index <= dispatch_end
                else ("combine" if index >= combine_begin else "moe")
            )
            stage_times[stage] += float(event.self_device_time_total)
        replay_times.append(
            Timing(
                dispatch_us=stage_times["dispatch"],
                moe_us=stage_times["moe"],
                combine_us=stage_times["combine"],
                e2e_us=sum(stage_times.values()),
            )
        )
        dispatch_start = stage_kernels["dispatch"][0].time_range.start
        dispatch_stop = max(
            event.time_range.end for event in stage_kernels["dispatch"]
        )
        moe_start = min(event.time_range.start for event in stage_kernels["moe"])
        moe_stop = max(event.time_range.end for event in stage_kernels["moe"])
        combine_start = min(
            event.time_range.start for event in stage_kernels["combine"]
        )
        combine_stop = max(
            event.time_range.end for event in stage_kernels["combine"]
        )
        busy_intervals = sorted(
            (event.time_range.start, event.time_range.end) for event in kernels
        )
        busy_us = 0.0
        busy_start, busy_stop = busy_intervals[0]
        for interval_start, interval_stop in busy_intervals[1:]:
            if interval_start > busy_stop:
                busy_us += busy_stop - busy_start
                busy_start, busy_stop = interval_start, interval_stop
            else:
                busy_stop = max(busy_stop, interval_stop)
        busy_us += busy_stop - busy_start
        replay_timelines.append(
            TimelineTiming(
                dispatch_span_us=dispatch_stop - dispatch_start,
                dispatch_moe_gap_us=moe_start - dispatch_stop,
                moe_span_us=moe_stop - moe_start,
                moe_combine_gap_us=combine_start - moe_stop,
                combine_span_us=combine_stop - combine_start,
                e2e_span_us=combine_stop - dispatch_start,
                device_idle_us=combine_stop - dispatch_start - busy_us,
            )
        )

    count = len(replay_times)
    return (
        Timing(
            dispatch_us=sum(item.dispatch_us for item in replay_times) / count,
            moe_us=sum(item.moe_us for item in replay_times) / count,
            combine_us=sum(item.combine_us for item in replay_times) / count,
            e2e_us=sum(item.e2e_us for item in replay_times) / count,
        ),
        TimelineTiming(
            *(
                sum(getattr(item, field) for item in replay_timelines) / count
                for field in TimelineTiming.__dataclass_fields__
            )
        ),
    )


def profile_pipeline_replays(
    captured: CapturedPipeline,
    iters: int,
    device: torch.device,
    sync_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, Timing]:
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    dist.barrier(group=sync_group)

    # Match SGLang decode: one complete forward is captured, then each decode
    # step replays that graph once. Prime before collecting profiler activity.
    captured.graph.replay()
    torch.cuda.synchronize()
    dist.barrier(group=sync_group)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        acc_events=True,
    ) as profiler:
        # Kineto initialization time varies by process. Align after every rank's
        # profiler is active so the first communication kernel does not measure
        # another rank's profiler startup delay.
        dist.barrier(group=sync_group)
        for _ in range(iters):
            captured.graph.replay()
        torch.cuda.synchronize()

    output = captured.output.clone()
    torch.cuda.synchronize()
    dist.barrier(group=sync_group)

    local_timing, local_timeline = summarize_profiled_replays(profiler, iters)
    dispatch_us, moe_us, combine_us, e2e_us = reduce_mean(
        [
            local_timing.dispatch_us,
            local_timing.moe_us,
            local_timing.combine_us,
            local_timing.e2e_us,
        ],
        device,
    )
    timeline_mean = reduce_mean(
        [
            local_timeline.dispatch_span_us,
            local_timeline.dispatch_moe_gap_us,
            local_timeline.moe_span_us,
            local_timeline.moe_combine_gap_us,
            local_timeline.combine_span_us,
            local_timeline.e2e_span_us,
            local_timeline.device_idle_us,
        ],
        device,
    )
    timeline_max = reduce_max(
        [
            local_timeline.dispatch_span_us,
            local_timeline.dispatch_moe_gap_us,
            local_timeline.moe_span_us,
            local_timeline.moe_combine_gap_us,
            local_timeline.combine_span_us,
            local_timeline.e2e_span_us,
            local_timeline.device_idle_us,
        ],
        device,
    )
    if dist.get_rank() == 0:
        print(
            "Torch Profiler CUDA kernel time per production-style graph replay: "
            f"dispatch={dispatch_us:.1f}us, MoE={moe_us:.1f}us, "
            f"combine={combine_us:.1f}us, E2E={e2e_us:.1f}us",
            flush=True,
        )
        print(
            "Torch Profiler CUDA timeline mean/max per graph replay: "
            f"dispatch_span={timeline_mean[0]:.1f}/{timeline_max[0]:.1f}us, "
            f"dispatch_to_MoE_gap={timeline_mean[1]:.1f}/{timeline_max[1]:.1f}us, "
            f"MoE_span={timeline_mean[2]:.1f}/{timeline_max[2]:.1f}us, "
            f"MoE_to_combine_gap={timeline_mean[3]:.1f}/{timeline_max[3]:.1f}us, "
            f"combine_span={timeline_mean[4]:.1f}/{timeline_max[4]:.1f}us, "
            f"device_graph_span={timeline_mean[5]:.1f}/{timeline_max[5]:.1f}us, "
            f"device_idle={timeline_mean[6]:.1f}/{timeline_max[6]:.1f}us",
            flush=True,
        )
    return (
        output,
        Timing(
            dispatch_us=dispatch_us,
            moe_us=moe_us,
            combine_us=combine_us,
            e2e_us=e2e_us,
        ),
    )


def time_pipeline_replays(
    captured: CapturedPipeline | list[CapturedPipeline],
    iters: int,
    repeats: int,
    device: torch.device,
    sync_group: dist.ProcessGroup,
    report_stage_events: bool = False,
) -> tuple[torch.Tensor, float]:
    captured_graphs = captured if isinstance(captured, list) else [captured]
    operations_per_replay = captured_graphs[0].operations_per_replay
    if any(
        item.operations_per_replay != operations_per_replay
        for item in captured_graphs
    ):
        raise ValueError("all cycled CUDA Graphs must contain the same operation count")
    torch.cuda.synchronize()
    dist.barrier(group=sync_group)

    for item in captured_graphs:
        item.graph.replay()
    torch.cuda.synchronize()
    dist.barrier(group=sync_group)

    round_us = []
    stage_rounds = []
    for _ in range(repeats):
        dist.barrier(group=sync_group)
        start = time.perf_counter()
        for index in range(iters):
            captured_graphs[index % len(captured_graphs)].graph.replay()
        torch.cuda.synchronize()
        elapsed_us = (
            (time.perf_counter() - start)
            * 1_000_000
            / iters
            / operations_per_replay
        )
        (elapsed_us,) = reduce_mean([elapsed_us], device)
        round_us.append(elapsed_us)
        if report_stage_events:
            local_stage_timings = [
                Timing(
                    dispatch_us=item.start.elapsed_time(item.dispatch_end) * 1e3,
                    moe_us=item.dispatch_end.elapsed_time(item.moe_end) * 1e3,
                    combine_us=item.moe_end.elapsed_time(item.end) * 1e3,
                    e2e_us=item.start.elapsed_time(item.end) * 1e3,
                )
                for item in captured_graphs
            ]
            count = len(local_stage_timings)
            local_mean = Timing(
                dispatch_us=sum(item.dispatch_us for item in local_stage_timings)
                / count,
                moe_us=sum(item.moe_us for item in local_stage_timings) / count,
                combine_us=sum(item.combine_us for item in local_stage_timings) / count,
                e2e_us=sum(item.e2e_us for item in local_stage_timings) / count,
            )
            stage_rounds.append(
                Timing(
                    *reduce_mean(
                        [
                            local_mean.dispatch_us,
                            local_mean.moe_us,
                            local_mean.combine_us,
                            local_mean.e2e_us,
                        ],
                        device,
                    )
                )
            )

    output = captured_graphs[(iters - 1) % len(captured_graphs)].output.clone()
    torch.cuda.synchronize()
    dist.barrier(group=sync_group)
    e2e_us = sorted(round_us)[len(round_us) // 2]
    if dist.get_rank() == 0:
        if operations_per_replay > 1:
            print(
                "Grouped CUDA Graph wall-clock E2E: "
                f"graph_replays={iters}, iterations_per_replay={operations_per_replay}, "
                f"total_iterations={iters * operations_per_replay}, "
                f"per_replay={e2e_us * operations_per_replay:.1f}us, "
                f"per_iteration_median={e2e_us:.1f}us "
                f"min={min(round_us):.1f}us max={max(round_us):.1f}us rounds="
                f"{','.join(f'{value:.1f}' for value in round_us)}",
                flush=True,
            )
        else:
            print(
                "Wall-clock E2E time per production-style graph replay: "
                f"median={e2e_us:.1f}us min={min(round_us):.1f}us "
                f"max={max(round_us):.1f}us rounds="
                f"{','.join(f'{value:.1f}' for value in round_us)}",
                flush=True,
            )
        if stage_rounds:
            median_round = sorted(
                range(len(round_us)), key=lambda index: round_us[index]
            )[len(round_us) // 2]
            stage = stage_rounds[median_round]
            print(
                "Embedded CUDA Graph stage intervals at median wall-clock round: "
                f"dispatch={stage.dispatch_us:.1f}us, MoE={stage.moe_us:.1f}us, "
                f"combine={stage.combine_us:.1f}us, device_graph={stage.e2e_us:.1f}us, "
                f"host_minus_device={e2e_us - stage.e2e_us:.1f}us",
                flush=True,
            )
    return output, e2e_us


def format_markdown_table(rows: list[list[str]]) -> str:
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]

    def format_row(row: list[str]) -> str:
        return (
            "| "
            + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
            + " |"
        )

    header = format_row(rows[0])
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([header, separator, *(format_row(row) for row in rows[1:])])


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    token_counts = parse_token_counts(args.tokens_per_rank)
    rank, world_size, _, cpu_group = initialize_distributed()
    validate_args(args, world_size)
    device = torch.device("cuda", torch.cuda.current_device())
    mscclpp_comm_group = None

    try:
        import flashinfer

        mscclpp = None
        if args.backend == "mscclpp":
            import mscclpp as imported_mscclpp

            mscclpp = imported_mscclpp
            mscclpp_comm_group = mscclpp.CommGroup(
                torch_group=cpu_group,
                rank=rank,
                size=world_size,
            )
        weights = make_local_weights(args, rank, world_size, device)
        torch.cuda.synchronize()
        dist.barrier()

        if rank == 0:
            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
            num_nodes = world_size // local_world_size
            weight_gib = (weights.w13.nbytes + weights.w2.nbytes) / (1024**3)
            print(
                "Configuration: "
                f"world_size={world_size}, nodes={num_nodes}, "
                f"backend={args.backend}, mnnvl={args.mnnvl}, dtype=bf16, "
                f"hidden={args.hidden_size}, intermediate={args.intermediate_size}, "
                f"experts={args.num_experts}, top_k={args.top_k}, "
                f"tokens_per_rank={token_counts}, warmup={args.warmup}, "
                f"graph_replays={args.iters}, routing_samples={args.routing_samples}, "
                f"graph_group_size={args.graph_group_size}, "
                f"torch_profiler={not args.disable_torch_profiler}, "
                f"moe_autotune={not args.skip_moe_autotune}"
            )
            versions = f"flashinfer={flashinfer.__version__}"
            if mscclpp is not None:
                versions += f", mscclpp={mscclpp.__version__}"
            print(f"Versions: {versions}")
            print(f"Rank-local expert weights: {weight_gib:.2f} GiB")
            if args.disable_torch_profiler:
                print("Wall-clock E2E time is averaged across ranks.\n")
            else:
                print("CUDA kernel time is averaged across per-rank profiler means.\n")
            if args.graph_group_size > 1:
                print(
                    f"One CUDA Graph contains {args.graph_group_size} independent-routing "
                    "dispatch->SGLang MoE runner->combine iterations. After one "
                    "unmeasured graph replay, repeated graph replays are timed and "
                    "reported per iteration.\n"
                )
            else:
                print(
                    "One complete dispatch->SGLang MoE runner->combine step is "
                    "captured, matching SGLang decode CUDA Graph boundaries. "
                    "After one unmeasured prime, repeated graph replays are timed.\n"
                )

        table = (
            [
                [
                    "tokens/rank",
                    "global tokens",
                    "backend",
                    "E2E (us)",
                    "global tok/s",
                    "graph/eager max abs",
                ]
            ]
            if args.disable_torch_profiler
            else [
                [
                    "tokens/rank",
                    "global tokens",
                    "backend",
                    "dispatch kernel (us)",
                    "MoE kernel (us)",
                    "combine kernel (us)",
                    "kernel sum (us)",
                    "global tok/s (kernel)",
                    "graph/eager max abs",
                ]
            ]
        )

        for num_tokens in token_counts:
            input_sample_count = (
                args.graph_group_size
                if args.graph_group_size > 1
                else args.routing_samples
            )
            routing_inputs = [
                make_inputs(args, rank, num_tokens, device, sample_index)
                for sample_index in range(input_sample_count)
            ]
            inputs = routing_inputs[0]
            if args.backend == "flashinfer":
                pipeline = FlashInferPipeline(
                    args=args,
                    rank=rank,
                    world_size=world_size,
                    num_tokens=num_tokens,
                    weights=weights,
                    group=dist.group.WORLD,
                )
            else:
                assert mscclpp_comm_group is not None
                pipeline = MscclppPipeline(
                    args=args,
                    rank=rank,
                    world_size=world_size,
                    num_tokens=num_tokens,
                    weights=weights,
                    comm_group=mscclpp_comm_group,
                    device=device,
                )

            torch.cuda.synchronize()
            dist.barrier()
            if not args.skip_moe_autotune:
                if rank == 0:
                    print(
                        "Autotuning FlashInfer CUTLASS MoE for "
                        f"{world_size * num_tokens} dispatched rows...",
                        flush=True,
                    )
                autotune_moe(
                    pipeline=pipeline,
                    inputs=inputs,
                    dispatched_tokens=world_size * num_tokens,
                    sync_group=cpu_group,
                )

            captured_graphs = []
            graph_max_abs = 0.0
            if args.graph_group_size > 1:
                for sample_inputs in routing_inputs:
                    eager_output = run_once(pipeline, sample_inputs)
                eager_output = eager_output.clone()
                torch.cuda.synchronize()
                dist.barrier()
                finite = torch.tensor(
                    int(torch.isfinite(eager_output).all()),
                    dtype=torch.int32,
                    device=device,
                )
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not bool(finite.item()):
                    raise AssertionError(
                        f"{pipeline.name} eager output contains NaN or Inf"
                    )

                captured = capture_grouped_pipeline(
                    pipeline=pipeline,
                    input_samples=routing_inputs,
                    warmup=args.warmup,
                    sync_group=cpu_group,
                )
                captured.graph.replay()
                torch.cuda.synchronize()
                dist.barrier()
                graph_output = captured.output.clone()
                sample_max_abs, _ = assert_outputs_close(
                    reference=eager_output,
                    candidate=graph_output,
                    rtol=args.rtol,
                    atol=args.atol,
                    device=device,
                    label=f"{pipeline.name} eager and grouped CUDA graph",
                )
                graph_max_abs = sample_max_abs
                captured_graphs.append(captured)
                del eager_output, graph_output
            else:
                for sample_index, sample_inputs in enumerate(routing_inputs):
                    eager_output = run_once(pipeline, sample_inputs).clone()
                    torch.cuda.synchronize()
                    dist.barrier()
                    finite = torch.tensor(
                        int(torch.isfinite(eager_output).all()),
                        dtype=torch.int32,
                        device=device,
                    )
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if not bool(finite.item()):
                        raise AssertionError(
                            f"{pipeline.name} eager output contains NaN or Inf"
                        )

                    captured = capture_pipeline(
                        pipeline=pipeline,
                        inputs=sample_inputs,
                        warmup=args.warmup if sample_index == 0 else 0,
                        sync_group=cpu_group,
                        record_stage_events=args.report_stage_events,
                    )
                    captured.graph.replay()
                    torch.cuda.synchronize()
                    dist.barrier()
                    graph_output = captured.output.clone()
                    sample_max_abs, _ = assert_outputs_close(
                        reference=eager_output,
                        candidate=graph_output,
                        rtol=args.rtol,
                        atol=args.atol,
                        device=device,
                        label=f"{pipeline.name} eager and CUDA graph sample {sample_index}",
                    )
                    graph_max_abs = max(graph_max_abs, sample_max_abs)
                    captured_graphs.append(captured)
                    del eager_output, graph_output

            if args.disable_torch_profiler:
                _, graph_e2e_us = time_pipeline_replays(
                    captured=captured_graphs,
                    iters=args.iters,
                    repeats=args.wall_clock_repeats,
                    device=device,
                    sync_group=cpu_group,
                    report_stage_events=args.report_stage_events,
                )
            else:
                _, graph_timing = profile_pipeline_replays(
                    captured=captured_graphs[0],
                    iters=args.iters,
                    device=device,
                    sync_group=cpu_group,
                )

            if rank == 0:
                global_tokens = num_tokens * world_size
                if args.disable_torch_profiler:
                    graph_throughput = global_tokens * 1_000_000 / graph_e2e_us
                    table.append(
                        [
                            str(num_tokens),
                            str(global_tokens),
                            pipeline.name,
                            f"{graph_e2e_us:.1f}",
                            f"{graph_throughput:,.0f}",
                            f"{graph_max_abs:.3g}",
                        ]
                    )
                    print(
                        f"Finished backend={args.backend} tokens_per_rank={num_tokens}: "
                        f"Graph E2E={graph_e2e_us:.1f}us"
                    )
                else:
                    graph_throughput = global_tokens * 1_000_000 / graph_timing.e2e_us
                    table.append(
                        [
                            str(num_tokens),
                            str(global_tokens),
                            pipeline.name,
                            f"{graph_timing.dispatch_us:.1f}",
                            f"{graph_timing.moe_us:.1f}",
                            f"{graph_timing.combine_us:.1f}",
                            f"{graph_timing.e2e_us:.1f}",
                            f"{graph_throughput:,.0f}",
                            f"{graph_max_abs:.3g}",
                        ]
                    )
                    print(
                        f"Finished backend={args.backend} tokens_per_rank={num_tokens}: "
                        f"Graph kernel sum={graph_timing.e2e_us:.1f}us"
                    )
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            for captured in captured_graphs:
                captured.graph.reset()
            if args.backend == "flashinfer":
                pipeline.close()
            del (
                pipeline,
                captured_graphs,
                routing_inputs,
            )
            gc.collect()
            torch.cuda.empty_cache()
            synchronize_stream_and_ranks(mscclpp_comm_group, cpu_group)

        if rank == 0:
            print()
            print(format_markdown_table(table))
        synchronize_stream_and_ranks(mscclpp_comm_group, cpu_group)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
