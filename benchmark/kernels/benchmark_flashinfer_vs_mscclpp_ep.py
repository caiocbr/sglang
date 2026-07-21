#!/usr/bin/env python3
"""Benchmark MoE EP dispatch/combine with a shared FlashInfer CUTLASS runner.

The two measured paths are:

1. FlashInfer ``MoeAlltoAll`` dispatch/combine + FlashInfer CUTLASS MoE.
2. MSCCL++ EP low-latency dispatch/combine + FlashInfer CUTLASS MoE.

Both paths use BF16 communication and BF16 expert compute with the same input,
routing decisions, routing weights, and rank-local expert weights. Both
dispatchers produce a fixed rank-major token buffer consumed by the same
FlashInfer CUTLASS MoE call; only dispatch/combine communication differs.

Pass ``--cuda-graph`` to additionally capture each complete
``dispatch -> MoE -> combine`` path in one CUDA graph and benchmark replay
latency.

Example (DeepSeek-like shape on one 8-GPU node):

    torchrun --standalone --nproc-per-node=8 \
      benchmark/kernels/benchmark_flashinfer_vs_mscclpp_ep.py \
      --cuda-graph

Small smoke run:

    torchrun --standalone --nproc-per-node=2 \
      benchmark/kernels/benchmark_flashinfer_vs_mscclpp_ep.py \
      --tokens-per-rank 8 --hidden-size 4096 --intermediate-size 512 \
      --num-experts 8 --top-k 2 --warmup 2 --iters 5

This benchmark is intentionally single-node. Its FlashInfer workspace uses
single-node CUDA virtual memory and POSIX FD exchange over Unix sockets, so it
does not require MNNVL fabric or ``SYS_PTRACE``.
"""

from __future__ import annotations

import argparse
import os
import statistics
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.distributed as dist

DTYPE = torch.bfloat16
MSCCLPP_LL_HIDDEN_SIZES = (4096, 6656, 7168, 8192, 9216)


@dataclass
class Inputs:
    hidden_states: torch.Tensor
    topk_ids_i64: torch.Tensor
    topk_ids_i32: torch.Tensor
    topk_weights: torch.Tensor


@dataclass
class Timing:
    dispatch_us: float
    moe_us: float
    combine_us: float
    e2e_us: float


@dataclass
class ExpertWeights:
    w13: torch.Tensor
    w2: torch.Tensor


@dataclass
class CapturedPipeline:
    graph: torch.cuda.CUDAGraph
    output: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Also benchmark one full dispatch + MoE + combine CUDA graph replay.",
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


def next_power_of_2(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


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
    if world_size != local_world_size:
        raise ValueError(
            "This benchmark only supports a single node: "
            f"world_size={world_size}, local_world_size={local_world_size}"
        )
    if args.num_experts % world_size != 0:
        raise ValueError(
            f"num_experts={args.num_experts} must be divisible by world_size={world_size}"
        )
    if not 0 < args.top_k <= min(9, args.num_experts):
        raise ValueError("--top-k must be in [1, min(9, num_experts)]")
    if args.hidden_size not in MSCCLPP_LL_HIDDEN_SIZES:
        raise ValueError(
            f"--hidden-size must be one of {MSCCLPP_LL_HIDDEN_SIZES} "
            "for MSCCL++ EP low-latency"
        )
    if args.intermediate_size <= 0:
        raise ValueError("--intermediate-size must be positive")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be non-negative and --iters must be positive")
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
) -> Inputs:
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 1_000_003 * rank + num_tokens)

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


class CutlassMoe:
    def __init__(
        self,
        weights: ExpertWeights,
        rank: int,
        world_size: int,
        max_dispatched_tokens: int,
        enable_alltoall: bool,
    ) -> None:
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer.fused_moe.core import ActivationType

        self.cutlass_fused_moe = cutlass_fused_moe
        self.activation_type = ActivationType.Swiglu
        self.w13 = weights.w13
        self.w2 = weights.w2
        if self.w13.dtype != DTYPE or self.w2.dtype != DTYPE:
            raise TypeError("FlashInfer CUTLASS BF16 MoE requires BF16 expert weights")
        self.rank = rank
        self.world_size = world_size
        self.enable_alltoall = enable_alltoall
        self.tune_max_num_tokens = next_power_of_2(max_dispatched_tokens)

    def __call__(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.dtype != DTYPE or output.dtype != DTYPE:
            raise TypeError("FlashInfer CUTLASS BF16 MoE requires BF16 input/output")
        result = self.cutlass_fused_moe(
            input=hidden_states,
            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=topk_weights,
            fc1_expert_weights=self.w13,
            fc2_expert_weights=self.w2,
            output_dtype=hidden_states.dtype,
            input_sf=None,
            quant_scales=None,
            ep_size=self.world_size,
            ep_rank=self.rank,
            tp_size=1,
            tp_rank=0,
            output=output,
            enable_alltoall=self.enable_alltoall,
            tune_max_num_tokens=self.tune_max_num_tokens,
            activation_type=self.activation_type,
        )
        return result[0]


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
        from flashinfer.comm import moe_a2a_get_workspace_size_per_rank
        from flashinfer.comm.mapping import Mapping

        from sglang.srt.layers.moe.token_dispatcher.flashinfer_utils import (
            TorchDistributedCommBackend,
        )

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
        self.a2a = make_single_node_moe_alltoall(
            mapping=mapping,
            max_num_tokens=num_tokens,
            top_k=args.top_k,
            num_experts=args.num_experts,
            workspace_size_per_rank=workspace_size,
            comm_backend=TorchDistributedCommBackend(group),
        )
        self.num_tokens = num_tokens
        self.hidden_size = args.hidden_size
        self.num_experts = args.num_experts
        self.world_size = world_size
        self.moe = CutlassMoe(
            weights=weights,
            rank=rank,
            world_size=world_size,
            max_dispatched_tokens=world_size * num_tokens,
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
                output_layout=DispatchLayout.TOKEN_MAJOR,
                token_major_init_padding=True,
                invalid_token_expert_id=args.num_experts,
                low_latency_num_blocks=args.low_latency_num_blocks,
                low_latency_combine_mode=CombineMode.RANK_LOCAL_REDUCE,
            )
        )
        if not self.communicator.is_available():
            raise RuntimeError("MSCCL++ EP low-latency runtime is unavailable")

        capacity = world_size * num_tokens
        self.dispatch_output = torch.empty(
            (capacity, args.hidden_size), dtype=DTYPE, device=device
        )
        self.expert_output = torch.empty_like(self.dispatch_output)
        self.combine_output = torch.empty(
            (num_tokens, args.hidden_size), dtype=DTYPE, device=device
        )
        self.moe = CutlassMoe(
            weights=weights,
            rank=rank,
            world_size=world_size,
            max_dispatched_tokens=capacity,
            enable_alltoall=False,
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
            raise RuntimeError("MSCCL++ TOKEN_MAJOR dispatch metadata is missing")
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


def reduce_max(values: Sequence[float], device: torch.device) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.cpu().tolist()


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


def check_correctness(
    flashinfer_pipeline: FlashInferPipeline,
    mscclpp_pipeline: MscclppPipeline,
    inputs: Inputs,
    rtol: float,
    atol: float,
    device: torch.device,
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    flashinfer_output = run_once(flashinfer_pipeline, inputs).clone()
    mscclpp_output = run_once(mscclpp_pipeline, inputs).clone()
    torch.cuda.synchronize()

    max_abs, max_rel = assert_outputs_close(
        reference=flashinfer_output,
        candidate=mscclpp_output,
        rtol=rtol,
        atol=atol,
        device=device,
        label="FlashInfer eager and MSCCL++ eager",
    )
    return max_abs, max_rel, flashinfer_output, mscclpp_output


def benchmark_pipeline(
    pipeline: Any,
    inputs: Inputs,
    warmup: int,
    iters: int,
    device: torch.device,
) -> Timing:
    for _ in range(warmup):
        run_once(pipeline, inputs)
        torch.cuda.synchronize()
        dist.barrier()

    dispatch_ms: list[float] = []
    moe_ms: list[float] = []
    combine_ms: list[float] = []
    e2e_ms: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        dispatch_end = torch.cuda.Event(enable_timing=True)
        moe_end = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        state = pipeline.dispatch(inputs)
        dispatch_end.record()
        expert_output = pipeline.run_moe(state)
        moe_end.record()
        pipeline.combine(state, expert_output)
        end.record()
        end.synchronize()

        dispatch_ms.append(start.elapsed_time(dispatch_end))
        moe_ms.append(dispatch_end.elapsed_time(moe_end))
        combine_ms.append(moe_end.elapsed_time(end))
        e2e_ms.append(start.elapsed_time(end))
        # Keep LL peers in lockstep without including the pacing barrier in timing.
        dist.barrier()

    dispatch_us, moe_us, combine_us, e2e_us = reduce_max(
        [
            statistics.mean(dispatch_ms) * 1000,
            statistics.mean(moe_ms) * 1000,
            statistics.mean(combine_ms) * 1000,
            statistics.mean(e2e_ms) * 1000,
        ],
        device,
    )
    return Timing(
        dispatch_us=dispatch_us,
        moe_us=moe_us,
        combine_us=combine_us,
        e2e_us=e2e_us,
    )


def capture_pipeline(pipeline: Any, inputs: Inputs) -> CapturedPipeline:
    run_once(pipeline, inputs)
    torch.cuda.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run_once(pipeline, inputs)

    torch.cuda.synchronize()
    dist.barrier()
    return CapturedPipeline(graph=graph, output=output)


def replay_and_clone(captured: CapturedPipeline) -> torch.Tensor:
    dist.barrier()
    captured.graph.replay()
    output = captured.output.clone()
    torch.cuda.synchronize()
    dist.barrier()
    return output


def check_cuda_graph_correctness(
    flashinfer_graph: CapturedPipeline,
    mscclpp_graph: CapturedPipeline,
    flashinfer_eager_output: torch.Tensor,
    mscclpp_eager_output: torch.Tensor,
    rtol: float,
    atol: float,
    device: torch.device,
) -> tuple[float, float]:
    flashinfer_graph_output = replay_and_clone(flashinfer_graph)
    mscclpp_graph_output = replay_and_clone(mscclpp_graph)

    assert_outputs_close(
        reference=flashinfer_eager_output,
        candidate=flashinfer_graph_output,
        rtol=rtol,
        atol=atol,
        device=device,
        label="FlashInfer eager and CUDA graph",
    )
    assert_outputs_close(
        reference=mscclpp_eager_output,
        candidate=mscclpp_graph_output,
        rtol=rtol,
        atol=atol,
        device=device,
        label="MSCCL++ eager and CUDA graph",
    )
    return assert_outputs_close(
        reference=flashinfer_graph_output,
        candidate=mscclpp_graph_output,
        rtol=rtol,
        atol=atol,
        device=device,
        label="FlashInfer CUDA graph and MSCCL++ CUDA graph",
    )


def benchmark_cuda_graph(
    captured: CapturedPipeline,
    warmup: int,
    iters: int,
    device: torch.device,
) -> float:
    for _ in range(warmup):
        captured.graph.replay()
        torch.cuda.synchronize()
        dist.barrier()

    latencies_ms: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        captured.graph.replay()
        end.record()
        end.synchronize()
        latencies_ms.append(start.elapsed_time(end))
        dist.barrier()

    (e2e_us,) = reduce_max(
        [statistics.mean(latencies_ms) * 1000],
        device,
    )
    return e2e_us


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

    try:
        import flashinfer
        import mscclpp

        mscclpp_comm_group = mscclpp.CommGroup(
            torch_group=cpu_group,
            rank=rank,
            size=world_size,
        )
        weights = make_local_weights(args, rank, world_size, device)
        torch.cuda.synchronize()
        dist.barrier()

        if rank == 0:
            weight_gib = (weights.w13.nbytes + weights.w2.nbytes) / (1024**3)
            print(
                "Configuration: "
                f"world_size={world_size}, dtype=bf16, "
                f"hidden={args.hidden_size}, intermediate={args.intermediate_size}, "
                f"experts={args.num_experts}, top_k={args.top_k}, "
                f"tokens_per_rank={token_counts}, warmup={args.warmup}, "
                f"iters={args.iters}, cuda_graph={args.cuda_graph}"
            )
            print(
                f"Versions: flashinfer={flashinfer.__version__}, "
                f"mscclpp={mscclpp.__version__}"
            )
            print(f"Rank-local expert weights: {weight_gib:.2f} GiB")
            print("Latency is the maximum per-rank mean GPU time.\n")

        table = [
            [
                "tokens/rank",
                "global tokens",
                "backend",
                "mode",
                "dispatch (us)",
                "MoE (us)",
                "combine (us)",
                "E2E (us)",
                "global tok/s",
                "E2E speedup",
                "max abs diff",
            ]
        ]

        for num_tokens in token_counts:
            inputs = make_inputs(args, rank, num_tokens, device)
            flashinfer_pipeline = FlashInferPipeline(
                args=args,
                rank=rank,
                world_size=world_size,
                num_tokens=num_tokens,
                weights=weights,
                group=dist.group.WORLD,
            )
            mscclpp_pipeline = MscclppPipeline(
                args=args,
                rank=rank,
                world_size=world_size,
                num_tokens=num_tokens,
                weights=weights,
                comm_group=mscclpp_comm_group,
                device=device,
            )

            (
                max_abs,
                _,
                flashinfer_eager_output,
                mscclpp_eager_output,
            ) = check_correctness(
                flashinfer_pipeline=flashinfer_pipeline,
                mscclpp_pipeline=mscclpp_pipeline,
                inputs=inputs,
                rtol=args.rtol,
                atol=args.atol,
                device=device,
            )
            flashinfer_timing = benchmark_pipeline(
                pipeline=flashinfer_pipeline,
                inputs=inputs,
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )
            mscclpp_timing = benchmark_pipeline(
                pipeline=mscclpp_pipeline,
                inputs=inputs,
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )

            flashinfer_graph = None
            mscclpp_graph = None
            graph_max_abs = None
            flashinfer_graph_us = None
            mscclpp_graph_us = None
            if args.cuda_graph:
                flashinfer_graph = capture_pipeline(flashinfer_pipeline, inputs)
                mscclpp_graph = capture_pipeline(mscclpp_pipeline, inputs)
                graph_max_abs, _ = check_cuda_graph_correctness(
                    flashinfer_graph=flashinfer_graph,
                    mscclpp_graph=mscclpp_graph,
                    flashinfer_eager_output=flashinfer_eager_output,
                    mscclpp_eager_output=mscclpp_eager_output,
                    rtol=args.rtol,
                    atol=args.atol,
                    device=device,
                )
                flashinfer_graph_us = benchmark_cuda_graph(
                    captured=flashinfer_graph,
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                )
                mscclpp_graph_us = benchmark_cuda_graph(
                    captured=mscclpp_graph,
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                )

            if rank == 0:
                global_tokens = num_tokens * world_size
                flashinfer_throughput = (
                    global_tokens * 1_000_000 / flashinfer_timing.e2e_us
                )
                mscclpp_throughput = global_tokens * 1_000_000 / mscclpp_timing.e2e_us
                speedup = flashinfer_timing.e2e_us / mscclpp_timing.e2e_us
                table.extend(
                    [
                        [
                            str(num_tokens),
                            str(global_tokens),
                            flashinfer_pipeline.name,
                            "eager",
                            f"{flashinfer_timing.dispatch_us:.1f}",
                            f"{flashinfer_timing.moe_us:.1f}",
                            f"{flashinfer_timing.combine_us:.1f}",
                            f"{flashinfer_timing.e2e_us:.1f}",
                            f"{flashinfer_throughput:,.0f}",
                            "1.000x",
                            f"{max_abs:.3g}",
                        ],
                        [
                            str(num_tokens),
                            str(global_tokens),
                            mscclpp_pipeline.name,
                            "eager",
                            f"{mscclpp_timing.dispatch_us:.1f}",
                            f"{mscclpp_timing.moe_us:.1f}",
                            f"{mscclpp_timing.combine_us:.1f}",
                            f"{mscclpp_timing.e2e_us:.1f}",
                            f"{mscclpp_throughput:,.0f}",
                            f"{speedup:.3f}x",
                            f"{max_abs:.3g}",
                        ],
                    ]
                )
                if (
                    flashinfer_graph_us is not None
                    and mscclpp_graph_us is not None
                    and graph_max_abs is not None
                ):
                    flashinfer_graph_throughput = (
                        global_tokens * 1_000_000 / flashinfer_graph_us
                    )
                    mscclpp_graph_throughput = (
                        global_tokens * 1_000_000 / mscclpp_graph_us
                    )
                    graph_speedup = flashinfer_graph_us / mscclpp_graph_us
                    table.extend(
                        [
                            [
                                str(num_tokens),
                                str(global_tokens),
                                flashinfer_pipeline.name,
                                "cuda_graph",
                                "-",
                                "-",
                                "-",
                                f"{flashinfer_graph_us:.1f}",
                                f"{flashinfer_graph_throughput:,.0f}",
                                "1.000x",
                                f"{graph_max_abs:.3g}",
                            ],
                            [
                                str(num_tokens),
                                str(global_tokens),
                                mscclpp_pipeline.name,
                                "cuda_graph",
                                "-",
                                "-",
                                "-",
                                f"{mscclpp_graph_us:.1f}",
                                f"{mscclpp_graph_throughput:,.0f}",
                                f"{graph_speedup:.3f}x",
                                f"{graph_max_abs:.3g}",
                            ],
                        ]
                    )
                print(
                    f"Finished tokens_per_rank={num_tokens}: "
                    f"MSCCL++ E2E speedup={speedup:.3f}x"
                )

            torch.cuda.synchronize()
            if flashinfer_graph is not None:
                flashinfer_graph.graph.reset()
            if mscclpp_graph is not None:
                mscclpp_graph.graph.reset()
            flashinfer_pipeline.close()
            del (
                flashinfer_pipeline,
                mscclpp_pipeline,
                flashinfer_eager_output,
                mscclpp_eager_output,
                flashinfer_graph,
                mscclpp_graph,
                inputs,
            )
            dist.barrier()

        if rank == 0:
            print()
            print(format_markdown_table(table))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
