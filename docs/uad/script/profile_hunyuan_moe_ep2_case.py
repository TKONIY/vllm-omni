#!/usr/bin/env python3
"""Profile one HunyuanImage3 FusedMoE layer with vLLM EP=2.

Run this script with torchrun. Each rank owns half of the experts through
vLLM expert parallelism; rank 0 writes one JSON report containing max-rank
latency, per-rank metadata, and per-rank profiler kernel rows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch
import torch.distributed as dist
from vllm.config import (
    DeviceConfig,
    KernelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.fused_moe import FusedMoE

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import bench_hunyuan_real_moe as real_moe  # noqa: E402
import profile_hunyuan_moe_balanced as balanced_profile  # noqa: E402
import profile_hunyuan_moe_kernels_html as moe_profile  # noqa: E402
import profile_hunyuan_moe_topk1_single_expert as topk1_profile  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=real_moe.DEFAULT_MODEL)
    parser.add_argument("--layer-id", type=int, default=15)
    parser.add_argument("--mode", choices=["topk8_balanced", "topk1_single"], required=True)
    parser.add_argument("--expert-id", type=int, default=0)
    parser.add_argument(
        "--tokens",
        default="16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--event-iters", type=int, default=20)
    parser.add_argument("--moe-backend", required=True)
    parser.add_argument("--all2all-backend", default="allgather_reducescatter")
    return parser.parse_args()


def parse_tokens(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def build_ep_vllm_config(
    *,
    max_tokens: int,
    device: torch.device,
    world_size: int,
    moe_backend: str,
    all2all_backend: str,
) -> VllmConfig:
    return VllmConfig(
        device_config=DeviceConfig(device=device),
        kernel_config=KernelConfig(moe_backend=moe_backend),
        parallel_config=ParallelConfig(
            tensor_parallel_size=world_size,
            enable_expert_parallel=True,
            is_moe_model=True,
            all2all_backend=all2all_backend,
            distributed_executor_backend="external_launcher",
        ),
        scheduler_config=SchedulerConfig(
            max_model_len=max_tokens,
            is_encoder_decoder=False,
            max_num_batched_tokens=max_tokens,
            max_num_seqs=1,
        ),
    )


def init_ep_distributed(vllm_config: VllmConfig, device: torch.device) -> tuple[int, int, int]:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(device)
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            backend="nccl",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=world_size,
            pipeline_model_parallel_size=1,
            backend="nccl",
        )
    return rank, local_rank, world_size


def create_ep_fused_moe(
    cfg: real_moe.HunyuanMoeConfig,
    vllm_config: VllmConfig,
    prefix: str,
    device: torch.device,
    world_size: int,
) -> FusedMoE:
    with set_current_vllm_config(vllm_config):
        moe = FusedMoE(
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            params_dtype=torch.bfloat16,
            renormalize=False,
            quant_config=None,
            tp_size=world_size,
            dp_size=1,
            pcp_size=1,
            prefix=prefix,
            custom_routing_function=real_moe._hunyuan_unpack_packed_topk,
            router_logits_dtype=torch.float32,
        )
    return moe.to(device)


def load_ep_local_expert_weights(
    *,
    model_dir: Path,
    layer_id: int,
    cfg: real_moe.HunyuanMoeConfig,
    moe: FusedMoE,
    device: torch.device,
) -> list[int]:
    keys: list[str] = []
    for expert_id in range(cfg.num_experts):
        local_id = moe._map_global_expert_id_to_local_expert_id(expert_id)
        if local_id == -1:
            continue
        keys.extend(
            [
                f"model.layers.{layer_id}.mlp.experts.{expert_id}.gate_and_up_proj.weight",
                f"model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight",
            ]
        )
    tensors = real_moe.load_tensors(model_dir, keys)
    loaded: list[int] = []
    with torch.no_grad():
        for expert_id in range(cfg.num_experts):
            local_id = moe._map_global_expert_id_to_local_expert_id(expert_id)
            if local_id == -1:
                continue
            gate_up = tensors[f"model.layers.{layer_id}.mlp.experts.{expert_id}.gate_and_up_proj.weight"]
            down = tensors[f"model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight"]
            # Hunyuan stores [up, gate]; vLLM FusedMoE expects [gate, up].
            up_full, gate_full = gate_up.chunk(2, dim=0)
            gate = gate_full.to(device=device, dtype=torch.bfloat16)
            up = up_full.to(device=device, dtype=torch.bfloat16)
            down_part = down.to(device=device, dtype=torch.bfloat16)
            moe.w13_weight[local_id, : cfg.intermediate_size].copy_(gate)
            moe.w13_weight[local_id, cfg.intermediate_size : 2 * cfg.intermediate_size].copy_(up)
            moe.w2_weight[local_id].copy_(down_part)
            loaded.append(expert_id)
    return loaded


def make_inputs(
    *,
    mode: str,
    tokens: int,
    cfg: real_moe.HunyuanMoeConfig,
    expert_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = torch.randn(tokens, cfg.hidden_size, device=device, dtype=torch.bfloat16)
    if mode == "topk8_balanced":
        packed, topk_ids = balanced_profile.make_balanced_topk(
            tokens, cfg.top_k, cfg.num_experts, device
        )
    elif mode == "topk1_single":
        packed, topk_ids = topk1_profile.make_single_expert_topk(tokens, expert_id, device)
    else:
        raise ValueError(f"unknown mode: {mode}")
    return hidden, packed, topk_ids


def aggregate_categories(rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        out[row["category"]] = out.get(row["category"], 0.0) + float(row["time_ms"])
    return out


def object_name(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, type):
        return value.__name__
    return value.__class__.__name__


def all_reduce_max_float(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def all_gather_jsonable(value: dict[str, Any]) -> list[dict[str, Any]]:
    gathered: list[dict[str, Any] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, value)
    return [item for item in gathered if item is not None]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    tokens_list = parse_tokens(args.tokens)
    torch.manual_seed(args.seed + local_rank)

    model_dir = real_moe.resolve_model_path(args.model_path)
    cfg = real_moe.read_hunyuan_moe_config(model_dir, args.layer_id)
    if args.mode == "topk1_single":
        cfg = replace(cfg, top_k=1)
    if max(tokens_list) * cfg.top_k % cfg.num_experts != 0 and args.mode == "topk8_balanced":
        raise ValueError("balanced topk8 requires tokens * top_k divisible by num_experts")

    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError(f"this script is for EP=2, got WORLD_SIZE={world_size}")

    vllm_config = build_ep_vllm_config(
        max_tokens=max(tokens_list),
        device=device,
        world_size=world_size,
        moe_backend=args.moe_backend,
        all2all_backend=args.all2all_backend,
    )
    rank, _, _ = init_ep_distributed(vllm_config, device)
    real_moe.init_workspace_manager(device)

    moe = create_ep_fused_moe(
        cfg=cfg,
        vllm_config=vllm_config,
        prefix=f"hunyuan_ep2_{args.mode}_{args.moe_backend}_l{args.layer_id}",
        device=device,
        world_size=world_size,
    )
    loaded_experts = load_ep_local_expert_weights(
        model_dir=model_dir,
        layer_id=args.layer_id,
        cfg=cfg,
        moe=moe,
        device=device,
    )
    with real_moe.set_current_vllm_config(vllm_config):
        moe.quant_method.process_weights_after_loading(moe)
    quant_method = moe.quant_method
    effective_backend_obj = getattr(quant_method, "unquantized_backend", None)
    effective_backend = getattr(effective_backend_obj, "value", str(effective_backend_obj))
    moe_kernel = getattr(quant_method, "moe_kernel", None)
    prepare_finalize = getattr(moe_kernel, "prepare_finalize", None)
    experts = getattr(moe_kernel, "experts", None)
    moe_parallel = moe.moe_parallel_config

    rank_results: list[dict[str, Any]] = []
    for tokens in tokens_list:
        hidden, packed, topk_ids = make_inputs(
            mode=args.mode,
            tokens=tokens,
            cfg=cfg,
            expert_id=args.expert_id,
            device=device,
        )
        global_counts = moe_profile.counts_list(topk_ids, cfg.num_experts)
        local_counts = [global_counts[expert_id] for expert_id in loaded_experts]

        def run_fused() -> torch.Tensor:
            with real_moe.set_forward_context(None, vllm_config, num_tokens=tokens):
                return moe(hidden, packed)

        for _ in range(args.warmup):
            _ = run_fused()
        torch.cuda.synchronize(device)
        dist.barrier()

        event_ms_rank = moe_profile.cuda_event_ms(run_fused, args.event_iters, device)
        event_ms_max = all_reduce_max_float(event_ms_rank, device)
        dist.barrier()

        prof_rows = moe_profile.profile_once(run_fused, device)
        by_category = aggregate_categories(prof_rows)
        gemm1_ms = by_category.get("gemm1_gate_up_activation", 0.0)
        gemm2_ms = by_category.get("gemm2_down", 0.0)
        profiler_sum_ms = sum(float(row["time_ms"]) for row in prof_rows)
        local_token_expert_rows = int(sum(local_counts))

        rank_payload = {
            "rank": rank,
            "tokens": tokens,
            "event_ms_rank": event_ms_rank,
            "event_ms_max": event_ms_max,
            "profiler_sum_ms": profiler_sum_ms,
            "gemm1_ms": gemm1_ms,
            "gemm2_ms": gemm2_ms,
            "category_ms": by_category,
            "loaded_experts": loaded_experts,
            "local_counts": local_counts,
            "local_token_expert_rows": local_token_expert_rows,
            "kernel_rows": prof_rows,
        }
        gathered = all_gather_jsonable(rank_payload)
        if rank == 0:
            rank_results.append(
                {
                    "tokens": tokens,
                    "event_ms_max": event_ms_max,
                    "global_counts": global_counts,
                    "ranks": sorted(gathered, key=lambda item: int(item["rank"])),
                }
            )
            print(
                f"mode={args.mode} backend={args.moe_backend} tokens={tokens:6d} "
                f"event_max={event_ms_max:.4f}ms",
                flush=True,
            )
        dist.barrier()

    if rank == 0:
        report = {
            "title": "HunyuanImage3 Layer-15 EP=2 FusedMoE Profile",
            "model_path": str(model_dir),
            "layer_id": args.layer_id,
            "mode": args.mode,
            "moe_backend": args.moe_backend,
            "all2all_backend": args.all2all_backend,
            "ep_size": world_size,
            "tokens": tokens_list,
            "config": {
                "hidden_size": cfg.hidden_size,
                "intermediate_size": cfg.intermediate_size,
                "num_experts": cfg.num_experts,
                "top_k": cfg.top_k,
                "num_shared_expert": cfg.num_shared_expert,
            },
            "runtime": {
                "requested_moe_backend": args.moe_backend,
                "effective_moe_backend": effective_backend,
                "quant_method": object_name(quant_method),
                "prepare_finalize": object_name(prepare_finalize),
                "experts": object_name(experts),
                "tp_size": moe_parallel.tp_size,
                "tp_rank": moe_parallel.tp_rank,
                "ep_size": moe_parallel.ep_size,
                "ep_rank": moe_parallel.ep_rank,
                "use_ep": moe_parallel.use_ep,
                "num_local_experts": len(loaded_experts),
                "use_all2all_kernels": moe_parallel.use_all2all_kernels,
            },
            "results": rank_results,
        }
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {output}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
