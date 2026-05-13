#!/usr/bin/env python3
"""Profile HunyuanImage3 FusedMoE as a top_k=1 single-expert control.

All input tokens route to one real Hunyuan expert. This is the FusedMoE-path
counterpart of the dense single-expert FFN tab: same layer, same expert,
same hidden/intermediate shape, different execution backend.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import bench_hunyuan_real_moe as real_moe  # noqa: E402
import profile_hunyuan_moe_kernels_html as moe_profile  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=real_moe.DEFAULT_MODEL)
    parser.add_argument("--layer-id", type=int, default=15)
    parser.add_argument("--expert-id", type=int, default=0)
    parser.add_argument(
        "--tokens",
        default="16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--event-iters", type=int, default=20)
    parser.add_argument("--moe-backend", default="auto")
    return parser.parse_args()


def parse_tokens(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def make_single_expert_topk(tokens: int, expert_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    topk_weights = torch.ones((tokens, 1), device=device, dtype=torch.float32)
    topk_ids = torch.full((tokens, 1), expert_id, device=device, dtype=torch.int64)
    packed = torch.cat([topk_weights, topk_ids.to(torch.float32)], dim=-1)
    return packed, topk_ids


def aggregate_categories(rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        out[row["category"]] = out.get(row["category"], 0.0) + float(row["time_ms"])
    return out


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    tokens_list = parse_tokens(args.tokens)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    model_dir, cfg, vllm_config, _, _ = moe_profile.setup_model(args, max(tokens_list), device)
    if not 0 <= args.expert_id < cfg.num_experts:
        raise ValueError(f"expert_id={args.expert_id} must be in [0, {cfg.num_experts})")
    cfg_topk1 = replace(cfg, top_k=1)
    moe = real_moe.create_fused_moe(
        cfg=cfg_topk1,
        vllm_config=vllm_config,
        prefix=f"hunyuan_topk1_single_l{args.layer_id}_e{args.expert_id}",
        local_intermediate_size=cfg.intermediate_size,
        device=device,
    )
    real_moe.load_real_expert_weights(
        model_dir=model_dir,
        layer_id=args.layer_id,
        cfg=cfg_topk1,
        moe=moe,
        tp_size=1,
        tp_rank=0,
        device=device,
    )
    with real_moe.set_current_vllm_config(vllm_config):
        moe.quant_method.process_weights_after_loading(moe)

    rows: list[dict[str, Any]] = []
    for tokens in tokens_list:
        hidden = torch.randn(tokens, cfg.hidden_size, device=device, dtype=torch.bfloat16)
        packed, topk_ids = make_single_expert_topk(tokens, args.expert_id, device)
        counts = moe_profile.counts_list(topk_ids, cfg.num_experts)

        def run_fused() -> torch.Tensor:
            with real_moe.set_forward_context(None, vllm_config, num_tokens=tokens):
                return moe(hidden, packed)

        for _ in range(args.warmup):
            _ = run_fused()
        torch.cuda.synchronize(device)

        event_ms = moe_profile.cuda_event_ms(run_fused, args.event_iters, device)
        prof_rows = moe_profile.profile_once(run_fused, device)
        profiler_sum_ms = sum(float(row["time_ms"]) for row in prof_rows)
        by_category = aggregate_categories(prof_rows)
        gemm1_ms = by_category.get("gemm1_gate_up_activation", 0.0)
        gemm2_ms = by_category.get("gemm2_down", 0.0)
        gemm1_flops = 2.0 * tokens * cfg.hidden_size * cfg.intermediate_size * 2
        gemm2_flops = 2.0 * tokens * cfg.intermediate_size * cfg.hidden_size
        gemm_total_flops = gemm1_flops + gemm2_flops
        gemm_total_tflops = (
            gemm_total_flops / ((gemm1_ms + gemm2_ms) / 1000.0) / 1e12
            if gemm1_ms + gemm2_ms > 0
            else 0.0
        )
        end_to_end_tflops = gemm_total_flops / (event_ms / 1000.0) / 1e12

        for row in prof_rows:
            rows.append(
                {
                    "model_path": str(model_dir),
                    "layer_id": args.layer_id,
                    "expert_id": args.expert_id,
                    "top_k": 1,
                    "tokens": tokens,
                    "hidden_size": cfg.hidden_size,
                    "intermediate_size": cfg.intermediate_size,
                    "active_experts": 1,
                    "rows_for_expert": counts[args.expert_id],
                    "cuda_event_topk1_moe_ms": event_ms,
                    "profiler_kernel_sum_ms": profiler_sum_ms,
                    "gemm1_gate_up_activation_ms": gemm1_ms,
                    "gemm2_down_ms": gemm2_ms,
                    "gemm_total_tflops": gemm_total_tflops,
                    "end_to_end_tflops": end_to_end_tflops,
                    **row,
                }
            )
        print(
            f"tokens={tokens:6d} expert={args.expert_id} event={event_ms:.4f}ms "
            f"gemm_total={gemm_total_tflops:.1f}TF",
            flush=True,
        )

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
