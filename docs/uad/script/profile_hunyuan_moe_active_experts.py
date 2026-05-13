#!/usr/bin/env python3
"""Profile HunyuanImage3 FusedMoE while varying active expert count."""

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
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--event-iters", type=int, default=20)
    parser.add_argument("--moe-backend", default="auto")
    parser.add_argument("--top-k-values", default="1,8")
    parser.add_argument("--active-experts", default="1,2,4,8,16,32,64")
    parser.add_argument("--total-token-expert-rows", default="16384,32768,65536")
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def make_active_topk(
    input_tokens: int,
    top_k: int,
    active_experts: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if active_experts < top_k:
        raise ValueError(f"active_experts={active_experts} must be >= top_k={top_k}")
    pair_count = input_tokens * top_k
    if pair_count % active_experts != 0:
        raise ValueError(f"pair_count={pair_count} must be divisible by active_experts={active_experts}")
    flat_ids = torch.arange(pair_count, device=device, dtype=torch.int64) % active_experts
    topk_ids = flat_ids.view(input_tokens, top_k).contiguous()
    topk_weights = torch.full((input_tokens, top_k), 1.0 / top_k, device=device, dtype=torch.float32)
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

    top_k_values = parse_int_list(args.top_k_values)
    active_expert_values = parse_int_list(args.active_experts)
    total_pair_values = parse_int_list(args.total_token_expert_rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    max_input_tokens = max(max(total_pair_values) // top_k for top_k in top_k_values)
    model_dir, cfg, vllm_config, _, _ = moe_profile.setup_model(args, max_input_tokens, device)
    rows: list[dict[str, Any]] = []

    for top_k in top_k_values:
        if any(total_pairs % top_k != 0 for total_pairs in total_pair_values):
            raise ValueError(f"All total-token-expert rows must be divisible by top_k={top_k}")
        cfg_topk = replace(cfg, top_k=top_k)
        moe = real_moe.create_fused_moe(
            cfg=cfg_topk,
            vllm_config=vllm_config,
            prefix=f"hunyuan_active_l{args.layer_id}_topk{top_k}",
            local_intermediate_size=cfg.intermediate_size,
            device=device,
        )
        real_moe.load_real_expert_weights(model_dir, args.layer_id, cfg_topk, moe, 1, 0, device)
        with real_moe.set_current_vllm_config(vllm_config):
            moe.quant_method.process_weights_after_loading(moe)

        for total_pairs in total_pair_values:
            input_tokens = total_pairs // top_k
            hidden = torch.randn(input_tokens, cfg.hidden_size, device=device, dtype=torch.bfloat16)
            for active_experts in active_expert_values:
                if active_experts < top_k:
                    continue
                packed, topk_ids = make_active_topk(input_tokens, top_k, active_experts, device)
                counts = moe_profile.counts_list(topk_ids, cfg.num_experts)
                active_counts = counts[:active_experts]

                def run_fused() -> torch.Tensor:
                    with real_moe.set_forward_context(None, vllm_config, num_tokens=input_tokens):
                        return moe(hidden, packed)

                for _ in range(args.warmup):
                    _ = run_fused()
                torch.cuda.synchronize(device)

                event_ms = moe_profile.cuda_event_ms(run_fused, args.event_iters, device)
                prof_rows = moe_profile.profile_once(run_fused, device)
                by_category = aggregate_categories(prof_rows)
                profiler_sum_ms = sum(float(row["time_ms"]) for row in prof_rows)
                gemm1_ms = by_category.get("gemm1_gate_up_activation", 0.0)
                gemm2_ms = by_category.get("gemm2_down", 0.0)
                gemm1_flops = 2.0 * total_pairs * cfg.hidden_size * cfg.intermediate_size * 2
                gemm2_flops = 2.0 * total_pairs * cfg.intermediate_size * cfg.hidden_size
                gemm_total_flops = gemm1_flops + gemm2_flops
                row = {
                    "model_path": str(model_dir),
                    "layer_id": args.layer_id,
                    "moe_backend": args.moe_backend,
                    "input_tokens": input_tokens,
                    "top_k": top_k,
                    "total_token_expert_rows": total_pairs,
                    "active_experts": active_experts,
                    "rows_per_active_expert": total_pairs / active_experts,
                    "active_rows_min": min(active_counts),
                    "active_rows_max": max(active_counts),
                    "active_rows_mean": sum(active_counts) / len(active_counts),
                    "cuda_event_fused_moe_ms": event_ms,
                    "profiler_kernel_sum_ms": profiler_sum_ms,
                    "prefix_sum_ms": by_category.get("prefix_sum", 0.0),
                    "dispatch_expand_ms": by_category.get("dispatch_expand", 0.0),
                    "gemm1_gate_up_activation_ms": gemm1_ms,
                    "gemm2_down_ms": gemm2_ms,
                    "finalize_combine_ms": by_category.get("finalize_combine", 0.0),
                    "gemm1_tflops": gemm1_flops / (gemm1_ms / 1000.0) / 1e12 if gemm1_ms > 0 else 0.0,
                    "gemm2_tflops": gemm2_flops / (gemm2_ms / 1000.0) / 1e12 if gemm2_ms > 0 else 0.0,
                    "gemm_total_tflops": gemm_total_flops / ((gemm1_ms + gemm2_ms) / 1000.0) / 1e12
                    if gemm1_ms + gemm2_ms > 0
                    else 0.0,
                    "end_to_end_tflops": gemm_total_flops / (event_ms / 1000.0) / 1e12,
                }
                rows.append(row)
                print(
                    "top_k={} total_pairs={:6d} input={:5d} active={:2d} rows/expert={:7.1f} "
                    "event={:8.3f}ms gemm_total={:7.1f}TF".format(
                        top_k,
                        total_pairs,
                        input_tokens,
                        active_experts,
                        row["rows_per_active_expert"],
                        event_ms,
                        row["gemm_total_tflops"],
                    ),
                    flush=True,
                )
        del moe
        torch.cuda.empty_cache()

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
