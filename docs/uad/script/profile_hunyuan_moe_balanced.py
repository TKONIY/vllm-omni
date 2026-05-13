#!/usr/bin/env python3
"""Profile one HunyuanImage3 FusedMoE layer with balanced TopK routing.

This is the source script for the report's TopK=8 MoE tabs. It routes every
input token to ``top_k`` experts while keeping the total token-expert rows per
expert exactly balanced, so the timing curve reflects backend/kernel behavior
rather than router imbalance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
    parser.add_argument(
        "--tokens",
        default="16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--event-iters", type=int, default=20)
    parser.add_argument("--moe-backend", default="auto")
    return parser.parse_args()


def parse_tokens(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def make_balanced_topk(
    tokens: int,
    top_k: int,
    num_experts: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    total = tokens * top_k
    if total % num_experts != 0:
        raise ValueError(
            f"tokens * top_k must be divisible by num_experts for exact balance: "
            f"{tokens} * {top_k} % {num_experts} != 0"
        )
    ids = torch.arange(total, device=device, dtype=torch.int64).remainder(num_experts)
    topk_ids = ids.reshape(tokens, top_k).contiguous()
    topk_weights = torch.full((tokens, top_k), 1.0 / top_k, device=device, dtype=torch.float32)
    packed = torch.cat([topk_weights, topk_ids.to(torch.float32)], dim=-1)
    return packed, topk_ids


def make_random_topk_meta(
    tokens: int,
    top_k: int,
    num_experts: int,
    device: torch.device,
) -> dict[str, float]:
    topk_ids = torch.randint(0, num_experts, (tokens, top_k), device=device)
    return counts_meta(moe_profile.counts_list(topk_ids, num_experts))


def counts_meta(counts: list[int]) -> dict[str, float]:
    values = torch.tensor(counts, dtype=torch.float32)
    active = int((values > 0).sum().item())
    mean = float(values.mean().item())
    std = float(values.std(unbiased=False).item())
    return {
        "active_experts": active,
        "min": int(values.min().item()),
        "p50": float(values.quantile(0.50).item()),
        "p90": float(values.quantile(0.90).item()),
        "p95": float(values.quantile(0.95).item()),
        "max": int(values.max().item()),
        "mean": mean,
        "std": std,
        "cv": std / mean if mean else 0.0,
        "max_over_mean": float(values.max().item()) / mean if mean else 0.0,
        "min_over_mean": float(values.min().item()) / mean if mean else 0.0,
    }


def build_series_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = str(row["label"])
        series = by_label.setdefault(
            label,
            {
                "label": label,
                "category": row.get("category", "other"),
                "order": int(row.get("order", 999)),
                "kernel": row.get("kernel", ""),
                "points": [],
            },
        )
        series["points"].append({"tokens": int(row["tokens"]), "ms": float(row["time_ms"])})
    for series in by_label.values():
        series["points"].sort(key=lambda p: p["tokens"])
    return sorted(by_label.values(), key=lambda s: (s["order"], s["label"]))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    tokens_list = parse_tokens(args.tokens)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)

    model_dir, cfg, vllm_config, moe, _ = moe_profile.setup_model(args, max(tokens_list), device)

    rows: list[dict[str, Any]] = []
    expert_counts: dict[str, list[int]] = {}
    token_meta: dict[str, dict[str, float]] = {}
    random_router_meta: dict[str, dict[str, float]] = {}

    for tokens in tokens_list:
        hidden = torch.randn(tokens, cfg.hidden_size, device=device, dtype=torch.bfloat16)
        packed, topk_ids = make_balanced_topk(tokens, cfg.top_k, cfg.num_experts, device)
        counts = moe_profile.counts_list(topk_ids, cfg.num_experts)
        expert_counts[str(tokens)] = counts
        random_router_meta[str(tokens)] = make_random_topk_meta(tokens, cfg.top_k, cfg.num_experts, device)

        def run_fused() -> torch.Tensor:
            with real_moe.set_forward_context(None, vllm_config, num_tokens=tokens):
                return moe(hidden, packed)

        for _ in range(args.warmup):
            _ = run_fused()
        torch.cuda.synchronize(device)

        event_ms = moe_profile.cuda_event_ms(run_fused, args.event_iters, device)
        prof_rows = moe_profile.profile_once(run_fused, device)
        profiler_sum_ms = sum(float(row["time_ms"]) for row in prof_rows)

        meta = counts_meta(counts)
        meta["cuda_event_fused_moe_ms"] = event_ms
        meta["profiler_kernel_sum_ms"] = profiler_sum_ms
        token_meta[str(tokens)] = meta

        for row in prof_rows:
            rows.append({"tokens": tokens, **row})
        print(
            f"tokens={tokens:6d} backend={args.moe_backend} event={event_ms:.4f}ms "
            f"profiler_sum={profiler_sum_ms:.4f}ms",
            flush=True,
        )

    report = {
        "title": f"HunyuanImage3 Layer-{args.layer_id} FusedMoE Kernel Profile",
        "model_path": str(model_dir),
        "layer_id": args.layer_id,
        "moe_backend": args.moe_backend,
        "routing_mode": "balanced_topk_for_timing",
        "tokens": tokens_list,
        "config": {
            "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "num_experts": cfg.num_experts,
            "top_k": cfg.top_k,
            "num_shared_expert": cfg.num_shared_expert,
        },
        "kernel_series": build_series_from_rows(rows),
        "expert_counts": expert_counts,
        "token_meta": token_meta,
        "random_router_meta": random_router_meta,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
