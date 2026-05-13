#!/usr/bin/env python3
"""Common profiler helpers for HunyuanImage3 one-layer FusedMoE experiments."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch
from torch.profiler import ProfilerActivity, profile

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import bench_hunyuan_real_moe as real_moe  # noqa: E402


def cuda_event_ms(fn, iters: int, device: torch.device, warmup: int = 0) -> float:
    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn()
    end.record()
    torch.cuda.synchronize(device)
    if out is not None:
        _ = out.shape
    return float(start.elapsed_time(end) / iters)


def canonical_kernel_name(name: str) -> str:
    if "blockExpertPrefixSumKernel" in name:
        return "blockExpertPrefixSumKernel"
    if "globalExpertPrefixSumKernel" in name:
        return "globalExpertPrefixSumKernel"
    if "mergeExpertPrefixSumKernel" in name:
        return "mergeExpertPrefixSumKernel"
    if "fusedBuildExpertMapsSortFirstTokenKernel" in name:
        return "fusedBuildExpertMapsSortFirstTokenKernel"
    if "expandInputRowsKernel" in name:
        return "expandInputRowsKernel"
    if "fused_moe::run_global" in name:
        return "fused_moe::run_global<GEMM1_gate_up_activation>"
    if "MoeFCGemm" in name:
        return "cutlass::Kernel<MoeFCGemm_GEMM2_down>"
    if "finalizeMoeRoutingKernel" in name:
        return "finalizeMoeRoutingKernel"
    if "Memcpy DtoD" in name:
        return "Memcpy DtoD"
    if name.startswith("Memset"):
        return "Memset"
    if "direct_copy_kernel" in name:
        return "copy/cast"
    return name.replace("\n", " ")[:180]


def kernel_category(name: str) -> str:
    if "copy/cast" in name:
        return "copy_cast"
    if "PrefixSumKernel" in name:
        return "prefix_sum"
    if "fusedBuildExpertMapsSortFirstTokenKernel" in name:
        return "expert_map_build"
    if "expandInputRowsKernel" in name:
        return "dispatch_expand"
    if "GEMM1" in name:
        return "gemm1_gate_up_activation"
    if "GEMM2" in name:
        return "gemm2_down"
    if "finalizeMoeRoutingKernel" in name:
        return "finalize_combine"
    if "Memcpy" in name:
        return "memcpy"
    if "Memset" in name:
        return "memset"
    return "other"


def profile_once(fn, device: torch.device) -> list[dict[str, Any]]:
    torch.cuda.synchronize(device)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        out = fn()
        torch.cuda.synchronize(device)
        _ = out.shape
    rows: list[dict[str, Any]] = []
    cuda_events = [
        event
        for event in prof.events()
        if getattr(getattr(event, "device_type", None), "name", "") == "CUDA"
    ]
    for index, event in enumerate(cuda_events, start=1):
        short = canonical_kernel_name(event.name)
        category = kernel_category(short)
        rows.append(
            {
                "order": index,
                "raw_name": event.name,
                "kernel": short,
                "category": category,
                "label": f"{index:02d}. {category} :: {short}",
                "time_ms": float(event.device_time_total) / 1000.0,
            }
        )
    return rows


def counts_list(topk_ids: torch.Tensor, num_experts: int) -> list[int]:
    counts = torch.bincount(topk_ids.reshape(-1), minlength=num_experts)
    return [int(v) for v in counts.cpu().tolist()]


def setup_model(args: argparse.Namespace, max_tokens: int, device: torch.device):
    model_dir = real_moe.resolve_model_path(args.model_path)
    cfg = real_moe.read_hunyuan_moe_config(model_dir, args.layer_id)
    vllm_config = real_moe.build_vllm_config(
        max_tokens=max_tokens,
        device=device,
        moe_backend=args.moe_backend,
    )
    real_moe.init_workspace_manager(device)
    moe = real_moe.create_fused_moe(
        cfg=cfg,
        vllm_config=vllm_config,
        prefix=f"hunyuan_profile_l{args.layer_id}",
        local_intermediate_size=cfg.intermediate_size,
        device=device,
    )
    real_moe.load_real_expert_weights(
        model_dir=model_dir,
        layer_id=args.layer_id,
        cfg=cfg,
        moe=moe,
        tp_size=1,
        tp_rank=0,
        device=device,
    )
    with real_moe.set_current_vllm_config(vllm_config):
        moe.quant_method.process_weights_after_loading(moe)
    return model_dir, cfg, vllm_config, moe, None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
