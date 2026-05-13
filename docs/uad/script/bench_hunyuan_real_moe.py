#!/usr/bin/env python3
"""Shared helpers for HunyuanImage3 one-layer MoE experiments."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch
from safetensors import safe_open
from vllm.config import (
    DeviceConfig,
    KernelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.v1.worker.workspace import init_workspace_manager

DEFAULT_MODEL = "tencent/HunyuanImage-3.0-Instruct"


@dataclass(frozen=True)
class HunyuanMoeConfig:
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    num_shared_expert: int


def resolve_model_path(model_path: str) -> Path:
    path = Path(model_path)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_path, local_files_only=False))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def layer_value(value: Any, layer_id: int) -> Any:
    return value[layer_id] if isinstance(value, list) else value


def read_hunyuan_moe_config(model_dir: Path, layer_id: int) -> HunyuanMoeConfig:
    cfg = load_json(model_dir / "config.json")
    return HunyuanMoeConfig(
        hidden_size=int(cfg["hidden_size"]),
        intermediate_size=int(layer_value(cfg["moe_intermediate_size"], layer_id)),
        num_experts=int(cfg["num_experts"]),
        top_k=int(layer_value(cfg["moe_topk"], layer_id)),
        num_shared_expert=int(layer_value(cfg.get("num_shared_expert", 0), layer_id)),
    )


def load_weight_map(model_dir: Path) -> dict[str, str]:
    return load_json(model_dir / "model.safetensors.index.json")["weight_map"]


def load_tensors(model_dir: Path, keys: list[str]) -> dict[str, torch.Tensor]:
    weight_map = load_weight_map(model_dir)
    by_file: dict[str, list[str]] = {}
    for key in keys:
        if key not in weight_map:
            raise KeyError(f"Missing checkpoint tensor: {key}")
        by_file.setdefault(weight_map[key], []).append(key)

    tensors: dict[str, torch.Tensor] = {}
    for filename, file_keys in sorted(by_file.items()):
        with safe_open(model_dir / filename, framework="pt", device="cpu") as f:
            for key in file_keys:
                tensors[key] = f.get_tensor(key)
    return tensors


def _hunyuan_unpack_packed_topk(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    del renormalize
    topk_weights = gating_output[:, :topk].contiguous()
    topk_indices = gating_output[:, topk:]
    return topk_weights.to(torch.float32), topk_indices.to(torch.int32)


def build_vllm_config(max_tokens: int, device: torch.device, moe_backend: str) -> VllmConfig:
    return VllmConfig(
        device_config=DeviceConfig(device=device),
        kernel_config=KernelConfig(moe_backend=moe_backend),
        scheduler_config=SchedulerConfig(
            max_model_len=max_tokens,
            is_encoder_decoder=False,
            max_num_batched_tokens=max_tokens,
            max_num_seqs=1,
        ),
    )


def create_fused_moe(
    cfg: HunyuanMoeConfig,
    vllm_config: VllmConfig,
    prefix: str,
    local_intermediate_size: int,
    device: torch.device,
) -> FusedMoE:
    with set_current_vllm_config(vllm_config):
        moe = FusedMoE(
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
            hidden_size=cfg.hidden_size,
            intermediate_size=local_intermediate_size,
            params_dtype=torch.bfloat16,
            renormalize=False,
            quant_config=None,
            tp_size=1,
            ep_size=1,
            dp_size=1,
            pcp_size=1,
            prefix=prefix,
            custom_routing_function=_hunyuan_unpack_packed_topk,
            router_logits_dtype=torch.float32,
        )
    return moe.to(device)


def load_real_expert_weights(
    model_dir: Path,
    layer_id: int,
    cfg: HunyuanMoeConfig,
    moe: FusedMoE,
    tp_size: int,
    tp_rank: int,
    device: torch.device,
) -> None:
    if cfg.intermediate_size % tp_size != 0:
        raise ValueError(f"intermediate_size={cfg.intermediate_size} not divisible by tp_size={tp_size}")
    local_inter = cfg.intermediate_size // tp_size
    start = tp_rank * local_inter
    keys: list[str] = []
    for expert_id in range(cfg.num_experts):
        keys.extend(
            [
                f"model.layers.{layer_id}.mlp.experts.{expert_id}.gate_and_up_proj.weight",
                f"model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight",
            ]
        )
    tensors = load_tensors(model_dir, keys)
    with torch.no_grad():
        for expert_id in range(cfg.num_experts):
            gate_up = tensors[f"model.layers.{layer_id}.mlp.experts.{expert_id}.gate_and_up_proj.weight"]
            down = tensors[f"model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight"]
            # Hunyuan stores [up, gate]; vLLM FusedMoE expects [gate, up].
            up_full, gate_full = gate_up.chunk(2, dim=0)
            gate = gate_full.narrow(0, start, local_inter).to(device=device, dtype=torch.bfloat16)
            up = up_full.narrow(0, start, local_inter).to(device=device, dtype=torch.bfloat16)
            down_part = down.narrow(1, start, local_inter).to(device=device, dtype=torch.bfloat16)
            moe.w13_weight[expert_id, :local_inter].copy_(gate)
            moe.w13_weight[expert_id, local_inter : 2 * local_inter].copy_(up)
            moe.w2_weight[expert_id].copy_(down_part)


def load_single_expert_dense_weights(
    model_dir: Path,
    layer_id: int,
    expert_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = [
        f"model.layers.{layer_id}.mlp.experts.{expert_id}.gate_and_up_proj.weight",
        f"model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight",
    ]
    tensors = load_tensors(model_dir, keys)
    raw_gate_up = tensors[keys[0]].to(device=device, dtype=torch.bfloat16)
    raw_down = tensors[keys[1]].to(device=device, dtype=torch.bfloat16)
    up_full, gate_full = raw_gate_up.chunk(2, dim=0)
    return torch.cat([gate_full, up_full], dim=0).t().contiguous(), raw_down.t().contiguous()
