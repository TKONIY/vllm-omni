"""Optional HunyuanImage3 MoE routing tracer.

The tracer is disabled unless ``HUNYUAN_MOE_ROUTE_TRACE_DIR`` is set.  It is
used by UAD profiling scripts to compare real request routing distributions
without changing normal serving behavior.
"""

from __future__ import annotations

import atexit
import json
import os
import socket
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

TRACE_DIR_ENV = "HUNYUAN_MOE_ROUTE_TRACE_DIR"
CAPTURE_FILE_ENV = "HUNYUAN_MOE_ROUTE_TRACE_CAPTURE_FILE"

LABEL_TO_MODALITY = {
    0: "dit_text",
    1: "dit_image",
    2: "dit_timestep",
    3: "dit_cond_image",
    4: "dit_other",
}

_current_labels: torch.Tensor | None = None
_trace: dict[str, Any] = {
    "schema_version": 1,
    "created_at": time.time(),
    "metadata": {},
    "stages": {},
}
_has_data = False
_flushed = False
_request_capture_depth = 0


def enabled() -> bool:
    if not os.environ.get(TRACE_DIR_ENV):
        return False
    capture_file = os.environ.get(CAPTURE_FILE_ENV)
    if capture_file and not Path(capture_file).exists():
        return False
    return True


def _rank_metadata() -> dict[str, Any]:
    data: dict[str, Any] = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    for key in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "VLLM_DP_RANK",
        "VLLM_DP_SIZE",
    ):
        value = os.environ.get(key)
        if value is not None:
            data[key.lower()] = value

    try:
        if torch.cuda.is_available():
            data["cuda_current_device"] = torch.accelerator.current_device_index()
    except Exception:
        pass

    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            data["dist_rank"] = dist.get_rank()
            data["dist_world_size"] = dist.get_world_size()
    except Exception:
        pass

    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        data["tp_rank"] = get_tensor_model_parallel_rank()
    except Exception:
        pass

    try:
        from vllm.distributed.parallel_state import get_ep_group

        ep_group = get_ep_group()
        data["ep_rank"] = ep_group.rank_in_group
        data["ep_size"] = ep_group.device_group.size()
    except Exception:
        pass

    return data


def _ensure_stage(stage: str, num_experts: int, top_k: int) -> dict[str, Any]:
    stages = _trace.setdefault("stages", {})
    entry = stages.setdefault(
        stage,
        {
            "num_experts": num_experts,
            "top_k": top_k,
            "num_calls": 0,
            "global": {},
            "by_layer": {},
        },
    )
    entry["num_experts"] = max(int(entry.get("num_experts", 0)), int(num_experts))
    entry["top_k"] = int(top_k)
    return entry


def _empty_modality_entry(num_experts: int) -> dict[str, Any]:
    return {
        "route_counts": [0 for _ in range(num_experts)],
        "token_positions": 0,
        "topk_assignments": 0,
        "calls": 0,
    }


def _accumulate(
    bucket: dict[str, Any],
    modality: str,
    counts: list[int],
    token_positions: int,
) -> None:
    entry = bucket.setdefault(modality, _empty_modality_entry(len(counts)))
    if len(entry["route_counts"]) < len(counts):
        entry["route_counts"].extend([0] * (len(counts) - len(entry["route_counts"])))
    for i, value in enumerate(counts):
        entry["route_counts"][i] += int(value)
    entry["token_positions"] += int(token_positions)
    entry["topk_assignments"] += int(sum(counts))
    entry["calls"] += 1


def _counts(topk_indices: torch.Tensor, num_experts: int) -> list[int]:
    flat = topk_indices.reshape(-1).to(torch.int64)
    counts = torch.bincount(flat, minlength=num_experts)
    return [int(v) for v in counts.detach().cpu().tolist()]


def record_routes(
    *,
    stage: str,
    layer_id: int,
    topk_indices: torch.Tensor,
    num_experts: int,
    default_modality: str,
    labels: torch.Tensor | None = None,
) -> None:
    if not enabled():
        return
    if stage == "dit" and _request_capture_depth <= 0:
        return

    global _has_data
    with torch.no_grad():
        topk_indices = topk_indices.detach()
        top_k = int(topk_indices.shape[-1]) if topk_indices.ndim > 1 else 1
        num_tokens = int(topk_indices.reshape(-1, top_k).shape[0])
        stage_entry = _ensure_stage(stage, num_experts, top_k)
        stage_entry["num_calls"] += 1
        layer_key = str(int(layer_id))
        layer_entry = stage_entry["by_layer"].setdefault(layer_key, {})

        active_labels = labels if labels is not None else _current_labels
        if active_labels is None:
            counts = _counts(topk_indices, num_experts)
            _accumulate(stage_entry["global"], default_modality, counts, num_tokens)
            _accumulate(layer_entry, default_modality, counts, num_tokens)
            _has_data = True
            return

        active_labels = active_labels.detach().reshape(-1).to(topk_indices.device)
        if active_labels.numel() != num_tokens:
            counts = _counts(topk_indices, num_experts)
            _accumulate(stage_entry["global"], default_modality, counts, num_tokens)
            _accumulate(layer_entry, default_modality, counts, num_tokens)
            stage_entry.setdefault("warnings", []).append(
                {
                    "layer_id": int(layer_id),
                    "message": (
                        f"label/token mismatch: labels={active_labels.numel()} "
                        f"tokens={num_tokens}; used {default_modality}"
                    ),
                }
            )
            _has_data = True
            return

        flat_topk = topk_indices.reshape(num_tokens, top_k)
        for label_id, modality in LABEL_TO_MODALITY.items():
            mask = active_labels == label_id
            token_positions = int(mask.sum().item())
            if token_positions == 0:
                continue
            counts = _counts(flat_topk[mask], num_experts)
            _accumulate(stage_entry["global"], modality, counts, token_positions)
            _accumulate(layer_entry, modality, counts, token_positions)
            _has_data = True


@contextmanager
def token_modality_labels(labels: torch.Tensor | None) -> Iterator[None]:
    global _current_labels
    if not enabled():
        yield
        return
    previous = _current_labels
    _current_labels = labels
    try:
        yield
    finally:
        _current_labels = previous


@contextmanager
def request_capture(request_id: str | None, request_ids: list[str] | None = None) -> Iterator[None]:
    global _request_capture_depth
    if not enabled():
        yield
        return

    ids = []
    if request_id:
        ids.append(request_id)
    if request_ids:
        ids.extend(request_ids)
    if any(str(value).startswith("dummy_req_id") for value in ids):
        yield
        return

    _request_capture_depth += 1
    try:
        yield
    finally:
        _request_capture_depth -= 1


def flush() -> None:
    global _flushed
    if _flushed or not enabled() or not _has_data:
        return
    _flushed = True
    out_dir = Path(os.environ[TRACE_DIR_ENV])
    out_dir.mkdir(parents=True, exist_ok=True)
    _trace["metadata"] = _rank_metadata()
    _trace["finished_at"] = time.time()
    stages = "-".join(sorted(_trace.get("stages", {}).keys())) or "empty"
    rank = _trace["metadata"].get("dist_rank", _trace["metadata"].get("rank", "na"))
    path = out_dir / f"hunyuan_moe_route_trace_{stages}_rank{rank}_pid{os.getpid()}.json"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(_trace, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


atexit.register(flush)
