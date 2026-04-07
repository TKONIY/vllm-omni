# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare vLLM-Omni DreamZero TP=1 vs TP=2 precision.

Run directly:
    PYTHONPATH=. .venv/bin/python tests/dreamzero/test_causal_wan_model_tp1_vs_tp2.py
"""

from __future__ import annotations

import os
import socket
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.multiprocessing as mp
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

os.environ.setdefault("ATTENTION_BACKEND", "torch")
os.environ.setdefault("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")

TP_SIZE = 2
DTYPE = torch.float32
ATOL = 1e-5
RTOL = 1e-5
FULL_MODEL_ATOL = 2e-4
FULL_MODEL_RTOL = 2e-4

TINY_CFG = dict(
    model_type="t2v",
    patch_size=(1, 2, 2),
    frame_seqlen=4,
    text_len=16,
    in_dim=4,
    dim=64,
    ffn_dim=128,
    freq_dim=32,
    text_dim=64,
    out_dim=4,
    num_heads=4,
    num_layers=2,
    qk_norm=True,
    cross_attn_norm=True,
    num_frame_per_block=1,
    action_dim=8,
    num_action_per_block=4,
    num_state_per_block=1,
    max_num_embodiments=4,
    hidden_size=32,
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _device(local_rank: int) -> torch.device:
    return torch.device(f"cuda:{local_rank}")


def _set_common_env(rank: int, world_size: int, master_port: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["ATTENTION_BACKEND"] = "torch"
    os.environ["DIFFUSION_ATTENTION_BACKEND"] = "TORCH_SDPA"


def _set_deterministic() -> None:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _load_vllm_param(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, loaded_weight.to(device=param.device, dtype=param.dtype))


def _slice_dim(tensor: torch.Tensor, rank: int, world_size: int, dim: int) -> torch.Tensor:
    dim %= tensor.ndim
    shard = tensor.shape[dim] // world_size
    return tensor.narrow(dim, rank * shard, shard).contiguous()


def _slice_heads(tensor: torch.Tensor, rank: int, world_size: int, head_dim: int) -> torch.Tensor:
    return _slice_dim(tensor, rank, world_size, head_dim)


def _make_empty_kv(
    num_layers: int,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    *,
    device: torch.device,
) -> list[torch.Tensor]:
    return [
        torch.zeros(2, batch_size, 0, num_heads, head_dim, device=device, dtype=DTYPE)
        for _ in range(num_layers)
    ]


def _make_crossattn_cache(num_layers: int) -> list[dict[str, object]]:
    return [{"is_init": False, "k": None, "v": None} for _ in range(num_layers)]


def _clone_crossattn_cache(caches: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "is_init": cache["is_init"],
            "k": None if cache["k"] is None else cache["k"].clone(),
            "v": None if cache["v"] is None else cache["v"].clone(),
        }
        for cache in caches
    ]


def _cache_to_cpu(caches: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "is_init": cache["is_init"],
            "k": None if cache["k"] is None else cache["k"].detach().cpu(),
            "v": None if cache["v"] is None else cache["v"].detach().cpu(),
        }
        for cache in caches
    ]


def _assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    max_error: torch.Tensor,
    *,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> None:
    actual = actual.detach().float()
    expected = expected.detach().float()
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: shape mismatch actual={tuple(actual.shape)}, expected={tuple(expected.shape)}")
    diff = 0.0 if actual.numel() == 0 else (actual - expected).abs().max().item()
    max_error.fill_(max(max_error.item(), diff))
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(f"{name}: max_diff={diff:.3e}, atol={atol}, rtol={rtol}")


def _assert_crossattn_cache_matches(
    name: str,
    actual: list[dict[str, object]],
    expected: list[dict[str, object]],
    *,
    local_rank: int,
    max_error: torch.Tensor,
) -> None:
    for idx, (actual_cache, expected_cache) in enumerate(zip(actual, expected, strict=True)):
        if actual_cache["is_init"] != expected_cache["is_init"]:
            raise AssertionError(
                f"{name}[{idx}].is_init mismatch: actual={actual_cache['is_init']}, expected={expected_cache['is_init']}"
            )
        if not actual_cache["is_init"]:
            if actual_cache["k"] is not None or actual_cache["v"] is not None:
                raise AssertionError(f"{name}[{idx}] should remain uninitialized")
            continue
        assert isinstance(actual_cache["k"], torch.Tensor)
        assert isinstance(actual_cache["v"], torch.Tensor)
        assert isinstance(expected_cache["k"], torch.Tensor)
        assert isinstance(expected_cache["v"], torch.Tensor)
        _assert_close(
            f"{name}[{idx}].k",
            actual_cache["k"],
            _slice_heads(expected_cache["k"], local_rank, TP_SIZE, head_dim=2).to(actual_cache["k"].device),
            max_error,
        )
        _assert_close(
            f"{name}[{idx}].v",
            actual_cache["v"],
            _slice_heads(expected_cache["v"], local_rank, TP_SIZE, head_dim=2).to(actual_cache["v"].device),
            max_error,
        )


def _init_model_parallel(rank: int, world_size: int, master_port: int) -> torch.device:
    from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
    from vllm_omni.platforms import current_omni_platform

    device = _device(rank)
    _set_common_env(rank, world_size, master_port)
    torch.cuda.set_device(device)
    current_omni_platform.set_device(device)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method="env://",
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        pipeline_model_parallel_size=1,
    )
    return device


def _build_tp1_reference(out_path: Path) -> None:
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import CausalWanModel

    master_port = _find_free_port()
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cuda"))):
        device = _init_model_parallel(rank=0, world_size=1, master_port=master_port)
        _set_deterministic()
        try:
            model = CausalWanModel(**TINY_CFG).to(device=device, dtype=DTYPE).eval()
            state = {name: param.detach().cpu() for name, param in model.named_parameters()}

            batch_size = 1
            num_heads = TINY_CFG["num_heads"]
            head_dim = TINY_CFG["dim"] // num_heads
            kv_cache = _make_empty_kv(TINY_CFG["num_layers"], batch_size, num_heads, head_dim, device=device)
            crossattn_cache = _make_crossattn_cache(TINY_CFG["num_layers"])

            x_prefill = torch.randn(batch_size, 4, 1, 4, 4, device=device, dtype=DTYPE)
            timestep_prefill = torch.tensor([[0]], device=device)
            context = torch.randn(batch_size, 16, 64, device=device, dtype=DTYPE)

            with torch.no_grad():
                video_1, action_1, kv_1 = model(
                    x=x_prefill,
                    timestep=timestep_prefill,
                    context=context,
                    seq_len=4,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start_frame=0,
                    y=None,
                    clip_feature=None,
                )

            crossattn_cache_prefill = _clone_crossattn_cache(crossattn_cache)

            x_step = torch.randn(batch_size, 4, 1, 4, 4, device=device, dtype=DTYPE)
            timestep_step = torch.tensor([[1]], device=device)
            action = torch.randn(batch_size, 4, TINY_CFG["action_dim"], device=device, dtype=DTYPE)
            timestep_action = torch.tensor([[0, 0, 1, 1]], device=device)
            state_input = torch.randn(batch_size, 1, 64, device=device, dtype=DTYPE)
            embodiment_id = torch.tensor([0], device=device)

            with torch.no_grad():
                video_2, action_2, kv_2 = model(
                    x=x_step,
                    timestep=timestep_step,
                    context=context,
                    seq_len=4,
                    kv_cache=[cache.clone() for cache in kv_1],
                    crossattn_cache=crossattn_cache,
                    current_start_frame=1,
                    action=action,
                    timestep_action=timestep_action,
                    state=state_input,
                    embodiment_id=embodiment_id,
                    y=None,
                    clip_feature=None,
                )

            reference = {
                "state": state,
                "inputs": {
                    "x_prefill": x_prefill.detach().cpu(),
                    "timestep_prefill": timestep_prefill.detach().cpu(),
                    "context": context.detach().cpu(),
                    "x_step": x_step.detach().cpu(),
                    "timestep_step": timestep_step.detach().cpu(),
                    "action": action.detach().cpu(),
                    "timestep_action": timestep_action.detach().cpu(),
                    "state_input": state_input.detach().cpu(),
                    "embodiment_id": embodiment_id.detach().cpu(),
                },
                "prefill": {
                    "video": video_1.detach().cpu(),
                    "action": None if action_1 is None else action_1.detach().cpu(),
                    "kv": [cache.detach().cpu() for cache in kv_1],
                    "crossattn_cache": _cache_to_cpu(crossattn_cache_prefill),
                },
                "step": {
                    "video": video_2.detach().cpu(),
                    "action": None if action_2 is None else action_2.detach().cpu(),
                    "kv": [cache.detach().cpu() for cache in kv_2],
                    "crossattn_cache": _cache_to_cpu(crossattn_cache),
                },
            }
            torch.save(reference, out_path)
        finally:
            cleanup_dist_env_and_memory()


def _run_tp2_worker(local_rank: int, world_size: int, master_port: int, reference_path: str) -> None:
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import CausalWanModel

    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cuda"))):
        device = _init_model_parallel(rank=local_rank, world_size=world_size, master_port=master_port)
        _set_deterministic()
        try:
            reference: dict[str, Any] = torch.load(reference_path, map_location="cpu")
            model = CausalWanModel(**TINY_CFG).to(device=device, dtype=DTYPE).eval()
            params = dict(model.named_parameters())
            for name, weight in reference["state"].items():
                _load_vllm_param(params[name], weight)

            inputs = {key: value.to(device=device) for key, value in reference["inputs"].items()}

            batch_size = 1
            tp_heads = TINY_CFG["num_heads"] // TP_SIZE
            head_dim = TINY_CFG["dim"] // TINY_CFG["num_heads"]
            kv_cache = _make_empty_kv(TINY_CFG["num_layers"], batch_size, tp_heads, head_dim, device=device)
            crossattn_cache = _make_crossattn_cache(TINY_CFG["num_layers"])

            max_error = torch.zeros(1, device=device, dtype=torch.float32)

            with torch.no_grad():
                video_1, action_1, kv_1 = model(
                    x=inputs["x_prefill"],
                    timestep=inputs["timestep_prefill"],
                    context=inputs["context"],
                    seq_len=4,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start_frame=0,
                    y=None,
                    clip_feature=None,
                )

            _assert_close(
                "prefill.video",
                video_1,
                reference["prefill"]["video"].to(device=device),
                max_error,
                atol=FULL_MODEL_ATOL,
                rtol=FULL_MODEL_RTOL,
            )
            if action_1 is not None or reference["prefill"]["action"] is not None:
                raise AssertionError("prefill.action should stay None for TP1 and TP2")
            for idx, (actual_kv, expected_kv) in enumerate(zip(kv_1, reference["prefill"]["kv"], strict=True)):
                _assert_close(
                    f"prefill.kv[{idx}]",
                    actual_kv,
                    _slice_heads(expected_kv.to(device=device), local_rank, TP_SIZE, head_dim=3),
                    max_error,
                )
            _assert_crossattn_cache_matches(
                "prefill.crossattn_cache",
                crossattn_cache,
                reference["prefill"]["crossattn_cache"],
                local_rank=local_rank,
                max_error=max_error,
            )

            crossattn_cache_prefill = _clone_crossattn_cache(crossattn_cache)

            with torch.no_grad():
                video_2, action_2, kv_2 = model(
                    x=inputs["x_step"],
                    timestep=inputs["timestep_step"],
                    context=inputs["context"],
                    seq_len=4,
                    kv_cache=[cache.clone() for cache in kv_1],
                    crossattn_cache=crossattn_cache,
                    current_start_frame=1,
                    action=inputs["action"],
                    timestep_action=inputs["timestep_action"],
                    state=inputs["state_input"],
                    embodiment_id=inputs["embodiment_id"],
                    y=None,
                    clip_feature=None,
                )

            _assert_close(
                "step.video",
                video_2,
                reference["step"]["video"].to(device=device),
                max_error,
                atol=FULL_MODEL_ATOL,
                rtol=FULL_MODEL_RTOL,
            )
            expected_action_2 = reference["step"]["action"]
            if expected_action_2 is None or action_2 is None:
                raise AssertionError("step.action should be present for TP1 and TP2")
            _assert_close(
                "step.action",
                action_2,
                expected_action_2.to(device=device),
                max_error,
                atol=FULL_MODEL_ATOL,
                rtol=FULL_MODEL_RTOL,
            )
            for idx, (actual_kv, expected_kv) in enumerate(zip(kv_2, reference["step"]["kv"], strict=True)):
                _assert_close(
                    f"step.kv[{idx}]",
                    actual_kv,
                    _slice_heads(expected_kv.to(device=device), local_rank, TP_SIZE, head_dim=3),
                    max_error,
                )
            _assert_crossattn_cache_matches(
                "step.crossattn_cache",
                crossattn_cache,
                reference["step"]["crossattn_cache"],
                local_rank=local_rank,
                max_error=max_error,
            )
            _assert_crossattn_cache_matches(
                "step.crossattn_cache_prefill",
                crossattn_cache_prefill,
                reference["prefill"]["crossattn_cache"],
                local_rank=local_rank,
                max_error=max_error,
            )

            torch.distributed.all_reduce(max_error, op=torch.distributed.ReduceOp.MAX)
            if local_rank == 0:
                print(f"TP1 vs TP2: PASS, max_diff={max_error.item():.3e}")
        finally:
            cleanup_dist_env_and_memory()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required.")
    if torch.cuda.device_count() < TP_SIZE:
        raise RuntimeError(f"TP=2 requires {TP_SIZE} GPUs.")

    with tempfile.TemporaryDirectory(prefix="dreamzero_tp1_vs_tp2_") as tmpdir:
        reference_path = Path(tmpdir) / "reference.pt"
        _build_tp1_reference(reference_path)
        master_port = _find_free_port()
        mp.spawn(
            _run_tp2_worker,
            args=(TP_SIZE, master_port, str(reference_path)),
            nprocs=TP_SIZE,
            join=True,
        )


if __name__ == "__main__":
    main()
