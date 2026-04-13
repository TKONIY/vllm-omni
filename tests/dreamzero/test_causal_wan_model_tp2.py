# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""TP=2 GPU precision tests for DreamZero CausalWanModel.

Run:
    PYTHONPATH=. /home/yangshen/miniconda3/envs/dreamzero/bin/python -m pytest \
        tests/dreamzero/test_causal_wan_model_tp2.py -v -s

Or run directly:
    PYTHONPATH=. /home/yangshen/miniconda3/envs/dreamzero/bin/python \
        tests/dreamzero/test_causal_wan_model_tp2.py
"""

from __future__ import annotations

import contextlib
import os
import socket
import sys

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

os.environ.setdefault("ATTENTION_BACKEND", "torch")
os.environ.setdefault("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../third_party/dreamzero"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../dreamzero"))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="2 GPUs required",
)

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


def _assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> None:
    actual = actual.detach().float()
    expected = expected.detach().float()
    assert actual.shape == expected.shape, (
        f"{name}: shape mismatch actual={tuple(actual.shape)}, expected={tuple(expected.shape)}"
    )
    max_diff = 0.0 if actual.numel() == 0 else (actual - expected).abs().max().item()
    assert torch.allclose(actual, expected, atol=atol, rtol=rtol), (
        f"{name}: max_diff={max_diff:.3e}, atol={atol}, rtol={rtol}"
    )


def _load_vllm_param(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, loaded_weight)


def _broadcast_module(module: torch.nn.Module) -> None:
    if not torch.distributed.is_initialized():
        return

    for param in module.parameters():
        torch.distributed.broadcast(param.data, src=0)
    for buffer in module.buffers():
        torch.distributed.broadcast(buffer.data, src=0)


def _sync_module(vllm_module: torch.nn.Module, dreamzero_module: torch.nn.Module) -> None:
    _broadcast_module(dreamzero_module)
    vllm_params = dict(vllm_module.named_parameters())
    dreamzero_params = dict(dreamzero_module.named_parameters())

    missing = sorted(set(dreamzero_params) - set(vllm_params))
    extra = sorted(set(vllm_params) - set(dreamzero_params))
    assert not missing, f"Missing params in vllm module: {missing}"
    assert not extra, f"Unexpected params in vllm module: {extra}"

    for name, dz_param in dreamzero_params.items():
        _load_vllm_param(vllm_params[name], dz_param.detach())


def _slice_dim(tensor: torch.Tensor, rank: int, world_size: int, dim: int) -> torch.Tensor:
    dim %= tensor.ndim
    assert tensor.shape[dim] % world_size == 0, (
        f"Cannot shard dim={dim} with size={tensor.shape[dim]} across world_size={world_size}"
    )
    shard = tensor.shape[dim] // world_size
    return tensor.narrow(dim, rank * shard, shard).contiguous()


def _slice_heads(tensor: torch.Tensor, rank: int, world_size: int, head_dim: int) -> torch.Tensor:
    return _slice_dim(tensor, rank, world_size, head_dim)


def _shared_randn(*shape: int, device: torch.device, dtype: torch.dtype = DTYPE) -> torch.Tensor:
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        tensor = torch.randn(*shape, device=device, dtype=dtype)
    else:
        tensor = torch.empty(*shape, device=device, dtype=dtype)
    if torch.distributed.is_initialized():
        torch.distributed.broadcast(tensor, src=0)
    return tensor


def _make_empty_kv(
    num_layers: int,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    *,
    device: torch.device,
) -> list[torch.Tensor]:
    return [torch.zeros(2, batch_size, 0, num_heads, head_dim, device=device, dtype=DTYPE) for _ in range(num_layers)]


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


def _assert_crossattn_cache_uninitialized(name: str, cache: dict[str, object]) -> None:
    assert cache["is_init"] is False, f"{name}.is_init should stay False"
    assert cache["k"] is None, f"{name}.k should stay None"
    assert cache["v"] is None, f"{name}.v should stay None"


def _assert_crossattn_cache_initialized(
    name: str,
    cache: dict[str, object],
    *,
    batch_size: int,
    context_tokens: int,
    num_heads: int,
    head_dim: int,
) -> None:
    assert cache["is_init"] is True, f"{name}.is_init should be True"
    assert isinstance(cache["k"], torch.Tensor), f"{name}.k should be a tensor"
    assert isinstance(cache["v"], torch.Tensor), f"{name}.v should be a tensor"
    expected_shape = (batch_size, context_tokens, num_heads, head_dim)
    assert cache["k"].shape == expected_shape, (
        f"{name}.k shape mismatch: actual={tuple(cache['k'].shape)}, expected={expected_shape}"
    )
    assert cache["v"].shape == expected_shape, (
        f"{name}.v shape mismatch: actual={tuple(cache['v'].shape)}, expected={expected_shape}"
    )


def _assert_crossattn_cache_reused(
    name: str,
    actual: dict[str, object],
    expected: dict[str, object],
    *,
    batch_size: int,
    context_tokens: int,
    num_heads: int,
    head_dim: int,
) -> None:
    _assert_crossattn_cache_initialized(
        name,
        actual,
        batch_size=batch_size,
        context_tokens=context_tokens,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    assert expected["is_init"] is True, f"{name}.expected.is_init should be True"
    assert isinstance(expected["k"], torch.Tensor), f"{name}.expected.k should be a tensor"
    assert isinstance(expected["v"], torch.Tensor), f"{name}.expected.v should be a tensor"
    _assert_close(f"{name}.k", actual["k"], expected["k"])
    _assert_close(f"{name}.v", actual["v"], expected["v"])


def _assert_crossattn_cache_matches(
    name: str,
    actual: dict[str, object],
    expected_full: dict[str, object],
    *,
    local_rank: int,
) -> None:
    assert actual["is_init"] == expected_full["is_init"], (
        f"{name}.is_init mismatch: actual={actual['is_init']}, expected={expected_full['is_init']}"
    )
    if not actual["is_init"]:
        assert actual["k"] is None and actual["v"] is None
        assert expected_full["k"] is None and expected_full["v"] is None
        return

    assert isinstance(actual["k"], torch.Tensor)
    assert isinstance(actual["v"], torch.Tensor)
    assert isinstance(expected_full["k"], torch.Tensor)
    assert isinstance(expected_full["v"], torch.Tensor)
    _assert_close(
        f"{name}.k",
        actual["k"],
        _slice_heads(expected_full["k"], local_rank, TP_SIZE, head_dim=2),
    )
    _assert_close(
        f"{name}.v",
        actual["v"],
        _slice_heads(expected_full["v"], local_rank, TP_SIZE, head_dim=2),
    )


@contextlib.contextmanager
def _patch_dreamzero_attention():
    import groot.vla.model.dreamzero.modules.attention as base_attention
    import groot.vla.model.dreamzero.modules.wan2_1_attention as wan_attention
    import groot.vla.model.dreamzero.modules.wan2_1_submodule as wan_submodule
    import groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk as wan_chunk

    old_backend = os.environ.get("ATTENTION_BACKEND")
    os.environ["ATTENTION_BACKEND"] = "torch"

    original_base_flash = base_attention.flash_attention
    original_wan_attention_flash = wan_attention.flash_attention
    original_wan_submodule_flash = wan_submodule.flash_attention
    original_attention_init = wan_attention.AttentionModule.__init__
    original_t2v_forward = wan_submodule.WanT2VCrossAttention.forward

    def sdpa_flash_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_lens=None,
        k_lens=None,
        dropout_p: float = 0.0,
        softmax_scale=None,
        q_scale=None,
        causal: bool = False,
        window_size=(-1, -1),
        deterministic: bool = False,
        dtype: torch.dtype = torch.float32,
        version=None,
    ) -> torch.Tensor:
        del window_size, deterministic, dtype, version
        assert q_lens is None and k_lens is None, "varlen attention is not covered in this test"
        out_dtype = q.dtype
        if q_scale is not None:
            q = q * q_scale
        q = q.transpose(1, 2).float()
        k = k.transpose(1, 2).float()
        v = v.transpose(1, 2).float()
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=causal,
            dropout_p=dropout_p,
            scale=softmax_scale,
        )
        return out.transpose(1, 2).contiguous().to(out_dtype)

    def patched_attention_init(
        self,
        num_heads: int,
        head_dim: int,
        dropout_p: float = 0.0,
        softmax_scale=None,
        q_scale=None,
        causal: bool = False,
        window_size=None,
        deterministic: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        backend: str | None = None,
    ) -> None:
        del dtype, backend
        original_attention_init(
            self,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=torch.float32,
            backend="torch",
        )

    def patched_t2v_forward(self, x, context, context_lens=None, crossattn_cache=None):
        return original_t2v_forward(self, x, context, context_lens, crossattn_cache)

    base_attention.flash_attention = sdpa_flash_attention
    wan_attention.flash_attention = sdpa_flash_attention
    wan_submodule.flash_attention = sdpa_flash_attention
    wan_attention.AttentionModule.__init__ = patched_attention_init
    wan_submodule.WanT2VCrossAttention.forward = patched_t2v_forward
    wan_chunk.WAN_CROSSATTENTION_CLASSES["t2v_cross_attn"].forward = patched_t2v_forward

    try:
        yield
    finally:
        base_attention.flash_attention = original_base_flash
        wan_attention.flash_attention = original_wan_attention_flash
        wan_submodule.flash_attention = original_wan_submodule_flash
        wan_attention.AttentionModule.__init__ = original_attention_init
        wan_submodule.WanT2VCrossAttention.forward = original_t2v_forward
        wan_chunk.WAN_CROSSATTENTION_CLASSES["t2v_cross_attn"].forward = original_t2v_forward
        if old_backend is None:
            os.environ.pop("ATTENTION_BACKEND", None)
        else:
            os.environ["ATTENTION_BACKEND"] = old_backend


def _init_tp_env(local_rank: int, world_size: int, master_port: int) -> torch.device:
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    from vllm_omni.platforms import current_omni_platform

    device = _device(local_rank)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["ATTENTION_BACKEND"] = "torch"
    os.environ["DIFFUSION_ATTENTION_BACKEND"] = "TORCH_SDPA"

    torch.cuda.set_device(device)
    current_omni_platform.set_device(device)

    init_distributed_environment(
        world_size=world_size,
        rank=local_rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        pipeline_model_parallel_size=1,
    )
    return device


def _make_vllm_model(device: torch.device):
    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import CausalWanModel

    model = CausalWanModel(**TINY_CFG).to(device=device, dtype=DTYPE)
    model.eval()
    return model


def _make_dreamzero_model(device: torch.device):
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel

    model = CausalWanModel(**TINY_CFG).to(device=device, dtype=DTYPE)
    model.eval()
    return model


def _test_hotpath_layer_types_tp2(device: torch.device) -> None:
    from vllm.model_executor.layers.conv import Conv3dLayer
    from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear

    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import DistributedRMSNorm

    model = _make_vllm_model(device)
    block = model.blocks[0]
    tp_heads = TINY_CFG["num_heads"] // TP_SIZE
    tp_dim = TINY_CFG["dim"] // TP_SIZE
    tp_ffn_dim = TINY_CFG["ffn_dim"] // TP_SIZE

    assert isinstance(model.patch_embedding, Conv3dLayer)
    assert isinstance(block.self_attn.q, ColumnParallelLinear)
    assert isinstance(block.self_attn.k, ColumnParallelLinear)
    assert isinstance(block.self_attn.v, ColumnParallelLinear)
    assert isinstance(block.self_attn.o, RowParallelLinear)
    assert isinstance(block.self_attn.norm_q, DistributedRMSNorm)
    assert isinstance(block.self_attn.norm_k, DistributedRMSNorm)
    assert isinstance(block.cross_attn.q, ColumnParallelLinear)
    assert isinstance(block.cross_attn.k, ColumnParallelLinear)
    assert isinstance(block.cross_attn.v, ColumnParallelLinear)
    assert isinstance(block.cross_attn.o, RowParallelLinear)
    assert isinstance(block.ffn[0], ColumnParallelLinear)
    assert isinstance(block.ffn[2], RowParallelLinear)

    assert block.self_attn.tp_num_heads == tp_heads
    assert block.self_attn.tp_inner_dim == tp_dim
    assert block.self_attn.q.weight.shape == (tp_dim, TINY_CFG["dim"])
    assert block.self_attn.k.weight.shape == (tp_dim, TINY_CFG["dim"])
    assert block.self_attn.v.weight.shape == (tp_dim, TINY_CFG["dim"])
    assert block.self_attn.o.weight.shape == (TINY_CFG["dim"], tp_dim)
    assert block.cross_attn.q.weight.shape == (tp_dim, TINY_CFG["dim"])
    assert block.cross_attn.o.weight.shape == (TINY_CFG["dim"], tp_dim)
    assert block.ffn[0].weight.shape == (tp_ffn_dim, TINY_CFG["dim"])
    assert block.ffn[2].weight.shape == (TINY_CFG["dim"], tp_ffn_dim)


def _test_distributed_rmsnorm_precision_tp2(local_rank: int, device: torch.device) -> None:
    from groot.vla.model.dreamzero.modules.wan2_1_submodule import WanRMSNorm

    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import DistributedRMSNorm

    shard_dim = 64 // TP_SIZE
    vllm_norm = DistributedRMSNorm(shard_dim, eps=1e-6).to(device=device, dtype=DTYPE)
    dreamzero_norm = WanRMSNorm(64, eps=1e-6).to(device=device, dtype=DTYPE)
    _load_vllm_param(vllm_norm.weight, dreamzero_norm.weight.detach())

    x_full = _shared_randn(2, 10, 64, device=device, dtype=DTYPE)
    x_local = _slice_dim(x_full, local_rank, TP_SIZE, dim=-1)
    expected = _slice_dim(dreamzero_norm(x_full), local_rank, TP_SIZE, dim=-1)
    _assert_close("DistributedRMSNorm.tp2", vllm_norm(x_local), expected)


def _test_t2v_cross_attn_precision_tp2(local_rank: int, device: torch.device) -> None:
    from groot.vla.model.dreamzero.modules.wan2_1_submodule import WanT2VCrossAttention as DreamZeroCrossAttention

    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import WanT2VCrossAttention

    vllm_attn = WanT2VCrossAttention(64, 4, qk_norm=True, eps=1e-6).to(device=device, dtype=DTYPE)
    dreamzero_attn = DreamZeroCrossAttention(64, 4, (-1, -1), True, 1e-6).to(device=device, dtype=DTYPE)
    _sync_module(vllm_attn, dreamzero_attn)

    x = _shared_randn(1, 8, 64, device=device, dtype=DTYPE)
    x_step = _shared_randn(1, 8, 64, device=device, dtype=DTYPE)
    context = _shared_randn(1, 16, 64, device=device, dtype=DTYPE)
    vllm_cache = {"is_init": False, "k": None, "v": None}
    dreamzero_cache = {"is_init": False, "k": None, "v": None}

    _assert_close(
        "WanT2VCrossAttention.prefill",
        vllm_attn(x, context, crossattn_cache=vllm_cache),
        dreamzero_attn(x, context, context_lens=None, crossattn_cache=dreamzero_cache),
    )
    _assert_crossattn_cache_matches(
        "WanT2VCrossAttention.cache",
        vllm_cache,
        dreamzero_cache,
        local_rank=local_rank,
    )
    _assert_close(
        "WanT2VCrossAttention.reuse",
        vllm_attn(x_step, context, crossattn_cache=vllm_cache),
        dreamzero_attn(x_step, context, context_lens=None, crossattn_cache=dreamzero_cache),
    )


def _test_i2v_cross_attn_precision_tp2(local_rank: int, device: torch.device) -> None:
    from groot.vla.model.dreamzero.modules.wan2_1_submodule import WanI2VCrossAttention as DreamZeroCrossAttention

    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import WanI2VCrossAttention

    vllm_attn = WanI2VCrossAttention(64, 4, qk_norm=True, eps=1e-6).to(device=device, dtype=DTYPE)
    dreamzero_attn = DreamZeroCrossAttention(64, 4, (-1, -1), True, 1e-6).to(device=device, dtype=DTYPE)
    _sync_module(vllm_attn, dreamzero_attn)

    x = _shared_randn(1, 8, 64, device=device, dtype=DTYPE)
    x_step = _shared_randn(1, 8, 64, device=device, dtype=DTYPE)
    context = _shared_randn(1, 257 + 16, 64, device=device, dtype=DTYPE)
    vllm_cache = {"is_init": False, "k": None, "v": None}
    dreamzero_cache = {"is_init": False, "k": None, "v": None}

    _assert_close(
        "WanI2VCrossAttention.prefill",
        vllm_attn(x, context, crossattn_cache=vllm_cache),
        dreamzero_attn(x, context, crossattn_cache=dreamzero_cache),
    )
    _assert_crossattn_cache_matches(
        "WanI2VCrossAttention.cache",
        vllm_cache,
        dreamzero_cache,
        local_rank=local_rank,
    )
    _assert_close(
        "WanI2VCrossAttention.reuse",
        vllm_attn(x_step, context, crossattn_cache=vllm_cache),
        dreamzero_attn(x_step, context, crossattn_cache=dreamzero_cache),
    )


def _test_self_attn_precision_tp2(local_rank: int, device: torch.device) -> None:
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
        CausalWanSelfAttention as DreamZeroSelfAttention,
    )

    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import (
        CausalWanSelfAttention,
        rope_params,
    )

    tp_num_heads = TINY_CFG["num_heads"] // TP_SIZE
    head_dim = TINY_CFG["dim"] // TINY_CFG["num_heads"]

    vllm_attn = CausalWanSelfAttention(64, 4, 4, num_action_per_block=4, num_state_per_block=1).to(
        device=device, dtype=DTYPE
    )
    dreamzero_attn = DreamZeroSelfAttention(64, 4, 4, num_action_per_block=4, num_state_per_block=1).to(
        device=device, dtype=DTYPE
    )
    _sync_module(vllm_attn, dreamzero_attn)

    kv_cache_vllm = torch.zeros(2, 1, 0, tp_num_heads, head_dim, device=device, dtype=DTYPE)
    kv_cache_dreamzero = torch.zeros(2, 1, 0, TINY_CFG["num_heads"], head_dim, device=device, dtype=DTYPE)
    freqs = rope_params(1024, head_dim)[:4].view(-1, 1, head_dim // 2).to(device=device)
    freqs_action = rope_params(10240, head_dim).to(device=device)
    freqs_state = rope_params(1024, head_dim).to(device=device)
    x = _shared_randn(1, 4, 64, device=device, dtype=DTYPE)

    vllm_out, vllm_kv = vllm_attn(x, freqs, freqs_action, freqs_state, None, kv_cache_vllm, current_start_frame=0)
    dreamzero_out, dreamzero_kv = dreamzero_attn(
        x,
        freqs,
        freqs_action,
        freqs_state,
        None,
        kv_cache_dreamzero,
        current_start_frame=0,
    )

    _assert_close("CausalWanSelfAttention.out.tp2", vllm_out, dreamzero_out)
    assert isinstance(vllm_kv, torch.Tensor)
    _assert_close(
        "CausalWanSelfAttention.kv.tp2",
        vllm_kv,
        _slice_heads(dreamzero_kv, local_rank, TP_SIZE, head_dim=3),
    )


def _test_attention_block_precision_tp2(local_rank: int, device: torch.device) -> None:
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
        CausalWanAttentionBlock as DreamZeroAttentionBlock,
    )

    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import (
        CausalWanAttentionBlock,
        rope_params,
    )

    tp_num_heads = TINY_CFG["num_heads"] // TP_SIZE
    head_dim = TINY_CFG["dim"] // TINY_CFG["num_heads"]

    vllm_block = CausalWanAttentionBlock(
        "t2v_cross_attn",
        64,
        128,
        4,
        4,
        num_action_per_block=4,
        num_state_per_block=1,
    ).to(device=device, dtype=DTYPE)
    dreamzero_block = DreamZeroAttentionBlock(
        "t2v_cross_attn",
        64,
        128,
        4,
        4,
        num_action_per_block=4,
        num_state_per_block=1,
    ).to(device=device, dtype=DTYPE)
    _sync_module(vllm_block, dreamzero_block)

    kv_cache_vllm = torch.zeros(2, 1, 0, tp_num_heads, head_dim, device=device, dtype=DTYPE)
    kv_cache_dreamzero = torch.zeros(2, 1, 0, TINY_CFG["num_heads"], head_dim, device=device, dtype=DTYPE)
    freqs = rope_params(1024, head_dim)[:4].view(-1, 1, head_dim // 2).to(device=device)
    freqs_action = rope_params(10240, head_dim).to(device=device)
    freqs_state = rope_params(1024, head_dim).to(device=device)
    x = _shared_randn(1, 4, 64, device=device, dtype=DTYPE)
    e = _shared_randn(1, 4, 6, 64, device=device, dtype=DTYPE)
    context = _shared_randn(1, 16, 64, device=device, dtype=DTYPE)

    vllm_out, vllm_kv = vllm_block(
        x=x,
        e=e,
        freqs=freqs,
        freqs_action=freqs_action,
        freqs_state=freqs_state,
        context=context,
        action_register_length=None,
        kv_cache=kv_cache_vllm,
        current_start_frame=0,
    )
    dreamzero_out, dreamzero_kv = dreamzero_block(
        x=x,
        e=e,
        freqs=freqs,
        freqs_action=freqs_action,
        freqs_state=freqs_state,
        action_register_length=None,
        context=context,
        kv_cache=kv_cache_dreamzero,
        current_start_frame=0,
    )

    _assert_close("CausalWanAttentionBlock.out.tp2", vllm_out, dreamzero_out)
    assert isinstance(vllm_kv, torch.Tensor)
    _assert_close(
        "CausalWanAttentionBlock.kv.tp2",
        vllm_kv,
        _slice_heads(dreamzero_kv, local_rank, TP_SIZE, head_dim=3),
    )


def _test_full_model_precision_prefill_and_ar_step_tp2(local_rank: int, device: torch.device) -> None:
    vllm_model = _make_vllm_model(device)
    dreamzero_model = _make_dreamzero_model(device)
    _sync_module(vllm_model, dreamzero_model)

    batch_size = 1
    num_heads = TINY_CFG["num_heads"]
    tp_num_heads = num_heads // TP_SIZE
    head_dim = TINY_CFG["dim"] // num_heads

    vllm_kv = _make_empty_kv(TINY_CFG["num_layers"], batch_size, tp_num_heads, head_dim, device=device)
    dreamzero_kv = _make_empty_kv(TINY_CFG["num_layers"], batch_size, num_heads, head_dim, device=device)
    vllm_crossattn_cache = _make_crossattn_cache(TINY_CFG["num_layers"])
    dreamzero_crossattn_cache = _make_crossattn_cache(TINY_CFG["num_layers"])

    x_prefill = _shared_randn(batch_size, 4, 1, 4, 4, device=device, dtype=DTYPE)
    timestep_prefill = torch.tensor([[0]], device=device)
    context = _shared_randn(batch_size, 16, 64, device=device, dtype=DTYPE)

    with torch.no_grad():
        vllm_video_1, vllm_action_1, vllm_kv_1 = vllm_model(
            x=x_prefill,
            timestep=timestep_prefill,
            context=context,
            seq_len=4,
            kv_cache=vllm_kv,
            crossattn_cache=vllm_crossattn_cache,
            current_start_frame=0,
            action=None,
            timestep_action=None,
            state=None,
            embodiment_id=None,
            y=None,
            clip_feature=None,
        )
        dreamzero_video_1, dreamzero_action_1, dreamzero_kv_1 = dreamzero_model(
            x=x_prefill,
            timestep=timestep_prefill,
            context=context,
            seq_len=4,
            kv_cache=dreamzero_kv,
            crossattn_cache=dreamzero_crossattn_cache,
            current_start_frame=0,
            action=None,
            timestep_action=None,
            state=None,
            embodiment_id=None,
            y=None,
            clip_feature=None,
        )

    assert vllm_action_1 is None
    assert dreamzero_action_1 is None
    _assert_close(
        "CausalWanModel.prefill.video.tp2",
        vllm_video_1,
        dreamzero_video_1,
        atol=FULL_MODEL_ATOL,
        rtol=FULL_MODEL_RTOL,
    )
    for idx, (vllm_layer_kv, dreamzero_layer_kv) in enumerate(zip(vllm_kv_1, dreamzero_kv_1, strict=True)):
        _assert_close(
            f"CausalWanModel.prefill.kv[{idx}].tp2",
            vllm_layer_kv,
            _slice_heads(dreamzero_layer_kv, local_rank, TP_SIZE, head_dim=3),
        )
    # Match upstream causal chunk behavior: cross-attention cache is not
    # threaded through `_forward_blocks()` and stays untouched.
    for idx, vllm_cache in enumerate(vllm_crossattn_cache):
        _assert_crossattn_cache_uninitialized(
            f"CausalWanModel.prefill.crossattn_cache[{idx}].tp2",
            vllm_cache,
        )
    for idx, dreamzero_cache in enumerate(dreamzero_crossattn_cache):
        _assert_crossattn_cache_uninitialized(
            f"CausalWanModel.prefill.dreamzero_crossattn_cache[{idx}].tp2",
            dreamzero_cache,
        )

    x_step = _shared_randn(batch_size, 4, 1, 4, 4, device=device, dtype=DTYPE)
    timestep_step = torch.tensor([[500]], device=device)
    action = _shared_randn(batch_size, 4, 8, device=device, dtype=DTYPE)
    timestep_action = torch.tensor([[500, 500, 500, 500]], device=device)
    state = _shared_randn(batch_size, 1, 64, device=device, dtype=DTYPE)
    embodiment_id = torch.tensor([0], device=device)

    with torch.no_grad():
        vllm_video_2, vllm_action_2, vllm_kv_2 = vllm_model(
            x=x_step,
            timestep=timestep_step,
            context=context,
            seq_len=4,
            kv_cache=[kv.clone() for kv in vllm_kv_1],
            crossattn_cache=vllm_crossattn_cache,
            current_start_frame=1,
            action=action,
            timestep_action=timestep_action,
            state=state,
            embodiment_id=embodiment_id,
            y=None,
            clip_feature=None,
        )
        dreamzero_video_2, dreamzero_action_2, dreamzero_kv_2 = dreamzero_model(
            x=x_step,
            timestep=timestep_step,
            context=context,
            seq_len=4,
            kv_cache=[kv.clone() for kv in dreamzero_kv_1],
            crossattn_cache=dreamzero_crossattn_cache,
            current_start_frame=1,
            action=action,
            timestep_action=timestep_action,
            state=state,
            embodiment_id=embodiment_id,
            y=None,
            clip_feature=None,
        )

    assert vllm_action_2 is not None
    assert dreamzero_action_2 is not None
    _assert_close(
        "CausalWanModel.step.video.tp2",
        vllm_video_2,
        dreamzero_video_2,
        atol=FULL_MODEL_ATOL,
        rtol=FULL_MODEL_RTOL,
    )
    _assert_close(
        "CausalWanModel.step.action.tp2",
        vllm_action_2,
        dreamzero_action_2,
        atol=FULL_MODEL_ATOL,
        rtol=FULL_MODEL_RTOL,
    )
    for idx, (vllm_layer_kv, dreamzero_layer_kv) in enumerate(zip(vllm_kv_2, dreamzero_kv_2, strict=True)):
        _assert_close(
            f"CausalWanModel.step.kv[{idx}].tp2",
            vllm_layer_kv,
            _slice_heads(dreamzero_layer_kv, local_rank, TP_SIZE, head_dim=3),
        )
    for idx, vllm_cache in enumerate(vllm_crossattn_cache):
        _assert_crossattn_cache_uninitialized(
            f"CausalWanModel.step.crossattn_cache[{idx}].tp2",
            vllm_cache,
        )
    for idx, dreamzero_cache in enumerate(dreamzero_crossattn_cache):
        _assert_crossattn_cache_uninitialized(
            f"CausalWanModel.step.dreamzero_crossattn_cache[{idx}].tp2",
            dreamzero_cache,
        )


def _run_worker(local_rank: int, world_size: int, master_port: int) -> None:
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cuda"))):
        device = _init_tp_env(local_rank, world_size, master_port)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        try:
            with _patch_dreamzero_attention():
                _test_hotpath_layer_types_tp2(device)
                _test_distributed_rmsnorm_precision_tp2(local_rank, device)
                _test_t2v_cross_attn_precision_tp2(local_rank, device)
                _test_i2v_cross_attn_precision_tp2(local_rank, device)
                _test_self_attn_precision_tp2(local_rank, device)
                _test_attention_block_precision_tp2(local_rank, device)
                _test_full_model_precision_prefill_and_ar_step_tp2(local_rank, device)
        finally:
            cleanup_dist_env_and_memory()


def _run_tp2_suite() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for TP=2 DreamZero precision tests.")
    if torch.cuda.device_count() < TP_SIZE:
        raise RuntimeError(f"TP=2 DreamZero precision tests require {TP_SIZE} GPUs.")

    master_port = _find_free_port()
    mp.spawn(_run_worker, args=(TP_SIZE, master_port), nprocs=TP_SIZE, join=True)


def test_causal_wan_model_tp2_precision():
    _run_tp2_suite()


if __name__ == "__main__":
    _run_tp2_suite()
    print("TP=2 DreamZero CausalWanModel precision suite: PASS")
