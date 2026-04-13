# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GPU precision tests for DreamZero CausalWanModel.

Run:
    PYTHONPATH=. /home/yangshen/miniconda3/envs/dreamzero/bin/python -m pytest         tests/dreamzero/test_causal_wan_model.py -v -s
"""

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

os.environ.setdefault("ATTENTION_BACKEND", "torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../third_party/dreamzero"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../dreamzero"))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

DEVICE = torch.device("cuda")
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


def _tiny_cfg(*, model_type: str) -> dict[str, object]:
    return {**TINY_CFG, "model_type": model_type}


def _assert_close(
    name: str, actual: torch.Tensor, expected: torch.Tensor, *, atol: float = ATOL, rtol: float = RTOL
) -> None:
    actual = actual.detach().float()
    expected = expected.detach().float()
    max_diff = (actual - expected).abs().max().item()
    assert torch.allclose(actual, expected, atol=atol, rtol=rtol), (
        f"{name}: max_diff={max_diff:.3e}, atol={atol}, rtol={rtol}"
    )
    print(f"{name}: max_diff={max_diff:.3e}")


def _load_vllm_param(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, loaded_weight)


def _sync_module(vllm_module: torch.nn.Module, dreamzero_module: torch.nn.Module) -> None:
    vllm_params = dict(vllm_module.named_parameters())
    dreamzero_params = dict(dreamzero_module.named_parameters())

    missing = sorted(set(dreamzero_params) - set(vllm_params))
    extra = sorted(set(vllm_params) - set(dreamzero_params))
    assert not missing, f"Missing params in vllm module: {missing}"
    assert not extra, f"Unexpected params in vllm module: {extra}"

    for name, dz_param in dreamzero_params.items():
        _load_vllm_param(vllm_params[name], dz_param.detach())


def _make_vllm_model():
    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import CausalWanModel

    model = CausalWanModel(**TINY_CFG).to(device=DEVICE, dtype=DTYPE)
    model.eval()
    return model


def _make_dreamzero_model():
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel

    model = CausalWanModel(**TINY_CFG).to(device=DEVICE, dtype=DTYPE)
    model.eval()
    return model


def _make_empty_kv(num_layers: int, batch_size: int, num_heads: int, head_dim: int) -> list[torch.Tensor]:
    return [torch.zeros(2, batch_size, 0, num_heads, head_dim, device=DEVICE, dtype=DTYPE) for _ in range(num_layers)]


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


@pytest.fixture(autouse=True)
def _manual_seed():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    yield


@pytest.fixture(autouse=True)
def _disable_tf32():
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = old_matmul
    torch.backends.cudnn.allow_tf32 = old_cudnn


@pytest.fixture(autouse=True)
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

    base_attention.flash_attention = sdpa_flash_attention
    wan_attention.flash_attention = sdpa_flash_attention
    wan_submodule.flash_attention = sdpa_flash_attention

    def patched_t2v_forward(self, x, context, context_lens=None, crossattn_cache=None):
        return original_t2v_forward(self, x, context, context_lens, crossattn_cache)

    wan_attention.AttentionModule.__init__ = patched_attention_init
    wan_submodule.WanT2VCrossAttention.forward = patched_t2v_forward
    wan_chunk.WAN_CROSSATTENTION_CLASSES["t2v_cross_attn"].forward = patched_t2v_forward

    yield

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


def test_hotpath_layer_types():
    from vllm.model_executor.layers.conv import Conv3dLayer
    from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear

    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import DistributedRMSNorm

    model = _make_vllm_model()
    block = model.blocks[0]

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


def test_img_emb_created_only_for_i2v():
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel as DreamZeroModel

    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import CausalWanModel as VllmModel

    for model_type, expected in (("t2v", False), ("i2v", True), ("ti2v", False)):
        vllm_model = VllmModel(**_tiny_cfg(model_type=model_type)).to(device=DEVICE, dtype=DTYPE)
        dreamzero_model = DreamZeroModel(**_tiny_cfg(model_type=model_type)).to(device=DEVICE, dtype=DTYPE)

        assert hasattr(vllm_model, "img_emb") is expected, f"vLLM model_type={model_type}"
        assert hasattr(dreamzero_model, "img_emb") is expected, f"DreamZero model_type={model_type}"


def test_init_weights_called_and_matches_upstream_scheme(monkeypatch):
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel as DreamZeroModel

    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import CausalWanModel as VllmModel

    called = 0
    original_init_weights = VllmModel.init_weights

    def wrapped_init_weights(self):
        nonlocal called
        called += 1
        return original_init_weights(self)

    monkeypatch.setattr(VllmModel, "init_weights", wrapped_init_weights)

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    vllm_model = VllmModel(**_tiny_cfg(model_type="i2v")).to(device=DEVICE, dtype=DTYPE)

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    dreamzero_model = DreamZeroModel(**_tiny_cfg(model_type="i2v")).to(device=DEVICE, dtype=DTYPE)

    assert called == 1

    assert torch.count_nonzero(vllm_model.head.head.weight).item() == 0
    assert torch.count_nonzero(dreamzero_model.head.head.weight).item() == 0

    zero_biases = [
        vllm_model.text_embedding[0].bias,
        vllm_model.time_embedding[0].bias,
        vllm_model.blocks[0].self_attn.q.bias,
        vllm_model.blocks[0].self_attn.o.bias,
        vllm_model.img_emb.fc1.bias,
        vllm_model.img_emb.fc2.bias,
        dreamzero_model.text_embedding[0].bias,
        dreamzero_model.time_embedding[0].bias,
        dreamzero_model.blocks[0].self_attn.q.bias,
        dreamzero_model.blocks[0].self_attn.o.bias,
        dreamzero_model.img_emb.proj[1].bias,
        dreamzero_model.img_emb.proj[3].bias,
    ]
    for bias in zero_biases:
        assert torch.count_nonzero(bias).item() == 0

    for weight in (
        vllm_model.text_embedding[0].weight,
        vllm_model.text_embedding[2].weight,
        vllm_model.time_embedding[0].weight,
        vllm_model.time_embedding[2].weight,
        dreamzero_model.text_embedding[0].weight,
        dreamzero_model.text_embedding[2].weight,
        dreamzero_model.time_embedding[0].weight,
        dreamzero_model.time_embedding[2].weight,
    ):
        std = weight.float().std().item()
        assert 0.015 <= std <= 0.025, f"expected normal std≈0.02, got {std:.4f}"

    for weight, fan_in, fan_out in (
        (
            vllm_model.blocks[0].self_attn.q.weight,
            vllm_model.blocks[0].self_attn.q.input_size,
            vllm_model.blocks[0].self_attn.q.output_size,
        ),
        (
            vllm_model.blocks[0].self_attn.o.weight,
            vllm_model.blocks[0].self_attn.o.input_size,
            vllm_model.blocks[0].self_attn.o.output_size,
        ),
        (
            dreamzero_model.blocks[0].self_attn.q.weight,
            dreamzero_model.blocks[0].self_attn.q.in_features,
            dreamzero_model.blocks[0].self_attn.q.out_features,
        ),
        (
            dreamzero_model.blocks[0].self_attn.o.weight,
            dreamzero_model.blocks[0].self_attn.o.in_features,
            dreamzero_model.blocks[0].self_attn.o.out_features,
        ),
    ):
        bound = math.sqrt(6.0 / float(fan_in + fan_out))
        assert weight.float().abs().max().item() <= bound + 1e-6

    conv_bias_bound = 1 / math.sqrt(
        vllm_model.patch_embedding.in_channels * math.prod(vllm_model.patch_embedding.kernel_size)
    )
    assert torch.isfinite(vllm_model.patch_embedding.bias).all()
    assert vllm_model.patch_embedding.bias.float().abs().max().item() <= conv_bias_bound + 1e-6
    assert torch.isfinite(dreamzero_model.patch_embedding.bias).all()


def test_distributed_rmsnorm_precision():
    from groot.vla.model.dreamzero.modules.wan2_1_submodule import WanRMSNorm

    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import DistributedRMSNorm

    vllm_norm = DistributedRMSNorm(64, eps=1e-6).to(device=DEVICE, dtype=DTYPE)
    dreamzero_norm = WanRMSNorm(64, eps=1e-6).to(device=DEVICE, dtype=DTYPE)
    _sync_module(vllm_norm, dreamzero_norm)

    x = torch.randn(2, 10, 64, device=DEVICE, dtype=DTYPE)
    _assert_close("DistributedRMSNorm", vllm_norm(x), dreamzero_norm(x))


def test_t2v_cross_attn_precision():
    from groot.vla.model.dreamzero.modules.wan2_1_submodule import WanT2VCrossAttention as DreamZeroCrossAttention

    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import WanT2VCrossAttention

    vllm_attn = WanT2VCrossAttention(64, 4, qk_norm=True, eps=1e-6).to(device=DEVICE, dtype=DTYPE)
    dreamzero_attn = DreamZeroCrossAttention(64, 4, (-1, -1), True, 1e-6).to(device=DEVICE, dtype=DTYPE)
    _sync_module(vllm_attn, dreamzero_attn)

    x = torch.randn(1, 8, 64, device=DEVICE, dtype=DTYPE)
    context = torch.randn(1, 16, 64, device=DEVICE, dtype=DTYPE)
    _assert_close(
        "WanT2VCrossAttention",
        vllm_attn(x, context),
        dreamzero_attn(x, context, context_lens=None),
    )


def test_t2v_cross_attn_cache_fill_and_reuse():
    from groot.vla.model.dreamzero.modules.wan2_1_submodule import WanT2VCrossAttention as DreamZeroCrossAttention

    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import WanT2VCrossAttention

    vllm_attn = WanT2VCrossAttention(64, 4, qk_norm=True, eps=1e-6).to(device=DEVICE, dtype=DTYPE)
    dreamzero_attn = DreamZeroCrossAttention(64, 4, (-1, -1), True, 1e-6).to(device=DEVICE, dtype=DTYPE)
    _sync_module(vllm_attn, dreamzero_attn)

    x = torch.randn(1, 8, 64, device=DEVICE, dtype=DTYPE)
    x_step = torch.randn(1, 8, 64, device=DEVICE, dtype=DTYPE)
    context = torch.randn(1, 16, 64, device=DEVICE, dtype=DTYPE)
    vllm_cache = {"is_init": False, "k": None, "v": None}
    dreamzero_cache = {"is_init": False, "k": None, "v": None}

    _assert_close(
        "WanT2VCrossAttention.prefill.cache",
        vllm_attn(x, context, crossattn_cache=vllm_cache),
        dreamzero_attn(x, context, context_lens=None, crossattn_cache=dreamzero_cache),
    )
    _assert_crossattn_cache_initialized(
        "WanT2VCrossAttention.vllm_cache",
        vllm_cache,
        batch_size=1,
        context_tokens=16,
        num_heads=4,
        head_dim=16,
    )
    _assert_crossattn_cache_initialized(
        "WanT2VCrossAttention.dreamzero_cache",
        dreamzero_cache,
        batch_size=1,
        context_tokens=16,
        num_heads=4,
        head_dim=16,
    )
    _assert_close("WanT2VCrossAttention.cache.k", vllm_cache["k"], dreamzero_cache["k"])
    _assert_close("WanT2VCrossAttention.cache.v", vllm_cache["v"], dreamzero_cache["v"])

    vllm_cache_prefill = _clone_crossattn_cache([vllm_cache])[0]
    dreamzero_cache_prefill = _clone_crossattn_cache([dreamzero_cache])[0]

    _assert_close(
        "WanT2VCrossAttention.reuse.cache",
        vllm_attn(x_step, context, crossattn_cache=vllm_cache),
        dreamzero_attn(x_step, context, context_lens=None, crossattn_cache=dreamzero_cache),
    )
    _assert_crossattn_cache_reused(
        "WanT2VCrossAttention.vllm_cache_reuse",
        vllm_cache,
        vllm_cache_prefill,
        batch_size=1,
        context_tokens=16,
        num_heads=4,
        head_dim=16,
    )
    _assert_crossattn_cache_reused(
        "WanT2VCrossAttention.dreamzero_cache_reuse",
        dreamzero_cache,
        dreamzero_cache_prefill,
        batch_size=1,
        context_tokens=16,
        num_heads=4,
        head_dim=16,
    )


def test_i2v_cross_attn_precision():
    from groot.vla.model.dreamzero.modules.wan2_1_submodule import WanI2VCrossAttention as DreamZeroCrossAttention

    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import WanI2VCrossAttention

    vllm_attn = WanI2VCrossAttention(64, 4, qk_norm=True, eps=1e-6).to(device=DEVICE, dtype=DTYPE)
    dreamzero_attn = DreamZeroCrossAttention(64, 4, (-1, -1), True, 1e-6).to(device=DEVICE, dtype=DTYPE)
    _sync_module(vllm_attn, dreamzero_attn)

    x = torch.randn(1, 8, 64, device=DEVICE, dtype=DTYPE)
    context = torch.randn(1, 257 + 16, 64, device=DEVICE, dtype=DTYPE)
    _assert_close("WanI2VCrossAttention", vllm_attn(x, context), dreamzero_attn(x, context))


def test_i2v_cross_attn_cache_fill_and_reuse():
    from groot.vla.model.dreamzero.modules.wan2_1_submodule import WanI2VCrossAttention as DreamZeroCrossAttention

    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import WanI2VCrossAttention

    vllm_attn = WanI2VCrossAttention(64, 4, qk_norm=True, eps=1e-6).to(device=DEVICE, dtype=DTYPE)
    dreamzero_attn = DreamZeroCrossAttention(64, 4, (-1, -1), True, 1e-6).to(device=DEVICE, dtype=DTYPE)
    _sync_module(vllm_attn, dreamzero_attn)

    x = torch.randn(1, 8, 64, device=DEVICE, dtype=DTYPE)
    x_step = torch.randn(1, 8, 64, device=DEVICE, dtype=DTYPE)
    context = torch.randn(1, 257 + 16, 64, device=DEVICE, dtype=DTYPE)
    vllm_cache = {"is_init": False, "k": None, "v": None}
    dreamzero_cache = {"is_init": False, "k": None, "v": None}

    _assert_close(
        "WanI2VCrossAttention.prefill.cache",
        vllm_attn(x, context, crossattn_cache=vllm_cache),
        dreamzero_attn(x, context, crossattn_cache=dreamzero_cache),
    )
    _assert_crossattn_cache_initialized(
        "WanI2VCrossAttention.vllm_cache",
        vllm_cache,
        batch_size=1,
        context_tokens=16,
        num_heads=4,
        head_dim=16,
    )
    _assert_crossattn_cache_initialized(
        "WanI2VCrossAttention.dreamzero_cache",
        dreamzero_cache,
        batch_size=1,
        context_tokens=16,
        num_heads=4,
        head_dim=16,
    )
    _assert_close("WanI2VCrossAttention.cache.k", vllm_cache["k"], dreamzero_cache["k"])
    _assert_close("WanI2VCrossAttention.cache.v", vllm_cache["v"], dreamzero_cache["v"])

    vllm_cache_prefill = _clone_crossattn_cache([vllm_cache])[0]
    dreamzero_cache_prefill = _clone_crossattn_cache([dreamzero_cache])[0]

    _assert_close(
        "WanI2VCrossAttention.reuse.cache",
        vllm_attn(x_step, context, crossattn_cache=vllm_cache),
        dreamzero_attn(x_step, context, crossattn_cache=dreamzero_cache),
    )
    _assert_crossattn_cache_reused(
        "WanI2VCrossAttention.vllm_cache_reuse",
        vllm_cache,
        vllm_cache_prefill,
        batch_size=1,
        context_tokens=16,
        num_heads=4,
        head_dim=16,
    )
    _assert_crossattn_cache_reused(
        "WanI2VCrossAttention.dreamzero_cache_reuse",
        dreamzero_cache,
        dreamzero_cache_prefill,
        batch_size=1,
        context_tokens=16,
        num_heads=4,
        head_dim=16,
    )


def test_self_attn_precision():
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
        CausalWanSelfAttention as DreamZeroSelfAttention,
    )

    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import (
        CausalWanSelfAttention,
        rope_params,
    )

    vllm_attn = CausalWanSelfAttention(64, 4, 4, num_action_per_block=4, num_state_per_block=1).to(
        device=DEVICE, dtype=DTYPE
    )
    dreamzero_attn = DreamZeroSelfAttention(64, 4, 4, num_action_per_block=4, num_state_per_block=1).to(
        device=DEVICE, dtype=DTYPE
    )
    _sync_module(vllm_attn, dreamzero_attn)

    kv_cache = torch.zeros(2, 1, 0, 4, 16, device=DEVICE, dtype=DTYPE)
    freqs = rope_params(1024, 16)[:4].view(-1, 1, 8).to(device=DEVICE)
    freqs_action = rope_params(10240, 16).to(device=DEVICE)
    freqs_state = rope_params(1024, 16).to(device=DEVICE)
    x = torch.randn(1, 4, 64, device=DEVICE, dtype=DTYPE)

    vllm_out, vllm_kv = vllm_attn(x, freqs, freqs_action, freqs_state, None, kv_cache, current_start_frame=0)
    dreamzero_out, dreamzero_kv = dreamzero_attn(
        x,
        freqs,
        freqs_action,
        freqs_state,
        None,
        kv_cache,
        current_start_frame=0,
    )

    _assert_close("CausalWanSelfAttention.out", vllm_out, dreamzero_out)
    _assert_close("CausalWanSelfAttention.kv", vllm_kv, dreamzero_kv)


def test_attention_block_precision():
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
        CausalWanAttentionBlock as DreamZeroAttentionBlock,
    )

    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import (
        CausalWanAttentionBlock,
        rope_params,
    )

    vllm_block = CausalWanAttentionBlock(
        "t2v_cross_attn",
        64,
        128,
        4,
        4,
        num_action_per_block=4,
        num_state_per_block=1,
    ).to(device=DEVICE, dtype=DTYPE)
    dreamzero_block = DreamZeroAttentionBlock(
        "t2v_cross_attn",
        64,
        128,
        4,
        4,
        num_action_per_block=4,
        num_state_per_block=1,
    ).to(device=DEVICE, dtype=DTYPE)
    _sync_module(vllm_block, dreamzero_block)

    kv_cache = torch.zeros(2, 1, 0, 4, 16, device=DEVICE, dtype=DTYPE)
    freqs = rope_params(1024, 16)[:4].view(-1, 1, 8).to(device=DEVICE)
    freqs_action = rope_params(10240, 16).to(device=DEVICE)
    freqs_state = rope_params(1024, 16).to(device=DEVICE)
    x = torch.randn(1, 4, 64, device=DEVICE, dtype=DTYPE)
    e = torch.randn(1, 4, 6, 64, device=DEVICE, dtype=DTYPE)
    context = torch.randn(1, 16, 64, device=DEVICE, dtype=DTYPE)

    vllm_out, vllm_kv = vllm_block(
        x=x,
        e=e,
        freqs=freqs,
        freqs_action=freqs_action,
        freqs_state=freqs_state,
        context=context,
        action_register_length=None,
        kv_cache=kv_cache,
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
        kv_cache=kv_cache,
        current_start_frame=0,
    )

    _assert_close("CausalWanAttentionBlock.out", vllm_out, dreamzero_out)
    _assert_close("CausalWanAttentionBlock.kv", vllm_kv, dreamzero_kv)


def test_full_model_precision_prefill_and_ar_step():
    vllm_model = _make_vllm_model()
    dreamzero_model = _make_dreamzero_model()
    _sync_module(vllm_model, dreamzero_model)

    batch_size = 1
    num_heads = TINY_CFG["num_heads"]
    head_dim = TINY_CFG["dim"] // TINY_CFG["num_heads"]

    vllm_kv = _make_empty_kv(TINY_CFG["num_layers"], batch_size, num_heads, head_dim)
    dreamzero_kv = _make_empty_kv(TINY_CFG["num_layers"], batch_size, num_heads, head_dim)
    vllm_crossattn_cache = _make_crossattn_cache(TINY_CFG["num_layers"])
    dreamzero_crossattn_cache = _make_crossattn_cache(TINY_CFG["num_layers"])

    x_prefill = torch.randn(batch_size, 4, 1, 4, 4, device=DEVICE, dtype=DTYPE)
    timestep_prefill = torch.tensor([[0]], device=DEVICE)
    context = torch.randn(batch_size, 16, 64, device=DEVICE, dtype=DTYPE)

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
        "CausalWanModel.prefill.video", vllm_video_1, dreamzero_video_1, atol=FULL_MODEL_ATOL, rtol=FULL_MODEL_RTOL
    )
    for idx, (vllm_layer_kv, dreamzero_layer_kv) in enumerate(zip(vllm_kv_1, dreamzero_kv_1, strict=True)):
        _assert_close(f"CausalWanModel.prefill.kv[{idx}]", vllm_layer_kv, dreamzero_layer_kv)
    # The causal DreamZero chunk model does not thread cross-attention cache
    # through `_forward_blocks()` in either upstream or this port, so the
    # cache must remain untouched on both sides.
    for idx, vllm_cache in enumerate(vllm_crossattn_cache):
        _assert_crossattn_cache_uninitialized(
            f"CausalWanModel.prefill.crossattn_cache[{idx}]",
            vllm_cache,
        )
    for idx, dreamzero_cache in enumerate(dreamzero_crossattn_cache):
        _assert_crossattn_cache_uninitialized(
            f"CausalWanModel.prefill.dreamzero_crossattn_cache[{idx}]",
            dreamzero_cache,
        )

    x_step = torch.randn(batch_size, 4, 1, 4, 4, device=DEVICE, dtype=DTYPE)
    timestep_step = torch.tensor([[500]], device=DEVICE)
    action = torch.randn(batch_size, 4, 8, device=DEVICE, dtype=DTYPE)
    timestep_action = torch.tensor([[500, 500, 500, 500]], device=DEVICE)
    state = torch.randn(batch_size, 1, 64, device=DEVICE, dtype=DTYPE)
    embodiment_id = torch.tensor([0], device=DEVICE)

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
        "CausalWanModel.step.video", vllm_video_2, dreamzero_video_2, atol=FULL_MODEL_ATOL, rtol=FULL_MODEL_RTOL
    )
    _assert_close(
        "CausalWanModel.step.action", vllm_action_2, dreamzero_action_2, atol=FULL_MODEL_ATOL, rtol=FULL_MODEL_RTOL
    )
    for idx, (vllm_layer_kv, dreamzero_layer_kv) in enumerate(zip(vllm_kv_2, dreamzero_kv_2, strict=True)):
        _assert_close(f"CausalWanModel.step.kv[{idx}]", vllm_layer_kv, dreamzero_layer_kv)
    for idx, vllm_cache in enumerate(vllm_crossattn_cache):
        _assert_crossattn_cache_uninitialized(
            f"CausalWanModel.step.crossattn_cache[{idx}]",
            vllm_cache,
        )
    for idx, dreamzero_cache in enumerate(dreamzero_crossattn_cache):
        _assert_crossattn_cache_uninitialized(
            f"CausalWanModel.step.dreamzero_crossattn_cache[{idx}]",
            dreamzero_cache,
        )
