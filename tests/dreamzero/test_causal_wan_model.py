# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GPU precision tests for DreamZero CausalWanModel.

Run:
    PYTHONPATH=. /home/yangshen/miniconda3/envs/dreamzero/bin/python -m pytest         tests/dreamzero/test_causal_wan_model.py -v -s
"""

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


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, *, atol: float = ATOL, rtol: float = RTOL) -> None:
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
    return [
        torch.zeros(2, batch_size, 0, num_heads, head_dim, device=DEVICE, dtype=DTYPE)
        for _ in range(num_layers)
    ]



def _make_crossattn_cache(num_layers: int) -> list[dict[str, object]]:
    return [{"is_init": False, "k": None, "v": None} for _ in range(num_layers)]


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



def test_i2v_cross_attn_precision():
    from groot.vla.model.dreamzero.modules.wan2_1_submodule import WanI2VCrossAttention as DreamZeroCrossAttention
    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import WanI2VCrossAttention

    vllm_attn = WanI2VCrossAttention(64, 4, qk_norm=True, eps=1e-6).to(device=DEVICE, dtype=DTYPE)
    dreamzero_attn = DreamZeroCrossAttention(64, 4, (-1, -1), True, 1e-6).to(device=DEVICE, dtype=DTYPE)
    _sync_module(vllm_attn, dreamzero_attn)

    x = torch.randn(1, 8, 64, device=DEVICE, dtype=DTYPE)
    context = torch.randn(1, 257 + 16, 64, device=DEVICE, dtype=DTYPE)
    _assert_close("WanI2VCrossAttention", vllm_attn(x, context), dreamzero_attn(x, context))



def test_self_attn_precision():
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanSelfAttention as DreamZeroSelfAttention
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
    _assert_close("CausalWanModel.prefill.video", vllm_video_1, dreamzero_video_1, atol=FULL_MODEL_ATOL, rtol=FULL_MODEL_RTOL)
    for idx, (vllm_layer_kv, dreamzero_layer_kv) in enumerate(zip(vllm_kv_1, dreamzero_kv_1, strict=True)):
        _assert_close(f"CausalWanModel.prefill.kv[{idx}]", vllm_layer_kv, dreamzero_layer_kv)

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
    _assert_close("CausalWanModel.step.video", vllm_video_2, dreamzero_video_2, atol=FULL_MODEL_ATOL, rtol=FULL_MODEL_RTOL)
    _assert_close("CausalWanModel.step.action", vllm_action_2, dreamzero_action_2, atol=FULL_MODEL_ATOL, rtol=FULL_MODEL_RTOL)
    for idx, (vllm_layer_kv, dreamzero_layer_kv) in enumerate(zip(vllm_kv_2, dreamzero_kv_2, strict=True)):
        _assert_close(f"CausalWanModel.step.kv[{idx}]", vllm_layer_kv, dreamzero_layer_kv)
