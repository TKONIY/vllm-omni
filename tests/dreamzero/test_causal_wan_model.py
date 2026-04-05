# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for CausalWanModel.

Shape tests run without GPU or dreamzero.
Precision alignment tests need dreamzero conda env.

Run: PYTHONPATH=. python tests/dreamzero/test_causal_wan_model.py
"""

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../third_party/dreamzero"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../dreamzero"))


# ── Tiny model config for shape tests ──────────────────────────────

TINY_CFG = dict(
    model_type="t2v",
    patch_size=(1, 2, 2),
    frame_seqlen=4,       # (4/2)*(4/2) = 4 tokens per frame
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


def _make_tiny_model():
    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import CausalWanModel
    model = CausalWanModel(**TINY_CFG)
    model.eval()
    return model


def _make_empty_kv(num_layers, batch_size, num_heads, head_dim):
    return [torch.zeros(2, batch_size, 0, num_heads, head_dim) for _ in range(num_layers)]


# ── Test 1: model init ─────────────────────────────────────────────

def test_init():
    model = _make_tiny_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Tiny model: {n_params:,} params, {len(model.blocks)} blocks")
    assert len(model.blocks) == 2
    print("✅ Init: PASS")


# ── Test 2: prefill (no action, current_start_frame=0) ─────────────

def test_prefill():
    model = _make_tiny_model()
    B, num_heads, head_dim = 1, 4, 16
    kv_cache = _make_empty_kv(2, B, num_heads, head_dim)
    crossattn_cache = [torch.zeros(2, B, 16, num_heads, head_dim) for _ in range(2)]

    # x: [B, C_in=4, T=1, H=4, W=4]
    x = torch.randn(B, 4, 1, 4, 4)
    timestep = torch.tensor([[0]])
    context = torch.randn(B, 16, 64)

    with torch.no_grad():
        video_pred, action_pred, updated_kv = model(
            x=x, timestep=timestep, context=context, seq_len=4,
            kv_cache=kv_cache, crossattn_cache=crossattn_cache,
            current_start_frame=0,
            action=None, timestep_action=None,
            state=None, embodiment_id=None,
            y=None, clip_feature=None,
        )

    assert video_pred is not None
    new_seq = updated_kv[0].shape[2]
    assert new_seq > 0, f"KV cache should grow, got {new_seq}"
    print(f"  Prefill: video={video_pred.shape}, KV 0→{new_seq}")
    print("✅ Prefill: PASS")


# ── Test 3: inference with action (current_start_frame=1) ───────────

def test_inference_with_action():
    model = _make_tiny_model()
    B, num_heads, head_dim = 1, 4, 16

    # Prefill first
    kv_cache = _make_empty_kv(2, B, num_heads, head_dim)
    crossattn_cache = [torch.zeros(2, B, 16, num_heads, head_dim) for _ in range(2)]

    with torch.no_grad():
        _, _, updated_kv = model(
            x=torch.randn(B, 4, 1, 4, 4), timestep=torch.tensor([[0]]),
            context=torch.randn(B, 16, 64), seq_len=4,
            kv_cache=kv_cache, crossattn_cache=crossattn_cache,
            current_start_frame=0,
            action=None, timestep_action=None,
            state=None, embodiment_id=None,
            y=None, clip_feature=None,
        )
    for i, kv in enumerate(updated_kv):
        kv_cache[i] = kv.clone()

    # Now inference with action
    with torch.no_grad():
        video_pred, action_pred, _ = model(
            x=torch.randn(B, 4, 1, 4, 4), timestep=torch.tensor([[500]]),
            context=torch.randn(B, 16, 64), seq_len=4,
            kv_cache=kv_cache, crossattn_cache=crossattn_cache,
            current_start_frame=1,
            action=torch.randn(B, 4, 8),
            timestep_action=torch.tensor([[500] * 4]),
            state=torch.randn(B, 1, 64),
            embodiment_id=torch.tensor([0]),
            y=None, clip_feature=None,
        )

    assert video_pred is not None
    assert action_pred is not None
    print(f"  Inference: video={video_pred.shape}, action={action_pred.shape}")
    print("✅ Inference with action: PASS")


# ── Test 4: KV cache grows across AR steps ──────────────────────────

def test_kv_cache_growth():
    model = _make_tiny_model()
    B, num_heads, head_dim = 1, 4, 16
    kv_cache = _make_empty_kv(2, B, num_heads, head_dim)
    crossattn_cache = [torch.zeros(2, B, 16, num_heads, head_dim) for _ in range(2)]
    context = torch.randn(B, 16, 64)

    sizes = []
    for step in range(3):
        action = torch.randn(B, 4, 8) if step > 0 else None
        timestep_action = torch.tensor([[500] * 4]) if step > 0 else None
        state = torch.randn(B, 1, 64) if step > 0 else None
        embodiment_id = torch.tensor([0]) if step > 0 else None

        with torch.no_grad():
            _, _, updated_kv = model(
                x=torch.randn(B, 4, 1, 4, 4),
                timestep=torch.tensor([[0 if step == 0 else 500]]),
                context=context, seq_len=4,
                kv_cache=kv_cache, crossattn_cache=crossattn_cache,
                current_start_frame=step,
                action=action, timestep_action=timestep_action,
                state=state, embodiment_id=embodiment_id,
                y=None, clip_feature=None,
            )
        for i, kv in enumerate(updated_kv):
            kv_cache[i] = kv.clone()
        sizes.append(kv_cache[0].shape[2])

    print(f"  KV sizes: {sizes}")
    assert all(sizes[i] < sizes[i + 1] for i in range(len(sizes) - 1))
    print("✅ KV cache growth: PASS")


# ── Test 5: RoPE precision alignment ────────────────────────────────

def test_rope_precision():
    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import (
        sinusoidal_embedding_1d as vllm_sin,
        rope_params as vllm_rope,
        rope_apply as vllm_rope_apply,
        rope_action_apply as vllm_rope_action,
        causal_rope_action_apply as vllm_causal_rope,
    )
    from groot.vla.model.dreamzero.modules.wan2_1_submodule import (
        sinusoidal_embedding_1d as dz_sin,
        rope_params_polar as dz_rope,
        rope_apply_polar as dz_rope_apply,
        rope_action_apply_polar as dz_rope_action,
    )
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
        causal_rope_action_apply_polar as dz_causal_rope,
    )

    # sinusoidal_embedding_1d
    pos = torch.arange(10).float()
    for dim in [64, 128, 256]:
        diff = (vllm_sin(dim, pos) - dz_sin(dim, pos)).abs().max().item()
        assert diff == 0, f"sinusoidal dim={dim}: {diff}"
        print(f"  sinusoidal_embedding_1d dim={dim}: OK ({diff:.2e})")

    # rope_params
    for d in [16, 32, 64, 128]:
        diff = (vllm_rope(1024, d) - dz_rope(1024, d)).abs().max().item()
        assert diff == 0, f"rope_params d={d}: {diff}"
        print(f"  rope_params d={d}: OK ({diff:.2e})")

    # rope_apply
    d = 16
    x = torch.randn(1, 8, 4, d)
    freqs = vllm_rope(1024, d)[:8].view(-1, 1, d // 2)
    v = vllm_rope_apply(x, freqs)
    dz = dz_rope_apply(x, freqs)
    diff = (v.float() - dz.float()).abs().max().item()
    assert diff < 1e-6, f"rope_apply: {diff}"
    print(f"  rope_apply: OK ({diff:.2e})")

    # rope_action_apply
    x = torch.randn(1, 13, 4, d)  # 8 spatial + 4 action + 1 state
    freqs = vllm_rope(1024, d)[:8].view(-1, 1, d // 2)
    freqs_a = vllm_rope(10240, d)
    freqs_s = vllm_rope(1024, d)
    v = vllm_rope_action(x, freqs, freqs_a, freqs_s, 5, 4, 1)
    dz = dz_rope_action(x, freqs, freqs_a, freqs_s, 5, 4, 1)
    diff = (v.float() - dz.float()).abs().max().item()
    assert diff < 1e-6, f"rope_action_apply: {diff}"
    print(f"  rope_action_apply: OK ({diff:.2e})")

    # causal_rope_action_apply
    x = torch.randn(1, 8, 4, d)
    freqs = vllm_rope(1024, d)[:3].view(-1, 1, d // 2)
    v = vllm_causal_rope(x, freqs, freqs_a, freqs_s, 5, 4, 1, 0)
    dz = dz_causal_rope(x, freqs, freqs_a, freqs_s, 5, 4, 1, 0)
    diff = (v.float() - dz.float()).abs().max().item()
    assert diff < 1e-6, f"causal_rope: {diff}"
    print(f"  causal_rope_action_apply: OK ({diff:.2e})")

    print("✅ RoPE + embedding precision: ALL PASS")


# ── Test 6: WanRMSNorm precision ────────────────────────────────────

def test_rmsnorm_precision():
    """RMSNorm precision — uses pytest default_vllm_config fixture."""
    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import RMSNorm
    from groot.vla.model.dreamzero.modules.wan2_1_submodule import WanRMSNorm as DzNorm

    for dim in [64, 128, 5120]:
        vl = RMSNorm(dim, eps=1e-5)
        dz = DzNorm(dim, eps=1e-5)
        vl.weight.data.copy_(dz.weight.data)
        x = torch.randn(2, 10, dim)
        diff = (vl(x) - dz(x)).abs().max().item()
        assert diff < 1e-5, f"RMSNorm dim={dim}: {diff}"


def test_mlpproj_precision():
    """MLPProj (ColumnParallel+RowParallel) vs DreamZero (nn.Sequential).
    Uses default_weight_loader for correct parallel weight copy.
    """
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import MLPProj as VllmMLP
    from groot.vla.model.dreamzero.modules.wan2_1_submodule import MLPProj as DzMLP

    vl = VllmMLP(1280, 5120)
    dz = DzMLP(1280, 5120)

    # Copy weights: dz.proj = Sequential(LN[0], Linear[1], GELU[2], Linear[3], LN[4])
    # norm1 ← proj[0], fc1 ← proj[1], fc2 ← proj[3], norm2 ← proj[4]
    default_weight_loader(vl.norm1.weight, dz.proj[0].weight.data)
    default_weight_loader(vl.norm1.bias, dz.proj[0].bias.data)
    default_weight_loader(vl.fc1.weight, dz.proj[1].weight.data)
    default_weight_loader(vl.fc1.bias, dz.proj[1].bias.data)
    default_weight_loader(vl.fc2.weight, dz.proj[3].weight.data)
    default_weight_loader(vl.fc2.bias, dz.proj[3].bias.data)
    default_weight_loader(vl.norm2.weight, dz.proj[4].weight.data)
    default_weight_loader(vl.norm2.bias, dz.proj[4].bias.data)

    x = torch.randn(1, 257, 1280)
    diff = (vl(x) - dz(x)).abs().max().item()
    assert diff < 1e-5, f"MLPProj: {diff}"


if __name__ == "__main__":
    # Use pytest to run: cd dreamzero && pytest /path/to/test_causal_wan_model.py -v
    print("Run with pytest (provides default_vllm_config fixture):")
    print("  pytest tests/dreamzero/test_causal_wan_model.py -v")
