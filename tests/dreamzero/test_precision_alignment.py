# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision alignment test: compare vllm-omni CausalWanModel vs DreamZero original.

Tests that identical weights + identical inputs produce identical outputs.
Runs on CPU with tiny model config (no GPU or model weights needed).
"""

import os
import sys

import torch
import torch.nn as nn

# Add third_party/dreamzero to path so we can import original code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../third_party/dreamzero"))


def _init_weights_deterministic(model: nn.Module, seed: int = 42):
    """Initialize all parameters with deterministic values."""
    gen = torch.Generator().manual_seed(seed)
    for name, param in model.named_parameters():
        param.data = torch.randn_like(param, generator=gen) * 0.02


def test_rope_params_alignment():
    """Verify rope_params produces identical frequencies."""
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import rope_params as dz_rope

    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import rope_params as vllm_rope

    for d in [16, 32, 64, 128]:
        vllm_freqs = vllm_rope(1024, d)
        dz_freqs = dz_rope(1024, d)
        assert torch.allclose(vllm_freqs, dz_freqs, atol=1e-10), (
            f"rope_params mismatch for d={d}: max_diff={torch.max(torch.abs(vllm_freqs - dz_freqs))}"
        )
    print("rope_params alignment: OK")


def test_causal_rope_action_apply_alignment():
    """Verify causal_rope_action_apply produces identical results."""
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
        causal_rope_action_apply_polar as dz_fn,
    )

    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import (
        causal_rope_action_apply as vllm_fn,
    )
    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import (
        rope_params as vllm_rope,
    )

    B, seq_len, n_heads, head_dim = 1, 8, 4, 16
    d = head_dim
    x = torch.randn(B, seq_len, n_heads, head_dim)

    # Create matching frequencies
    freqs = vllm_rope(1024, d)[: seq_len - 5].view(-1, 1, d // 2)  # spatial tokens
    freqs_action = vllm_rope(10240, d)
    freqs_state = vllm_rope(1024, d)

    # With action/state tokens
    vllm_out = vllm_fn(
        x,
        freqs,
        freqs_action,
        freqs_state,
        action_register_length=5,  # 4 action + 1 state
        num_action_per_block=4,
        num_state_per_block=1,
        action_state_index=0,
    )
    dz_out = dz_fn(
        x,
        freqs,
        freqs_action,
        freqs_state,
        action_register_length=5,
        num_action_per_block=4,
        num_state_per_block=1,
        action_state_index=0,
    )

    max_diff = torch.max(torch.abs(vllm_out.float() - dz_out.float()))
    assert torch.allclose(vllm_out.float(), dz_out.float(), atol=1e-6), (
        f"causal_rope_action_apply mismatch: max_diff={max_diff}"
    )
    print(f"causal_rope_action_apply alignment: OK (max_diff={max_diff:.2e})")


def test_rope_action_apply_alignment():
    """Verify rope_action_apply (training mode) produces identical results."""
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
        rope_action_apply as dz_fn,
    )

    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import (
        rope_action_apply as vllm_fn,
    )
    from vllm_omni.diffusion.models.dreamzero.modeling.causal_wan_model import (
        rope_params as vllm_rope,
    )

    B, n_heads, head_dim = 1, 4, 16
    d = head_dim
    # 8 spatial + 4 action + 1 state = 13
    seq_len = 13
    x = torch.randn(B, seq_len, n_heads, head_dim)

    freqs = vllm_rope(1024, d)[:8].view(-1, 1, d // 2)
    freqs_action = vllm_rope(10240, d)
    freqs_state = vllm_rope(1024, d)

    vllm_out = vllm_fn(
        x,
        freqs,
        freqs_action,
        freqs_state,
        action_register_length=5,  # 4 action + 1 state
        num_action_per_block=4,
        num_state_per_block=1,
    )
    dz_out = dz_fn(
        x,
        freqs,
        freqs_action,
        freqs_state,
        action_register_length=5,
        num_action_per_block=4,
        num_state_per_block=1,
    )

    max_diff = torch.max(torch.abs(vllm_out.float() - dz_out.float()))
    assert torch.allclose(vllm_out.float(), dz_out.float(), atol=1e-6), (
        f"rope_action_apply mismatch: max_diff={max_diff}"
    )
    print(f"rope_action_apply alignment: OK (max_diff={max_diff:.2e})")


def test_action_encoder_alignment():
    """Verify MultiEmbodimentActionEncoder produces identical results with same weights."""
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
        MultiEmbodimentActionEncoder as DzEncoder,
    )

    from vllm_omni.diffusion.models.dreamzero.modeling.action_encoder import (
        MultiEmbodimentActionEncoder as VllmEncoder,
    )

    # Create with same config
    kwargs = dict(action_dim=8, hidden_size=64, num_embodiments=4)
    vllm_enc = VllmEncoder(**kwargs)
    dz_enc = DzEncoder(**kwargs)

    # Copy weights from vllm to dz
    vllm_state = vllm_enc.state_dict()
    dz_enc.load_state_dict(vllm_state, strict=False)

    # Same input
    torch.manual_seed(42)
    action = torch.randn(1, 24, 8)
    timestep = torch.randint(0, 1000, (1, 24))
    cat_ids = torch.tensor([0])

    vllm_out = vllm_enc(action, timestep, cat_ids)
    dz_out = dz_enc(action, timestep, cat_ids)

    max_diff = torch.max(torch.abs(vllm_out - dz_out))
    assert torch.allclose(vllm_out, dz_out, atol=1e-6), f"ActionEncoder mismatch: max_diff={max_diff}"
    print(f"ActionEncoder alignment: OK (max_diff={max_diff:.2e})")


def test_category_specific_mlp_alignment():
    """Verify CategorySpecificMLP produces identical results."""
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CategorySpecificMLP as DzMLP

    from vllm_omni.diffusion.models.dreamzero.modeling.action_encoder import CategorySpecificMLP as VllmMLP

    kwargs = dict(num_categories=4, input_dim=8, hidden_dim=32, output_dim=16)
    vllm_mlp = VllmMLP(**kwargs)
    dz_mlp = DzMLP(**kwargs)

    # Copy weights
    vllm_state = vllm_mlp.state_dict()
    dz_mlp.load_state_dict(vllm_state, strict=False)

    torch.manual_seed(42)
    x = torch.randn(2, 5, 8)
    cat_ids = torch.tensor([1, 3])

    vllm_out = vllm_mlp(x, cat_ids)
    dz_out = dz_mlp(x, cat_ids)

    max_diff = torch.max(torch.abs(vllm_out - dz_out))
    assert torch.allclose(vllm_out, dz_out, atol=1e-6), f"CategorySpecificMLP mismatch: max_diff={max_diff}"
    print(f"CategorySpecificMLP alignment: OK (max_diff={max_diff:.2e})")


if __name__ == "__main__":
    test_rope_params_alignment()
    test_causal_rope_action_apply_alignment()
    test_rope_action_apply_alignment()
    test_action_encoder_alignment()
    test_category_specific_mlp_alignment()
    print("\nAll precision alignment tests passed!")
