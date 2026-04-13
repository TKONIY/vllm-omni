# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Precision alignment tests for action encoder components.

Tests that vllm-omni implementations produce identical outputs
to DreamZero original code with the same weights and inputs.

Run: PYTHONPATH=. python tests/dreamzero/test_action_encoder.py
Requires: dreamzero repo at ../dreamzero (for original imports)
"""

import os
import sys

import torch

# Add dreamzero to path for original imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../third_party/dreamzero"))
# Also try sibling directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../dreamzero"))


def _copy_weights(src, dst):
    """Copy state_dict from src to dst (same architecture, same keys)."""
    dst.load_state_dict(src.state_dict())


def _load_dz_classes():
    """Load DreamZero original classes without triggering flash_attn import.

    The classes we need (CategorySpecificLinear etc.) are defined at the top
    of wan_video_dit_action_casual_chunk.py (L31-90) and only depend on
    torch/nn/F + SinusoidalPositionalEncoding + swish. We exec just those
    lines to avoid importing flash_attn.
    """
    import importlib

    # SinusoidalPositionalEncoding and swish come from a clean module
    ae_mod = importlib.import_module("groot.vla.model.n1_5.modules.action_encoder")
    SinusoidalPositionalEncoding = ae_mod.SinusoidalPositionalEncoding
    swish_fn = ae_mod.swish

    # Read the source file and exec only the class definitions
    import pathlib

    src_path = pathlib.Path(
        os.path.join(
            os.path.dirname(__file__),
            "../../third_party/dreamzero",
            "groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py",
        )
    ).resolve()
    if not src_path.exists():
        src_path = pathlib.Path(
            os.path.join(
                os.path.dirname(__file__),
                "../../../dreamzero",
                "groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py",
            )
        ).resolve()

    source = src_path.read_text()
    # Extract lines 31-90 (the 4 classes we need)
    lines = source.split("\n")
    # Find class boundaries
    start = next(i for i, line in enumerate(lines) if "class CategorySpecificLinear" in line)
    end = next(i for i, line in enumerate(lines) if "def causal_rope_action_apply" in line)
    snippet = "\n".join(lines[start:end])

    ns = {
        "torch": torch,
        "nn": torch.nn,
        "F": torch.nn.functional,
        "SinusoidalPositionalEncoding": SinusoidalPositionalEncoding,
        "swish": swish_fn,
    }
    exec(snippet, ns)
    return ns


# ── Test 1: SinusoidalPositionalEncoding ────────────────────────────


def test_sinusoidal_positional_encoding():
    from groot.vla.model.n1_5.modules.action_encoder import (
        SinusoidalPositionalEncoding as DzSPE,
    )

    from vllm_omni.diffusion.models.dreamzero.modeling.action_encoder import (
        SinusoidalPositionalEncoding as VllmSPE,
    )

    for dim in [64, 128, 256, 5120]:
        vllm_enc = VllmSPE(dim)
        dz_enc = DzSPE(dim)

        timesteps = torch.tensor([[0, 100, 500, 999], [10, 20, 30, 40]])  # (2, 4)
        vllm_out = vllm_enc(timesteps)
        dz_out = dz_enc(timesteps)

        max_diff = (vllm_out - dz_out).abs().max().item()
        assert torch.allclose(vllm_out, dz_out, atol=1e-6), f"SinusoidalPE dim={dim}: max_diff={max_diff}"
        print(f"  SinusoidalPE dim={dim}: OK (max_diff={max_diff:.2e})")

    print("✅ SinusoidalPositionalEncoding: PASS")


# ── Test 2: CategorySpecificLinear ──────────────────────────────────


def test_category_specific_linear():
    from vllm_omni.diffusion.models.dreamzero.modeling.action_encoder import (
        CategorySpecificLinear as VllmCSL,
    )

    dz_ns = _load_dz_classes()
    DzCSL = dz_ns["CategorySpecificLinear"]

    vllm_layer = VllmCSL(num_categories=4, input_dim=8, hidden_dim=16)
    dz_layer = DzCSL(num_categories=4, input_dim=8, hidden_dim=16)
    _copy_weights(vllm_layer, dz_layer)

    x = torch.randn(2, 5, 8)
    cat_ids = torch.tensor([0, 3])

    vllm_out = vllm_layer(x, cat_ids)
    dz_out = dz_layer(x, cat_ids)

    max_diff = (vllm_out - dz_out).abs().max().item()
    assert torch.allclose(vllm_out, dz_out, atol=1e-6), f"CategorySpecificLinear: max_diff={max_diff}"
    print(f"✅ CategorySpecificLinear: PASS (max_diff={max_diff:.2e})")


# ── Test 3: CategorySpecificMLP ─────────────────────────────────────


def test_category_specific_mlp():
    from vllm_omni.diffusion.models.dreamzero.modeling.action_encoder import (
        CategorySpecificMLP as VllmMLP,
    )

    dz_ns = _load_dz_classes()
    DzMLP = dz_ns["CategorySpecificMLP"]

    vllm_mlp = VllmMLP(num_categories=4, input_dim=8, hidden_dim=32, output_dim=16)
    dz_mlp = DzMLP(num_categories=4, input_dim=8, hidden_dim=32, output_dim=16)
    _copy_weights(vllm_mlp, dz_mlp)

    x = torch.randn(2, 5, 8)
    cat_ids = torch.tensor([1, 2])

    vllm_out = vllm_mlp(x, cat_ids)
    dz_out = dz_mlp(x, cat_ids)

    max_diff = (vllm_out - dz_out).abs().max().item()
    assert torch.allclose(vllm_out, dz_out, atol=1e-6), f"CategorySpecificMLP: max_diff={max_diff}"
    print(f"✅ CategorySpecificMLP: PASS (max_diff={max_diff:.2e})")


# ── Test 4: MultiEmbodimentActionEncoder ────────────────────────────


def test_multi_embodiment_action_encoder():
    from vllm_omni.diffusion.models.dreamzero.modeling.action_encoder import (
        MultiEmbodimentActionEncoder as VllmEnc,
    )

    dz_ns = _load_dz_classes()
    DzEnc = dz_ns["MultiEmbodimentActionEncoder"]

    vllm_enc = VllmEnc(action_dim=32, hidden_size=64, num_embodiments=4)
    dz_enc = DzEnc(action_dim=32, hidden_size=64, num_embodiments=4)
    _copy_weights(vllm_enc, dz_enc)

    torch.manual_seed(42)
    actions = torch.randn(2, 24, 32)  # (B, T, action_dim)
    timesteps = torch.randint(0, 1000, (2, 24))  # (B, T)
    cat_ids = torch.tensor([0, 2])  # (B,)

    vllm_out = vllm_enc(actions, timesteps, cat_ids)
    dz_out = dz_enc(actions, timesteps, cat_ids)

    max_diff = (vllm_out - dz_out).abs().max().item()
    assert torch.allclose(vllm_out, dz_out, atol=1e-6), f"MultiEmbodimentActionEncoder: max_diff={max_diff}"
    print(f"✅ MultiEmbodimentActionEncoder: PASS (max_diff={max_diff:.2e})")


# ── Test 5: Shape tests (no original code needed) ──────────────────


def test_shapes():
    from vllm_omni.diffusion.models.dreamzero.modeling.action_encoder import (
        CategorySpecificLinear,
        CategorySpecificMLP,
        MultiEmbodimentActionEncoder,
        SinusoidalPositionalEncoding,
    )

    # SinusoidalPE
    spe = SinusoidalPositionalEncoding(128)
    assert spe(torch.zeros(2, 10)).shape == (2, 10, 128)

    # CategorySpecificLinear
    csl = CategorySpecificLinear(4, 8, 16)
    assert csl(torch.zeros(2, 5, 8), torch.tensor([0, 1])).shape == (2, 5, 16)

    # CategorySpecificMLP
    mlp = CategorySpecificMLP(4, 8, 32, 16)
    assert mlp(torch.zeros(2, 5, 8), torch.tensor([0, 1])).shape == (2, 5, 16)

    # MultiEmbodimentActionEncoder
    enc = MultiEmbodimentActionEncoder(32, 64, 4)
    out = enc(torch.zeros(2, 24, 32), torch.zeros(2, 24).long(), torch.tensor([0, 1]))
    assert out.shape == (2, 24, 64)

    print("✅ Shape tests: PASS")


if __name__ == "__main__":
    print("=== Action Encoder Tests ===\n")

    # Shape tests (always run, no dreamzero dependency)
    test_shapes()

    # Precision alignment tests (need dreamzero repo)
    try:
        import groot  # noqa: F401

        has_dreamzero = True
    except ImportError:
        has_dreamzero = False
        print("\n⚠️  dreamzero not available, skipping precision alignment tests")

    if has_dreamzero:
        test_sinusoidal_positional_encoding()
        test_category_specific_linear()
        test_category_specific_mlp()
        test_multi_embodiment_action_encoder()

    print("\n=== ALL TESTS PASSED ===")
