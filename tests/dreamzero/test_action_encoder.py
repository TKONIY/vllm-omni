# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for action encoder/decoder components."""

import torch


def test_category_specific_linear():
    from vllm_omni.diffusion.models.dreamzero.modeling.action_encoder import CategorySpecificLinear

    layer = CategorySpecificLinear(num_categories=4, input_dim=8, hidden_dim=16)
    x = torch.randn(2, 5, 8)
    cat_ids = torch.tensor([0, 2])
    out = layer(x, cat_ids)
    assert out.shape == (2, 5, 16), f"Expected (2,5,16), got {out.shape}"
    print("CategorySpecificLinear: OK")


def test_category_specific_mlp():
    from vllm_omni.diffusion.models.dreamzero.modeling.action_encoder import CategorySpecificMLP

    mlp = CategorySpecificMLP(num_categories=4, input_dim=8, hidden_dim=32, output_dim=16)
    x = torch.randn(2, 5, 8)
    cat_ids = torch.tensor([1, 3])
    out = mlp(x, cat_ids)
    assert out.shape == (2, 5, 16), f"Expected (2,5,16), got {out.shape}"
    print("CategorySpecificMLP: OK")


def test_sinusoidal_positional_encoding():
    from vllm_omni.diffusion.models.dreamzero.modeling.action_encoder import SinusoidalPositionalEncoding

    enc = SinusoidalPositionalEncoding(embedding_dim=64)
    timesteps = torch.tensor([[0, 1, 2], [10, 20, 30]])
    out = enc(timesteps)
    assert out.shape == (2, 3, 64), f"Expected (2,3,64), got {out.shape}"
    print("SinusoidalPositionalEncoding: OK")


def test_multi_embodiment_action_encoder():
    from vllm_omni.diffusion.models.dreamzero.modeling.action_encoder import MultiEmbodimentActionEncoder

    encoder = MultiEmbodimentActionEncoder(
        action_dim=8,
        hidden_size=64,
        num_embodiments=4,
    )
    action = torch.randn(2, 24, 8)
    timestep = torch.randint(0, 1000, (2, 24))
    embodiment_id = torch.tensor([0, 2])
    out = encoder(action, timestep, embodiment_id)
    assert out.shape == (2, 24, 64), f"Expected (2,24,64), got {out.shape}"
    print("MultiEmbodimentActionEncoder: OK")


if __name__ == "__main__":
    test_category_specific_linear()
    test_category_specific_mlp()
    test_sinusoidal_positional_encoding()
    test_multi_embodiment_action_encoder()
    print("\nAll action encoder tests passed!")
