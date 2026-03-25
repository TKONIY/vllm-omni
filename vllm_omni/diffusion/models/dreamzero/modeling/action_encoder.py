# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Action and state encoder/decoder components for DreamZero."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for timesteps.

    Input:  timesteps of shape ``[B, T]``
    Output: positional embeddings of shape ``[B, T, dim]``
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # timesteps: (B, T)
        timesteps = timesteps.float()
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * (math.log(10000.0) / half_dim)
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class CategorySpecificLinear(nn.Module):
    """Per-category (embodiment) linear layer.

    Stores ``W: [num_categories, out_dim, in_dim]`` and
    ``b: [num_categories, out_dim]``.  Forward selects weights by
    ``category_id`` and applies a batched matmul per sample.
    """

    def __init__(
        self,
        num_categories: int,
        input_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.num_categories = num_categories
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_dim), cat_ids: (B,)
        selected_W = self.W[cat_ids]  # (B, in_dim, hidden_dim)
        selected_b = self.b[cat_ids]  # (B, hidden_dim)
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    """Two-layer MLP using :class:`CategorySpecificLinear`.

    Used for ``state_encoder`` and ``action_decoder``.
    """

    def __init__(
        self,
        num_categories: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    """Encodes action vectors with embodiment-specific MLPs and sinusoidal
    timestep encoding.

    Input:
        action      -- ``[B, T, action_dim]``
        timestep    -- ``[B, T]``
        embodiment_id -- ``[B]``
    Output:
        ``[B, T, hidden_dim]``

    Structure: three layers of :class:`CategorySpecificLinear` with SiLU
    activations and sinusoidal positional encoding injected between the
    first and second layers.
    """

    def __init__(
        self,
        action_dim: int,
        hidden_size: int,
        num_embodiments: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: action_dim -> hidden_size
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        # W2: 2 * hidden_size -> hidden_size (concat of action emb + pos enc)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)
        # W3: hidden_size -> hidden_size
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        cat_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            actions:   ``[B, T, action_dim]``
            timesteps: ``[B, T]`` timestep indices
            cat_ids:   ``[B]`` embodiment ids
        Returns:
            ``[B, T, hidden_size]``
        """
        # Project actions: (B, T, action_dim) -> (B, T, hidden_size)
        a_emb = self.W1(actions, cat_ids)

        # Sinusoidal timestep encoding: (B, T) -> (B, T, hidden_size)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # Concat + project: (B, T, 2*hidden_size) -> (B, T, hidden_size)
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # Final projection
        x = self.W3(x, cat_ids)
        return x
