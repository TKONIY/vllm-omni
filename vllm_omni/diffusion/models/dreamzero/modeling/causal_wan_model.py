# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CausalWanModel — 40-layer DiT with causal attention and KV cache.

Adapted from: dreamzero/groot/vla/model/dreamzero/modules/
              wan_video_dit_action_casual_chunk.py L1218-2200

Key differences from WanTransformer3DModel (wan2_2_transformer.py):
- Causal self-attention (new frames only see history)
- KV cache for streaming inference
- Action/state token support (appended after video tokens)
- Extended RoPE with action/state-specific frequencies
- Dual forward: _forward_inference (KV cache) / _forward_train
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni.diffusion.models.dreamzero.modeling.action_encoder import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)


# ── RoPE utilities ──────────────────────────────────────────────────
# Source: wan2_1_submodule.py rope_params / rope_action_apply
#         wan_video_dit_action_casual_chunk.py L93-185 causal_rope_action_apply


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Sinusoidal positional embedding for timesteps.
    Source: wan2_1_submodule.py L16-26
    """
    assert dim % 2 == 0                                          # L18
    half = dim // 2                                              # L19
    position = position.type(torch.float64)                      # L20
    sinusoid = torch.outer(                                      # L23-24
        position,
        torch.pow(10000, -torch.arange(half, dtype=position.dtype, device=position.device).div(half)),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)  # L25
    return x


def rope_params(max_seq_len: int, dim: int) -> torch.Tensor:
    """Precompute complex-valued RoPE frequencies (polar form).
    Source: wan2_1_submodule.py L37-44 (rope_params_polar)
    Returns: complex tensor [max_seq_len, dim // 2]
    """
    assert dim % 2 == 0                                          # L38
    freqs = torch.outer(                                         # L39-42
        torch.arange(max_seq_len),
        1.0 / torch.pow(10000, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)           # L43
    return freqs


def rope_apply(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x using precomputed complex freqs.
    Source: wan2_1_submodule.py L64-75 (rope_apply_polar)
    """
    B, seq_len, n, _ = x.shape                                  # L65
    x = torch.view_as_complex(                                   # L68-70
        x.to(torch.float64).reshape(B, seq_len, n, -1, 2)
    )
    freqs = freqs.unsqueeze(0)                                   # L73
    x = torch.view_as_real(x * freqs).flatten(3)                 # L74
    return x


def rope_action_apply(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int = 32,
    num_state_per_block: int = 1,
) -> torch.Tensor:
    """RoPE with action/state frequency tables (training mode).
    Source: wan2_1_submodule.py L130-159 (rope_action_apply_polar)
    """
    B, seq_len, n, _ = x.shape                                  # L139
    x = torch.view_as_complex(                                   # L142-144
        x.to(torch.float64).reshape(B, seq_len, n, -1, 2)
    )
    if action_register_length is not None:                       # L146
        assert num_action_per_block is not None                  # L147
        assert num_state_per_block is not None                   # L148
        chunk_size = action_register_length // (num_action_per_block + num_state_per_block)  # L150
        freqs_1d_action = freqs_action[:chunk_size * num_action_per_block].view(  # L152
            chunk_size * num_action_per_block, 1, -1)
        freqs_1d_state = freqs_state[:chunk_size * num_state_per_block].view(     # L153
            chunk_size * num_state_per_block, 1, -1)
        freqs = torch.cat([freqs, freqs_1d_action, freqs_1d_state], dim=0)       # L154
    freqs = freqs.unsqueeze(0)                                   # L157
    x = torch.view_as_real(x * freqs).flatten(3)                 # L158
    return x


def causal_rope_action_apply(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int,
    num_state_per_block: int,
    action_state_index: int,
) -> torch.Tensor:
    """RoPE for single inference step (causal / KV-cache mode).
    Source: wan_video_dit_action_casual_chunk.py L153-185 (causal_rope_action_apply_polar)
    """
    B, seq_len, n, _ = x.shape                                  # L163
    x = torch.view_as_complex(                                   # L166-168
        x.to(torch.float64).reshape(B, seq_len, n, -1, 2)
    )
    if action_register_length is not None:                       # L170
        assert action_register_length == (num_action_per_block + num_state_per_block)  # L171
        freqs_action = freqs_action[                             # L172-174
            action_state_index * num_action_per_block:(action_state_index + 1) * num_action_per_block
        ]
        freqs_state = freqs_state[                               # L175-177
            action_state_index * num_state_per_block:(action_state_index + 1) * num_state_per_block
        ]
        freqs_1d = torch.cat([freqs_action, freqs_state], dim=0).view(  # L178
            action_register_length, 1, -1)
        freqs = torch.cat([freqs, freqs_1d], dim=0)             # L179
    freqs = freqs.unsqueeze(0)                                   # L182
    x = torch.view_as_real(x * freqs).flatten(3)                 # L183
    return x


# ── Normalization ───────────────────────────────────────────────────
# Reuse wan2_2's DistributedRMSNorm (supports TP, no vllm config needed)
# Source: wan2_1_submodule.py L162-178 (WanRMSNorm)
#         wan2_2_transformer.py L65-95 (DistributedRMSNorm — TP-aware version)

from vllm.model_executor.layers.layernorm import RMSNorm as WanRMSNorm  # noqa: E402
# Source: wan2_1_submodule.py L162-178 (WanRMSNorm)
# Using vllm's RMSNorm which supports TP and custom kernels.
# Requires VllmConfig context (auto-set during serving, use set_current_vllm_config in tests).


class WanLayerNorm(nn.LayerNorm):
    """Source: wan2_1_submodule.py L181-184"""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False) -> None:
        super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine)


# ── Projections ─────────────────────────────────────────────────────


class MLPProj(nn.Module):
    """CLIP feature projection for i2v.
    Source: wan2_1_submodule.py L565-577
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(                                   # L570-573
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        return self.proj(image_embeds)                               # L576


# ── Cross-Attention ─────────────────────────────────────────────────
# Source: wan_video_dit_action_casual_chunk.py L1087-1190 (referenced)
# T2V and I2V cross-attention variants


class WanT2VCrossAttention(nn.Module):
    """Text-to-video cross-attention.
    Source: wan_video_dit_action_casual_chunk.py (T2V cross-attn block)
    """

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(
        self, x: torch.Tensor, context: torch.Tensor,
        crossattn_cache: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class WanI2VCrossAttention(nn.Module):
    """Image-to-video cross-attention (splits first 257 image tokens).
    Source: wan_video_dit_action_casual_chunk.py (I2V cross-attn block)
    """

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(
        self, x: torch.Tensor, context: torch.Tensor,
        crossattn_cache: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


# ── Self-Attention with causal masking + KV cache ───────────────────
# Source: wan_video_dit_action_casual_chunk.py L188-1085


class CausalWanSelfAttention(nn.Module):
    """Causal self-attention with KV cache + action/state tokens.
    Source: wan_video_dit_action_casual_chunk.py L188-1085

    Two modes:
    - Training (kv_cache=None): blockwise causal with teacher forcing
    - Inference (kv_cache provided): incremental KV cache update
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        frame_seqlen: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        num_frame_per_block: int = 1,
        qk_norm: bool = True,
        eps: float = 1e-6,
        num_action_per_block: int = 32,
        num_state_per_block: int = 1,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        freqs_state: torch.Tensor,
        context: torch.Tensor,
        action_register_length: int | None,
        kv_cache: torch.Tensor | None = None,
        current_start_frame: int = 0,
        is_tf: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns: (output, updated_kv_cache_or_None)
        """
        raise NotImplementedError


# ── Attention Block ─────────────────────────────────────────────────
# Source: wan_video_dit_action_casual_chunk.py L1087-1190


class CausalWanAttentionBlock(nn.Module):
    """Transformer block: self-attn + cross-attn + FFN with 6-param modulation.
    Source: wan_video_dit_action_casual_chunk.py L1087-1190
    """

    def __init__(
        self,
        cross_attn_type: str,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        frame_seqlen: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        num_frame_per_block: int = 1,
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        num_action_per_block: int = 32,
        num_state_per_block: int = 1,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        freqs_state: torch.Tensor,
        context: torch.Tensor,
        action_register_length: int | None = None,
        kv_cache: torch.Tensor | None = None,
        crossattn_cache: torch.Tensor | None = None,
        current_start_frame: int = 0,
        is_tf: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns: (output, updated_kv_cache_or_None)
        """
        raise NotImplementedError


# ── Output Head ─────────────────────────────────────────────────────
# Source: wan_video_dit_action_casual_chunk.py L1190-1215


class CausalHead(nn.Module):
    """Output norm + linear with 2-param modulation.
    Source: wan_video_dit_action_casual_chunk.py L1190-1215
    """

    def __init__(self, dim: int, out_dim: int, patch_size: tuple, eps: float = 1e-6) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ── Main Model ──────────────────────────────────────────────────────
# Source: wan_video_dit_action_casual_chunk.py L1218-2200


class CausalWanModel(nn.Module):
    """Causal video diffusion transformer for DreamZero.

    Source: wan_video_dit_action_casual_chunk.py L1218-2200
    Architecture (14B): 40 layers, dim=5120, heads=40, ffn=13824

    __init__ params match original L1230-1256:
        model_type, patch_size, frame_seqlen, text_len, in_dim, dim,
        ffn_dim, freq_dim, text_dim, out_dim, num_heads, num_layers,
        max_chunk_size, sink_size, qk_norm, cross_attn_norm, eps,
        num_frame_per_block, action_dim, num_registers, max_state_dim,
        max_num_embodiments, hidden_size, diffusion_model_pretrained_path,
        num_action_per_block, num_state_per_block
    """

    def __init__(
        self,
        model_type: str = "t2v",
        patch_size: tuple[int, int, int] = (1, 2, 2),
        frame_seqlen: int = 220,
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 2048,
        ffn_dim: int = 8192,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 16,
        num_layers: int = 32,
        max_chunk_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        num_frame_per_block: int = 1,
        action_dim: int = 32,
        num_registers: int = 8,
        max_state_dim: int = 64,
        max_num_embodiments: int = 32,
        hidden_size: int = 1024,
        diffusion_model_pretrained_path: str | None = None,
        num_action_per_block: int = 32,
        num_state_per_block: int = 1,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def _create_freqs(self, grid_size: torch.Tensor, start_frame: int) -> torch.Tensor:
        """Create 3D RoPE frequency tensor.
        Source: wan_video_dit_action_casual_chunk.py L2151-2174
        """
        raise NotImplementedError

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor) -> torch.Tensor:
        """Reconstruct video from patch embeddings.
        Source: wan_video_dit_action_casual_chunk.py L2127-2149
        """
        raise NotImplementedError

    def _forward_blocks(
        self,
        x: torch.Tensor,
        seq_len: int,
        freqs: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: torch.Tensor | None,
        embodiment_id: torch.Tensor | None,
        action: torch.Tensor | None,
        timestep_action: torch.Tensor | None,
        state: torch.Tensor | None,
        kv_cache: list[torch.Tensor] | None,
        current_start_frame: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        """Process through all transformer blocks.
        Source: wan_video_dit_action_casual_chunk.py L1691-1780
        Returns: (x_video, action_noise_pred, updated_kv_caches)
        """
        raise NotImplementedError

    def _forward_inference(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        seq_len: int,
        kv_cache: list[torch.Tensor],
        crossattn_cache: list[torch.Tensor],
        current_start_frame: int,
        y: torch.Tensor | None = None,
        clip_feature: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        timestep_action: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        embodiment_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        """Inference with KV cache.
        Source: wan_video_dit_action_casual_chunk.py L1863-1950
        Returns: (video_noise_pred, action_noise_pred, updated_kv_caches)
        """
        raise NotImplementedError

    def _forward_train(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        seq_len: int,
        clean_x: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        clip_feature: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        timestep_action: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        embodiment_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Training with full sequence (optional teacher forcing).
        Source: wan_video_dit_action_casual_chunk.py L1952-2115
        Returns: (video_noise_pred, action_noise_pred)
        """
        raise NotImplementedError

    def forward(self, *args: Any, **kwargs: Any):
        """Route to inference or train based on kv_cache presence.
        Source: wan_video_dit_action_casual_chunk.py L2117-2125
        """
        if kwargs.get("kv_cache", None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)
