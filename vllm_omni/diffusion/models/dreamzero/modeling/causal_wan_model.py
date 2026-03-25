# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CausalWanModel -- causal video diffusion transformer for DreamZero.

Adapted from:
  third_party/dreamzero/groot/vla/model/dreamzero/modules/
      wan_video_dit_action_casual_chunk.py

Style reference:
  vllm_omni/diffusion/models/wan2_2/wan2_2_transformer.py
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

# ---------------------------------------------------------------------------
# RoPE utilities
# ---------------------------------------------------------------------------


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Sinusoidal positional embedding.

    Args:
        dim: Embedding dimension (must be even).
        position: Flat tensor of positions.

    Returns:
        Tensor of shape ``[len(position), dim]``.
    """
    assert dim % 2 == 0
    half = dim // 2
    position = position.to(torch.float64)
    sinusoid = torch.outer(
        position,
        torch.pow(
            10000,
            -torch.arange(half, dtype=position.dtype, device=position.device).div(half),
        ),
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


def rope_params(max_seq_len: int, dim: int, theta: float = 10000) -> torch.Tensor:
    """Precompute complex-valued RoPE frequencies (polar form).

    Returns:
        Complex tensor of shape ``[max_seq_len, dim // 2]``.
    """
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0
        / torch.pow(
            theta,
            torch.arange(0, dim, 2).to(torch.float64).div(dim),
        ),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def rope_apply(
    x: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to *x* using precomputed complex *freqs*.

    Args:
        x: ``[B, seq_len, num_heads, head_dim]``
        freqs: ``[seq_len, head_dim // 2]`` (complex)

    Returns:
        Same shape as *x*, with rotary embeddings applied.
    """
    B, seq_len, n, _ = x.shape
    x_complex = torch.view_as_complex(x.to(torch.float64).reshape(B, seq_len, n, -1, 2))
    freqs = freqs.unsqueeze(0)  # [1, seq_len, 1, head_dim//2]
    out = torch.view_as_real(x_complex * freqs).flatten(3)
    return out


def rope_action_apply(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int = 32,
    num_state_per_block: int = 1,
) -> torch.Tensor:
    """Apply RoPE with separate action/state frequency tables.

    When ``action_register_length`` is not None the action and state
    frequency rows are concatenated after the spatial frequencies so
    that action/state tokens get their own positional encoding.
    """
    B, seq_len, n, _ = x.shape
    x_complex = torch.view_as_complex(x.to(torch.float64).reshape(B, seq_len, n, -1, 2))

    if action_register_length is not None and action_register_length > 0:
        total_per_block = num_action_per_block + num_state_per_block
        chunk_size = action_register_length // total_per_block if total_per_block > 0 else 0
        n_action = chunk_size * num_action_per_block
        n_state = chunk_size * num_state_per_block
        parts = [freqs]
        if n_action > 0:
            parts.append(freqs_action[:n_action].view(n_action, 1, -1))
        if n_state > 0:
            parts.append(freqs_state[:n_state].view(n_state, 1, -1))
        freqs = torch.cat(parts, dim=0)

    freqs = freqs.unsqueeze(0)
    out = torch.view_as_real(x_complex * freqs).flatten(3)
    return out


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
    """RoPE for a single inference step (causal / KV-cache mode).

    Unlike :func:`rope_action_apply`, this selects exactly the action/state
    frequencies corresponding to the current ``action_state_index`` and
    appends them to the spatial frequencies for the new tokens.
    """
    B, seq_len, n, _ = x.shape
    x_complex = torch.view_as_complex(x.to(torch.float64).reshape(B, seq_len, n, -1, 2))

    if action_register_length is not None:
        assert action_register_length == (num_action_per_block + num_state_per_block)
        freqs_action_slice = freqs_action[
            action_state_index * num_action_per_block : (action_state_index + 1) * num_action_per_block
        ]
        freqs_state_slice = freqs_state[
            action_state_index * num_state_per_block : (action_state_index + 1) * num_state_per_block
        ]
        freqs_1d = torch.cat([freqs_action_slice, freqs_state_slice], dim=0).view(action_register_length, 1, -1)
        freqs = torch.cat([freqs, freqs_1d], dim=0)

    freqs = freqs.unsqueeze(0)
    out = torch.view_as_real(x_complex * freqs).flatten(3)
    return out


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


class WanRMSNorm(nn.Module):
    """Root-mean-square layer normalization."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.weight


class WanLayerNorm(nn.LayerNorm):
    """LayerNorm with *elementwise_affine=False* by default."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
    ) -> None:
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)


# ---------------------------------------------------------------------------
# MLP projection for CLIP features (i2v)
# ---------------------------------------------------------------------------


class MLPProj(nn.Module):
    """Two-layer MLP with LayerNorm for projecting CLIP image features."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        return self.proj(image_embeds)


# ---------------------------------------------------------------------------
# Cross-attention modules
# ---------------------------------------------------------------------------


class WanT2VCrossAttention(nn.Module):
    """Text-to-video cross-attention (text as context)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        crossattn_cache: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        # Scaled dot-product attention
        q = q.transpose(1, 2)  # [B, n, L_q, d]
        k = k.transpose(1, 2)  # [B, n, L_k, d]
        v = v.transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).contiguous()  # [B, L_q, n, d]

        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(nn.Module):
    """Image-to-video cross-attention (CLIP image + text as context)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        # Additional projections for image embeddings
        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        crossattn_cache: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        # Split context: first 257 tokens are image, rest is text
        context_img = context[:, :257]
        context_txt = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context_txt)).view(b, -1, n, d)
                v = self.v(context_txt).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context_txt)).view(b, -1, n, d)
            v = self.v(context_txt).view(b, -1, n, d)

        # Text cross-attention
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        x_txt = F.scaled_dot_product_attention(q_t, k_t, v_t)
        x_txt = x_txt.transpose(1, 2).contiguous().flatten(2)

        # Image cross-attention
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        q_i = q.transpose(1, 2)
        k_i = k_img.transpose(1, 2)
        v_i = v_img.transpose(1, 2)
        x_img = F.scaled_dot_product_attention(q_i, k_i, v_i)
        x_img = x_img.transpose(1, 2).contiguous().flatten(2)

        x = self.o(x_txt + x_img)
        return x


WAN_CROSSATTENTION_CLASSES: dict[str, type[nn.Module]] = {
    "t2v_cross_attn": WanT2VCrossAttention,
    "i2v_cross_attn": WanI2VCrossAttention,
}


# ---------------------------------------------------------------------------
# Self-attention with causal masking and KV cache
# ---------------------------------------------------------------------------


def _sdpa_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Standard multi-head attention via ``F.scaled_dot_product_attention``.

    Args:
        q, k, v: ``[B, L, num_heads, head_dim]``

    Returns:
        ``[B, L_q, num_heads, head_dim]``
    """
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).contiguous()


def _sdpa_causal_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Causal multi-head attention via ``F.scaled_dot_product_attention``.

    Args:
        q, k, v: ``[B, L, num_heads, head_dim]``
            (q and k/v may have different sequence lengths; when L_q < L_k
            only the last L_q positions of the causal mask are used.)

    Returns:
        ``[B, L_q, num_heads, head_dim]``
    """
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=(q.shape[2] == k.shape[2]))
    return out.transpose(1, 2).contiguous()


class CausalWanSelfAttention(nn.Module):
    """Self-attention with blockwise causal masking and KV cache support.

    In *training* mode (no KV cache), the full sequence (video + action +
    state tokens) is processed at once with a blockwise causal mask where
    each frame block can only attend to itself and previous frame blocks
    (plus the first conditioning frame).

    In *inference* mode (KV cache provided), new tokens are appended to the
    cache and full-sequence attention is computed over the cached context.
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
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.num_frame_per_block = num_frame_per_block
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 21 * frame_seqlen if local_attn_size == -1 else local_attn_size * frame_seqlen
        self.frame_seqlen = frame_seqlen
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block

        # Projections
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    # ------------------------------------------------------------------
    # Blockwise causal attention (training / full-sequence)
    # ------------------------------------------------------------------

    def _blockwise_causal_flash_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        frame_seqlen: int,
        num_frame_per_block: int = 1,
        action_horizon: int | None = None,
        state_horizon: int | None = None,
        num_action_per_block: int | None = None,
        num_state_per_block: int | None = None,
    ) -> torch.Tensor:
        """Blockwise causal attention over the full sequence.

        Token layout: ``[first_frame | image_blocks | action_blocks | state_blocks]``

        Image block *i* can attend to: first frame + all image blocks <= i +
        action block *i* + state block *i*.

        Action block *i* can attend to: first frame + all image blocks <= i +
        action block *i* + state block *i*.

        State blocks only self-attend.

        When *action_horizon* is None, the sequence has no action/state tokens
        and a simple blockwise causal mask over frames is used.
        """
        b, total_len, n, d = q.shape
        has_action_state = action_horizon is not None and state_horizon is not None

        if not has_action_state:
            # Simple blockwise causal attention over image frames only
            num_frames = total_len // frame_seqlen
            block_size = frame_seqlen * num_frame_per_block
            num_blocks = (num_frames - 1) // num_frame_per_block

            if num_blocks <= 0:
                return _sdpa_attn(q, k, v)

            if self.local_attn_size == -1:
                return _sdpa_causal_attn(q, k, v)

            output = torch.empty_like(q)
            # First frame (conditioning): self-attention only
            output[:, :frame_seqlen] = _sdpa_attn(q[:, :frame_seqlen], k[:, :frame_seqlen], v[:, :frame_seqlen])
            for block_idx in range(num_blocks):
                block_start = frame_seqlen + block_idx * block_size
                block_end = min(block_start + block_size, total_len)
                kv_start = max(0, block_end - self.local_attn_size * frame_seqlen)
                output[:, block_start:block_end] = _sdpa_attn(
                    q[:, block_start:block_end],
                    k[:, kv_start:block_end],
                    v[:, kv_start:block_end],
                )
            return output

        # Multi-modal: [first_frame | image_blocks | action_blocks | state_blocks]
        assert num_action_per_block is not None and num_state_per_block is not None

        first_image_len = frame_seqlen
        action_len = action_horizon
        state_len = state_horizon
        image_blocks_len = total_len - first_image_len - action_len - state_len

        num_image_blocks = image_blocks_len // (num_frame_per_block * frame_seqlen)
        num_action_blocks = action_horizon // num_action_per_block
        num_state_blocks = state_horizon // num_state_per_block
        assert num_image_blocks == num_action_blocks == num_state_blocks

        first_image_start = 0
        first_image_end = first_image_len
        image_blocks_start = first_image_end
        image_blocks_end = image_blocks_start + image_blocks_len
        action_start = image_blocks_end
        state_start = action_start + action_len

        output = torch.empty_like(q)

        # First frame: self-attention
        output[:, first_image_start:first_image_end] = _sdpa_attn(
            q[:, first_image_start:first_image_end],
            k[:, first_image_start:first_image_end],
            v[:, first_image_start:first_image_end],
        )

        # Pre-compute block boundary indices
        image_block_starts = [
            image_blocks_start + i * num_frame_per_block * frame_seqlen for i in range(num_image_blocks)
        ]
        image_block_ends = [
            image_blocks_start + (i + 1) * num_frame_per_block * frame_seqlen for i in range(num_image_blocks)
        ]
        if self.local_attn_size != -1:
            image_kv_starts = [
                max(image_blocks_start, end - self.local_attn_size * frame_seqlen) for end in image_block_ends
            ]
        else:
            image_kv_starts = [image_blocks_start] * num_image_blocks

        action_block_starts = [action_start + i * num_action_per_block for i in range(num_action_blocks)]
        action_block_ends = [action_start + (i + 1) * num_action_per_block for i in range(num_action_blocks)]
        state_block_starts = [state_start + i * num_state_per_block for i in range(num_state_blocks)]
        state_block_ends = [state_start + (i + 1) * num_state_per_block for i in range(num_state_blocks)]

        # Process image blocks
        for idx in range(num_image_blocks):
            bs = image_block_starts[idx]
            be = image_block_ends[idx]
            kv_s = image_kv_starts[idx]
            abs_ = action_block_starts[idx]
            abe = action_block_ends[idx]
            sbs = state_block_starts[idx]
            sbe = state_block_ends[idx]

            k_ctx = torch.cat(
                [
                    k[:, first_image_start:first_image_end],
                    k[:, kv_s:be],
                    k[:, abs_:abe],
                    k[:, sbs:sbe],
                ],
                dim=1,
            )
            v_ctx = torch.cat(
                [
                    v[:, first_image_start:first_image_end],
                    v[:, kv_s:be],
                    v[:, abs_:abe],
                    v[:, sbs:sbe],
                ],
                dim=1,
            )
            output[:, bs:be] = _sdpa_attn(q[:, bs:be], k_ctx, v_ctx)

        # Process action blocks
        for idx in range(num_action_blocks):
            abs_ = action_block_starts[idx]
            abe = action_block_ends[idx]
            img_be = image_block_ends[idx]
            sbs = state_block_starts[idx]
            sbe = state_block_ends[idx]
            if self.local_attn_size != -1:
                img_kv_s = max(image_blocks_start, img_be - self.local_attn_size * frame_seqlen)
            else:
                img_kv_s = image_blocks_start

            k_ctx = torch.cat(
                [
                    k[:, first_image_start:first_image_end],
                    k[:, img_kv_s:img_be],
                    k[:, abs_:abe],
                    k[:, sbs:sbe],
                ],
                dim=1,
            )
            v_ctx = torch.cat(
                [
                    v[:, first_image_start:first_image_end],
                    v[:, img_kv_s:img_be],
                    v[:, abs_:abe],
                    v[:, sbs:sbe],
                ],
                dim=1,
            )
            output[:, abs_:abe] = _sdpa_attn(q[:, abs_:abe], k_ctx, v_ctx)

        # Process state blocks (self-attention only)
        for idx in range(num_state_blocks):
            sbs = state_block_starts[idx]
            sbe = state_block_ends[idx]
            output[:, sbs:sbe] = _sdpa_attn(q[:, sbs:sbe], k[:, sbs:sbe], v[:, sbs:sbe])

        return output

    # ------------------------------------------------------------------
    # Training helpers for teacher-forcing mode
    # ------------------------------------------------------------------

    def _process_clean_image_only(
        self,
        clean_q: torch.Tensor,
        clean_k: torch.Tensor,
        clean_v: torch.Tensor,
        clean_frames: int,
    ) -> torch.Tensor:
        """Blockwise causal attention over clean (context) image tokens."""
        block_size = self.frame_seqlen * self.num_frame_per_block
        num_blocks = (clean_frames - 1) // self.num_frame_per_block

        if num_blocks == 0:
            return _sdpa_attn(
                clean_q[:, : self.frame_seqlen],
                clean_k[:, : self.frame_seqlen],
                clean_v[:, : self.frame_seqlen],
            )

        b, total_len, n, d = clean_q.shape
        output = torch.empty_like(clean_q)

        # First frame: self-attention
        output[:, : self.frame_seqlen] = _sdpa_attn(
            clean_q[:, : self.frame_seqlen],
            clean_k[:, : self.frame_seqlen],
            clean_v[:, : self.frame_seqlen],
        )

        if self.local_attn_size == -1:
            # Global causal: all remaining tokens attend to first frame + all previous
            output[:, self.frame_seqlen :] = _sdpa_causal_attn(
                clean_q[:, self.frame_seqlen :],
                clean_k,
                clean_v,
            )
        else:
            for block_idx in range(num_blocks):
                block_start = self.frame_seqlen + block_idx * block_size
                block_end = min(block_start + block_size, total_len)
                image_kv_start = max(self.frame_seqlen, block_end - self.local_attn_size * self.frame_seqlen)
                k_ctx = torch.cat(
                    [clean_k[:, : self.frame_seqlen], clean_k[:, image_kv_start:block_end]],
                    dim=1,
                )
                v_ctx = torch.cat(
                    [clean_v[:, : self.frame_seqlen], clean_v[:, image_kv_start:block_end]],
                    dim=1,
                )
                output[:, block_start:block_end] = _sdpa_attn(clean_q[:, block_start:block_end], k_ctx, v_ctx)

        return output

    def _process_noisy_image_blocks(
        self,
        noisy_image_q: torch.Tensor,
        noisy_image_k: torch.Tensor,
        noisy_image_v: torch.Tensor,
        clean_image_k: torch.Tensor,
        clean_image_v: torch.Tensor,
        noisy_action_k: torch.Tensor,
        noisy_action_v: torch.Tensor,
        noisy_state_k: torch.Tensor,
        noisy_state_v: torch.Tensor,
        half_frames: int,
        action_horizon: int,
        state_horizon: int,
    ) -> torch.Tensor:
        """Process noisy image blocks with teacher forcing pattern.

        Block *i* attends to: clean_blocks[0..i-1] + current noisy block +
        action[i] + state[i].
        """
        block_size = self.frame_seqlen * self.num_frame_per_block
        num_blocks = (half_frames - 1) // self.num_frame_per_block

        output = torch.empty_like(noisy_image_q)

        # First noisy frame: self-attention only
        output[:, : self.frame_seqlen] = _sdpa_attn(
            noisy_image_q[:, : self.frame_seqlen],
            noisy_image_k[:, : self.frame_seqlen],
            noisy_image_v[:, : self.frame_seqlen],
        )

        if num_blocks == 0:
            return output

        for block_idx in range(num_blocks):
            noisy_start = self.frame_seqlen + block_idx * block_size
            noisy_end = min(noisy_start + block_size, noisy_image_q.shape[1])
            clean_end = self.frame_seqlen + block_idx * block_size
            act_start = block_idx * self.num_action_per_block
            act_end = act_start + self.num_action_per_block
            st_start = block_idx * self.num_state_per_block
            st_end = st_start + self.num_state_per_block

            k_ctx = torch.cat(
                [
                    clean_image_k[:, :clean_end],
                    noisy_image_k[:, noisy_start:noisy_end],
                    noisy_action_k[:, act_start:act_end],
                    noisy_state_k[:, st_start:st_end],
                ],
                dim=1,
            )
            v_ctx = torch.cat(
                [
                    clean_image_v[:, :clean_end],
                    noisy_image_v[:, noisy_start:noisy_end],
                    noisy_action_v[:, act_start:act_end],
                    noisy_state_v[:, st_start:st_end],
                ],
                dim=1,
            )
            output[:, noisy_start:noisy_end] = _sdpa_attn(noisy_image_q[:, noisy_start:noisy_end], k_ctx, v_ctx)

        return output

    def _process_noisy_action_blocks(
        self,
        noisy_action_q: torch.Tensor,
        noisy_action_k: torch.Tensor,
        noisy_action_v: torch.Tensor,
        clean_image_k: torch.Tensor,
        clean_image_v: torch.Tensor,
        noisy_image_k: torch.Tensor,
        noisy_image_v: torch.Tensor,
        noisy_state_k: torch.Tensor,
        noisy_state_v: torch.Tensor,
        half_frames: int,
        action_horizon: int,
        state_horizon: int,
    ) -> torch.Tensor:
        """Process noisy action blocks with teacher forcing pattern."""
        num_blocks = (half_frames - 1) // self.num_frame_per_block

        if num_blocks == 0:
            return torch.empty_like(noisy_action_q)

        output = torch.empty_like(noisy_action_q)

        for block_idx in range(num_blocks):
            act_start = block_idx * self.num_action_per_block
            act_end = act_start + self.num_action_per_block
            clean_end = self.frame_seqlen + block_idx * self.frame_seqlen * self.num_frame_per_block
            noisy_img_start = self.frame_seqlen + block_idx * self.frame_seqlen * self.num_frame_per_block
            noisy_img_end = noisy_img_start + self.frame_seqlen * self.num_frame_per_block
            st_start = block_idx * self.num_state_per_block
            st_end = st_start + self.num_state_per_block

            k_ctx = torch.cat(
                [
                    clean_image_k[:, :clean_end],
                    noisy_image_k[:, noisy_img_start:noisy_img_end],
                    noisy_action_k[:, act_start:act_end],
                    noisy_state_k[:, st_start:st_end],
                ],
                dim=1,
            )
            v_ctx = torch.cat(
                [
                    clean_image_v[:, :clean_end],
                    noisy_image_v[:, noisy_img_start:noisy_img_end],
                    noisy_action_v[:, act_start:act_end],
                    noisy_state_v[:, st_start:st_end],
                ],
                dim=1,
            )
            output[:, act_start:act_end] = _sdpa_attn(noisy_action_q[:, act_start:act_end], k_ctx, v_ctx)

        return output

    def _process_state_blocks(
        self,
        state_q: torch.Tensor,
        state_k: torch.Tensor,
        state_v: torch.Tensor,
        state_horizon: int,
    ) -> torch.Tensor:
        """Process state blocks: self-attention only per block."""
        num_blocks = state_horizon // self.num_state_per_block

        if num_blocks == 1:
            return _sdpa_attn(state_q, state_k, state_v)

        output = torch.empty_like(state_q)
        for block_idx in range(num_blocks):
            s = block_idx * self.num_state_per_block
            e = s + self.num_state_per_block
            output[:, s:e] = _sdpa_attn(state_q[:, s:e], state_k[:, s:e], state_v[:, s:e])
        return output

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        freqs_state: torch.Tensor,
        action_register_length: int | None,
        kv_cache: torch.Tensor | None = None,
        current_start_frame: int = 0,
        is_tf: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Self-attention forward with optional KV cache.

        Args:
            x: Hidden states ``[B, L, C]``.
            freqs: RoPE frequencies for spatial tokens.
            freqs_action: RoPE frequencies for action tokens.
            freqs_state: RoPE frequencies for state tokens.
            action_register_length: Total action + state register length
                (None if no action/state tokens).
            kv_cache: ``[2, B, cached_len, num_heads, head_dim]`` or None.
            current_start_frame: Frame index for the start of the current chunk
                (used in inference mode for RoPE indexing).
            is_tf: Whether this is teacher-forcing mode (training).

        Returns:
            ``(output, updated_kv_cache)`` where *updated_kv_cache* is None
            in training mode.
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        def qkv_fn(x_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            q = self.norm_q(self.q(x_in)).view(b, -1, n, d)
            k = self.norm_k(self.k(x_in)).view(b, -1, n, d)
            v = self.v(x_in).view(b, -1, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        updated_kv_cache: torch.Tensor | None = None

        if kv_cache is None:
            # ---- Full-sequence (training / non-cached) path ----
            if is_tf:
                # Teacher forcing: split into clean / noisy halves
                if action_register_length is not None:
                    q_context = q[:, : (s - action_register_length) // 2]
                    k_context = k[:, : (s - action_register_length) // 2]
                    q_noisy = q[:, (s - action_register_length) // 2 :]
                    k_noisy = k[:, (s - action_register_length) // 2 :]
                else:
                    q_context = q[:, : s // 2]
                    k_context = k[:, : s // 2]
                    q_noisy = q[:, s // 2 :]
                    k_noisy = k[:, s // 2 :]

                # RoPE is the same for both clean and noisy halves
                rq_context = rope_action_apply(
                    q_context, freqs, freqs_action, freqs_state, action_register_length=None
                ).type_as(v)
                rk_context = rope_action_apply(
                    k_context, freqs, freqs_action, freqs_state, action_register_length=None
                ).type_as(v)
                rq_noisy = rope_action_apply(
                    q_noisy,
                    freqs,
                    freqs_action,
                    freqs_state,
                    action_register_length=action_register_length,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                ).type_as(v)
                rk_noisy = rope_action_apply(
                    k_noisy,
                    freqs,
                    freqs_action,
                    freqs_state,
                    action_register_length=action_register_length,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                ).type_as(v)

                roped_query = torch.cat([rq_context, rq_noisy], dim=1)
                roped_key = torch.cat([rk_context, rk_noisy], dim=1)

                half_seq_len = (s - (action_register_length if action_register_length is not None else 0)) // 2

                if action_register_length is not None:
                    # --- Teacher forcing with action/state tokens ---
                    clean_image_seq_len = half_seq_len
                    clean_frames = clean_image_seq_len // self.frame_seqlen
                    noisy_image_seq_len = half_seq_len
                    noisy_frames = noisy_image_seq_len // self.frame_seqlen
                    num_image_blocks = (noisy_frames - 1) // self.num_frame_per_block
                    action_horizon = num_image_blocks * self.num_action_per_block
                    state_horizon = num_image_blocks * self.num_state_per_block

                    # Split clean/noisy parts
                    clean_image_q = roped_query[:, :clean_image_seq_len]
                    clean_image_k = roped_key[:, :clean_image_seq_len]
                    clean_image_v = v[:, :clean_image_seq_len]

                    noisy_off = half_seq_len
                    noisy_image_q = roped_query[:, noisy_off : noisy_off + noisy_image_seq_len]
                    noisy_action_q = roped_query[
                        :, noisy_off + noisy_image_seq_len : noisy_off + noisy_image_seq_len + action_horizon
                    ]
                    noisy_state_q = roped_query[:, noisy_off + noisy_image_seq_len + action_horizon :]

                    noisy_image_k = roped_key[:, noisy_off : noisy_off + noisy_image_seq_len]
                    noisy_action_k = roped_key[
                        :, noisy_off + noisy_image_seq_len : noisy_off + noisy_image_seq_len + action_horizon
                    ]
                    noisy_state_k = roped_key[:, noisy_off + noisy_image_seq_len + action_horizon :]

                    noisy_image_v = v[:, noisy_off : noisy_off + noisy_image_seq_len]
                    noisy_action_v = v[
                        :, noisy_off + noisy_image_seq_len : noisy_off + noisy_image_seq_len + action_horizon
                    ]
                    noisy_state_v = v[:, noisy_off + noisy_image_seq_len + action_horizon :]

                    clean_out = self._process_clean_image_only(
                        clean_image_q, clean_image_k, clean_image_v, clean_frames
                    )
                    noisy_img_out = self._process_noisy_image_blocks(
                        noisy_image_q,
                        noisy_image_k,
                        noisy_image_v,
                        clean_image_k,
                        clean_image_v,
                        noisy_action_k,
                        noisy_action_v,
                        noisy_state_k,
                        noisy_state_v,
                        noisy_frames,
                        action_horizon,
                        state_horizon,
                    )
                    noisy_act_out = self._process_noisy_action_blocks(
                        noisy_action_q,
                        noisy_action_k,
                        noisy_action_v,
                        clean_image_k,
                        clean_image_v,
                        noisy_image_k,
                        noisy_image_v,
                        noisy_state_k,
                        noisy_state_v,
                        noisy_frames,
                        action_horizon,
                        state_horizon,
                    )
                    noisy_state_out = self._process_state_blocks(
                        noisy_state_q, noisy_state_k, noisy_state_v, state_horizon
                    )

                    attn_out = torch.cat(
                        [clean_out, noisy_img_out, noisy_act_out, noisy_state_out],
                        dim=1,
                    )
                else:
                    # No action/state tokens -- simple teacher-forcing
                    half_seq_len // self.frame_seqlen
                    clean_q = roped_query[:, :half_seq_len]
                    clean_k = roped_key[:, :half_seq_len]
                    clean_v = v[:, :half_seq_len]
                    noisy_q = roped_query[:, half_seq_len:]
                    noisy_k = roped_key[:, half_seq_len:]
                    noisy_v = v[:, half_seq_len:]

                    x_clean = self._blockwise_causal_flash_attn(
                        clean_q,
                        clean_k,
                        clean_v,
                        self.frame_seqlen,
                        self.num_frame_per_block,
                    )
                    full_k = torch.cat([clean_k, noisy_k], dim=1)
                    full_v = torch.cat([clean_v, noisy_v], dim=1)
                    x_noisy = _sdpa_attn(noisy_q, full_k, full_v)
                    attn_out = torch.cat([x_clean, x_noisy], dim=1)

            else:
                # Non-teacher-forcing: standard blockwise causal
                roped_query = rope_action_apply(
                    q,
                    freqs,
                    freqs_action,
                    freqs_state,
                    action_register_length=action_register_length,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                ).type_as(v)
                roped_key = rope_action_apply(
                    k,
                    freqs,
                    freqs_action,
                    freqs_state,
                    action_register_length=action_register_length,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                ).type_as(v)

                if action_register_length is not None:
                    chunk_size = action_register_length // (self.num_action_per_block + self.num_state_per_block)
                    action_horizon = chunk_size * self.num_action_per_block
                    state_horizon = chunk_size * self.num_state_per_block
                else:
                    action_horizon = None
                    state_horizon = None

                attn_out = self._blockwise_causal_flash_attn(
                    roped_query,
                    roped_key,
                    v,
                    self.frame_seqlen,
                    self.num_frame_per_block,
                    action_horizon=action_horizon,
                    state_horizon=state_horizon,
                    num_action_per_block=self.num_action_per_block if action_register_length else None,
                    num_state_per_block=self.num_state_per_block if action_register_length else None,
                )

        else:
            # ---- Inference path with KV cache ----
            action_state_index = max(0, (current_start_frame - 1) // self.num_frame_per_block)

            roped_query = causal_rope_action_apply(
                q,
                freqs,
                freqs_action,
                freqs_state,
                action_register_length=action_register_length,
                num_action_per_block=self.num_action_per_block,
                num_state_per_block=self.num_state_per_block,
                action_state_index=action_state_index,
            ).type_as(v)
            roped_key = causal_rope_action_apply(
                k,
                freqs,
                freqs_action,
                freqs_state,
                action_register_length=action_register_length,
                num_action_per_block=self.num_action_per_block,
                num_state_per_block=self.num_state_per_block,
                action_state_index=action_state_index,
            ).type_as(v)

            # Split off action/state register tokens
            roped_action_query: torch.Tensor | None = None
            roped_action_key: torch.Tensor | None = None
            action_v: torch.Tensor | None = None

            if action_register_length is not None:
                roped_action_query = roped_query[:, -action_register_length:]
                roped_query = roped_query[:, :-action_register_length]
                roped_action_key = roped_key[:, -action_register_length:]
                roped_key = roped_key[:, :-action_register_length]
                action_v = v[:, -action_register_length:]
                v = v[:, :-action_register_length]

            # Update KV cache
            cached_k = kv_cache[0]  # [B, cached_len, n, d]
            cached_v = kv_cache[1]
            new_k = torch.cat([cached_k, roped_key], dim=1)
            new_v = torch.cat([cached_v, v], dim=1)

            # Truncate if exceeding max attention window
            new_k = new_k[:, -self.max_attention_size :]
            new_v = new_v[:, -self.max_attention_size :]

            if action_register_length is not None:
                assert roped_action_query is not None
                assert roped_action_key is not None
                assert action_v is not None
                attn_out = _sdpa_attn(
                    torch.cat([roped_query, roped_action_query], dim=1),
                    torch.cat([new_k, roped_action_key], dim=1),
                    torch.cat([new_v, action_v], dim=1),
                )
            else:
                attn_out = _sdpa_attn(roped_query, new_k, new_v)

            updated_kv_cache = torch.stack([new_k, new_v], dim=0)

        # Output projection
        out = attn_out.flatten(2)
        out = self.o(out)
        return out, updated_kv_cache


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------


class CausalWanAttentionBlock(nn.Module):
    """Transformer block with causal self-attention, cross-attention, and FFN.

    Uses scale-shift modulation from timestep embeddings (6 modulation params).
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
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        num_action_per_block: int = 32,
        num_state_per_block: int = 1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads

        # Self-attention
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim=dim,
            num_heads=num_heads,
            frame_seqlen=frame_seqlen,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            num_frame_per_block=num_frame_per_block,
            qk_norm=qk_norm,
            eps=eps,
            num_action_per_block=num_action_per_block,
            num_state_per_block=num_state_per_block,
        )

        # Cross-attention
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps)

        # Feed-forward
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        # Scale-shift modulation (6 parameters)
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        freqs_state: torch.Tensor,
        action_register_length: int | None,
        context: torch.Tensor,
        kv_cache: torch.Tensor | None = None,
        crossattn_cache: dict[str, Any] | None = None,
        current_start_frame: int = 0,
        is_tf: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass through one transformer block.

        Args:
            x: ``[B, L, C]``
            e: ``[B, F, 6, C]`` timestep modulation.
            freqs: Spatial RoPE frequencies.
            freqs_action: Action RoPE frequencies.
            freqs_state: State RoPE frequencies.
            action_register_length: Total action + state token count (or None).
            context: Cross-attention context ``[B, L_ctx, C]``.
            kv_cache: Per-layer KV cache or None.
            crossattn_cache: Optional cross-attention cache dict.
            current_start_frame: Current frame index (inference).
            is_tf: Whether teacher-forcing mode.

        Returns:
            ``(x, updated_kv_cache)``
        """
        # Modulation
        e_mod = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # Align modulation sequence length to x
        L = x.shape[1]
        aligned = []
        for part in e_mod:
            L_e = part.shape[1]
            if L_e == L:
                aligned.append(part)
            elif L_e >= L:
                aligned.append(part[:, :L])
            else:
                repeat = (L + L_e - 1) // L_e
                aligned.append(part.repeat_interleave(repeat, dim=1)[:, :L])
        e_mod = tuple(aligned)

        # 1. Self-attention
        y, updated_kv_cache = self.self_attn(
            x=(self.norm1(x) * (1 + e_mod[1].squeeze(2)) + e_mod[0].squeeze(2)),
            freqs=freqs,
            freqs_action=freqs_action,
            freqs_state=freqs_state,
            action_register_length=action_register_length,
            kv_cache=kv_cache,
            is_tf=is_tf,
            current_start_frame=current_start_frame,
        )
        x = x + (y * e_mod[2].squeeze(2))

        # 2. Cross-attention
        x = x + self.cross_attn(self.norm3(x), context)

        # 3. Feed-forward
        y = self.ffn(self.norm2(x) * (1 + e_mod[4].squeeze(2)) + e_mod[3].squeeze(2))
        x = x + (y * e_mod[5].squeeze(2))

        return x, updated_kv_cache


# ---------------------------------------------------------------------------
# Output head
# ---------------------------------------------------------------------------


class CausalHead(nn.Module):
    """Output head: LayerNorm -> Linear -> (unpatchified externally).

    Applies 2-parameter scale-shift modulation from timestep embeddings.
    """

    def __init__(
        self,
        dim: int,
        out_dim: int,
        patch_size: tuple[int, int, int],
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size

        out_channels = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_channels)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[B, L, C]``
            e: ``[B, F, 1, C]`` timestep modulation.

        Returns:
            ``[B, L, out_channels]``
        """
        e_mod = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)

        L = x.shape[1]
        aligned = []
        for part in e_mod:
            L_e = part.shape[1]
            if L_e == L:
                aligned.append(part)
            elif L_e >= L:
                aligned.append(part[:, :L])
            else:
                repeat = (L + L_e - 1) // L_e
                aligned.append(part.repeat_interleave(repeat, dim=1)[:, :L])
        e_mod = tuple(aligned)

        x = self.head(self.norm(x) * (1 + e_mod[1].squeeze(2)) + e_mod[0].squeeze(2))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class CausalWanModel(nn.Module):
    """Causal video diffusion transformer for DreamZero.

    Supports both text-to-video (``t2v``) and image-to-video (``i2v``)
    generation with optional action/state conditioning for embodied AI.

    Architecture (14B variant):
        - 40 transformer blocks with causal self-attn + cross-attn + FFN
        - Hidden dim 5120, FFN dim 13824, 40 heads, head_dim 128
        - Patch embedding: Conv3d(16, 5120, kernel=(1,2,2), stride=(1,2,2))
        - Text embedding: Linear(4096, 5120) -> GELU -> Linear(5120, 5120)
        - Time embedding: Linear(256, 5120) -> SiLU -> Linear(5120, 5120*6)
        - Image embedding (i2v): MLPProj(1280, 5120)
        - Action encoder: MultiEmbodimentActionEncoder
        - State encoder / Action decoder: CategorySpecificMLP
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
        concat_first_frame_latent: bool = True,
    ) -> None:
        super().__init__()

        assert model_type in ("t2v", "i2v", "ti2v")
        self.model_type = model_type

        # Store config as simple namespace (matches WanTransformer3DModel style)
        self.config = type(
            "Config",
            (),
            {
                "model_type": model_type,
                "patch_size": patch_size,
                "frame_seqlen": frame_seqlen,
                "text_len": text_len,
                "in_dim": in_dim,
                "dim": dim,
                "ffn_dim": ffn_dim,
                "freq_dim": freq_dim,
                "text_dim": text_dim,
                "out_dim": out_dim,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "max_chunk_size": max_chunk_size,
                "sink_size": sink_size,
                "qk_norm": qk_norm,
                "cross_attn_norm": cross_attn_norm,
                "eps": eps,
                "num_frame_per_block": num_frame_per_block,
                "action_dim": action_dim,
                "num_registers": num_registers,
                "max_state_dim": max_state_dim,
                "max_num_embodiments": max_num_embodiments,
                "hidden_size": hidden_size,
                "num_action_per_block": num_action_per_block,
                "num_state_per_block": num_state_per_block,
                "concat_first_frame_latent": concat_first_frame_latent,
            },
        )()

        self.patch_size = patch_size
        self.frame_seqlen = frame_seqlen
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = max_chunk_size * num_frame_per_block + 1 if max_chunk_size != -1 else -1
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.num_frame_per_block = num_frame_per_block
        self.diffusion_model_pretrained_path = diffusion_model_pretrained_path
        self.action_dim = action_dim
        self.num_registers = num_registers
        self.max_state_dim = max_state_dim
        self.max_num_embodiments = max_num_embodiments
        self.hidden_size = hidden_size
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block
        self.concat_first_frame_latent = concat_first_frame_latent

        # Override to 1 embodiment internally (matches source)
        _num_embodiments = 1

        # ---- Action / State encoders & decoder ----
        self.state_encoder = CategorySpecificMLP(
            num_categories=_num_embodiments,
            input_dim=max_state_dim,
            hidden_dim=hidden_size,
            output_dim=dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=action_dim,
            hidden_size=dim,
            num_embodiments=_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=_num_embodiments,
            input_dim=dim,
            hidden_dim=hidden_size,
            output_dim=action_dim,
        )

        # ---- Embeddings ----
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )

        # ---- Transformer blocks ----
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                CausalWanAttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    frame_seqlen,
                    self.local_attn_size,
                    sink_size,
                    num_frame_per_block,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    num_action_per_block,
                    num_state_per_block,
                )
                for _ in range(num_layers)
            ]
        )

        # ---- Output head ----
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # ---- RoPE buffers (non-registered to avoid dtype changes in .to()) ----
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs_action = rope_params(1024 * 10, d)
        self.freqs_state = rope_params(1024, d)
        self.freqs = [
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ]

        # Image embedding for i2v / ti2v
        if model_type in ("i2v", "ti2v"):
            self.img_emb = MLPProj(1280, dim)

        self.init_weights()

        self.gradient_checkpointing = True
        self.independent_first_frame = num_frame_per_block != 1

    # ------------------------------------------------------------------
    # Weight initialization
    # ------------------------------------------------------------------

    def init_weights(self) -> None:
        """Xavier initialization for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        nn.init.zeros_(self.head.head.weight)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_freqs(
        self,
        grid_size: torch.Tensor,
        start_frame: int,
    ) -> torch.Tensor:
        """Create 3D RoPE frequency tensor for the given grid size."""
        device = self.patch_embedding.weight.device
        if any(freq.device != device for freq in self.freqs):
            self.freqs = [freq.to(device) for freq in self.freqs]
        if self.freqs_action.device != device:
            self.freqs_action = self.freqs_action.to(device)
        if self.freqs_state.device != device:
            self.freqs_state = self.freqs_state.to(device)

        f, h, w = grid_size.tolist()
        freqs = torch.cat(
            [
                self.freqs[0][start_frame : start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1)
        return freqs

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor) -> torch.Tensor:
        """Reconstruct video tensors from patch embeddings.

        Args:
            x: ``[B, L, C_out * prod(patch_size)]``
            grid_size: ``[3]`` -- (F_patches, H_patches, W_patches)

        Returns:
            ``[B, C_out, F, H, W]``
        """
        B = x.shape[0]
        c = self.out_dim
        gs = grid_size.tolist()
        assert x.shape[1] == math.prod(gs)
        x = x.view(B, *gs, *self.patch_size, c)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        x = x.reshape(B, c, *[i * j for i, j in zip(gs, self.patch_size)])
        return x

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    # ------------------------------------------------------------------
    # Forward through blocks (shared by train and inference)
    # ------------------------------------------------------------------

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
        kv_cache: list[torch.Tensor | None],
        current_start_frame: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor | None]]:
        """Forward through all transformer blocks with action/state handling."""
        x = x.flatten(start_dim=2).transpose(1, 2)
        B = x.shape[0]
        F_t = timestep.shape[1]

        if action is not None:
            embodiment_id_local = torch.tensor([0], device=x.device).repeat(B)
            action_features = self.action_encoder(action, timestep_action, embodiment_id_local)
            state_features = self.state_encoder(state, embodiment_id_local)
            action_register = torch.cat([action_features, state_features], dim=1)
            action_length = action_features.shape[1]
            action_register_length = action_register.shape[1]
            x = torch.cat([x, action_register], dim=1)
        else:
            action_features = None
            state_features = None
            action_length = 0
            action_register_length = None

        # Expand timestep to match seq_len
        if F_t <= seq_len:
            repeat_factor = (seq_len + F_t - 1) // F_t
            timestep = timestep.repeat_interleave(repeat_factor, dim=1)[:, :seq_len]
        else:
            indices = torch.linspace(0, F_t - 1, seq_len, device=timestep.device, dtype=torch.long)
            timestep = timestep[:, indices]

        if action is not None:
            assert timestep_action is not None
            assert state_features is not None
            stride = timestep_action.shape[1] // state_features.shape[1]
            timestep_state = timestep_action[:, ::stride]
            timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)

        # Time embeddings
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x))
        e = e.unflatten(dim=0, sizes=(B, -1))
        e0 = self.time_projection(e)
        e0 = e0.unflatten(dim=2, sizes=(6, self.dim))

        # Text / image context
        context = self.text_embedding(context)
        if clip_feature is not None:
            clip_embedding = self.img_emb(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)

        # Run through blocks
        updated_kv_caches: list[torch.Tensor | None] = []
        for block_index, block in enumerate(self.blocks):
            x, updated_kv = block(
                x=x,
                e=e0,
                freqs=freqs,
                freqs_action=self.freqs_action,
                freqs_state=self.freqs_state,
                context=context,
                action_register_length=action_register_length,
                kv_cache=kv_cache[block_index],
                current_start_frame=current_start_frame,
            )
            updated_kv_caches.append(updated_kv)

        # Decode actions
        if action is not None:
            action_noise_pred = x[:, seq_len : seq_len + action_length]
            action_noise_pred = self.action_decoder(action_noise_pred, embodiment_id_local)
        else:
            action_noise_pred = None

        # Extract video tokens and apply output head
        x_video = x[:, :seq_len]
        e_video = e[:, :seq_len]
        x_video = self.head(x_video, e_video.unsqueeze(2))

        return x_video, action_noise_pred, updated_kv_caches

    # ------------------------------------------------------------------
    # Inference forward (with KV cache)
    # ------------------------------------------------------------------

    def _forward_inference(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        seq_len: int,
        kv_cache: list[torch.Tensor],
        crossattn_cache: list[dict[str, Any]] | None,
        current_start_frame: int,
        y: torch.Tensor | None = None,
        clip_feature: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        timestep_action: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        embodiment_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor | None]]:
        """Inference forward with KV caching (streaming / autoregressive).

        Processes latent frames incrementally, appending to the per-layer
        KV cache on each call.

        Args:
            x: Noisy latent ``[B, C_in, F, H, W]``.
            timestep: ``[B, F]`` diffusion timesteps.
            context: Text embeddings ``[B, text_len, text_dim]``.
            seq_len: Expected flattened sequence length after patching.
            kv_cache: List of per-layer KV caches (length = num_layers).
            crossattn_cache: Optional cross-attention caches.
            current_start_frame: Frame index offset for RoPE.
            y: Conditioning video (first frame latent) for i2v.
            clip_feature: CLIP image features for i2v.
            action: ``[B, H, action_dim]`` or None.
            timestep_action: ``[B, H]`` or None.
            state: ``[B, 1, state_dim]`` or None.
            embodiment_id: ``[B]`` or None.

        Returns:
            ``(video_noise_pred, action_noise_pred, updated_kv_caches)``
        """
        if self.model_type == "i2v":
            assert clip_feature is not None and y is not None
        assert context.shape[1] == self.text_len

        # Concat first-frame latent if applicable
        if y is not None and self.concat_first_frame_latent:
            x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)

        # Patch embedding
        x = self.patch_embedding(x)
        grid_size = torch.tensor(x.shape[2:], dtype=torch.long)

        freqs = self._create_freqs(grid_size=grid_size, start_frame=current_start_frame)

        x_video, action_noise_pred, updated_kv_caches = self._forward_blocks(
            x=x,
            seq_len=seq_len,
            freqs=freqs,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            embodiment_id=embodiment_id,
            action=action,
            timestep_action=timestep_action,
            state=state,
            kv_cache=kv_cache,
            current_start_frame=current_start_frame,
        )

        x_video = x_video.clone()
        if action_noise_pred is not None:
            action_noise_pred = action_noise_pred.clone()

        video_noise_pred = self.unpatchify(x_video, grid_size)
        return video_noise_pred, action_noise_pred, updated_kv_caches

    # ------------------------------------------------------------------
    # Training forward (full sequence, teacher forcing)
    # ------------------------------------------------------------------

    def _forward_train(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        timestep_action: torch.Tensor | None,
        context: torch.Tensor,
        seq_len: int,
        clean_x: torch.Tensor | None = None,
        aug_t: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        clip_feature: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        embodiment_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Training forward with optional teacher forcing.

        When ``clean_x`` is provided, the model operates in teacher-forcing
        mode: clean (context) frames are concatenated before the noisy
        frames, and the causal attention mask ensures noisy frames can only
        attend to preceding clean frames.

        Args:
            x: Noisy latent ``[B, C_in, F, H, W]``.
            timestep: ``[B, F]`` diffusion timesteps per frame.
            timestep_action: ``[B, T_action]`` or None.
            context: Text embeddings ``[B, text_len, text_dim]``.
            seq_len: Expected sequence length after patching.
            clean_x: Clean latent (same shape as x) for teacher forcing.
            aug_t: Augmentation timesteps for the clean context.
            y: First-frame latent for i2v.
            clip_feature: CLIP features for i2v.
            action: ``[B, T_action, action_dim]`` or None.
            state: ``[B, 1, state_dim]`` or None.
            embodiment_id: ``[B]`` or None.

        Returns:
            ``(video_noise_pred, action_noise_pred)``
        """
        if self.model_type == "i2v":
            assert clip_feature is not None and y is not None

        # Concat first-frame latent if applicable
        if y is not None and self.concat_first_frame_latent:
            x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)

        # Patch embedding
        x = self.patch_embedding(x)
        grid_size = torch.tensor(x.shape[2:], dtype=torch.long)
        freqs = self._create_freqs(grid_size=grid_size, start_frame=0)

        x = x.flatten(start_dim=2).transpose(1, 2)
        assert x.shape[1] == seq_len

        B = x.shape[0]
        F_t = timestep.shape[1]

        # Action / state encoding
        if action is not None:
            embodiment_id_local = torch.tensor([0], device=embodiment_id.device).repeat(B)
            action_features = self.action_encoder(action, timestep_action, embodiment_id_local)
            action_length = action_features.shape[1]
            state_features = self.state_encoder(state, embodiment_id_local)
            action_register = torch.cat([action_features, state_features], dim=1)
            action_register_length = action_register.shape[1]
            x = torch.cat([x, action_register], dim=1)
        else:
            action_features = None
            action_length = None
            state_features = None
            action_register_length = None

        # Time embeddings
        timestep_expanded = timestep.unsqueeze(-1).expand(B, F_t, seq_len // F_t).reshape(B, -1)
        timestep_original = timestep_expanded.clone()

        if action is not None:
            assert timestep_action is not None
            assert state_features is not None
            stride = timestep_action.shape[1] // state_features.shape[1]
            timestep_state = timestep_action[:, ::stride]
            timestep_expanded = torch.cat([timestep_expanded, timestep_action, timestep_state], dim=1)

        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep_expanded.flatten()).type_as(x))
        e = e.unflatten(dim=0, sizes=(B, -1))
        e0 = self.time_projection(e)
        e0 = e0.unflatten(dim=2, sizes=(6, self.dim))

        # Context
        assert context.shape[1] == self.text_len
        context = self.text_embedding(context)
        if clip_feature is not None:
            clip_embedding = self.img_emb(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)

        # Teacher forcing: prepend clean tokens
        if clean_x is not None:
            if y is not None and self.concat_first_frame_latent:
                clean_x = torch.cat([clean_x, y.to(dtype=clean_x.dtype)], dim=1)
            clean_x = self.patch_embedding(clean_x)
            clean_x = clean_x.flatten(start_dim=2).transpose(1, 2)
            assert clean_x.shape[1] == seq_len

            x = torch.cat([clean_x, x], dim=1)

            if aug_t is None:
                aug_t = torch.zeros_like(timestep_original)

            e_clean = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e_clean = e_clean.unflatten(dim=0, sizes=timestep_original.shape)
            e0_clean = self.time_projection(e_clean)
            e0_clean = e0_clean.unflatten(dim=2, sizes=(6, self.dim))
            e0 = torch.cat([e0_clean, e0], dim=1)

        kwargs = dict(
            e=e0,
            freqs=freqs,
            freqs_action=self.freqs_action,
            freqs_state=self.freqs_state,
            action_register_length=action_register_length,
            context=context,
            is_tf=clean_x is not None,
        )

        def create_custom_forward(module: nn.Module):
            def custom_forward(*inputs, **kw):
                outputs, updated_kv_cache = module(*inputs, **kw)
                assert updated_kv_cache is None
                return outputs

            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **kwargs,
                    use_reentrant=False,
                )
            else:
                x, _ = block(x, **kwargs)

        # Strip clean prefix if teacher forcing
        if clean_x is not None:
            x = x[:, clean_x.shape[1] :]

        # Decode actions
        if action is not None:
            action_noise_pred = x[:, seq_len : seq_len + action_length]
            action_noise_pred = self.action_decoder(action_noise_pred, embodiment_id_local)
        else:
            action_noise_pred = None

        # Output head + unpatchify
        x_video = x[:, :seq_len]
        e_video = e[:, :seq_len]
        x_video = self.head(x_video, e_video.unsqueeze(2))
        video_noise_pred = self.unpatchify(x_video, grid_size)

        return video_noise_pred, action_noise_pred

    # ------------------------------------------------------------------
    # Public forward (dispatch)
    # ------------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any):
        """Dispatch to ``_forward_inference`` or ``_forward_train``."""
        if kwargs.get("kv_cache", None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)
