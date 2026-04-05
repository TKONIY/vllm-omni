# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DreamZero pipeline persistent state.

Consolidates all cross-forward() state that was originally scattered across:
- ARDroidRoboarenaPolicy._frame_buffers   (socket_test_optimized_AR.py)
- WANPolicyHead.kv_cache1/kv_cache_neg    (wan_flow_matching_action_tf.py)
- WANPolicyHead.clip_feas/ys              (wan_flow_matching_action_tf.py)
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Unified image keys (after transform, dataset-agnostic)
IMAGE_KEYS = [
    "images/exterior_0",
    "images/exterior_1",
    "images/wrist",
]

# Number of frames per chunk for subsequent calls (first call uses 1)
# Corresponds to: ARDroidRoboarenaPolicy.FRAMES_PER_CHUNK = 4
FRAMES_PER_CHUNK = 4


class DreamZeroState:
    """Pipeline persistent state across forward() calls.

    Lifecycle:
        - Created once in DreamZeroPipeline.__init__()
        - Mutated every forward() call (frame append, KV cache grow)
        - reset() on new session / language change / local_attn_size exceeded
    """

    def __init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------
    # Frame accumulation
    # Corresponds to: socket_test_optimized_AR.py L110-144
    #   ARDroidRoboarenaPolicy._convert_observation (frame buffer part only)
    # ------------------------------------------------------------------

    def accumulate_frames(self, openpi_obs: dict) -> dict[str, np.ndarray]:
        """Accumulate frames and return stacked video arrays.

        Each call appends new frames to buffers. Returns dict mapping
        each image key to a (T, H, W, 3) ndarray.
        T = 1 for first call, FRAMES_PER_CHUNK thereafter.
        If buffer has fewer than needed, pads by repeating first frame.

        Corresponds to: socket_test_optimized_AR.py L110-144
        """
        # L110-120: append new frames to buffers
        for key in IMAGE_KEYS:
            if key in openpi_obs:
                data = openpi_obs[key]
                if isinstance(data, np.ndarray):
                    if data.ndim == 4:
                        # Multiple frames (T, H, W, 3) — L116-117
                        self.frame_buffers[key].extend(list(data))
                    else:
                        # Single frame (H, W, 3) — L119-120
                        self.frame_buffers[key].append(data)

        # L122-128: determine frame count
        num_frames = 1 if self.call_count == 0 else FRAMES_PER_CHUNK

        # L130-144: build stacked video from buffer
        result: dict[str, np.ndarray] = {}
        for key, buffer in self.frame_buffers.items():
            if len(buffer) > 0:
                if len(buffer) >= num_frames:
                    # L133-135: take last N frames
                    frames = buffer[-num_frames:]
                else:
                    # L137-141: pad by repeating first frame
                    frames = buffer.copy()
                    while len(frames) < num_frames:
                        frames.insert(0, buffer[0])
                # L143: stack to (T, H, W, C)
                result[key] = np.stack(frames, axis=0)

        self.call_count += 1
        return result

    # ------------------------------------------------------------------
    # Reset / should_reset
    # Corresponds to: wan_flow_matching_action_tf.py L968-981
    #   WANPolicyHead.lazy_joint_video_action (reset condition block)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all state.

        Corresponds to:
        - socket_test_optimized_AR.py L302-330: ARDroidRoboarenaPolicy._reset_state
        - wan_flow_matching_action_tf.py L185-199: WANPolicyHead.__init__ state fields
        """
        # Frame buffers — from ARDroidRoboarenaPolicy.__init__ L63-66
        self.frame_buffers: dict[str, list[np.ndarray]] = {
            key: [] for key in IMAGE_KEYS
        }
        self.call_count: int = 0

        # KV cache — from WANPolicyHead.__init__ L185-188
        self.kv_cache: list[torch.Tensor] | None = None
        self.kv_cache_neg: list[torch.Tensor] | None = None
        self.crossattn_cache: list[torch.Tensor] | None = None
        self.crossattn_cache_neg: list[torch.Tensor] | None = None
        self.current_start_frame: int = 0  # WANPolicyHead L199

        # Encoding cache — from WANPolicyHead.__init__ L197-200
        self.clip_feas: torch.Tensor | None = None
        self.ys: torch.Tensor | None = None
        self.language: torch.Tensor | None = None  # WANPolicyHead L200

    def should_reset(self, text_tokens: torch.Tensor | None, num_video_frames: int, local_attn_size: int) -> bool:
        """Determine if state should be reset before this forward().

        Corresponds to: wan_flow_matching_action_tf.py L968-981
            if self.language is None:                         → first call
                self.current_start_frame = 0
            elif not torch.equal(self.language, data["text"]): → language changed
                self.current_start_frame = 0
            elif videos.shape[2] == 1:                        → single-frame (new episode)
                self.current_start_frame = 0
            elif self.current_start_frame >= self.model.local_attn_size: → overflow
                self.current_start_frame = 0
        """
        # L968-971: first call (language not set yet)
        if self.language is None:
            logger.info("language is None, resetting")
            return True

        # L972-975: language changed
        if text_tokens is not None and not torch.equal(self.language, text_tokens):
            logger.info("language changed, resetting")
            return True

        # L976-978: single-frame input (signals new episode in real-world eval)
        if num_video_frames == 1:
            logger.info("single frame input, resetting")
            return True

        # L979-981: KV cache exceeded local attention window
        if self.current_start_frame >= local_attn_size:
            logger.info("current_start_frame %d >= local_attn_size %d, resetting",
                        self.current_start_frame, local_attn_size)
            return True

        return False

    # ------------------------------------------------------------------
    # KV cache management
    # Corresponds to: wan_flow_matching_action_tf.py L480-512
    #   WANPolicyHead._create_kv_caches / _create_crossattn_caches
    # ------------------------------------------------------------------

    def create_kv_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        num_layers: int,
        num_heads: int,
        head_dim: int,
    ) -> None:
        """Initialize empty KV caches and cross-attention caches.

        Corresponds to: wan_flow_matching_action_tf.py L480-512
            _create_kv_caches:      [2, B, 0, num_heads, head_dim] per layer  (L486-491)
            _create_crossattn_caches: [2, B, 512, num_heads, head_dim] per layer (L504-510)
        """
        # L486-491: KV caches start with seq_len=0, grow during prefill
        self.kv_cache = [
            torch.zeros(2, batch_size, 0, num_heads, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.kv_cache_neg = [
            torch.zeros(2, batch_size, 0, num_heads, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]

        # L504-510: cross-attn caches have fixed seq_len=512 (text length)
        self.crossattn_cache = [
            torch.zeros(2, batch_size, 512, num_heads, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.crossattn_cache_neg = [
            torch.zeros(2, batch_size, 512, num_heads, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]

    def update_kv_cache(
        self,
        layer_index: int,
        updated_kv: torch.Tensor,
        is_negative: bool = False,
    ) -> None:
        """Update a single layer's KV cache after prefill.

        Corresponds to: wan_flow_matching_action_tf.py L856-858
            if kv_cache_metadata["update_kv_cache"]:
                for block_index, updated_kv_cache in enumerate(updated_kv_caches):
                    kv_cache[block_index] = updated_kv_cache.clone()
        """
        cache = self.kv_cache_neg if is_negative else self.kv_cache
        assert cache is not None, "KV caches not initialized, call create_kv_caches first"
        cache[layer_index] = updated_kv.clone()

    def get_kv_caches(self, is_negative: bool = False) -> list[torch.Tensor]:
        """Get KV caches for the specified branch.

        Corresponds to: wan_flow_matching_action_tf.py L776-791
            _get_caches — in vllm-omni, CFGParallelMixin handles rank dispatch,
            so we just return the requested branch directly.
        """
        cache = self.kv_cache_neg if is_negative else self.kv_cache
        assert cache is not None, "KV caches not initialized"
        return cache

    def get_crossattn_caches(self, is_negative: bool = False) -> list[torch.Tensor]:
        """Get cross-attention caches for the specified branch.

        Same pattern as get_kv_caches.
        """
        cache = self.crossattn_cache_neg if is_negative else self.crossattn_cache
        assert cache is not None, "Cross-attn caches not initialized"
        return cache
