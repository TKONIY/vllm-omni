# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class LingbotWorldFastRuntimeConfig:
    session_id: str
    rendered_prompt: str
    text_layers: dict[str, str]
    width: int
    height: int
    fps: int
    chunk_size: int
    seed: int | None
    shift: float
    max_attention_size: int | None

    @property
    def signature(self) -> tuple[Any, ...]:
        return (
            self.rendered_prompt,
            tuple(self.text_layers.items()),
            self.width,
            self.height,
            self.fps,
            self.chunk_size,
            self.seed,
            self.shift,
            self.max_attention_size,
        )


@dataclass
class LingbotWorldFastRuntimeState:
    config: LingbotWorldFastRuntimeConfig
    prompt_context: torch.Tensor
    condition_latents: torch.Tensor
    zero_condition_video: torch.Tensor
    generator: torch.Generator
    scheduler: Any
    encoder_feat_cache: list[Any]
    decoder_feat_cache: list[Any]
    kv_cache: list[dict[str, torch.Tensor]]
    crossattn_cache: list[dict[str, Any]]
    latent_height: int
    latent_width: int
    frame_seqlen: int
    max_sequence_length: int
    current_start: int = 0
    generated_latent_frames: int = 0
    generated_chunks: int = 0
    kv_capacity_tokens: int = 0
