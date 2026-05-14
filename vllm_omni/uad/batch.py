from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from vllm_omni.uad.request import UADPhase

UADInputKind = Literal["token_ids", "latent_timestep"]
UADAttentionKind = Literal["causal_paged", "dit_chunk_bidirectional"]


@dataclass(frozen=True)
class UADBatchItem:
    """One scheduled item after runner-side batch packing."""

    request_id: str
    phase: UADPhase
    output_index: int
    num_tokens: int
    token_start: int
    token_end: int
    num_computed_tokens: int
    persist: bool
    input_kind: UADInputKind
    attention_kind: UADAttentionKind
    dit_step_index: int | None = None
    total_dit_steps: int | None = None


@dataclass(frozen=True)
class UADBatchInputs:
    """Batch-first input contract consumed by UAD model shells.

    `input_token_ids` is the flattened token-slot buffer for the whole runner
    tick. AR slots contain real token IDs. DiT slots are fake latent slots in
    the current toy path and are marked by `input_kind` rather than token ID.
    """

    items: tuple[UADBatchItem, ...]
    input_token_ids: torch.Tensor
    token_item_indices: torch.Tensor
    token_positions: torch.Tensor

    @property
    def num_ar_tokens(self) -> int:
        return sum(item.num_tokens for item in self.items if item.input_kind == "token_ids")

    @property
    def num_dit_tokens(self) -> int:
        return sum(item.num_tokens for item in self.items if item.input_kind == "latent_timestep")

    @property
    def num_ffn_tokens(self) -> int:
        return int(self.token_item_indices.numel())

    @property
    def causal_attention_item_indices(self) -> tuple[int, ...]:
        return tuple(range(len(self.items)))

    @property
    def bidirectional_attention_item_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, item in enumerate(self.items)
            if item.attention_kind == "dit_chunk_bidirectional"
        )

    @property
    def num_causal_attention_tokens(self) -> int:
        return self.num_ffn_tokens

    @property
    def num_bidirectional_attention_tokens(self) -> int:
        return sum(
            self.items[index].num_tokens
            for index in self.bidirectional_attention_item_indices
        )


@dataclass(frozen=True)
class UADBatchItemOutput:
    """Raw model output for one packed item."""

    request_id: str
    phase: UADPhase
    output_index: int
    num_scheduled_tokens: int
    next_token_id: int | None = None


@dataclass(frozen=True)
class UADBatchOutputs:
    """Raw batch output plus observability counters for tests and tracing."""

    item_outputs: tuple[UADBatchItemOutput, ...]
    num_ffn_tokens: int
    num_ar_tokens: int
    num_dit_tokens: int
    num_causal_attention_tokens: int
    num_bidirectional_attention_tokens: int
