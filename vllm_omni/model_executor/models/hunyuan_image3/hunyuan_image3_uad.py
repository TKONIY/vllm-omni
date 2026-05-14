from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

UADPhase = Literal["ar_prefill", "ar_decode", "dit_step"]
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
    """Batch-first input contract consumed by `HunyuanImage3UADModel`.

    `input_token_ids` is the flattened token-slot buffer for the whole runner
    tick. AR slots contain real token IDs. DiT slots are fake latent slots in
    the Step 4 toy path and are marked by `input_kind` rather than token ID.
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
    """Raw batch output plus observability counters for Step 4 tests."""

    item_outputs: tuple[UADBatchItemOutput, ...]
    num_ffn_tokens: int
    num_ar_tokens: int
    num_dit_tokens: int
    num_causal_attention_tokens: int
    num_bidirectional_attention_tokens: int


class HunyuanImage3UADModel(nn.Module):
    """Toy UAD-native HunyuanImage3 batch model shell.

    This is shaped like a concrete vLLM model-executor `nn.Module`, but it does
    not load HunyuanImage3 weights yet. The important Step 4 behavior is that
    AR token slots and DiT latent slots enter one batch contract before the
    fake model output is scattered back to requests.
    """

    def __init__(self, vocab_size: int = 32000) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.forward_calls = 0
        self.last_batch_inputs: UADBatchInputs | None = None
        self.last_batch_outputs: UADBatchOutputs | None = None

    def forward(
        self,
        batch_inputs: UADBatchInputs,
    ) -> UADBatchOutputs:
        self.forward_calls += 1
        self.last_batch_inputs = batch_inputs

        item_outputs: list[UADBatchItemOutput] = []
        for item in batch_inputs.items:
            next_token_id: int | None = None
            if item.input_kind == "token_ids":
                if item.num_tokens <= 0:
                    raise ValueError("AR UAD item requires at least one token")
                item_token_ids = batch_inputs.input_token_ids[item.token_start : item.token_end]
                last_token_id = int(item_token_ids.reshape(-1)[-1].item())
                next_token_id = (last_token_id + 1) % self.vocab_size

            item_outputs.append(
                UADBatchItemOutput(
                    request_id=item.request_id,
                    phase=item.phase,
                    output_index=item.output_index,
                    num_scheduled_tokens=item.num_tokens,
                    next_token_id=next_token_id,
                )
            )

        batch_outputs = UADBatchOutputs(
            item_outputs=tuple(item_outputs),
            num_ffn_tokens=batch_inputs.num_ffn_tokens,
            num_ar_tokens=batch_inputs.num_ar_tokens,
            num_dit_tokens=batch_inputs.num_dit_tokens,
            num_causal_attention_tokens=batch_inputs.num_causal_attention_tokens,
            num_bidirectional_attention_tokens=batch_inputs.num_bidirectional_attention_tokens,
        )
        self.last_batch_outputs = batch_outputs
        return batch_outputs


HunyuanImage3UADForConditionalGeneration = HunyuanImage3UADModel
