from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from vllm_omni.uad.request import UADPhase

UADInputKind = Literal["token_ids", "latent_timestep"]


@dataclass(frozen=True)
class UADBatchItem:
    """One scheduled item after runner-side batch packing.

    Attributes:
        request_id: Request this packed item belongs to.
        phase: Request phase selected by scheduler for this item.
        output_index: Original scheduler item index used to scatter outputs
            back to scheduler order.
        num_tokens: Number of token slots represented by this item.
        token_start: Inclusive start offset in flattened token-slot tensors.
        token_end: Exclusive end offset in flattened token-slot tensors.
        num_computed_tokens: Reusable context prefix length when this item was
            scheduled. Runner uses it to derive positions without reading the
            request object.
        persist: Whether successful execution should commit reusable context
            and advance request `num_computed_tokens`.
        input_kind: Input recipe for this item. AR uses token IDs; DiT uses
            latent/timestep slots.
        dit_step_index: Current DiT denoise step for DiT items.
        total_dit_steps: Total denoise steps for DiT items.
        ar_sampler_token_ids: AR generated-token history used to build
            vLLM `SamplingMetadata.output_token_ids`.
        sample_token_offset: Token offset inside this scheduled item whose
            hidden state should be projected to logits, matching vLLM's
            `logits_indices` role.
    """

    request_id: str
    phase: UADPhase
    output_index: int
    num_tokens: int
    token_start: int
    token_end: int
    num_computed_tokens: int
    persist: bool
    input_kind: UADInputKind
    dit_step_index: int | None = None
    total_dit_steps: int | None = None
    ar_sampler_token_ids: tuple[int, ...] = ()
    sample_token_offset: int | None = None


@dataclass(frozen=True)
class UADBatchInputs:
    """Batch-first input contract consumed by UAD model shells.

    `input_token_ids` is the flattened token-slot buffer for the whole runner
    tick. AR slots contain real token IDs. DiT slots are fake latent slots in
    the current toy path and are marked by `input_kind` rather than token ID.

    Attributes:
        items: Per-scheduler-item metadata after runner packing.
        input_token_ids: Flattened token-slot IDs. AR slots are real token IDs;
            toy DiT slots use placeholder IDs until real latent inputs land.
        token_item_indices: For each flattened token slot, the index of the
            owning `UADBatchItem`.
        token_positions: Per-slot logical positions derived from
            `num_computed_tokens` and item length.
    """

    items: tuple[UADBatchItem, ...]
    input_token_ids: torch.Tensor
    token_item_indices: torch.Tensor
    token_positions: torch.Tensor

    @property
    def num_ar_tokens(self) -> int:
        """Number of flattened token slots using the AR token-id recipe."""
        return sum(item.num_tokens for item in self.items if item.input_kind == "token_ids")

    @property
    def num_dit_tokens(self) -> int:
        """Number of flattened token slots using the DiT latent/timestep recipe."""
        return sum(item.num_tokens for item in self.items if item.input_kind == "latent_timestep")

    @property
    def num_ffn_tokens(self) -> int:
        """Total flattened tokens that can enter shared projection/FFN/MoE work."""
        return int(self.token_item_indices.numel())

    @property
    def ar_item_indices(self) -> tuple[int, ...]:
        """Item indices for AR phases.

        Attention execution derives the AR recipe from these phases: causal
        paged attention over the persisted prefix plus the scheduled suffix.
        """
        return tuple(
            index
            for index, item in enumerate(self.items)
            if item.phase in ("ar_prefill", "ar_decode")
        )

    @property
    def dit_item_indices(self) -> tuple[int, ...]:
        """Item indices for DiT phases.

        Attention execution derives the DiT recipe from this phase: read the
        persisted prefix with non-causal paged attention, run chunk-local
        bidirectional attention, then merge the two paths with LSE.
        """
        return tuple(
            index
            for index, item in enumerate(self.items)
            if item.phase == "dit_step"
        )


@dataclass(frozen=True)
class UADBatchItemOutput:
    """Raw model output for one packed item.

    Attributes:
        request_id: Request this output belongs to.
        phase: Phase executed for the item.
        output_index: Scheduler item index used by runner scatter.
        num_scheduled_tokens: Number of token slots completed by the item.
        next_token_id: Toy sampled token for AR items when no backend is used.
        hidden_states: Backend forward output for runner-side logits/sampling.
    """

    request_id: str
    phase: UADPhase
    output_index: int
    num_scheduled_tokens: int
    next_token_id: int | None = None
    hidden_states: torch.Tensor | None = None


@dataclass(frozen=True)
class UADBatchOutputs:
    """Raw batch output plus observability counters for tests and tracing.

    Attributes:
        item_outputs: Raw per-item outputs in model-produced order.
        num_ffn_tokens: Total token slots exposed to shared FFN/MoE work.
        num_ar_tokens: Token slots using AR token-id inputs.
        num_dit_tokens: Token slots using DiT latent/timestep inputs.
    """

    item_outputs: tuple[UADBatchItemOutput, ...]
    num_ffn_tokens: int
    num_ar_tokens: int
    num_dit_tokens: int
