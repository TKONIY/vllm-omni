from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

UADPhase = Literal["ar_prefill", "ar_decode", "dit_step"]
UADTokenModality = Literal["text", "image"]


@dataclass(frozen=True)
class UADToken:
    """A logical token in the unified multimodal engine ledger."""

    modality: UADTokenModality
    token_id: int


@dataclass(frozen=True)
class UADPhaseUpdate:
    """State-machine update emitted by a runner after executing a work item."""

    phase: UADPhase | None = None
    dit_step_index: int | None = None
    total_dit_steps: int | None = None
    image_ratio_token_id: int | None = None
    image_ratio_index: int | None = None
    image_width: int | None = None
    image_height: int | None = None
    image_token_width: int | None = None
    image_token_height: int | None = None
    image_base_size: int | None = None
    image_context_token_count: int | None = None
    latent_shape: tuple[int, ...] | None = None
    seed: int | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    pending_image_context_commit: bool | None = None


@dataclass
class UADRequestState:
    """Minimal request ledger for the UAD toy engine.

    `engine_tokens` is the full logical context. `materialized_tokens` is the
    subset visible to the caller. `num_computed_tokens` tracks the committed
    prefix of `engine_tokens` that has already been run by the model.
    """

    request_id: str
    engine_tokens: list[UADToken]
    materialized_tokens: list[UADToken] = field(default_factory=list)
    phase: UADPhase = "ar_prefill"
    num_computed_tokens: int = 0
    finished: bool = False
    dit_step_index: int = 0
    total_dit_steps: int = 0
    image_ratio_token_id: int | None = None
    image_ratio_index: int | None = None
    image_width: int | None = None
    image_height: int | None = None
    image_token_width: int | None = None
    image_token_height: int | None = None
    image_base_size: int | None = None
    image_context_token_count: int = 0
    latent_shape: tuple[int, ...] | None = None
    seed: int | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    pending_image_context_commit: bool = False

    @classmethod
    def from_prompt_token_ids(
        cls,
        request_id: str,
        prompt_token_ids: list[int],
    ) -> UADRequestState:
        return cls(
            request_id=request_id,
            engine_tokens=[UADToken(modality="text", token_id=token_id) for token_id in prompt_token_ids],
        )

    @property
    def num_tokens(self) -> int:
        return len(self.engine_tokens)

    def pending_token_ids(self) -> list[int]:
        return [token.token_id for token in self.engine_tokens[self.num_computed_tokens :]]

    def append_engine_tokens(self, tokens: list[UADToken]) -> None:
        self.engine_tokens.extend(tokens)

    def append_materialized_tokens(self, tokens: list[UADToken]) -> None:
        self.materialized_tokens.extend(tokens)

    def advance_computed_tokens(self, num_tokens: int) -> None:
        next_value = self.num_computed_tokens + num_tokens
        if next_value > len(self.engine_tokens):
            raise ValueError(
                f"num_computed_tokens would exceed engine token ledger: {next_value} > {len(self.engine_tokens)}"
            )
        self.num_computed_tokens = next_value
        if self.num_computed_tokens == len(self.engine_tokens) and not self.finished:
            self.phase = "ar_decode"

    def apply_phase_update(self, update: UADPhaseUpdate) -> None:
        if update.phase is not None:
            self.phase = update.phase
        if update.dit_step_index is not None:
            self.dit_step_index = update.dit_step_index
        if update.total_dit_steps is not None:
            self.total_dit_steps = update.total_dit_steps
        if update.image_ratio_token_id is not None:
            self.image_ratio_token_id = update.image_ratio_token_id
        if update.image_ratio_index is not None:
            self.image_ratio_index = update.image_ratio_index
        if update.image_width is not None:
            self.image_width = update.image_width
        if update.image_height is not None:
            self.image_height = update.image_height
        if update.image_token_width is not None:
            self.image_token_width = update.image_token_width
        if update.image_token_height is not None:
            self.image_token_height = update.image_token_height
        if update.image_base_size is not None:
            self.image_base_size = update.image_base_size
        if update.image_context_token_count is not None:
            self.image_context_token_count = update.image_context_token_count
        if update.latent_shape is not None:
            self.latent_shape = update.latent_shape
        if update.seed is not None:
            self.seed = update.seed
        if update.num_inference_steps is not None:
            self.num_inference_steps = update.num_inference_steps
        if update.guidance_scale is not None:
            self.guidance_scale = update.guidance_scale
        if update.pending_image_context_commit is not None:
            self.pending_image_context_commit = update.pending_image_context_commit
