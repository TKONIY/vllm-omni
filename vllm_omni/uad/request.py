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


@dataclass
class UADRequestState:
    """Minimal request ledger for Step 0.

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
