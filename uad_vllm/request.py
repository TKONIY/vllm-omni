from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class UADPhase(str, Enum):
    """Request phase tracked by the UAD scheduler."""

    AR_PREFILL = "ar_prefill"
    AR_DECODE = "ar_decode"
    DIT_STEP = "dit_step"
    ARTIFACT_DECODE = "artifact_decode"


UADTokenModality = Literal["text", "image", "video", "audio", "latent", "control"]


@dataclass(frozen=True)
class UADToken:
    """Logical token in the unified multimodal engine ledger."""

    modality: UADTokenModality
    token_id: int | None = None
    payload: Any | None = None


@dataclass
class UADRequestState:
    """UAD-owned state attached to a vLLM request lifecycle."""

    request_id: str
    phase: UADPhase = UADPhase.AR_PREFILL
    engine_tokens: list[UADToken] = field(default_factory=list)
    materialized_tokens: list[UADToken] = field(default_factory=list)
    dit_step_index: int = 0
    dit_num_steps: int = 0
    dit_query_tokens: int = 0
    runtime_state: dict[str, Any] = field(default_factory=dict)
