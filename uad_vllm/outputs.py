from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from typing import Any

from vllm.v1.outputs import ModelRunnerOutput

from uad_vllm.request import UADPhase, UADToken


@dataclass
class UADPhaseUpdate:
    """State update emitted after one scheduled UAD item finishes."""

    new_engine_tokens: list[UADToken] = field(default_factory=list)
    new_materialized_tokens: list[UADToken] = field(default_factory=list)
    num_new_computed_tokens: int = 0
    next_phase: UADPhase | None = None
    runtime_state_delta: dict[str, Any] = field(default_factory=dict)


@dataclass
class UADPhaseOutput:
    """Per-item output before scheduler state application."""

    request_id: str
    phase: UADPhase
    update: UADPhaseUpdate = field(default_factory=UADPhaseUpdate)
    raw_output: Any | None = None


@dataclass
class UADModelRunnerOutput(ModelRunnerOutput):
    """v1 ModelRunnerOutput with UAD phase outputs attached."""

    phase_outputs: list[UADPhaseOutput] = field(default_factory=list)

    @classmethod
    def from_base(
        cls,
        base_output: ModelRunnerOutput,
        phase_outputs: list[UADPhaseOutput] | None = None,
    ) -> UADModelRunnerOutput:
        base_values = {
            field_info.name: getattr(base_output, field_info.name)
            for field_info in dataclass_fields(ModelRunnerOutput)
        }
        return cls(**base_values, phase_outputs=list(phase_outputs or []))

    @classmethod
    def make_empty(cls, phase_outputs: list[UADPhaseOutput] | None = None) -> UADModelRunnerOutput:
        return cls(req_ids=[], req_id_to_index={}, phase_outputs=list(phase_outputs or []))
