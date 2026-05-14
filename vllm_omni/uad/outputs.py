from __future__ import annotations

from dataclasses import dataclass, field

from vllm_omni.uad.request import UADPhase, UADPhaseUpdate, UADToken


@dataclass
class UADRunnerOutput:
    """Raw per-item result from runner execution.

    This object is intentionally model-semantic-free. For AR phases it carries
    the sampled token; for DiT phases it records only that the scheduled work
    item completed. The engine/model state machine converts this into request
    state updates.
    """

    request_id: str
    phase: UADPhase
    num_scheduled_tokens: int
    sampled_token: UADToken | None = None


@dataclass
class UADRunnerStepOutput:
    outputs: list[UADRunnerOutput] = field(default_factory=list)

    @property
    def request_ids(self) -> list[str]:
        return [output.request_id for output in self.outputs]


@dataclass
class UADModelOutput:
    request_id: str
    new_engine_tokens: list[UADToken] = field(default_factory=list)
    new_materialized_tokens: list[UADToken] = field(default_factory=list)
    num_computed_tokens_delta: int = 0
    phase_update: UADPhaseUpdate | None = None
    finished: bool = False


@dataclass
class UADStepOutput:
    outputs: list[UADModelOutput] = field(default_factory=list)

    @property
    def request_ids(self) -> list[str]:
        return [output.request_id for output in self.outputs]
