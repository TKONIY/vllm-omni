from __future__ import annotations

from dataclasses import dataclass, field

from vllm_omni.uad.request import UADPhase, UADPhaseUpdate, UADToken


@dataclass
class UADModelRunnerItemOutput:
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
class UADModelRunnerOutput:
    """Batch output from `UADRunner.execute_model()`.

    This mirrors vLLM's `ModelRunnerOutput` layer: it is the raw runner result
    for one scheduler batch, before scheduler/state-machine request updates.
    """

    outputs: list[UADModelRunnerItemOutput] = field(default_factory=list)

    @property
    def request_ids(self) -> list[str]:
        return [output.request_id for output in self.outputs]


@dataclass
class UADStateUpdate:
    """Request-state delta produced by a model-specific state machine."""

    request_id: str
    new_engine_tokens: list[UADToken] = field(default_factory=list)
    new_materialized_tokens: list[UADToken] = field(default_factory=list)
    new_ar_sampler_token_ids: list[int] = field(default_factory=list)
    phase_update: UADPhaseUpdate | None = None
    finished: bool = False


@dataclass
class UADEngineCoreOutputs:
    """UAD engine-core batch output after scheduler state update.

    This mirrors vLLM's `EngineCoreOutputs` layer. The current toy UAD engine
    carries `UADStateUpdate` objects directly; serving/output-processor mapping
    to user-facing request outputs is still a later step.
    """

    outputs: list[UADStateUpdate] = field(default_factory=list)

    @property
    def request_ids(self) -> list[str]:
        return [output.request_id for output in self.outputs]
