from __future__ import annotations

from dataclasses import dataclass, field

from vllm_omni.uad.request import UADToken


@dataclass
class UADModelOutput:
    request_id: str
    new_engine_tokens: list[UADToken] = field(default_factory=list)
    new_materialized_tokens: list[UADToken] = field(default_factory=list)
    num_computed_tokens_delta: int = 0
    finished: bool = False


@dataclass
class UADStepOutput:
    outputs: list[UADModelOutput] = field(default_factory=list)

    @property
    def request_ids(self) -> list[str]:
        return [output.request_id for output in self.outputs]
