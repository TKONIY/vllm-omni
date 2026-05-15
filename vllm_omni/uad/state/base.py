from __future__ import annotations

from abc import ABC, abstractmethod

from vllm_omni.uad.outputs import UADModelRunnerItemOutput, UADStateUpdate
from vllm_omni.uad.request import UADRequestState


class UADModelStateMachine(ABC):
    """Base class for model-specific UAD request-state transitions.

    The runner owns execution mechanics and returns semantic-free raw outputs.
    The scheduler calls this base class from `update_from_output()` to convert
    raw outputs into request-state deltas while keeping model-private phase
    rules out of generic scheduling and runner code.
    """

    @abstractmethod
    def update_request_state(
        self,
        *,
        request: UADRequestState,
        runner_output: UADModelRunnerItemOutput,
    ) -> UADStateUpdate:
        """Convert one raw runner output into a request-state delta."""
        raise NotImplementedError
