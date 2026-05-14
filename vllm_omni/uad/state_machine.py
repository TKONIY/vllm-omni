from __future__ import annotations

from typing import Protocol

from vllm_omni.uad.outputs import UADModelOutput, UADRunnerOutput
from vllm_omni.uad.request import UADRequestState


class UADModelStateMachine(Protocol):
    """Model-specific phase and output-ledger policy used by scheduler update.

    The runner owns execution mechanics and returns semantic-free raw outputs.
    The scheduler calls this protocol from `update_from_output()` to interpret
    raw outputs, compute request-state deltas, and keep model-private phase
    rules out of the generic runner.
    """

    def update_request_state(
        self,
        *,
        request: UADRequestState,
        runner_output: UADRunnerOutput,
    ) -> UADModelOutput:
        """Convert one raw runner output into a request-state delta."""
        ...
