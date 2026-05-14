from __future__ import annotations

from typing import Protocol

from vllm_omni.uad.outputs import UADModelOutput
from vllm_omni.uad.request import UADRequestState, UADToken


class UADModelStateMachine(Protocol):
    """Model-specific phase and output-ledger policy used by `UADRunner`.

    The runner owns execution mechanics. This protocol owns model-defined
    interpretation of sampled AR tokens, phase transitions, and per-phase
    request ledger updates.
    """

    def on_ar_token_sampled(
        self,
        *,
        request: UADRequestState,
        sampled_token: UADToken,
        num_scheduled_tokens: int,
    ) -> UADModelOutput:
        """Convert one sampled AR token into a request-state update."""
        ...

    def on_dit_step_completed(
        self,
        *,
        request: UADRequestState,
        num_scheduled_tokens: int,
    ) -> UADModelOutput:
        """Convert one completed DiT scheduler item into a request-state update."""
        ...
