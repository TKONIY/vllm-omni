from __future__ import annotations

from dataclasses import dataclass, field

from vllm_omni.uad.request import UADPhase, UADRequestState


@dataclass(frozen=True)
class UADScheduleItem:
    request_id: str
    phase: UADPhase
    token_ids: list[int]
    num_scheduled_tokens: int


@dataclass
class UADSchedulerOutput:
    scheduled_items: list[UADScheduleItem] = field(default_factory=list)


class UADToyScheduler:
    """Token-budget-free scheduler used only for Step 0 smoke tests."""

    def schedule(self, requests: list[UADRequestState]) -> UADSchedulerOutput:
        items: list[UADScheduleItem] = []
        for request in requests:
            if request.finished:
                continue

            token_ids = request.pending_token_ids()
            if not token_ids:
                continue

            phase: UADPhase = "ar_prefill" if request.num_computed_tokens == 0 else "ar_decode"
            request.phase = phase
            items.append(
                UADScheduleItem(
                    request_id=request.request_id,
                    phase=phase,
                    token_ids=token_ids,
                    num_scheduled_tokens=len(token_ids),
                )
            )

        return UADSchedulerOutput(scheduled_items=items)
