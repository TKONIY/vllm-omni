from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm_omni.uad.request import UADPhase, UADRequestState


@dataclass(frozen=True)
class UADScheduleItem:
    request_id: str
    phase: UADPhase
    token_ids: list[int]
    num_scheduled_tokens: int
    num_computed_tokens: int


@dataclass
class UADSchedulerOutput:
    base_output: Any | None = None
    scheduled_items: list[UADScheduleItem] = field(default_factory=list)

    @property
    def request_ids(self) -> list[str]:
        return [item.request_id for item in self.scheduled_items]

    @property
    def total_num_scheduled_tokens(self) -> int:
        return sum(item.num_scheduled_tokens for item in self.scheduled_items)

    @property
    def num_scheduled_tokens_by_request(self) -> dict[str, int]:
        return {item.request_id: item.num_scheduled_tokens for item in self.scheduled_items}


class UADShadowScheduler:
    """Build UAD scheduler metadata beside an existing scheduler decision."""

    def build_shadow_output(
        self,
        requests: list[UADRequestState],
        base_output: Any | None = None,
    ) -> UADSchedulerOutput:
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
                    num_computed_tokens=request.num_computed_tokens,
                )
            )

        return UADSchedulerOutput(base_output=base_output, scheduled_items=items)


class UADToyScheduler(UADShadowScheduler):
    """Token-budget-free scheduler used only for Step 0/1 smoke tests."""

    def schedule(
        self,
        requests: list[UADRequestState],
        base_output: Any | None = None,
    ) -> UADSchedulerOutput:
        return self.build_shadow_output(requests, base_output=base_output)
