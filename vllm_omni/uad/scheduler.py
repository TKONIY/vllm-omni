from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm_omni.uad.omni.hunyuan_image3 import HunyuanImage3UADStateMachine
from vllm_omni.uad.outputs import UADModelOutput, UADRunnerOutput, UADRunnerStepOutput, UADStepOutput
from vllm_omni.uad.request import UADPhase, UADRequestState
from vllm_omni.uad.state_machine import UADModelStateMachine


@dataclass(frozen=True)
class UADScheduleItem:
    """One request's work item for a scheduler tick.

    `request_id` identifies the request. `phase` selects the runner path.
    `token_ids` are the logical input token ids when the item is token-backed
    AR work; DiT denoise items can leave this empty because their input comes
    from latent/timestep state. `num_scheduled_tokens` is the token budget cost
    of this item. `num_computed_tokens` snapshots the reusable context prefix
    length before execution.

    `persist` is the hard context-commit bit. If it is true and execution
    succeeds, the item must write reusable engine context/KV and advance the
    request's `num_computed_tokens`. If it is false, the item is transient
    compute and must not advance `num_computed_tokens`.

    Examples:
    - AR prefill/decode: `persist=True`; scheduled tokens write paged KV.
    - DiT non-final denoise: `persist=False`; only diffusion state changes.
    - DiT final context write: `persist=True`; image context becomes reusable.
    """

    request_id: str
    phase: UADPhase
    token_ids: list[int]
    num_scheduled_tokens: int
    num_computed_tokens: int
    persist: bool = True


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
            if request.phase == "dit_step":
                if request.dit_step_index >= request.total_dit_steps:
                    continue
                is_final_dit_step = request.dit_step_index + 1 >= request.total_dit_steps
                pending_token_ids = request.pending_token_ids()
                num_scheduled_tokens = (
                    len(pending_token_ids)
                    if is_final_dit_step
                    else request.image_context_token_count or len(pending_token_ids)
                )
                items.append(
                    UADScheduleItem(
                        request_id=request.request_id,
                        phase="dit_step",
                        token_ids=[],
                        num_scheduled_tokens=num_scheduled_tokens,
                        num_computed_tokens=request.num_computed_tokens,
                        persist=is_final_dit_step,
                    )
                )
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
    """Token-budget-free scheduler used only for UAD toy smoke tests."""

    def __init__(
        self,
        state_machine: UADModelStateMachine | None = None,
    ) -> None:
        self.state_machine = state_machine or HunyuanImage3UADStateMachine()
        self.requests: dict[str, UADRequestState] = {}

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: list[int],
    ) -> UADRequestState:
        if request_id in self.requests:
            raise ValueError(f"duplicate UAD request_id: {request_id}")
        request = UADRequestState.from_prompt_token_ids(request_id, prompt_token_ids)
        self.requests[request_id] = request
        return request

    def get_request(self, request_id: str) -> UADRequestState:
        return self.requests[request_id]

    def schedule(
        self,
        base_output: Any | None = None,
    ) -> UADSchedulerOutput:
        return self.build_shadow_output(list(self.requests.values()), base_output=base_output)

    def update_from_output(
        self,
        scheduler_output: UADSchedulerOutput,
        runner_output: UADRunnerStepOutput,
    ) -> UADStepOutput:
        if len(scheduler_output.scheduled_items) != len(runner_output.outputs):
            raise ValueError(
                "scheduler/runner output length mismatch: "
                f"{len(scheduler_output.scheduled_items)} scheduled items vs {len(runner_output.outputs)} outputs"
            )

        item_outputs = list(zip(scheduler_output.scheduled_items, runner_output.outputs, strict=True))
        outputs = [self._process_runner_output(item, output) for item, output in item_outputs]
        for item, output in zip(scheduler_output.scheduled_items, outputs, strict=True):
            self._apply_model_output(item, output)
        return UADStepOutput(outputs=outputs)

    def _process_runner_output(self, item: UADScheduleItem, output: UADRunnerOutput) -> UADModelOutput:
        if item.request_id != output.request_id or item.phase != output.phase:
            raise ValueError(
                "scheduler/runner output item mismatch: "
                f"scheduled ({item.request_id}, {item.phase}) but got ({output.request_id}, {output.phase})"
            )
        request = self.requests[output.request_id]
        return self.state_machine.update_request_state(
            request=request,
            runner_output=output,
        )

    def _apply_model_output(self, item: UADScheduleItem, output: UADModelOutput) -> None:
        request = self.requests[output.request_id]
        if item.persist:
            request.advance_computed_tokens(item.num_scheduled_tokens)
        request.append_engine_tokens(output.new_engine_tokens)
        request.append_materialized_tokens(output.new_materialized_tokens)
        if output.phase_update is not None:
            request.apply_phase_update(output.phase_update)
        request.finished = output.finished
