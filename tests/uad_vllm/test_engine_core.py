from __future__ import annotations

from contextlib import nullcontext
from types import MethodType
from typing import Any

import pytest
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput

from uad_vllm.outputs import UADModelRunnerOutput
from uad_vllm.scheduler import UADSchedulerOutput

pytestmark = pytest.mark.cpu
engine_core_mod = pytest.importorskip("uad_vllm.engine_core", exc_type=ImportError)
UADEngineCore = engine_core_mod.UADEngineCore


class _Future:
    def __init__(self, value: object) -> None:
        self.value = value

    def result(self) -> object:
        return self.value


class _Scheduler:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        base_output = SchedulerOutput.make_empty()
        base_output.num_scheduled_tokens = {"req": 1}
        base_output.total_num_scheduled_tokens = 1
        self.output = UADSchedulerOutput.from_base(base_output)

    def has_requests(self) -> bool:
        self.calls.append("has_requests")
        return True

    def schedule(self) -> UADSchedulerOutput:
        self.calls.append("schedule")
        return self.output

    def get_grammar_bitmask(self, scheduler_output: UADSchedulerOutput) -> str:
        assert scheduler_output is self.output
        self.calls.append("grammar")
        return "grammar"

    def update_from_output(
        self,
        scheduler_output: UADSchedulerOutput,
        runner_output: UADModelRunnerOutput,
    ) -> dict[int, str]:
        assert scheduler_output is self.output
        assert runner_output.req_ids == ["req"]
        self.calls.append("update")
        return {0: "engine-output"}


class _Executor:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def execute_model(self, scheduler_output: UADSchedulerOutput, *, non_block: bool = False) -> _Future:
        assert scheduler_output.total_num_scheduled_tokens == 1
        assert non_block
        self.calls.append("execute")
        return _Future(None)

    def sample_tokens(self, grammar_output: object, *, non_block: bool = False) -> ModelRunnerOutput:
        assert grammar_output == "grammar"
        assert not non_block
        self.calls.append("sample")
        return ModelRunnerOutput(req_ids=["req"], req_id_to_index={"req": 0})


class _Runner:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def process_outputs(self, scheduler_output: UADSchedulerOutput, raw_model_output: Any) -> UADModelRunnerOutput:
        assert scheduler_output.total_num_scheduled_tokens == 1
        assert raw_model_output.req_ids == ["req"]
        self.calls.append("runner")
        return UADModelRunnerOutput.from_base(raw_model_output)


def test_uad_engine_core_step_calls_uad_modules_in_order() -> None:
    calls: list[str] = []
    engine = object.__new__(UADEngineCore)
    engine.uad_scheduler = _Scheduler(calls)
    engine.uad_executor = _Executor(calls)
    engine.uad_runner = _Runner(calls)
    engine.log_error_detail = MethodType(lambda self, output: nullcontext(), engine)
    engine.log_iteration_details = MethodType(lambda self, output: nullcontext(), engine)
    engine._process_aborts_queue = MethodType(lambda self: calls.append("aborts"), engine)

    outputs, model_executed = UADEngineCore.step(engine)

    assert outputs == {0: "engine-output"}
    assert model_executed is True
    assert calls == [
        "has_requests",
        "schedule",
        "execute",
        "grammar",
        "sample",
        "aborts",
        "runner",
        "update",
    ]


def test_uad_engine_core_keeps_v1_scheduler_interface(monkeypatch: pytest.MonkeyPatch) -> None:
    base_scheduler = object()
    base_executor = object()

    def fake_stage_init(self: object, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.scheduler = base_scheduler  # type: ignore[attr-defined]
        self.model_executor = base_executor  # type: ignore[attr-defined]
        self.batch_queue = None  # type: ignore[attr-defined]

    monkeypatch.setattr(engine_core_mod.StageEngineCoreProc, "__init__", fake_stage_init)

    engine = UADEngineCore()

    assert engine.scheduler is base_scheduler
    assert engine.model_executor is base_executor
    assert not hasattr(engine.uad_scheduler, "base_scheduler")
    assert not hasattr(engine.uad_executor, "base_executor")
