"""UAD EngineCore integration scaffold.

This package is intentionally separate from the earlier toy UAD engine under
``vllm_omni.uad``.  The classes here are shaped around vLLM's EngineCore
runtime boundary so serving can instantiate a UAD-aware EngineCore while still
using the normal EngineCoreRequest / EngineCoreOutputs protocol.
"""

from typing import Any

from uad_vllm.config import UAD_ENV_VAR, UADConfig, configure_uad_engine_env, should_use_uad_engine
from uad_vllm.executor import UADExecutor
from uad_vllm.outputs import UADModelRunnerOutput
from uad_vllm.runner import UADGPUModelRunner, UADRunner
from uad_vllm.scheduler import UADScheduleItem, UADScheduler, UADSchedulerOutput

__all__ = [
    "UAD_ENV_VAR",
    "UADConfig",
    "UADEngineCore",
    "UADExecutor",
    "UADGPUModelRunner",
    "UADModelRunnerOutput",
    "UADRunner",
    "UADScheduleItem",
    "UADScheduler",
    "UADSchedulerOutput",
    "configure_uad_engine_env",
    "should_use_uad_engine",
]


def __getattr__(name: str) -> Any:
    if name == "UADEngineCore":
        from uad_vllm.engine_core import UADEngineCore

        return UADEngineCore
    raise AttributeError(name)
