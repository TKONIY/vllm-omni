from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

UAD_ENV_VAR = "VLLM_OMNI_USE_UAD_ENGINE"
_TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class UADConfig:
    """Runtime switch for the EngineCore-compatible UAD path."""

    enabled: bool = False

    @classmethod
    def from_env(cls) -> UADConfig:
        return cls(enabled=_env_enabled())


def _env_enabled() -> bool:
    return os.environ.get(UAD_ENV_VAR, "").strip().lower() in _TRUE_VALUES


def configure_uad_engine_env(enabled: bool | None) -> None:
    """Enable the UAD process path for child EngineCore processes.

    ``False`` intentionally leaves the environment unchanged so deployments can
    still opt in via ``VLLM_OMNI_USE_UAD_ENGINE=1``.
    """

    if enabled:
        os.environ[UAD_ENV_VAR] = "1"


def should_use_uad_engine(vllm_config: Any | None = None) -> bool:
    """Return whether StageEngineCoreProc should instantiate UADEngineCore."""

    if _env_enabled():
        return True

    # Keep an object-level hook for later config plumbing without depending on
    # a vLLM config schema change in this scaffold step.
    if vllm_config is not None:
        if bool(getattr(vllm_config, "uad_engine", False)):
            return True
        omni_config = getattr(vllm_config, "omni_config", None)
        if bool(getattr(omni_config, "uad_engine", False)):
            return True

    return False
