from __future__ import annotations

from types import SimpleNamespace

import pytest

from uad_vllm.config import UAD_ENV_VAR, configure_uad_engine_env, should_use_uad_engine

pytestmark = pytest.mark.cpu


def test_uad_engine_env_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(UAD_ENV_VAR, raising=False)

    assert not should_use_uad_engine()

    configure_uad_engine_env(True)

    assert should_use_uad_engine()


def test_stage_engine_core_resolves_uad_proc_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    stage_proc_mod = pytest.importorskip("vllm_omni.engine.stage_engine_core_proc", exc_type=ImportError)
    resolve_stage_engine_core_cls = stage_proc_mod.resolve_stage_engine_core_cls
    monkeypatch.setenv(UAD_ENV_VAR, "1")

    assert resolve_stage_engine_core_cls(SimpleNamespace()).__name__ == "UADEngineCore"


def test_stage_engine_core_defaults_to_existing_proc(monkeypatch: pytest.MonkeyPatch) -> None:
    stage_proc_mod = pytest.importorskip("vllm_omni.engine.stage_engine_core_proc", exc_type=ImportError)
    StageEngineCoreProc = stage_proc_mod.StageEngineCoreProc
    resolve_stage_engine_core_cls = stage_proc_mod.resolve_stage_engine_core_cls
    monkeypatch.delenv(UAD_ENV_VAR, raising=False)

    assert resolve_stage_engine_core_cls(SimpleNamespace()) is StageEngineCoreProc


def test_uad_engine_can_be_enabled_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(UAD_ENV_VAR, raising=False)

    vllm_config = SimpleNamespace(uad_engine=True)

    assert should_use_uad_engine(vllm_config)
