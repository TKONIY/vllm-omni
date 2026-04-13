from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.entrypoints.openai.realtime.robot.openpi_serving import (
    ServingRealtimeRobotOpenPI,
)
from vllm_omni.entrypoints.openai.realtime.robot.transform.base import (
    RobotPolicyTransform,
)


class _FakeTransform:
    def transform_input(self, obs: dict) -> dict:
        return {
            "prompt": f"wrapped:{obs['prompt']}",
            "images": obs["images"],
            "state": obs["state"],
            "embodiment_name": "fake",
        }

    def transform_output(self, result) -> np.ndarray:
        return np.asarray(result.multimodal_output["actions"], dtype=np.float32)


class _BaseTransform(RobotPolicyTransform):
    IMAGE_KEY_MAP = {}
    EMBODIMENT_NAME = "fake"
    ACTION_DIM = 2

    def _stitch_views(self, images: dict[str, np.ndarray]) -> np.ndarray:
        return np.zeros((1, 1, 1, 3), dtype=np.uint8)

    def _language_template(self, prompt: str) -> str:
        return prompt

    def _extract_raw_state(self, obs: dict) -> np.ndarray:
        return np.zeros((0,), dtype=np.float64)


class _FakeAsyncOmni:
    def __init__(self) -> None:
        self.requests = []

    async def generate(self, *, prompt, request_id, sampling_params_list):
        self.requests.append(
            SimpleNamespace(
                prompts=[prompt],
                request_ids=[request_id],
                sampling_params=sampling_params_list[0],
            ),
        )
        yield SimpleNamespace(
            multimodal_output={"actions": [[1.0, 2.0], [3.0, 4.0]]},
        )


def test_infer_uses_async_omni_generate(monkeypatch):
    engine = _FakeAsyncOmni()
    serving = ServingRealtimeRobotOpenPI(engine_client=engine, model_name="dreamzero-droid")
    monkeypatch.setattr(serving, "_get_transform", lambda obs: _FakeTransform())

    obs = {
        "prompt": "pick up the cup",
        "images": np.zeros((1, 4, 4, 3), dtype=np.uint8),
        "state": np.zeros((7,), dtype=np.float32),
        "session_id": "session-0",
    }

    actions = asyncio.run(serving.infer(obs))

    assert len(engine.requests) == 1
    request = engine.requests[0]
    assert request.prompts == ["wrapped:pick up the cup"]
    assert request.request_ids == ["robot-session-0"]
    assert request.sampling_params.extra_args["reset"] is True
    assert request.sampling_params.extra_args["session_id"] == "session-0"
    assert request.sampling_params.extra_args["unified_obs"]["embodiment_name"] == "fake"
    np.testing.assert_array_equal(
        actions,
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )


def test_infer_reset_flag_tracks_session_boundaries(monkeypatch):
    engine = _FakeAsyncOmni()
    serving = ServingRealtimeRobotOpenPI(engine_client=engine, model_name="dreamzero-droid")
    monkeypatch.setattr(serving, "_get_transform", lambda obs: _FakeTransform())

    base_obs = {
        "prompt": "move forward",
        "images": np.zeros((1, 2, 2, 3), dtype=np.uint8),
        "state": np.zeros((3,), dtype=np.float32),
    }

    asyncio.run(serving.infer({**base_obs, "session_id": "session-a"}))
    asyncio.run(serving.infer({**base_obs, "session_id": "session-a"}))
    asyncio.run(serving.infer({**base_obs, "session_id": "session-b"}))

    resets = [request.sampling_params.extra_args["reset"] for request in engine.requests]
    assert resets == [True, False, True]


def test_transform_output_requires_actions():
    transform = _BaseTransform()
    with pytest.raises(RuntimeError, match="Missing multimodal_output in robot policy result"):
        transform.transform_output(SimpleNamespace())
    with pytest.raises(RuntimeError, match="Missing multimodal_output\\['actions'\\]"):
        transform.transform_output(SimpleNamespace(multimodal_output={}))
