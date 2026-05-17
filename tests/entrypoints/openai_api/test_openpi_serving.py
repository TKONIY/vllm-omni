import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from vllm_omni.entrypoints.openai.realtime.robot import openpi_serving

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

TEST_POLICY_SERVER_CONFIG = {
    "image_resolution": (180, 320),
    "n_external_cameras": 2,
    "needs_wrist_camera": True,
    "needs_stereo_camera": False,
    "needs_session_id": True,
    "action_space": "joint_position",
}


def _engine_with_policy_config(policy_config=None):
    od_config = SimpleNamespace(model_config={"policy_server_config": policy_config or TEST_POLICY_SERVER_CONFIG})
    return SimpleNamespace(get_diffusion_od_config=lambda: od_config)


def test_policy_server_config_reads_diffusion_model_config():
    policy_config = {
        "image_resolution": [64, 64],
        "n_external_cameras": 1,
        "custom_model_key": {"nested": True},
    }
    od_config = SimpleNamespace(model_config={"policy_server_config": policy_config})
    engine_client = SimpleNamespace(get_diffusion_od_config=lambda: od_config)

    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine_client)

    assert serving.policy_server_config.to_dict() == policy_config


def test_policy_server_config_reads_stage_config_model_config():
    policy_config = {"custom_model_key": "from-stage-config"}
    engine_client = SimpleNamespace(
        get_diffusion_od_config=lambda: None,
        stage_configs=[
            SimpleNamespace(
                stage_type="diffusion",
                engine_args=SimpleNamespace(model_config={"policy_server_config": policy_config}),
            )
        ],
    )

    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine_client)

    assert serving.policy_server_config.to_dict() == policy_config


def test_policy_server_config_reads_omegaconf_stage_config():
    engine_client = SimpleNamespace(
        get_diffusion_od_config=lambda: None,
        stage_configs=[
            SimpleNamespace(
                stage_type="diffusion",
                engine_args=SimpleNamespace(
                    model_config=OmegaConf.create({"policy_server_config": {"custom_model_key": "from-omegaconf"}})
                ),
            )
        ],
    )

    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine_client)

    assert serving.policy_server_config.to_dict() == {"custom_model_key": "from-omegaconf"}


def test_policy_server_config_is_required():
    od_config = SimpleNamespace(model_config={})
    engine_client = SimpleNamespace(get_diffusion_od_config=lambda: od_config)

    with pytest.raises(ValueError) as exc_info:
        openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine_client)

    assert "policy_server_config" in str(exc_info.value)


def test_create_policy_server_returns_none_without_policy_config():
    od_config = SimpleNamespace(model_config={})
    engine_client = SimpleNamespace(get_diffusion_od_config=lambda: od_config)

    serving = openpi_serving.ServingRealtimeRobotOpenPI.create_policy_server(
        engine_client=engine_client,
        model_name="generic-model",
    )

    assert serving is None


def test_policy_server_config_reads_engine_model_config():
    policy_config = {"custom_model_key": "custom-value"}
    engine_client = SimpleNamespace(model_config=SimpleNamespace(policy_server_config=policy_config))

    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine_client)

    assert serving.policy_server_config.to_dict() == policy_config


def test_build_request_forwards_connection_session_state():
    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=_engine_with_policy_config())

    request = serving._build_request(
        {"prompt": "pick up the object"},
        session_id="session-a",
        reset=True,
    )

    assert request.sampling_params.extra_args["reset"] is True
    assert request.sampling_params.extra_args["session_id"] == "session-a"
    assert request.sampling_params.extra_args["robot_obs"]["prompt"] == "pick up the object"
    assert request.request_ids == ["robot-session-a"]


def test_infer_extracts_actions_from_generic_multimodal_output():
    class FakeEngineClient:
        def get_diffusion_od_config(self):
            return SimpleNamespace(model_config={"policy_server_config": TEST_POLICY_SERVER_CONFIG})

        async def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            yield SimpleNamespace(multimodal_output={"actions": [[1.0, 2.0, 3.0]]})

    engine_client = FakeEngineClient()
    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine_client)

    actions = asyncio.run(serving.infer({"prompt": "pick up"}, session_id="session-a", reset=True))

    np.testing.assert_allclose(actions, np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    assert engine_client.generate_kwargs["prompt"] == "pick up"
    assert engine_client.generate_kwargs["request_id"] == "robot-session-a"


def test_infer_preserves_dict_actions_from_multimodal_output():
    class FakeEngineClient:
        def get_diffusion_od_config(self):
            return SimpleNamespace(model_config={"policy_server_config": TEST_POLICY_SERVER_CONFIG})

        async def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            yield SimpleNamespace(
                multimodal_output={
                    "actions": {
                        "left_arm": [[1.0, 2.0]],
                        "right_arm": np.array([[3.0, 4.0]], dtype=np.float64),
                    }
                }
            )

    engine_client = FakeEngineClient()
    serving = openpi_serving.ServingRealtimeRobotOpenPI(engine_client=engine_client)

    actions = asyncio.run(serving.infer({"prompt": "pick up"}, session_id="session-a", reset=True))

    assert isinstance(actions, dict)
    assert set(actions) == {"left_arm", "right_arm"}
    np.testing.assert_allclose(actions["left_arm"], np.array([[1.0, 2.0]], dtype=np.float32))
    np.testing.assert_allclose(actions["right_arm"], np.array([[3.0, 4.0]], dtype=np.float32))
    assert actions["left_arm"].dtype == np.float32
    assert actions["right_arm"].dtype == np.float32
