from argparse import Namespace
import asyncio

from starlette.datastructures import State

from vllm_omni.entrypoints.openai.api_server import omni_init_app_state
from vllm_omni.entrypoints.openai.realtime.robot.openpi_serving import (
    ServingRealtimeRobotOpenPI,
)


class _FakeDiffusionEngine:
    stage_configs = [{"stage_type": "diffusion"}]

    async def get_vllm_config(self):
        return None


def test_pure_diffusion_init_registers_robot_serving():
    engine = _FakeDiffusionEngine()
    state = State()
    args = Namespace(
        model="dreamzero-local",
        served_model_name=["dreamzero-droid"],
        enable_log_requests=False,
        disable_log_stats=False,
        enable_server_load_tracking=False,
    )

    asyncio.run(omni_init_app_state(engine, state, args))

    assert state.diffusion_engine is engine
    assert isinstance(state.openai_serving_realtime_robot, ServingRealtimeRobotOpenPI)
    assert state.openai_serving_realtime_robot.engine_client is engine
    assert state.openai_serving_realtime_robot.model_name == "dreamzero-droid"
