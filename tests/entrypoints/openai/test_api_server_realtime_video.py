import asyncio
from argparse import Namespace

from starlette.datastructures import State

from vllm_omni.entrypoints.openai.api_server import omni_init_app_state
from vllm_omni.entrypoints.openai.realtime.video.serving import RealtimeVideoServing


class _FakeDiffusionEngine:
    stage_configs = [{"stage_type": "diffusion"}]

    async def get_vllm_config(self):
        return None


def test_pure_diffusion_init_registers_realtime_video_serving():
    engine = _FakeDiffusionEngine()
    state = State()
    args = Namespace(
        model="robbyant/lingbot-world-base-cam",
        served_model_name=["robbyant/lingbot-world-base-cam"],
        enable_log_requests=False,
        disable_log_stats=False,
        enable_server_load_tracking=False,
    )

    asyncio.run(omni_init_app_state(engine, state, args))

    assert isinstance(state.openai_serving_realtime_video, RealtimeVideoServing)
    assert state.openai_serving_realtime_video.engine_client is engine
    assert state.openai_serving_realtime_video.model_name == "robbyant/lingbot-world-base-cam"
