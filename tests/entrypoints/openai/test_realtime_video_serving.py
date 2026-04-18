from __future__ import annotations

import asyncio
import base64
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI, WebSocket
from PIL import Image
from starlette.testclient import TestClient

from vllm_omni.diffusion.models.lingbot_world_fast import normalize_lingbot_control_chunk
from vllm_omni.entrypoints.openai.realtime.video.connection import RealtimeVideoConnection
from vllm_omni.entrypoints.openai.realtime.video.protocol import (
    RealtimeVideoGenerationRequest,
    RealtimeVideoSession,
    render_text_layers,
)
from vllm_omni.entrypoints.openai.realtime.video.serving import RealtimeVideoServing


class _FakeEngine:
    def __init__(self) -> None:
        self.requests = []

    async def generate(self, *, prompt, request_id, sampling_params_list):
        self.requests.append(
            SimpleNamespace(
                prompt=prompt,
                request_id=request_id,
                sampling=sampling_params_list[0],
            )
        )
        yield SimpleNamespace(
            images=[],
            custom_output={
                "video_chunk": np.zeros((2, 4, 4, 3), dtype=np.uint8),
                "realtime_video": {"generated_chunks": 1},
            },
            multimodal_output={"fps": 16},
        )


class _FakeCollectiveEngine:
    def __init__(self) -> None:
        self.stage_configs = [{"stage_type": "llm"}, {"stage_type": "diffusion"}]
        self.rpc_calls = []
        self.engine = SimpleNamespace(
            stage_configs=self.stage_configs,
            collective_rpc=self.collective_rpc,
        )

    def collective_rpc(self, *, method, args, stage_ids):
        self.rpc_calls.append(
            SimpleNamespace(method=method, args=args, stage_ids=stage_ids)
        )
        return [True]


class _FakeServing:
    def __init__(self) -> None:
        self.model_name = "robbyant/lingbot-world-base-cam"
        self.requests: list[RealtimeVideoGenerationRequest] = []
        self.reset_calls: list[str] = []

    def reset(self, session_id: str) -> None:
        self.reset_calls.append(session_id)

    async def generate(self, request: RealtimeVideoGenerationRequest):
        self.requests.append(request)
        return SimpleNamespace(
            raw_output=SimpleNamespace(
                images=[],
                custom_output={"video_chunk": np.zeros((2, 4, 4, 3), dtype=np.uint8)},
                multimodal_output={},
            )
        )


def _encode_test_image() -> str:
    image = Image.new("RGB", (8, 8), color=(12, 34, 56))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_render_text_layers_preserves_hierarchy_order():
    rendered = render_text_layers(
        {
            "scene": "A forest path at sunset.",
            "motion": "The camera moves forward smoothly.",
            "style": "Photorealistic.",
        },
        fallback_prompt="ignored fallback",
    )
    assert rendered == (
        "A forest path at sunset.\n\n"
        "The camera moves forward smoothly.\n\n"
        "Photorealistic.\n\n"
        "ignored fallback"
    )


def test_build_engine_request_carries_layers_and_control():
    engine = _FakeEngine()
    serving = RealtimeVideoServing(engine_client=engine, model_name="robbyant/lingbot-world-base-cam")
    session = RealtimeVideoSession(model="robbyant/lingbot-world-base-cam")
    session.apply_update(
        {
            "text_layers": {
                "scene": "A calm lake.",
                "motion": "The viewpoint drifts left.",
            },
            "width": 640,
            "height": 368,
            "fps": 12,
            "num_frames": 9,
            "chunk_size": 2,
            "seed": 7,
            "shift": 3.0,
            "max_attention_size": 2048,
        }
    )
    session.image.image = Image.new("RGB", (8, 8), color=(255, 0, 0))
    session.append_control({"poses": [[[1, 0, 0, 0]]], "intrinsics": [[1, 1, 0, 0]]})

    request = RealtimeVideoGenerationRequest.from_session(session)
    engine_request = serving.build_engine_request(request)

    assert engine_request.request_ids == [request.request_id]
    assert engine_request.prompts[0]["prompt"] == request.rendered_prompt
    assert engine_request.prompts[0]["multi_modal_data"]["image"] is session.image.image
    rt = engine_request.sampling_params.extra_args["realtime_video"]
    assert rt["text_layers"] == session.text_layers
    assert rt["control"] == session.control
    assert rt["chunk_size"] == 2
    assert rt["shift"] == 3.0
    assert rt["max_attention_size"] == 2048


def test_normalize_lingbot_control_chunk_broadcasts_intrinsics_and_action_alias():
    chunk = normalize_lingbot_control_chunk(
        {
            "poses": np.repeat(np.eye(4, dtype=np.float32)[None, ...], 5, axis=0),
            "intrinsics": [[400.0, 401.0, 200.0, 201.0]],
            "action": np.zeros((5, 4), dtype=np.float32),
        }
    )

    assert chunk.poses.shape == (5, 4, 4)
    assert chunk.intrinsics.shape == (5, 4)
    assert chunk.wasd_action is not None
    assert chunk.wasd_action.shape == (5, 4)
    assert chunk.control_type == "act"


def test_normalize_lingbot_control_chunk_rejects_frame_mismatch():
    with pytest.raises(ValueError, match="intrinsics frame count"):
        normalize_lingbot_control_chunk(
            {
                "poses": np.repeat(np.eye(4, dtype=np.float32)[None, ...], 3, axis=0),
                "intrinsics": np.zeros((2, 4), dtype=np.float32),
            }
        )


def test_realtime_serving_keeps_lingbot_payload_generic():
    engine = _FakeEngine()
    serving = RealtimeVideoServing(engine_client=engine, model_name="robbyant/lingbot-world-fast")
    session = RealtimeVideoSession(model="robbyant/lingbot-world-fast")
    session.apply_update(
        {
            "text_layers": {"scene": "A rainy alley.", "motion": "Move forward."},
            "width": 640,
            "height": 368,
            "fps": 12,
            "num_frames": 13,
            "chunk_size": 3,
            "seed": 7,
        }
    )
    session.image.image = Image.new("RGB", (8, 8), color=(255, 0, 0))
    session.append_control(
        {
            "poses": np.repeat(np.eye(4, dtype=np.float32)[None, ...], 13, axis=0),
            "intrinsics": [[400.0, 401.0, 200.0, 201.0]],
        }
    )

    request = RealtimeVideoGenerationRequest.from_session(session)
    engine_request = serving.build_engine_request(request)
    rt = engine_request.sampling_params.extra_args["realtime_video"]

    assert "backend" not in rt
    assert "session_state" not in rt
    assert rt["text_layers"] == session.text_layers
    assert rt["control"] == session.control
    assert rt["control"][0]["poses"].shape == (13, 4, 4)

    result = asyncio.run(serving.generate(request))
    assert result.chunk_index == 0
    assert engine.requests[0].sampling.extra_args["realtime_video"]["control"] == session.control


def test_realtime_serving_reset_uses_generic_diffusion_rpc():
    engine = _FakeCollectiveEngine()
    serving = RealtimeVideoServing(engine_client=engine, model_name="robbyant/lingbot-world-fast")

    serving.reset("rtvideo_123")

    assert len(engine.rpc_calls) == 1
    assert engine.rpc_calls[0].method == "reset_realtime_video_session"
    assert engine.rpc_calls[0].args == ("rtvideo_123",)
    assert engine.rpc_calls[0].stage_ids == [1]


def test_realtime_video_connection_event_flow():
    app = FastAPI()
    serving = _FakeServing()

    @app.websocket("/v1/realtime/video")
    async def ws_endpoint(websocket: WebSocket):
        handler = RealtimeVideoConnection(websocket, serving)
        await handler.handle_connection()

    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime/video") as ws:
            created = ws.receive_json()
            assert created["type"] == "session.created"

            ws.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "text_layers": {
                            "scene": "A neon city street.",
                            "motion": "The view moves forward.",
                        },
                        "fps": 10,
                        "num_frames": 13,
                    },
                }
            )
            updated = ws.receive_json()
            assert updated["type"] == "session.updated"
            assert updated["session"]["text"]["layers"]["scene"] == "A neon city street."

            ws.send_json(
                {
                    "type": "input.image.set",
                    "image": {"b64_json": _encode_test_image()},
                }
            )
            image_updated = ws.receive_json()
            assert image_updated["type"] == "input.image.updated"
            assert image_updated["has_image"] is True

            ws.send_json(
                {
                    "type": "input.control.append",
                    "control": {"poses": [[[1, 0, 0, 0]]], "intrinsics": [[1, 1, 0, 0]]},
                }
            )
            control_ack = ws.receive_json()
            assert control_ack["type"] == "input.control.appended"
            assert control_ack["buffered_segments"] == 1

            ws.send_json({"type": "response.create"})
            response_created = ws.receive_json()
            assert response_created["type"] == "response.created"
            response_delta = ws.receive_json()
            assert response_delta["type"] == "response.output_video.delta"
            assert response_delta["response"]["text"]["layers"]["motion"] == "The view moves forward."
            assert len(response_delta["response"]["video"]["frames_b64_jpeg"]) == 2
            response_done = ws.receive_json()
            assert response_done["type"] == "response.completed"

            assert len(serving.requests) == 1
            assert serving.requests[0].reset is True

            ws.send_json({"type": "session.reset"})
            reset_done = ws.receive_json()
            assert reset_done["type"] == "session.reset.completed"
            assert serving.reset_calls == [serving.requests[0].session_id]
