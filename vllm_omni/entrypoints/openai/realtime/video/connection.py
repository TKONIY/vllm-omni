# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.video_api_utils import _normalize_video_array

from .protocol import (
    RealtimeVideoGenerationRequest,
    RealtimeVideoInputImage,
    RealtimeVideoSession,
)
from .serving import RealtimeVideoServing

logger = init_logger(__name__)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        shape = tuple(int(dim) for dim in value.shape)
        return {
            "type": type(value).__name__,
            "shape": list(shape),
            "dtype": str(value.dtype),
        }
    return value


class RealtimeVideoConnection:
    """JSON websocket protocol for realtime interactive video generation."""

    def __init__(
        self,
        websocket: WebSocket,
        serving: RealtimeVideoServing,
    ) -> None:
        self.websocket = websocket
        self.serving = serving
        self.session = RealtimeVideoSession(model=serving.model_name)

    async def _send_error(self, message: str, *, code: str = "invalid_request_error") -> None:
        await self.websocket.send_json(
            {
                "type": "error",
                "error": {
                    "code": code,
                    "message": message,
                },
            }
        )

    async def _send_session_created(self) -> None:
        await self.websocket.send_json(
            {
                "type": "session.created",
                "session": {
                    "id": self.session.id,
                    "model": self.session.model,
                    "modalities": ["video"],
                    "video": {
                        "width": self.session.width,
                        "height": self.session.height,
                        "fps": self.session.fps,
                        "num_frames": self.session.num_frames,
                        "chunk_size": self.session.chunk_size,
                    },
                    "text": {
                        "layers": self.session.text_layers,
                        "rendered": self.session.rendered_prompt,
                    },
                    "capabilities": [
                        "session.update",
                        "input.image.set",
                        "input.control.append",
                        "response.create",
                        "session.reset",
                    ],
                },
            }
        )

    @staticmethod
    def _extract_payload(message: dict[str, Any]) -> dict[str, Any]:
        if "text" in message and message["text"]:
            return json.loads(message["text"])
        raise ValueError("Realtime video websocket expects JSON text frames.")

    async def _send_generation_result(self, output: Any, request: RealtimeVideoGenerationRequest) -> None:
        custom_output = getattr(output, "custom_output", {}) or {}
        multimodal_output = getattr(output, "multimodal_output", {}) or {}
        video = custom_output.get("video_chunk")
        if video is None:
            images = getattr(output, "images", None) or []
            if images:
                video = images[0]
        if video is None:
            video = multimodal_output.get("video")

        encoded_frames: list[str] = []
        if video is not None:
            normalized = _normalize_video_array(video)
            if isinstance(normalized, list):
                if normalized:
                    normalized = normalized[0]
            for frame in normalized:
                from io import BytesIO
                import base64
                from PIL import Image

                buffer = BytesIO()
                Image.fromarray((frame * 255).clip(0, 255).astype("uint8")).save(buffer, format="JPEG", quality=90)
                encoded_frames.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

        await self.websocket.send_json(
            {
                "type": "response.output_video.delta",
                "response": {
                    "id": request.request_id,
                    "session_id": request.session_id,
                    "chunk_index": self.session.generation_index,
                    "text": {
                        "layers": request.text_layers,
                        "rendered": request.rendered_prompt,
                    },
                    "video": {
                        "fps": request.fps,
                        "width": request.width,
                        "height": request.height,
                        "frames_b64_jpeg": encoded_frames,
                    },
                    "custom_output": _json_safe(custom_output),
                },
            }
        )

        await self.websocket.send_json(
            {
                "type": "response.completed",
                "response": {
                    "id": request.request_id,
                    "session_id": request.session_id,
                },
            }
        )

    async def handle_connection(self) -> None:
        await self.websocket.accept()
        await self._send_session_created()

        try:
            while True:
                message = await self.websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                try:
                    event = self._extract_payload(message)
                except Exception as exc:
                    await self._send_error(str(exc))
                    continue

                event_type = event.get("type")
                if event_type == "session.update":
                    session_payload = event.get("session", {}) or {}
                    self.session.apply_update(session_payload)
                    await self.websocket.send_json(
                        {
                            "type": "session.updated",
                            "session": {
                                "id": self.session.id,
                                "model": self.session.model,
                                "text": {
                                    "layers": self.session.text_layers,
                                    "rendered": self.session.rendered_prompt,
                                },
                                "video": {
                                    "width": self.session.width,
                                    "height": self.session.height,
                                    "fps": self.session.fps,
                                    "num_frames": self.session.num_frames,
                                    "chunk_size": self.session.chunk_size,
                                },
                            },
                        }
                    )
                    continue

                if event_type == "input.image.set":
                    self.session.image = RealtimeVideoInputImage.from_event(event.get("image"))
                    self.session.needs_reset = True
                    await self.websocket.send_json(
                        {
                            "type": "input.image.updated",
                            "session_id": self.session.id,
                            "has_image": self.session.image.image is not None,
                            "source": self.session.image.source,
                        }
                    )
                    continue

                if event_type == "input.control.append":
                    control_payload = event.get("control", {}) or {}
                    self.session.append_control(control_payload)
                    await self.websocket.send_json(
                        {
                            "type": "input.control.appended",
                            "session_id": self.session.id,
                            "buffered_segments": len(self.session.control),
                        }
                    )
                    continue

                if event_type == "session.reset":
                    self.session.reset_buffers()
                    self.serving.reset(self.session.id)
                    await self.websocket.send_json(
                        {
                            "type": "session.reset.completed",
                            "session_id": self.session.id,
                        }
                    )
                    continue

                if event_type == "response.create":
                    request = RealtimeVideoGenerationRequest.from_session(self.session)
                    if not request.rendered_prompt:
                        await self._send_error("Missing prompt/text_layers for realtime video generation.")
                        continue
                    if request.image is None:
                        await self._send_error("Missing input image for realtime video generation.")
                        continue
                    if not request.control:
                        await self._send_error("Missing control buffer for realtime video generation.")
                        continue

                    await self.websocket.send_json(
                        {
                            "type": "response.created",
                            "response": {
                                "id": request.request_id,
                                "session_id": request.session_id,
                            },
                        }
                    )
                    result = await self.serving.generate(request)
                    await self._send_generation_result(result.raw_output, request)
                    self.session.generation_index += 1
                    self.session.needs_reset = False
                    self.session.control.clear()
                    continue

                await self._send_error(f"Unsupported event type: {event_type!r}", code="unsupported")

        except WebSocketDisconnect:
            pass
