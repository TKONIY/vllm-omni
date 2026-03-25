# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""WebSocket handler for DreamZero world model inference (OpenPI-style).

Protocol (msgpack over WebSocket binary frames):

    Client -> Server:
        {
            "endpoint": "infer" | "reset",
            "session_id": str,
            "observation/exterior_image_0_left": ndarray(H,W,3),
            "observation/exterior_image_1_left": ndarray(H,W,3),
            "observation/wrist_image_left": ndarray(H,W,3),
            "observation/joint_position": ndarray(7,),
            "observation/gripper_position": ndarray(1,),
            "prompt": str,
        }

    Server -> Client (on connect):
        {"image_resolution": [H, W], "n_cameras": 3,
         "action_dim": 8, "action_horizon": 24}

    Server -> Client (per infer):
        {"actions": ndarray(N, 8), "session_id": str,
         "timing": {"infer_ms": float, "preprocess_ms": float}}
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.session_manager import (
    DreamZeroSessionState,
    WorldSessionStore,
)

logger = init_logger(__name__)

_DEFAULT_IDLE_TIMEOUT = 30.0  # seconds


def _pack(obj: Any) -> bytes:
    """Serialize dict with numpy arrays to msgpack bytes."""
    import msgpack
    import msgpack_numpy

    return msgpack.packb(obj, default=msgpack_numpy.encode)


def _unpack(data: bytes) -> dict:
    """Deserialize msgpack bytes to dict with numpy arrays."""
    import msgpack
    import msgpack_numpy

    return msgpack.unpackb(data, object_hook=msgpack_numpy.decode, raw=False)


class OmniWorldStreamHandler:
    """Handles WebSocket sessions for OpenPI-style world model inference.

    Each WebSocket connection maps to one session. The handler:
    1. Sends server metadata on connect
    2. Loops: receive observation (msgpack) → infer → send actions (msgpack)
    3. Manages session state (create/reset/destroy)

    Args:
        engine: The diffusion engine for running inference.
        session_store: Session store for managing state across calls.
        idle_timeout: Max seconds to wait for a message before closing.
    """

    def __init__(
        self,
        engine: Any,
        session_store: WorldSessionStore | None = None,
        idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
    ) -> None:
        self._engine = engine
        self._session_store = session_store or WorldSessionStore()
        self._idle_timeout = idle_timeout

    async def handle_session(self, websocket: WebSocket) -> None:
        """Main session loop for a single WebSocket connection."""
        await websocket.accept()
        session_id: str | None = None

        try:
            # Send server metadata
            # Metadata must be compatible with DreamZero's PolicyServerConfig
            # so that test_client_AR.py can do PolicyServerConfig(**metadata)
            metadata = {
                "image_resolution": (180, 320),
                "n_external_cameras": 2,
                "needs_wrist_camera": True,
                "needs_stereo_camera": False,
                "needs_session_id": True,
                "action_space": "joint_position",
            }
            await websocket.send_bytes(_pack(metadata))

            # Main loop: receive observation, infer, send actions
            while True:
                try:
                    raw = await asyncio.wait_for(
                        websocket.receive_bytes(),
                        timeout=self._idle_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning("Session %s idle timeout", session_id)
                    break

                msg = _unpack(raw)
                endpoint = msg.get("endpoint", "infer")
                session_id = msg.get("session_id", "default")

                # Remove endpoint from obs before processing (DreamZero convention)
                del msg["endpoint"]

                if endpoint == "reset":
                    self._session_store.reset(session_id)
                    # DreamZero protocol: reset returns a plain string, not msgpack
                    await websocket.send("reset successful")
                    continue

                # endpoint == "infer"
                time.perf_counter()

                session = self._session_store.get_or_create(session_id, factory=DreamZeroSessionState)

                # Build OmniDiffusionRequest from observation
                t_preprocess = time.perf_counter()
                request = self._build_request(msg, session)
                (time.perf_counter() - t_preprocess) * 1000

                # Run inference (sync engine call in thread pool)
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self._engine.step, request)

                # Extract actions from result
                actions = self._extract_actions(result)

                # DreamZero protocol: infer returns msgpack(ndarray) directly,
                # not a dict. test_client_AR.py expects raw ndarray from unpackb().
                await websocket.send_bytes(_pack(actions))

        except WebSocketDisconnect:
            logger.info("Session %s disconnected", session_id)
        except Exception:
            logger.exception("Error in session %s", session_id)
        finally:
            if session_id:
                self._session_store.destroy(session_id)

    def _build_request(self, msg: dict, session: DreamZeroSessionState) -> Any:
        """Convert OpenPI-style observation dict to OmniDiffusionRequest.

        Key mapping:
            observation/exterior_image_0_left → video.exterior_image_1_left
            observation/exterior_image_1_left → video.exterior_image_2_left
            observation/wrist_image_left → video.wrist_image_left
            observation/joint_position → state.joint_position
            observation/gripper_position → state.gripper_position
            prompt → annotation.language.action_text
        """
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        prompt = msg.get("prompt", "")

        extra_args = {
            "reset": session.needs_reset,
            "session_id": session.session_id,
            "images": {
                "exterior_image_1_left": msg.get("observation/exterior_image_0_left"),
                "exterior_image_2_left": msg.get("observation/exterior_image_1_left"),
                "wrist_image_left": msg.get("observation/wrist_image_left"),
            },
            "state": {
                "joint_position": msg.get("observation/joint_position"),
                "gripper_position": msg.get("observation/gripper_position"),
            },
        }

        sampling_params = OmniDiffusionSamplingParams(
            extra_args=extra_args,
        )

        return OmniDiffusionRequest(
            prompts=[prompt],
            sampling_params=sampling_params,
            request_ids=[f"world-{session.session_id}-{session.call_count}"],
        )

    def _extract_actions(self, result: Any) -> np.ndarray:
        """Extract action ndarray from DiffusionOutput."""
        if hasattr(result, "__iter__"):
            result = list(result)
            if result:
                result = result[0]

        if hasattr(result, "custom_output"):
            actions = result.custom_output.get("actions")
            if actions is not None:
                if hasattr(actions, "numpy"):
                    return actions.numpy()
                return np.asarray(actions)

        # Fallback: empty actions
        logger.warning("No actions in result, returning zeros")
        return np.zeros((24, 8), dtype=np.float32)
