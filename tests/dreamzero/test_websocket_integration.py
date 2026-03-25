# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration test: WebSocket endpoint matching DreamZero protocol exactly.

Protocol (from eval_utils/policy_server.py + test_client_AR.py):
  Connect:  server sends msgpack(PolicyServerConfig fields)
  Infer:    client sends msgpack(obs_dict with endpoint="infer")
            server sends msgpack(ndarray)  ← raw action array, NOT a dict
  Reset:    client sends msgpack({"endpoint": "reset"})
            server sends string "reset successful"  ← plain text, NOT msgpack
"""

import asyncio

import msgpack
import msgpack_numpy
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.testclient import TestClient


def _pack(obj):
    return msgpack.packb(obj, default=msgpack_numpy.encode)


def _unpack(data):
    return msgpack.unpackb(data, object_hook=msgpack_numpy.decode, raw=False)


class DreamZeroCompatHandler:
    """Handler that exactly matches DreamZero's WebsocketPolicyServer protocol."""

    def __init__(self, action_fn):
        self._action_fn = action_fn

    async def handle_session(self, websocket: WebSocket):
        await websocket.accept()
        try:
            # Send PolicyServerConfig as metadata
            metadata = {
                "image_resolution": (180, 320),
                "n_external_cameras": 2,
                "needs_wrist_camera": True,
                "needs_stereo_camera": False,
                "needs_session_id": True,
                "action_space": "joint_position",
            }
            await websocket.send_bytes(_pack(metadata))

            while True:
                raw = await asyncio.wait_for(websocket.receive_bytes(), timeout=5.0)
                msg = _unpack(raw)
                endpoint = msg.pop("endpoint", "infer")

                if endpoint == "reset":
                    # DreamZero: returns plain string
                    await websocket.send_text("reset successful")
                    continue

                # infer: return raw ndarray via msgpack
                actions = self._action_fn(msg)
                await websocket.send_bytes(_pack(actions))

        except (WebSocketDisconnect, asyncio.TimeoutError):
            pass


def _dummy_action(msg):
    return np.ones((24, 8), dtype=np.float32) * 0.5


def _build_app():
    app = FastAPI()
    handler = DreamZeroCompatHandler(action_fn=_dummy_action)

    @app.websocket("/v1/world/roboarena")
    async def ws(websocket: WebSocket):
        await handler.handle_session(websocket)

    return app


def test_metadata_matches_policy_server_config():
    """Metadata can be parsed by DreamZero's PolicyServerConfig(**metadata)."""
    app = _build_app()
    with TestClient(app).websocket_connect("/v1/world/roboarena") as ws:
        metadata = _unpack(ws.receive_bytes())
        # These are the exact fields PolicyServerConfig expects
        assert metadata["n_external_cameras"] == 2
        assert metadata["needs_wrist_camera"] is True
        assert metadata["needs_stereo_camera"] is False
        assert metadata["needs_session_id"] is True
        assert metadata["action_space"] == "joint_position"
        assert metadata["image_resolution"] == (180, 320) or metadata["image_resolution"] == [180, 320]
    print("Metadata matches PolicyServerConfig: OK")


def test_reset_returns_string():
    """Reset returns plain string 'reset successful' (not msgpack)."""
    app = _build_app()
    with TestClient(app).websocket_connect("/v1/world/roboarena") as ws:
        ws.receive_bytes()  # metadata
        ws.send_bytes(_pack({"endpoint": "reset", "session_id": "s1"}))
        # DreamZero client.reset() calls ws.recv() and gets a string
        resp = ws.receive_text()
        assert resp == "reset successful", f"Expected 'reset successful', got: {resp!r}"
    print("Reset returns string: OK")


def test_infer_returns_raw_ndarray():
    """Infer returns msgpack(ndarray), not msgpack(dict)."""
    app = _build_app()
    with TestClient(app).websocket_connect("/v1/world/roboarena") as ws:
        ws.receive_bytes()  # metadata
        obs = {
            "endpoint": "infer",
            "session_id": "s2",
            "observation/exterior_image_0_left": np.zeros((180, 320, 3), dtype=np.uint8),
            "observation/exterior_image_1_left": np.zeros((180, 320, 3), dtype=np.uint8),
            "observation/wrist_image_left": np.zeros((180, 320, 3), dtype=np.uint8),
            "observation/joint_position": np.zeros(7, dtype=np.float32),
            "observation/gripper_position": np.zeros(1, dtype=np.float32),
            "prompt": "pick up the red block",
        }
        ws.send_bytes(_pack(obs))
        raw = ws.receive_bytes()
        actions = _unpack(raw)
        # DreamZero: actions is a raw ndarray, NOT a dict
        assert isinstance(actions, np.ndarray), f"Expected ndarray, got {type(actions)}"
        assert actions.shape == (24, 8), f"Expected (24,8), got {actions.shape}"
        assert actions.dtype == np.float32
        assert np.allclose(actions, 0.5)
    print(f"Infer returns raw ndarray: OK — shape={actions.shape}")


def test_multi_round_like_test_client_ar():
    """Simulate test_client_AR.py: reset + N infer rounds."""
    app = _build_app()
    with TestClient(app).websocket_connect("/v1/world/roboarena") as ws:
        # Step 0: receive metadata
        metadata = _unpack(ws.receive_bytes())
        assert "n_external_cameras" in metadata

        session_id = "test-ar-session"

        # Step 1: initial infer (single frame)
        obs = {
            "endpoint": "infer",
            "session_id": session_id,
            "observation/exterior_image_0_left": np.zeros((180, 320, 3), dtype=np.uint8),
            "observation/exterior_image_1_left": np.zeros((180, 320, 3), dtype=np.uint8),
            "observation/wrist_image_left": np.zeros((180, 320, 3), dtype=np.uint8),
            "observation/joint_position": np.zeros(7, dtype=np.float32),
            "observation/gripper_position": np.zeros(1, dtype=np.float32),
            "prompt": "pick up the red block",
        }
        ws.send_bytes(_pack(obs))
        actions = _unpack(ws.receive_bytes())
        assert isinstance(actions, np.ndarray) and actions.shape == (24, 8)

        # Step 2-4: subsequent infers (multi-frame)
        for i in range(3):
            obs["observation/exterior_image_0_left"] = np.zeros((4, 180, 320, 3), dtype=np.uint8)
            obs["observation/exterior_image_1_left"] = np.zeros((4, 180, 320, 3), dtype=np.uint8)
            obs["observation/wrist_image_left"] = np.zeros((4, 180, 320, 3), dtype=np.uint8)
            obs["endpoint"] = "infer"
            ws.send_bytes(_pack(obs))
            actions = _unpack(ws.receive_bytes())
            assert isinstance(actions, np.ndarray) and actions.shape == (24, 8)

        # Step 5: reset (triggers video save in DreamZero)
        ws.send_bytes(_pack({"endpoint": "reset"}))
        resp = ws.receive_text()
        assert resp == "reset successful"

    print("Multi-round test_client_AR flow: OK (1 init + 3 chunks + reset)")


def test_msgpack_numpy_roundtrip():
    """numpy arrays survive pack/unpack."""
    arr = np.random.randn(24, 8).astype(np.float32)
    recovered = _unpack(_pack(arr))
    assert np.array_equal(arr, recovered)
    assert recovered.dtype == np.float32
    print("msgpack_numpy roundtrip: OK")


if __name__ == "__main__":
    test_metadata_matches_policy_server_config()
    test_reset_returns_string()
    test_infer_returns_raw_ndarray()
    test_multi_round_like_test_client_ar()
    test_msgpack_numpy_roundtrip()
    print("\nAll WebSocket integration tests passed!")
