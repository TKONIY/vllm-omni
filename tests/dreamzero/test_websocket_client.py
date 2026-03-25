# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration test: WebSocket client for DreamZero world model.

Tests that vllm-omni's /v1/world/openpi endpoint is protocol-compatible
with DreamZero's test_client_AR.py. Run against a live server.

Usage:
    python tests/dreamzero/test_websocket_client.py --url ws://localhost:8000/v1/world/openpi
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def _pack(obj):
    import msgpack
    import msgpack_numpy

    return msgpack.packb(obj, default=msgpack_numpy.encode)


def _unpack(data):
    import msgpack
    import msgpack_numpy

    return msgpack.unpackb(data, object_hook=msgpack_numpy.decode, raw=False)


def run_client_test(url: str, num_rounds: int = 3, save_path: str | None = None):
    """Connect to server, send observations, collect action history."""
    import websocket

    ws = websocket.create_connection(url)

    # Receive server metadata
    metadata = _unpack(ws.recv())
    print(f"Server metadata: {metadata}")

    action_history = []

    for i in range(num_rounds):
        # Build observation (fixed dummy data for reproducibility)
        rng = np.random.RandomState(seed=42 + i)
        obs = {
            "endpoint": "infer" if i > 0 else "reset",
            "session_id": "test-session-001",
            "observation/exterior_image_0_left": rng.randint(0, 255, (180, 320, 3), dtype=np.uint8),
            "observation/exterior_image_1_left": rng.randint(0, 255, (180, 320, 3), dtype=np.uint8),
            "observation/wrist_image_left": rng.randint(0, 255, (180, 320, 3), dtype=np.uint8),
            "observation/joint_position": rng.randn(7).astype(np.float32),
            "observation/gripper_position": rng.randn(1).astype(np.float32),
            "prompt": "pick up the red block",
        }

        # First round: reset, subsequent: infer
        if i > 0:
            obs["endpoint"] = "infer"

        t0 = time.perf_counter()
        ws.send_binary(_pack(obs))
        response = _unpack(ws.recv())
        dt = (time.perf_counter() - t0) * 1000

        actions = response.get("actions")
        timing = response.get("timing", {})
        print(
            f"Round {i}: actions shape={actions.shape if actions is not None else None}, "
            f"infer_ms={timing.get('infer_ms', '?'):.1f}, "
            f"round_trip_ms={dt:.1f}"
        )

        if actions is not None:
            action_history.append(actions)

    ws.close()

    if save_path:
        np.savez(save_path, *action_history)
        print(f"Saved action history to {save_path}")

    return action_history


def compare_histories(baseline_path: str, test_path: str, atol: float = 1e-5):
    """Compare two action histories for alignment."""
    baseline_data = np.load(baseline_path)
    baseline = [baseline_data[k] for k in sorted(baseline_data.keys())]
    test_data = np.load(test_path)
    test = [test_data[k] for k in sorted(test_data.keys())]

    assert len(baseline) == len(test), f"Length mismatch: {len(baseline)} vs {len(test)}"

    all_close = True
    for i, (b, t) in enumerate(zip(baseline, test)):
        if not np.allclose(b, t, atol=atol):
            max_diff = np.max(np.abs(b - t))
            print(f"Round {i}: MISMATCH (max_diff={max_diff:.6f})")
            all_close = False
        else:
            print(f"Round {i}: OK")

    if all_close:
        print("All rounds match!")
    else:
        print("ALIGNMENT FAILED — some rounds differ")

    return all_close


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8000/v1/world/openpi")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--save", default=None, help="Save action history to pickle file")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "TEST"), help="Compare two action history files")
    args = parser.parse_args()

    if args.compare:
        compare_histories(args.compare[0], args.compare[1])
    else:
        run_client_test(args.url, args.rounds, args.save)
