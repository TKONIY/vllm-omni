# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _to_float32_array(value: Any, *, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D array, but got shape {array.shape}.")
    return array


@dataclass
class LingbotWorldFastControlChunk:
    """Normalized control chunk for one realtime turn."""

    poses: np.ndarray
    intrinsics: np.ndarray
    wasd_action: np.ndarray | None = None
    ijkl_action: np.ndarray | None = None
    control_type: str = "cam"

    @property
    def frame_count(self) -> int:
        return int(self.poses.shape[0])


def normalize_lingbot_control_chunk(payload: dict[str, Any]) -> LingbotWorldFastControlChunk:
    poses = _to_float32_array(payload.get("poses"), ndim=3, name="poses")
    if poses.shape[-2:] != (4, 4):
        raise ValueError(f"poses must have shape [F, 4, 4], but got {poses.shape}.")

    intrinsics = _to_float32_array(payload.get("intrinsics"), ndim=2, name="intrinsics")
    if intrinsics.shape[-1] != 4:
        raise ValueError(f"intrinsics must have shape [F, 4] or [1, 4], but got {intrinsics.shape}.")
    if intrinsics.shape[0] == 1:
        intrinsics = np.repeat(intrinsics, poses.shape[0], axis=0)
    if intrinsics.shape[0] != poses.shape[0]:
        raise ValueError(
            "intrinsics frame count must match poses frame count "
            f"({intrinsics.shape[0]} != {poses.shape[0]})."
        )

    wasd_action = None
    raw_wasd_action = payload.get("wasd_action", payload.get("action"))
    if raw_wasd_action is not None:
        wasd_action = _to_float32_array(raw_wasd_action, ndim=2, name="wasd_action")
        if wasd_action.shape[0] != poses.shape[0]:
            raise ValueError(
                "wasd_action frame count must match poses frame count "
                f"({wasd_action.shape[0]} != {poses.shape[0]})."
            )

    ijkl_action = None
    raw_ijkl_action = payload.get("ijkl_action")
    if raw_ijkl_action is not None:
        ijkl_action = _to_float32_array(raw_ijkl_action, ndim=2, name="ijkl_action")
        if ijkl_action.shape[0] != poses.shape[0]:
            raise ValueError(
                "ijkl_action frame count must match poses frame count "
                f"({ijkl_action.shape[0]} != {poses.shape[0]})."
            )

    control_type = "act" if wasd_action is not None else "cam"
    return LingbotWorldFastControlChunk(
        poses=poses,
        intrinsics=intrinsics,
        wasd_action=wasd_action,
        ijkl_action=ijkl_action,
        control_type=control_type,
    )


@dataclass
class LingbotWorldFastSessionConfig:
    """Stable session parameters that change only on reset."""

    session_id: str
    rendered_prompt: str
    text_layers: dict[str, str]
    width: int
    height: int
    fps: int
    chunk_size: int
    seed: int | None = None
    shift: float | None = None
    max_attention_size: int | None = None

    @property
    def signature(self) -> tuple[Any, ...]:
        return (
            self.rendered_prompt,
            tuple(self.text_layers.items()),
            self.width,
            self.height,
            self.fps,
            self.chunk_size,
            self.seed,
            self.shift,
            self.max_attention_size,
        )


@dataclass
class LingbotWorldFastSessionState:
    """Persistent causal generation state across websocket turns."""

    config: LingbotWorldFastSessionConfig
    control_history: list[LingbotWorldFastControlChunk] = field(default_factory=list)
    current_chunk_index: int = 0
    generated_frame_count: int = 0
    prompt_changed: bool = True
    image_changed: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def append_control(self, chunk: LingbotWorldFastControlChunk) -> None:
        self.control_history.append(chunk)

    def reset_runtime(self) -> None:
        self.control_history.clear()
        self.current_chunk_index = 0
        self.generated_frame_count = 0
        self.extra.clear()
        self.prompt_changed = True
        self.image_changed = True

    @property
    def total_control_frames(self) -> int:
        return int(sum(chunk.frame_count for chunk in self.control_history))

    def mark_chunk_generated(self, *, produced_frames: int) -> None:
        self.current_chunk_index += 1
        self.generated_frame_count += int(produced_frames)
        self.prompt_changed = False
        self.image_changed = False
