# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from PIL import Image

from vllm_omni.entrypoints.openai.image_api_utils import parse_size


def render_text_layers(
    layers: Mapping[str, Any] | None,
    *,
    fallback_prompt: str | None = None,
) -> str:
    """Render hierarchical/layered text into a single prompt string.

    The websocket protocol preserves the original layered structure under
    ``session.text_layers`` while the backend receives a deterministic rendered
    prompt string. The rendering order follows the user-provided JSON order.
    """

    rendered_parts: list[str] = []
    if layers is not None:
        for key, value in layers.items():
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            rendered_parts.append(text)

    if fallback_prompt:
        fallback = fallback_prompt.strip()
        if fallback and fallback not in rendered_parts:
            rendered_parts.append(fallback)

    return "\n\n".join(rendered_parts)


@dataclass
class RealtimeVideoInputImage:
    """Normalized session image input."""

    image: Image.Image | None = None
    source: str | None = None

    @classmethod
    def from_event(cls, payload: Mapping[str, Any] | None) -> "RealtimeVideoInputImage":
        if payload is None:
            return cls()

        if "b64_json" in payload and payload["b64_json"] is not None:
            raw = base64.b64decode(str(payload["b64_json"]))
            from io import BytesIO

            image = Image.open(BytesIO(raw)).convert("RGB")
            return cls(image=image, source="b64_json")

        if "image" in payload and isinstance(payload["image"], Image.Image):
            return cls(image=payload["image"].convert("RGB"), source="image")

        if "image_url" in payload:
            return cls(source="image_url")

        if "file_id" in payload:
            return cls(source="file_id")

        return cls()


@dataclass
class RealtimeVideoSession:
    """Connection-local realtime video session state."""

    id: str = field(default_factory=lambda: f"rtvideo_{uuid4().hex}")
    model: str | None = None
    prompt: str = ""
    text_layers: dict[str, str] = field(default_factory=dict)
    rendered_prompt: str = ""
    width: int = 832
    height: int = 480
    fps: int = 16
    num_frames: int = 13
    chunk_size: int = 3
    seed: int | None = None
    shift: float | None = None
    max_attention_size: int | None = None
    image: RealtimeVideoInputImage = field(default_factory=RealtimeVideoInputImage)
    control: list[dict[str, Any]] = field(default_factory=list)
    generation_index: int = 0
    needs_reset: bool = True

    @property
    def size(self) -> str:
        return f"{self.width}x{self.height}"

    def apply_update(self, payload: Mapping[str, Any]) -> None:
        if "model" in payload:
            self.model = payload.get("model")

        if "prompt" in payload and payload.get("prompt") is not None:
            self.prompt = str(payload["prompt"])

        text_layers = payload.get("text_layers")
        if isinstance(text_layers, Mapping):
            self.text_layers = {
                str(key): str(value)
                for key, value in text_layers.items()
                if value is not None and str(value).strip()
            }

        size = payload.get("size")
        if isinstance(size, str) and "x" in size:
            self.width, self.height = parse_size(size)

        for key in ("width", "height", "fps", "num_frames", "chunk_size"):
            value = payload.get(key)
            if value is not None:
                setattr(self, key, int(value))

        if "seed" in payload:
            seed = payload.get("seed")
            self.seed = None if seed is None else int(seed)

        if "shift" in payload:
            shift = payload.get("shift")
            self.shift = None if shift is None else float(shift)

        if "max_attention_size" in payload:
            value = payload.get("max_attention_size")
            self.max_attention_size = None if value is None else int(value)

        self.rendered_prompt = render_text_layers(self.text_layers, fallback_prompt=self.prompt)
        self.needs_reset = True

    def append_control(self, payload: Mapping[str, Any]) -> None:
        item = {str(key): value for key, value in payload.items()}
        self.control.append(item)

    def reset_buffers(self) -> None:
        self.control.clear()
        self.generation_index = 0
        self.needs_reset = True


@dataclass
class RealtimeVideoGenerationRequest:
    """One generation turn assembled from websocket session state."""

    session_id: str
    request_id: str
    rendered_prompt: str
    text_layers: dict[str, str]
    image: Image.Image | None
    control: list[dict[str, Any]]
    width: int
    height: int
    fps: int
    num_frames: int
    chunk_size: int
    seed: int | None
    shift: float | None
    max_attention_size: int | None
    reset: bool = False

    @classmethod
    def from_session(cls, session: RealtimeVideoSession) -> "RealtimeVideoGenerationRequest":
        request_id = f"rtvideo-{session.id}-{session.generation_index}"
        return cls(
            session_id=session.id,
            request_id=request_id,
            rendered_prompt=session.rendered_prompt or session.prompt,
            text_layers=dict(session.text_layers),
            image=session.image.image,
            control=list(session.control),
            width=session.width,
            height=session.height,
            fps=session.fps,
            num_frames=session.num_frames,
            chunk_size=session.chunk_size,
            seed=session.seed,
            shift=session.shift,
            max_attention_size=session.max_attention_size,
            reset=session.needs_reset,
        )
