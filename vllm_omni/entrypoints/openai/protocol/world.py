# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Protocol definitions for world model WebSocket interface."""

from __future__ import annotations

from pydantic import BaseModel, Field


class WorldModelMetadata(BaseModel):
    """Server metadata sent to client on connection."""

    image_resolution: list[int] = Field(default=[180, 320])
    n_cameras: int = 3
    action_dim: int = 8
    action_horizon: int = 24
