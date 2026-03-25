# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Session state management for stateful world model inference.

DreamZero (and future world models) maintain KV cache across multiple
inference calls within a session. This module manages session lifecycle
(create, reuse, reset, expire, destroy).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class WorldSessionState:
    """Base session state for world model inference."""

    session_id: str
    call_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)
    needs_reset: bool = True  # First call always resets KV cache

    def touch(self) -> None:
        self.last_active_at = time.time()
        self.call_count += 1
        # After first call, no auto-reset
        self.needs_reset = False

    def mark_reset(self) -> None:
        self.needs_reset = True
        self.call_count = 0


@dataclass
class DreamZeroSessionState(WorldSessionState):
    """DreamZero-specific session state with frame buffers for 3 cameras."""

    exterior_image_1_left: list[np.ndarray] = field(default_factory=list)
    exterior_image_2_left: list[np.ndarray] = field(default_factory=list)
    wrist_image_left: list[np.ndarray] = field(default_factory=list)

    def mark_reset(self) -> None:
        super().mark_reset()
        self.exterior_image_1_left.clear()
        self.exterior_image_2_left.clear()
        self.wrist_image_left.clear()


class WorldSessionStore:
    """Thread-safe session store with TTL-based expiration."""

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self._sessions: dict[str, WorldSessionState] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def get_or_create(
        self,
        session_id: str,
        factory: type[WorldSessionState] = DreamZeroSessionState,
    ) -> WorldSessionState:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = factory(session_id=session_id)
                logger.info("Created session %s", session_id)
            session = self._sessions[session_id]
            session.touch()
            return session

    def reset(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].mark_reset()
                logger.info("Reset session %s", session_id)

    def destroy(self, session_id: str) -> None:
        with self._lock:
            if self._sessions.pop(session_id, None) is not None:
                logger.info("Destroyed session %s", session_id)

    def cleanup_expired(self) -> int:
        now = time.time()
        expired = []
        with self._lock:
            for sid, session in self._sessions.items():
                if now - session.last_active_at > self._ttl:
                    expired.append(sid)
            for sid in expired:
                del self._sessions[sid]
        if expired:
            logger.info("Expired %d sessions: %s", len(expired), expired)
        return len(expired)
