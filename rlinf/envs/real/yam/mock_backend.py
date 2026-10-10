# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""In-memory YAM backends for unit tests and ``is_dummy`` environments."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .types import YamArmState


class MockYamFollower:
    """A deterministic follower whose measured state tracks accepted commands."""

    def __init__(self) -> None:
        self.connected = False
        self.closed = False
        self.target = np.zeros(7, dtype=np.float64)

    def connect(self) -> None:
        self.connected = True
        self.closed = False

    def read_state(self) -> YamArmState:
        self._require_connected()
        return YamArmState(self.target[:6], float(self.target[6]), time.time())

    def command(self, target: np.ndarray) -> None:
        self._require_connected()
        self.target = np.asarray(target, dtype=np.float64).reshape(7).copy()

    def joint_limits(self) -> np.ndarray:
        """Return permissive finite limits matching the dummy env defaults."""
        return np.column_stack(
            [np.full(6, -np.pi, dtype=np.float64), np.full(6, np.pi)]
        )

    def hold(self) -> None:
        self._require_connected()

    def assert_healthy(self, max_feedback_age_s: float) -> None:
        del max_feedback_age_s
        self._require_connected()

    def close(self) -> None:
        self.connected = False
        self.closed = True

    def _require_connected(self) -> None:
        if not self.connected:
            raise RuntimeError("mock YAM follower is not connected")


class MockYamBackendFactory:
    """Factory retaining created devices so tests can drive their state."""

    def __init__(self) -> None:
        self.followers: list[MockYamFollower] = []

    def create_follower(self, device_config: Any) -> MockYamFollower:
        del device_config
        follower = MockYamFollower()
        self.followers.append(follower)
        return follower
