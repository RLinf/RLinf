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
"""Foot-pedal-gated wrapper for autonomous policy eval.

Pedal: ``a`` starts a rollout from idle; ``c`` ends with reward=1
("success"); ``b`` ends with reward=0 ("failure"). On end, returns
``terminated=True`` so the outer ``auto_reset`` can return the robot home.
"""

import time
from typing import Any, Optional, SupportsFloat

from gymnasium.core import ActType, Env, ObsType

from .session import KeyboardSession


class KeyboardEvalControlWrapper(KeyboardSession):
    """Foot-pedal-gated start/stop for autonomous policy eval rollouts."""

    IDLE_POLL_S = 0.05
    WAIT_HEARTBEAT_S = 10.0

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self._running = False
        self._last_obs: Any = None

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        # Reset first, then wait for explicit operator confirmation.
        self.drain()
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = obs
        # Emit a heartbeat while the homed robot waits for the start signal.
        self.log(
            "Arms homed and idle. Arrange the scene, then press pedal 'a' "
            "to start the next rollout (Ctrl-C to abort)."
        )
        last_heartbeat = time.monotonic()
        while True:
            time.sleep(self.IDLE_POLL_S)
            now = time.monotonic()
            if now - last_heartbeat >= self.WAIT_HEARTBEAT_S:
                last_heartbeat = now
                self.log("Still waiting for pedal 'a' to start the rollout...")
            for key in self.listener.pop_pressed_keys():
                if key == "a":
                    self._running = True
                    self.log("Pedal 'a' pressed; starting rollout.")
                    return obs, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if not self._running:
            # Keep the robot idle while polling for the start signal.
            time.sleep(self.IDLE_POLL_S)
            for key in self.presses():
                if key == "a":
                    self._running = True
                    return self._idle_response(event="start")
            return self._idle_response(event=None)

        # Forward policy actions until the operator reports an outcome.
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs

        terminated = False
        truncated = False

        result: str | None = None
        for key in self.presses():
            if key == "c":
                terminated = True
                reward = 1.0
                result = "success"
                self._running = False
                break
            if key == "b":
                terminated = True
                reward = 0.0
                result = "failure"
                self._running = False
                break

        info["eval_phase"] = "rec" if self._running else "pre"
        info["eval_result"] = result
        return obs, reward, terminated, truncated, info

    def _idle_response(
        self, event: str | None
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        info = {"eval_phase": "pre", "eval_event": event, "eval_result": None}
        return self._last_obs, 0.0, False, False, info
