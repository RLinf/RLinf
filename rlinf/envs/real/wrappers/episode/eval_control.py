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
"""Keyboard-gated wrapper for autonomous policy eval.

Controls: ``a`` starts a rollout from idle; ``c`` ends with reward=1
("success"); ``b`` ends with reward=0 ("failure"). On end, returns
``terminated=True`` so the outer ``auto_reset`` can return the robot home.
"""

import time
from typing import Any, Optional, SupportsFloat

from gymnasium.core import ActType, Env, ObsType

from .session import KeyboardAbort, KeyboardSession


class KeyboardEvalControlWrapper(KeyboardSession):
    """Keyboard-controlled start, pause, resume, and stop for eval rollouts."""

    IDLE_POLL_S = 0.05
    WAIT_HEARTBEAT_S = 10.0
    REQUIRED_KEY_NAMES = ("KEY_A", "KEY_B", "KEY_C", "KEY_P", "KEY_Q", "KEY_R")

    def __init__(self, env: Env) -> None:
        super().__init__(env, required_key_names=self.REQUIRED_KEY_NAMES)
        self._running = False
        self._paused = False
        self._last_obs: Any = None
        self._last_action: Any = None

    @staticmethod
    def _operator_help() -> str:
        return (
            "Operator controls: [a] start | [p] pause/hold | [r] resume | "
            "[c] success | [b] failure | [q] abort and exit | [Ctrl-C] emergency stop."
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        # Reset first, then wait for explicit operator confirmation.
        self.drain()
        self._running = False
        self._paused = False
        self._last_action = None
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = obs
        # Emit a heartbeat while the homed robot waits for the start signal.
        self.operator_log(
            "Arms homed and idle. Arrange the scene, then press [a] to start."
        )
        self.operator_log(self._operator_help())
        last_heartbeat = time.monotonic()
        while True:
            time.sleep(self.IDLE_POLL_S)
            now = time.monotonic()
            if now - last_heartbeat >= self.WAIT_HEARTBEAT_S:
                last_heartbeat = now
                self.operator_log("Waiting for [a] start. " + self._operator_help())
            for key in self.listener.pop_pressed_keys():
                if key == "q":
                    self.operator_log("Abort requested; stopping evaluation.")
                    raise KeyboardAbort("Operator requested evaluation abort.")
                if key == "a":
                    self._running = True
                    self.operator_log(
                        "Started rollout. Press [p] to pause and hold the arm."
                    )
                    return obs, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if not self._running:
            # Keep the robot idle while polling for the start signal.
            time.sleep(self.IDLE_POLL_S)
            for key in self.presses():
                if key == "q":
                    self.operator_log("Abort requested; stopping evaluation.")
                    raise KeyboardAbort("Operator requested evaluation abort.")
                if key == "a":
                    self._running = True
                    self._paused = False
                    self.operator_log(
                        "Started rollout. Press [p] to pause and hold the arm."
                    )
                    return self._idle_response(event="start")
            return self._idle_response(event=None)

        events = list(self.presses())
        for key in events:
            if key == "q":
                self.operator_log("Abort requested; stopping evaluation.")
                raise KeyboardAbort("Operator requested evaluation abort.")
            if key == "p" and not self._paused:
                self._paused = True
                self.operator_log(
                    "Paused: holding the latest arm command. Press [r] to resume."
                )
            elif key == "r" and self._paused:
                self._paused = False
                self.operator_log("Resumed rollout. Press [p] to pause again.")

        if self._paused and self._last_action is not None:
            action = self._last_action
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs
        self._last_action = action

        terminated = False
        truncated = False

        result: str | None = None
        for key in events:
            if key == "c":
                terminated = True
                reward = 1.0
                result = "success"
                self._running = False
                self._paused = False
                self.operator_log("Marked episode SUCCESS; returning to idle.")
                break
            if key == "b":
                terminated = True
                reward = 0.0
                result = "failure"
                self._running = False
                self._paused = False
                self.operator_log("Marked episode FAILURE; returning to idle.")
                break

        info["eval_phase"] = (
            "paused" if self._paused else ("rec" if self._running else "pre")
        )
        info["eval_result"] = result
        return obs, reward, terminated, truncated, info

    def _idle_response(
        self, event: str | None
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        info = {"eval_phase": "pre", "eval_event": event, "eval_result": None}
        return self._last_obs, 0.0, False, False, info
