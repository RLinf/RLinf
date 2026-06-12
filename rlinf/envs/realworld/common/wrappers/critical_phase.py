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

"""Critical-phase metadata for realworld RLT rollouts."""

from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener


class CriticalPhaseWrapper(gym.Wrapper):
    """Adds RLT critical-phase state to realworld env ``info``.

    ``critical_phase`` means the actor is responsible immediately after reset.
    ``full_task`` means the reference/base policy runs the prefix, and replay
    recording starts only after the operator enters the critical phase.
    """

    VALID_TASK_MODES = {"critical_phase", "full_task"}

    def __init__(
        self,
        env: gym.Env,
        *,
        task_mode: str = "critical_phase",
        critical_phase_key: str = "v",
        record_prefix_before_critical_phase: bool = False,
    ):
        super().__init__(env)
        self.task_mode = str(task_mode)
        if self.task_mode not in self.VALID_TASK_MODES:
            raise ValueError(
                "Unsupported realworld RLT task_mode. Expected one of "
                f"{sorted(self.VALID_TASK_MODES)}, got {self.task_mode!r}."
            )
        self.critical_phase_key = str(critical_phase_key).lower()
        self.record_prefix_before_critical_phase = bool(
            record_prefix_before_critical_phase
        )
        self.in_critical_phase = self.task_mode == "critical_phase"
        self.critical_phase_started_once = self.in_critical_phase
        base_env = getattr(env, "unwrapped", env)
        config = getattr(base_env, "config", None)
        self._listener = (
            None
            if bool(getattr(config, "is_dummy", False))
            or self.task_mode != "full_task"
            else KeyboardListener()
        )
        self._key_was_down = False

    def reset(self, *, seed=None, options=None):
        self.in_critical_phase = self.task_mode == "critical_phase"
        self.critical_phase_started_once = self.in_critical_phase
        self._key_was_down = False
        return self.env.reset(seed=seed, options=options)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self._poll_critical_phase_key()
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._inject_policy_info(info)
        return observation, reward, terminated, truncated, info

    def enter_critical_phase(self) -> None:
        self.in_critical_phase = True
        self.critical_phase_started_once = True

    def _poll_critical_phase_key(self) -> None:
        if self._listener is None:
            return
        key = self._listener.get_key()
        is_down = key == self.critical_phase_key
        if is_down and not self._key_was_down:
            self.enter_critical_phase()
            print(
                "Critical phase entered via key "
                f"{self.critical_phase_key!r}; actor control/replay is now enabled."
            )
        self._key_was_down = is_down

    def _inject_policy_info(self, info: dict[str, Any]) -> None:
        record_transition = (
            self.record_prefix_before_critical_phase or self.in_critical_phase
        )
        policy_info = info.get("policy_info")
        if not isinstance(policy_info, dict):
            policy_info = {}
            info["policy_info"] = policy_info
        policy_info["in_critical_phase"] = bool(self.in_critical_phase)
        policy_info["critical_phase_started"] = bool(
            self.critical_phase_started_once
        )
        policy_info["record_transition"] = bool(record_transition)
        info["in_critical_phase"] = policy_info["in_critical_phase"]
        info["record_transition"] = policy_info["record_transition"]
