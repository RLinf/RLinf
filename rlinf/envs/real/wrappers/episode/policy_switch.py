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

from typing import Any, SupportsFloat

from gymnasium.core import ActType, Env, ObsType

from .session import KeyboardSession


class KeyboardRLTPolicySwitchWrapper(KeyboardSession):
    """Press ``b`` to enter the RLT critical phase."""

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self._rlt_switch_flags = False

    @property
    def rlt_switch_flags(self) -> bool:
        """Return whether the Stage2 actor is active."""
        return self._rlt_switch_flags

    def begin_episode(self) -> None:
        """Restore the Stage1 policy at the start of an episode."""
        self._rlt_switch_flags = False

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        event: str | None = None
        for key in self.presses():
            if key == "b":
                if not self._rlt_switch_flags:
                    event = "enter_actor"
                    self._rlt_switch_flags = True
                    self.log(
                        "Pedal 'b' pressed; switching RLT rollout to Stage2 actor."
                    )
                else:
                    event = "actor_already_active"

        info["rlt_switch_flags"] = self._rlt_switch_flags
        info["rlt_policy_switch_event"] = event
        return obs, reward, terminated, truncated, info
