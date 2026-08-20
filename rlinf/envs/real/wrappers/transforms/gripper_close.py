# Copyright 2025 The RLinf Authors.
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

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class GripperCloseEnv(gym.ActionWrapper):
    """
    Use this wrapper to task that requires the gripper to be closed
    """

    #: Applied when this env-config flag is set. A wrapper knowing its own
    #: switch is what lets one stack builder serve every robot.
    CONFIG_FLAG = "no_gripper"
    CONFIG_DEFAULT = True

    def __init__(self, env):
        super().__init__(env)
        ub = self.env.action_space
        assert ub.shape == (7,)
        self.action_space = Box(ub.low[:6], ub.high[:6])

    def action_parts(self):
        """The env's parts, minus the gripper this wrapper holds shut.

        A wrapper that changes the action space says so, for the same reason
        an env declares its own: whoever drives the action has to be told what
        the numbers mean, and here there is one fewer of them.
        """
        return tuple(
            part
            for part in self.env.get_wrapper_attr("action_parts")()
            if part.name not in ("end_effector", "hand")
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        new_action = np.zeros((7,), dtype=np.float32)
        new_action[:6] = action.copy()
        return new_action

    def step(self, action):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"][:6]
        return obs, rew, done, truncated, info
