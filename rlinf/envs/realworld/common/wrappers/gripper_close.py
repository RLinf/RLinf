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


class GripperCloseEnv(gym.ActionWrapper):
    """Force the gripper channel closed without shrinking the action space.

    Keeps the full 7D action vector (arm 6D + gripper). The 7th dim is always
    set to -1 (closed under Franka's binary gripper convention). Teleop or
    policy may still send 6D; missing gripper is padded as -1.

    Pair with ``env.no_gripper=True`` so task ``go_to_rest`` does not open the
    jaw on reset; this wrapper only constrains runtime actions.
    """

    _GRIPPER_CLOSED = -1.0

    def __init__(self, env):
        super().__init__(env)
        ub = self.env.action_space
        assert ub.shape == (7,), f"expected 7D action space, got {ub.shape}"
        # Keep 7D so intervene_action / policy tensors stay aligned.
        self.action_space = ub

    def _force_closed(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] == 7:
            out = action.copy()
        elif action.shape[0] == 6:
            out = np.zeros((7,), dtype=np.float32)
            out[:6] = action
        else:
            raise ValueError(
                "GripperCloseEnv expects action of shape (6,) or (7,), "
                f"got {action.shape}"
            )
        out[6] = self._GRIPPER_CLOSED
        return out

    def action(self, action: np.ndarray) -> np.ndarray:
        return self._force_closed(action)

    def step(self, action):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if "intervene_action" in info and info["intervene_action"] is not None:
            info["intervene_action"] = self._force_closed(info["intervene_action"])
        return obs, rew, done, truncated, info
