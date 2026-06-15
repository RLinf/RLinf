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

"""GELLO intervention wrapper for absolute joint-target action spaces."""

from __future__ import annotations

import time

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.common.gello.gello_expert import GelloExpert


class GelloJointTargetIntervention(gym.ActionWrapper):
    """Override policy actions with GELLO absolute joint targets."""

    def __init__(self, env, port: str, gripper_enabled: bool = True):
        super().__init__(env)
        self.gripper_enabled = gripper_enabled
        self.expert = GelloExpert(port=port)
        self.last_intervene = 0.0

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        if not self.expert.ready:
            return action, False

        joints, gripper = self.expert.get_joint_action()
        expected_dim = int(np.prod(self.action_space.shape))
        joint_dim = expected_dim - (1 if self.gripper_enabled else 0)
        if joints.shape[0] < joint_dim:
            raise ValueError(
                "GELLO joint target intervention received too few joint values: "
                f"got {joints.shape[0]}, need {joint_dim} for action_dim={expected_dim}."
            )

        expert_a = np.asarray(action, dtype=np.float64).reshape(-1).copy()
        expert_a[:joint_dim] = joints[:joint_dim]
        gripper_active = False
        if self.gripper_enabled:
            expert_a[joint_dim] = float(np.asarray(gripper).reshape(-1)[0])
            gripper_active = bool(
                abs(expert_a[joint_dim] - action.reshape(-1)[joint_dim]) > 1e-3
            )

        expert_a = np.clip(expert_a, self.action_space.low, self.action_space.high)
        joint_active = bool(
            np.linalg.norm(
                expert_a[:joint_dim] - np.asarray(action).reshape(-1)[:joint_dim]
            )
            > 1e-3
        )
        if joint_active or gripper_active:
            self.last_intervene = time.time()

        if time.time() - self.last_intervene < 0.5:
            return expert_a.astype(action.dtype, copy=False), True

        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        return obs, rew, done, truncated, info
