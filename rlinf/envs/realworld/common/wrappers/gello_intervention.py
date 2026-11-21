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

import time

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.common.gello.gello_expert import GelloExpert


class GelloIntervention(gym.ActionWrapper):
    """Action wrapper that overrides the policy action with GELLO teleop input.

    Args:
        env: The wrapped environment.
        port: Serial port of the GELLO device.  Must be provided
            (typically from the ``gello_port`` field in the env YAML config).
        gripper_enabled: Whether the gripper channel is present in the action
            space.  Determined by the ``no_gripper`` env config field.
    """

    def __init__(self, env, port: str, gripper_enabled: bool = True):
        super().__init__(env)

        self.gripper_enabled = gripper_enabled
        self.expert = GelloExpert(port=port)
        self.last_intervene = 0

    def action(self, action: np.ndarray) -> np.ndarray:
        if not self.expert.ready:
            return action, False

        target_pos, target_quat, target_gripper = self.expert.get_action()
        r_target = R.from_quat(target_quat.copy())
        tcp_pose = self.get_wrapper_attr("get_tcp_pose")()

        tcp_pos = tcp_pose[:3]
        tcp_quat = tcp_pose[3:]
        r_tcp = R.from_quat(tcp_quat.copy())

        r_delta = r_target * r_tcp.inv()
        delta_euler = r_delta.as_euler("xyz")

        delta_pos = target_pos - tcp_pos

        action_scale = self.get_wrapper_attr("get_action_scale")()
        delta_pos = delta_pos / action_scale[0]
        delta_euler = delta_euler / action_scale[1]

        expert_a = np.concatenate((delta_pos, delta_euler), axis=0)
        expert_a = np.clip(expert_a, -1.0, 1.0)

        gripper_active = False
        if self.gripper_enabled:
            target_gripper = target_gripper / action_scale[2]
            target_gripper = -(2 * target_gripper - 1.0)
            target_gripper = np.clip(target_gripper, -1.0, 1.0)
            gripper_active = np.abs(target_gripper).item() > 0.5
            expert_a = np.concatenate((expert_a, target_gripper), axis=0)

        if np.linalg.norm(expert_a[:6]) > 0.001 or gripper_active:
            self.last_intervene = time.time()

        if time.time() - self.last_intervene < 0.5:
            return expert_a, True

        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action

        return obs, rew, done, truncated, info
