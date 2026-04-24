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

"""Relative-frame wrappers for dual-arm environments with 14D tcp_pose.

These wrappers mirror :class:`RelativeFrame` and :class:`RelativeTargetFrame`
but operate on the concatenated ``(14,)`` tcp_pose produced by
dual-arm real-world environments.  Each arm's adjoint / homogeneous transform
is computed independently.
"""

import gymnasium as gym
import numpy as np
from gymnasium import Env
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.franka.utils import (
    construct_adjoint_matrix,
    construct_homogeneous_matrix,
)

NUM_ARMS = 2


class DualRelativeFrame(gym.Wrapper):
    """Transform observations and actions between base and end-effector frames
    for a dual-arm environment with ``(14,)`` tcp_pose.

    The adjoint matrix is maintained per arm.  Actions are expected to be
    ``(12,)`` (no gripper) or ``(14,)`` (with gripper), laid out as
    ``[left_6d, (grip), right_6d, (grip)]``.
    """

    def __init__(self, env: Env, include_relative_pose=True):
        super().__init__(env)
        self.adjoint_matrices = [np.zeros((6, 6)) for _ in range(NUM_ARMS)]

        self.include_relative_pose = include_relative_pose
        if self.include_relative_pose:
            self.T_b_r_invs = [np.zeros((4, 4)) for _ in range(NUM_ARMS)]

    def step(self, action: np.ndarray):
        transformed_action = self.transform_action(action)
        obs, reward, done, truncated, info = self.env.step(transformed_action)

        if "intervene_action" in info:
            info["intervene_action"] = self.transform_action_inv(
                info["intervene_action"]
            )

        self._update_adjoint(obs["state"]["tcp_pose"])
        transformed_obs = self.transform_observation(obs)
        return transformed_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        tcp_pose = obs["state"]["tcp_pose"]

        self._update_adjoint(tcp_pose)
        if self.include_relative_pose:
            for arm in range(NUM_ARMS):
                pose7 = tcp_pose[arm * 7 : arm * 7 + 7]
                self.T_b_r_invs[arm] = np.linalg.inv(
                    construct_homogeneous_matrix(pose7)
                )

        return self.transform_observation(obs), info

    # -------------------------------------------------------------- #

    def _update_adjoint(self, tcp_pose: np.ndarray):
        for arm in range(NUM_ARMS):
            self.adjoint_matrices[arm] = construct_adjoint_matrix(
                tcp_pose[arm * 7 : arm * 7 + 7]
            )

    def _right_arm_motion_slice(self, action: np.ndarray) -> slice:
        """Return the slice for the right arm's 6D motion in *action*.

        With gripper channels the layout is ``[left_6, grip, right_6, grip]``
        (len 14, right starts at 7).  Without grippers it is ``[left_6, right_6]``
        (len 12, right starts at 6).
        """
        start = 7 if len(action) == 14 else 6
        return slice(start, start + 6)

    def transform_observation(self, obs: dict) -> dict:
        """Transform dual-arm observations from base frame to end-effector frame."""
        tcp_pose = obs["state"]["tcp_pose"]
        tcp_vel = obs["state"].get("tcp_vel")

        out_pose_parts = []
        out_vel_parts = []
        for arm in range(NUM_ARMS):
            pose7 = tcp_pose[arm * 7 : arm * 7 + 7]
            adj_inv = np.linalg.inv(self.adjoint_matrices[arm])

            if tcp_vel is not None:
                vs, ve = arm * 6, arm * 6 + 6
                out_vel_parts.append(adj_inv @ tcp_vel[vs:ve])

            if self.include_relative_pose:
                T_b_o = construct_homogeneous_matrix(pose7)
                T_r_o = self.T_b_r_invs[arm] @ T_b_o
                p = T_r_o[:3, 3]
                q = R.from_matrix(T_r_o[:3, :3].copy()).as_quat()
                out_pose_parts.append(np.concatenate((p, q)))

        if self.include_relative_pose:
            obs["state"]["tcp_pose"] = np.concatenate(out_pose_parts)
        if out_vel_parts:
            obs["state"]["tcp_vel"] = np.concatenate(out_vel_parts)
        return obs

    def transform_action(self, action: np.ndarray) -> np.ndarray:
        """Transform action from end-effector frame to base frame."""
        action = np.array(action)
        action[:6] = self.adjoint_matrices[0] @ action[:6]
        rs = self._right_arm_motion_slice(action)
        action[rs] = self.adjoint_matrices[1] @ action[rs]
        return action

    def transform_action_inv(self, action: np.ndarray) -> np.ndarray:
        """Transform action from base frame to end-effector frame."""
        action = np.array(action)
        action[:6] = np.linalg.inv(self.adjoint_matrices[0]) @ action[:6]
        rs = self._right_arm_motion_slice(action)
        action[rs] = np.linalg.inv(self.adjoint_matrices[1]) @ action[rs]
        return action


class DualRelativeTargetFrame(DualRelativeFrame):
    """Like :class:`DualRelativeFrame`, but the adjoint is computed from the
    *target* end-effector pose rather than the current pose on reset."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        target = self.env.target_ee_pose  # (14,) quaternion form
        for arm in range(NUM_ARMS):
            pose7 = target[arm * 7 : arm * 7 + 7]
            self.adjoint_matrices[arm] = construct_adjoint_matrix(pose7)
            if self.include_relative_pose:
                self.T_b_r_invs[arm] = np.linalg.inv(
                    construct_homogeneous_matrix(pose7)
                )

        return self.transform_observation(obs), info
