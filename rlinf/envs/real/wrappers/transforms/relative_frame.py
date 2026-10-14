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

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import Env
from scipy.spatial.transform import Rotation as R

from rlinf.envs.real.utils.pose import (
    construct_adjoint_matrix,
    construct_homogeneous_matrix,
)


class RelativeFrame(gym.Wrapper):
    """Express Cartesian observations and actions in the end-effector frame.

    When ``include_relative_pose`` is enabled, ``tcp_pose`` is also expressed
    relative to the pose recorded at reset. The wrapped environment must expose
    a seven-value ``xyz + quaternion`` pose and at least six Cartesian action
    values.
    """

    #: Environment configuration flag that enables this wrapper.
    CONFIG_FLAG = "use_relative_frame"
    CONFIG_DEFAULT = True

    def __init__(self, env: Env, include_relative_pose: bool = True) -> None:
        super().__init__(env)
        self.adjoint_matrix = np.zeros((6, 6))

        self.include_relative_pose = include_relative_pose
        if self.include_relative_pose:
            # Transform from the base frame to the reset-relative frame.
            self.T_b_r_inv = np.zeros((4, 4))

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        # Convert the Cartesian action from end-effector to base frame.
        transformed_action = self.transform_action(action)

        obs, reward, done, truncated, info = self.env.step(transformed_action)

        # Report intervention actions in the wrapper's public frame.
        if "intervene_action" in info:
            info["intervene_action"] = self.transform_action_inv(
                info["intervene_action"]
            )

        # Update the frame transform from the latest pose.
        self.adjoint_matrix = construct_adjoint_matrix(obs["state"]["tcp_pose"])

        transformed_obs = self.transform_observation(obs)
        return transformed_obs, reward, done, truncated, info

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)

        self.adjoint_matrix = construct_adjoint_matrix(obs["state"]["tcp_pose"])
        if self.include_relative_pose:
            # Record the reset pose as the origin of the relative frame.
            self.T_b_r_inv = np.linalg.inv(
                construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            )

        return self.transform_observation(obs), info

    def transform_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Transform observations from the base to end-effector frame."""
        adjoint_inv = np.linalg.inv(self.adjoint_matrix)
        if "tcp_vel" in obs["state"]:
            obs["state"]["tcp_vel"] = adjoint_inv @ obs["state"]["tcp_vel"]

        if self.include_relative_pose:
            T_b_o = construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            T_r_o = self.T_b_r_inv @ T_b_o

            p_r_o = T_r_o[:3, 3]
            quat_r_o = R.from_matrix(T_r_o[:3, :3].copy()).as_quat()
            obs["state"]["tcp_pose"] = np.concatenate((p_r_o, quat_r_o))

        return obs

    def transform_action(self, action: np.ndarray) -> np.ndarray:
        """Transform an action from the end-effector to base frame."""
        # Copy because JAX may provide a read-only array.
        action = np.array(action)
        action[:6] = self.adjoint_matrix @ action[:6]
        return action

    def transform_action_inv(self, action: np.ndarray) -> np.ndarray:
        """Transform an action from the base to end-effector frame."""
        action = np.array(action)
        action[:6] = np.linalg.inv(self.adjoint_matrix) @ action[:6]
        return action


class RelativeTargetFrame(RelativeFrame):
    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)

        self.adjoint_matrix = construct_adjoint_matrix(self.env.target_ee_pose)
        if self.include_relative_pose:
            self.T_b_r_inv = np.linalg.inv(
                construct_homogeneous_matrix(self.env.target_ee_pose)
            )

        return self.transform_observation(obs), info
