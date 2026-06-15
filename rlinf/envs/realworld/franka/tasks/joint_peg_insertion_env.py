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

"""Franka peg insertion environment with absolute joint-target actions."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.franka.franka_env import FrankaEnv, FrankaRobotConfig
from rlinf.envs.realworld.franka.franka_robot_state import FrankaRobotState

_FRANKA_JOINT_LIMIT_LOW = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    dtype=np.float64,
)
_FRANKA_JOINT_LIMIT_HIGH = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    dtype=np.float64,
)
_DEFAULT_FRANKA_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]


@dataclass
class FrankaJointPegInsertionConfig(FrankaRobotConfig):
    """Configuration for 8D realworld RLT joint-control peg insertion."""

    task_description: str = "insert the peg in the hole"
    target_ee_pose: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.0, 0.1, -3.14, 0.0, 0.0])
    )
    reset_ee_pose: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.0, 0.2, -3.14, 0.0, 0.0])
    )
    reward_threshold: np.ndarray = field(
        default_factory=lambda: np.array([0.015, 0.015, 0.03, 0.2, 0.2, 0.2])
    )
    joint_limit_low: np.ndarray = field(
        default_factory=lambda: _FRANKA_JOINT_LIMIT_LOW.copy()
    )
    joint_limit_high: np.ndarray = field(
        default_factory=lambda: _FRANKA_JOINT_LIMIT_HIGH.copy()
    )
    reset_joint_qpos: list[float] = field(
        default_factory=lambda: [0.0, -0.785, 0.0, -2.35, 0.0, 1.57, 0.785]
    )
    max_joint_delta: float | list[float] = 0.08
    joint_move_timeout: float = 1.5
    reset_gripper_action: float = -1.0
    check_orientation_success: bool = False
    joint_command_topic: str | None = "/joint_states_gripper"
    joint_command_joint_names: list[str] | None = field(
        default_factory=lambda: _DEFAULT_FRANKA_JOINT_NAMES.copy()
    )
    enable_gripper_penalty: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.joint_limit_low = np.asarray(self.joint_limit_low, dtype=np.float64)
        self.joint_limit_high = np.asarray(self.joint_limit_high, dtype=np.float64)
        if self.joint_limit_low.shape != (7,) or self.joint_limit_high.shape != (7,):
            raise ValueError(
                "Franka joint limits must both be 7D, got "
                f"{self.joint_limit_low.shape=} and {self.joint_limit_high.shape=}."
            )
        self.reset_joint_qpos = [
            float(value) for value in np.asarray(self.reset_joint_qpos).reshape(-1)
        ]
        if len(self.reset_joint_qpos) != 7:
            raise ValueError(
                "reset_joint_qpos must be 7D for Franka joint control, got "
                f"{len(self.reset_joint_qpos)}."
            )


class FrankaJointPegInsertionEnv(FrankaEnv):
    """Single Franka peg insertion task with 8D absolute joint-target actions.

    Action layout:
        ``actions[0:7]`` are absolute Franka arm joint targets in radians.
        ``actions[7]`` is the gripper command, using the same binary semantics
        as :class:`FrankaEnv`.
    """

    CONFIG_CLS = FrankaJointPegInsertionConfig

    def _init_action_obs_spaces(self):
        """Initialize joint-target action and RLT-compatible observation spaces."""
        self._joint_limit_low = np.asarray(
            self.config.joint_limit_low,
            dtype=np.float64,
        )
        self._joint_limit_high = np.asarray(
            self.config.joint_limit_high,
            dtype=np.float64,
        )

        action_low = np.concatenate([self._joint_limit_low, [-1.0]]).astype(np.float32)
        action_high = np.concatenate([self._joint_limit_high, [1.0]]).astype(
            np.float32
        )
        self.action_space = gym.spaces.Box(action_low, action_high)

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "arm_joint_position": gym.spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(7,),
                        ),
                        "arm_joint_velocity": gym.spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(7,),
                        ),
                        "gripper_position": gym.spaces.Box(-np.inf, np.inf, shape=(1,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        camera_info.name: gym.spaces.Box(
                            0,
                            255,
                            shape=(128, 128, 3),
                            dtype=np.uint8,
                        )
                        for camera_info in self._camera_infos
                    }
                ),
            }
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    def step(self, action: np.ndarray):
        """Execute one 8D absolute joint-target action."""
        start_time = time.time()

        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (8,):
            raise ValueError(
                "FrankaJointPegInsertionEnv expects 8D actions "
                "[7 joint targets, gripper], got "
                f"{action.shape}."
            )
        action = np.clip(action, self.action_space.low, self.action_space.high)

        current_q = np.asarray(
            self._franka_state.arm_joint_position,
            dtype=np.float64,
        )
        q_target = self._clip_joint_target(current_q, action[:7])
        gripper_action = float(action[7])
        executed_action = np.concatenate([q_target, [gripper_action]])

        is_gripper_action_effective = False
        if not self.config.is_dummy:
            self._move_joint_action(q_target)
            is_gripper_action_effective = self._end_effector_action(
                np.array([gripper_action], dtype=np.float64)
            )
        else:
            dt = 1.0 / max(float(self.config.step_frequency), 1e-6)
            self._franka_state.arm_joint_velocity = (q_target - current_q) / dt
            self._franka_state.arm_joint_position = q_target.copy()
            self._franka_state.gripper_position = gripper_action
            self._franka_state.gripper_open = (
                gripper_action >= self.config.binary_gripper_threshold
            )

        self._num_steps += 1
        step_time = time.time() - start_time
        time.sleep(max(0.0, (1.0 / self.config.step_frequency) - step_time))

        if not self.config.is_dummy:
            self._franka_state = self._controller.get_state().wait()[0]

        observation = self._get_observation()
        reward = self._calc_step_reward(observation, is_gripper_action_effective)
        terminated = (reward == 1.0) and (
            self._success_hold_counter >= self.config.success_hold_steps
        )
        truncated = self._num_steps >= self.config.max_num_steps
        reward *= self.config.reward_scale
        info = {"executed_action": executed_action.astype(np.float32)}
        return observation, reward, terminated, truncated, info

    def go_to_rest(self, joint_reset=False):
        """Reset using joint-space targets instead of Cartesian EE deltas."""
        if self.config.is_dummy:
            q_target = np.asarray(self.config.reset_joint_qpos, dtype=np.float64)
            self._franka_state.arm_joint_position = q_target
            self._franka_state.arm_joint_velocity = np.zeros(7)
            self._num_steps = 0
            return

        if joint_reset:
            self._controller.reset_joint(self.config.joint_reset_qpos).wait()
            time.sleep(0.5)

        self._interpolate_joint_move(
            np.asarray(self.config.reset_joint_qpos, dtype=np.float64),
            timeout=float(self.config.joint_move_timeout),
        )
        if not self._is_hand:
            self._end_effector_action(
                np.array([float(self.config.reset_gripper_action)], dtype=np.float64)
            )

    def _interpolate_move(self, pose: np.ndarray, timeout: float = 1.5):
        """Use joint reset during base ``FrankaEnv`` initialization."""
        del pose
        if self.config.is_dummy:
            return
        self._interpolate_joint_move(
            np.asarray(self.config.reset_joint_qpos, dtype=np.float64),
            timeout=timeout,
        )

    def _interpolate_joint_move(self, q_target: np.ndarray, timeout: float = 1.5):
        q_target = np.clip(q_target, self._joint_limit_low, self._joint_limit_high)
        if self.config.joint_command_topic is None:
            self._controller.reset_joint(q_target.astype(float).tolist()).wait()
            return

        self._franka_state: FrankaRobotState = self._controller.get_state().wait()[0]
        current_q = np.asarray(self._franka_state.arm_joint_position, dtype=np.float64)
        num_steps = max(1, int(timeout * self.config.step_frequency))
        for q in np.linspace(current_q, q_target, num_steps + 1)[1:]:
            self._move_joint_action(q)
            time.sleep(1.0 / self.config.step_frequency)
        self._franka_state = self._controller.get_state().wait()[0]

    def _move_joint_action(self, q_target: np.ndarray):
        self._clear_error()
        self._controller.move_joints(q_target.astype(np.float32)).wait()

    def _clip_joint_target(
        self,
        current_q: np.ndarray,
        q_target: np.ndarray,
    ) -> np.ndarray:
        q_target = np.clip(q_target, self._joint_limit_low, self._joint_limit_high)

        max_delta = np.asarray(self.config.max_joint_delta, dtype=np.float64)
        if max_delta.size == 1:
            max_delta = np.full(7, float(max_delta.reshape(-1)[0]))
        max_delta = np.abs(max_delta.reshape(-1))
        if max_delta.shape != (7,):
            raise ValueError(
                "max_joint_delta must be scalar or 7D, got "
                f"{max_delta.shape}."
            )
        finite_delta = np.isfinite(max_delta)
        if finite_delta.any():
            lower = current_q.copy()
            upper = current_q.copy()
            lower[finite_delta] -= max_delta[finite_delta]
            upper[finite_delta] += max_delta[finite_delta]
            q_target = np.clip(q_target, lower, upper)
        return np.clip(q_target, self._joint_limit_low, self._joint_limit_high)

    def _calc_step_reward(
        self,
        observation: dict[str, np.ndarray | FrankaRobotState],
        is_gripper_action_effective: bool = False,
    ) -> float:
        """Sparse success reward for realworld peg insertion."""
        if self.config.use_reward_model:
            reward = self._compute_reward_model(observation)
            if reward >= 1.0:
                self._success_hold_counter += 1
            else:
                self._success_hold_counter = 0
            return reward

        if self.config.is_dummy:
            return 0.0

        euler_angles = np.abs(
            R.from_quat(self._franka_state.tcp_pose[3:].copy()).as_euler("xyz")
        )
        position = np.hstack([self._franka_state.tcp_pose[:3], euler_angles])
        target_delta = np.abs(position - self.config.target_ee_pose)
        success_mask = target_delta <= self.config.reward_threshold
        is_success = bool(
            success_mask.all()
            if self.config.check_orientation_success
            else success_mask[:3].all()
        )

        if is_success:
            self._success_hold_counter += 1
            reward = 1.0
        else:
            self._success_hold_counter = 0
            if self.config.use_dense_reward:
                reward = float(np.exp(-500 * np.sum(np.square(target_delta[:3]))))
            else:
                reward = 0.0

        if self.config.enable_gripper_penalty and is_gripper_action_effective:
            reward -= self.config.gripper_penalty
        return reward

    def _get_observation(self) -> dict:
        observation = super()._get_observation()
        if self.config.is_dummy:
            return observation

        state = observation["state"]
        state["arm_joint_position"] = self._franka_state.arm_joint_position
        state["arm_joint_velocity"] = self._franka_state.arm_joint_velocity
        return copy.deepcopy(observation)
