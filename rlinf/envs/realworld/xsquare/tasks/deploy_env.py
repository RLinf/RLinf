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
from dataclasses import dataclass, field
from typing import Literal

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.xsquare.turtle2_env import Turtle2Env, Turtle2RobotConfig

NUM_ARMS = 2
POSE_DIM = 6
ACTION_DIM_PER_ARM = POSE_DIM + 1


@dataclass
class Turtle2DeployEnvConfig(Turtle2RobotConfig):
    use_camera_ids: list[int] = field(default_factory=lambda: [0, 1, 2])
    use_arm_ids: list[int] = field(default_factory=lambda: [0, 1])
    enforce_gripper_close: bool = False
    task_description: str = ""
    action_mode: Literal["absolute_pose", "relative_pose"] = "relative_pose"

    def __post_init__(self):
        self.target_ee_pose = np.asarray(self.target_ee_pose, dtype=np.float64).reshape(
            NUM_ARMS, POSE_DIM
        )
        self.reset_ee_pose = np.asarray(self.reset_ee_pose, dtype=np.float64).reshape(
            NUM_ARMS, POSE_DIM
        )
        self.ee_pose_limit_min = np.asarray(
            self.ee_pose_limit_min, dtype=np.float64
        ).reshape(NUM_ARMS, POSE_DIM)
        self.ee_pose_limit_max = np.asarray(
            self.ee_pose_limit_max, dtype=np.float64
        ).reshape(NUM_ARMS, POSE_DIM)
        self.reward_threshold = np.asarray(
            self.reward_threshold, dtype=np.float64
        ).reshape(NUM_ARMS, POSE_DIM)
        self.action_scale = np.asarray(self.action_scale, dtype=np.float64)


class Turtle2DeployEnv(Turtle2Env):
    CONFIG_CLS = Turtle2DeployEnvConfig

    def __init__(self, override_cfg, worker_info=None, hardware_info=None, env_idx=0):
        config = self.CONFIG_CLS(**override_cfg)
        super().__init__(config, worker_info, hardware_info, env_idx)

    @property
    def task_description(self):
        return self.config.task_description

    def _init_action_obs_spaces(self):
        super()._init_action_obs_spaces()
        self._relative_pose_action_space = self.action_space
        action_low = []
        action_high = []
        for arm_id in self.config.use_arm_ids:
            action_low.append(
                np.concatenate(
                    [
                        self.config.ee_pose_limit_min[arm_id],
                        np.array([self.config.gripper_width_limit_min]),
                    ]
                )
            )
            action_high.append(
                np.concatenate(
                    [
                        self.config.ee_pose_limit_max[arm_id],
                        np.array([self.config.gripper_width_limit_max]),
                    ]
                )
            )
        self._absolute_pose_action_space = gym.spaces.Box(
            low=np.concatenate(action_low).astype(np.float32),
            high=np.concatenate(action_high).astype(np.float32),
            dtype=np.float32,
        )

    def get_absolute_pose_action_space(self) -> gym.spaces.Box:
        return self._absolute_pose_action_space

    def get_relative_pose_action_space(self) -> gym.spaces.Box:
        return self._relative_pose_action_space

    def _current_absolute_pose_action(self) -> np.ndarray:
        poses = [self._turtle2_state.follow1_pos, self._turtle2_state.follow2_pos]
        return np.concatenate(
            [poses[arm_id] for arm_id in self.config.use_arm_ids]
        ).astype(np.float32, copy=False)

    def _last_published_absolute_pose_action(self) -> np.ndarray:
        last_action = getattr(self, "_last_published_action", None)
        if last_action is None:
            return self._current_absolute_pose_action()
        last_action = np.asarray(last_action, dtype=np.float32).reshape(-1).copy()
        expected_dim = len(self.config.use_arm_ids) * ACTION_DIM_PER_ARM
        if last_action.shape != (expected_dim,):
            return self._current_absolute_pose_action()
        return last_action

    def _direct_pose_rejection_reason(
        self, raw_action: np.ndarray, expected_shape: tuple[int, ...]
    ) -> str | None:
        if raw_action.shape != expected_shape:
            return f"invalid_shape:{raw_action.shape}"
        if not np.all(np.isfinite(raw_action)):
            return "non_finite"
        low = self._absolute_pose_action_space.low
        high = self._absolute_pose_action_space.high
        if np.any(raw_action < low) or np.any(raw_action > high):
            return "outside_absolute_pose_action_space"
        if self.config.enforce_gripper_close:
            grip_min = float(self.config.gripper_width_limit_min)
            gripper_values = raw_action.reshape(-1, ACTION_DIM_PER_ARM)[:, POSE_DIM]
            if np.any(np.abs(gripper_values - grip_min) > 1e-6):
                return "enforce_gripper_close"
        return None

    def step_absolute_pose(self, action: np.ndarray):
        expected_shape = (len(self.config.use_arm_ids) * ACTION_DIM_PER_ARM,)
        start_time = time.time()
        raw_action = np.asarray(action, dtype=np.float32)
        pose_control_backend = str(self.config.pose_control_backend)

        if pose_control_backend == "direct":
            rejection_reason = self._direct_pose_rejection_reason(
                raw_action,
                expected_shape,
            )
            last_published_action = self._last_published_absolute_pose_action()
            if rejection_reason is None:
                executed_action = raw_action.reshape(-1).copy()
                action_rejected = False
                next_position = executed_action.reshape(-1, ACTION_DIM_PER_ARM)
                next_positions = {
                    0: self._turtle2_state.follow1_pos.copy(),
                    1: self._turtle2_state.follow2_pos.copy(),
                }
                for action_row, arm_id in zip(
                    next_position, self.config.use_arm_ids, strict=False
                ):
                    next_positions[arm_id] = action_row.copy()
                next_position1 = next_positions[0]
                next_position2 = next_positions[1]
                if not self.config.is_dummy:
                    self._controller.move_arm(
                        next_position1.tolist(), next_position2.tolist()
                    ).wait()
                else:
                    self._turtle2_state.follow1_pos = next_position1.copy()
                    self._turtle2_state.follow2_pos = next_position2.copy()
                self._last_published_action = executed_action.copy()
                last_published_action = executed_action.copy()
            else:
                executed_action = last_published_action.copy()
                action_rejected = True

            self._num_steps += 1
            step_time = time.time() - start_time
            time.sleep(max(0, (1.0 / self.config.step_frequency) - step_time))

            if not self.config.is_dummy:
                self._turtle2_state = self._controller.get_state().wait()[0]
            observation = self._get_observation()
            reward = self._calc_step_reward(observation)
            terminated = False
            truncated = self._num_steps >= self.config.max_num_steps
            info = {
                "raw_action": raw_action.reshape(-1).copy(),
                "executed_action": executed_action,
                "action_clipped": False,
                "clip_delta_max": 0.0,
                "action_rejected": action_rejected,
                "rejection_reason": rejection_reason,
                "last_published_action": last_published_action,
                "pose_control_backend": pose_control_backend,
            }
            return observation, reward, terminated, truncated, info

        assert raw_action.shape == expected_shape, (
            f"Action shape must be {expected_shape}, but got {raw_action.shape}."
        )
        raw_action = raw_action.reshape(-1).copy()

        action = np.clip(
            raw_action,
            self._absolute_pose_action_space.low,
            self._absolute_pose_action_space.high,
        ).astype(np.float32, copy=False)
        action = action.reshape(-1, ACTION_DIM_PER_ARM)
        next_positions = {
            0: self._turtle2_state.follow1_pos.copy(),
            1: self._turtle2_state.follow2_pos.copy(),
        }
        for action_row, arm_id in zip(action, self.config.use_arm_ids, strict=False):
            next_positions[arm_id][:POSE_DIM] = action_row[:POSE_DIM]
            if self.config.enforce_gripper_close:
                next_positions[arm_id][POSE_DIM] = self.config.gripper_width_limit_min
            else:
                next_positions[arm_id][POSE_DIM] = action_row[POSE_DIM]

        next_position = self._clip_position_to_safety_box(
            np.stack([next_positions[0], next_positions[1]])
        )
        next_position = next_position.astype(np.float32, copy=False)
        executed_action = next_position.reshape(-1).copy()
        clip_delta = np.abs(executed_action - raw_action)
        clip_delta_max = float(np.max(clip_delta)) if clip_delta.size else 0.0
        action_clipped = bool(clip_delta_max > 1e-6)
        next_position1 = next_position[0]
        next_position2 = next_position[1]

        if not self.config.is_dummy:
            self._controller.move_arm(
                next_position1.tolist(), next_position2.tolist()
            ).wait()
        else:
            self._turtle2_state.follow1_pos = next_position1.copy()
            self._turtle2_state.follow2_pos = next_position2.copy()

        self._num_steps += 1
        step_time = time.time() - start_time
        time.sleep(max(0, (1.0 / self.config.step_frequency) - step_time))

        if not self.config.is_dummy:
            self._turtle2_state = self._controller.get_state().wait()[0]
        observation = self._get_observation()
        reward = self._calc_step_reward(observation)
        terminated = False
        truncated = self._num_steps >= self.config.max_num_steps
        info = {
            "raw_action": raw_action,
            "executed_action": executed_action,
            "action_clipped": action_clipped,
            "clip_delta_max": clip_delta_max,
            "action_rejected": False,
            "rejection_reason": None,
            "last_published_action": executed_action.copy(),
            "pose_control_backend": pose_control_backend,
        }
        self._last_published_action = executed_action.copy()
        return observation, reward, terminated, truncated, info

    def step_relative_pose(self, action: np.ndarray):
        return super().step(action)

    def step(self, action: np.ndarray):
        if self.config.action_mode == "absolute_pose":
            return self.step_absolute_pose(action)
        if self.config.action_mode == "relative_pose":
            return self.step_relative_pose(action)
        raise ValueError(
            f"Unsupported action_mode={self.config.action_mode!r}. "
            "Expected one of {'absolute_pose', 'relative_pose'}."
        )

    def _calc_step_reward(self, observation) -> float:
        return 0.0
