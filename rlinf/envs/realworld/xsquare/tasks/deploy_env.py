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

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.xsquare.turtle2_env import Turtle2Env, Turtle2RobotConfig


@dataclass
class Turtle2DeployEnvConfig(Turtle2RobotConfig):
    use_camera_ids: list[int] = field(default_factory=lambda: [0, 1, 2])
    use_arm_ids: list[int] = field(default_factory=lambda: [0, 1])
    enforce_gripper_close: bool = False
    task_description: str = ""


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
        self.action_space = gym.spaces.Box(
            low=np.concatenate(action_low).astype(np.float32),
            high=np.concatenate(action_high).astype(np.float32),
            dtype=np.float32,
        )

    def step(self, action: np.ndarray):
        assert action.shape == (len(self.config.use_arm_ids) * 7,), (
            f"Action shape must be {(len(self.config.use_arm_ids) * 7,)}, but got {action.shape}."
        )

        start_time = time.time()

        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = action.reshape(-1, 7)
        next_positions = {
            0: self._turtle2_state.follow1_pos.copy(),
            1: self._turtle2_state.follow2_pos.copy(),
        }
        for action_row, arm_id in zip(action, self.config.use_arm_ids, strict=False):
            next_positions[arm_id][:6] = action_row[:6]
            if self.config.enforce_gripper_close:
                next_positions[arm_id][6] = self.config.gripper_width_limit_min
            else:
                next_positions[arm_id][6] = action_row[6]

        next_position = self._clip_position_to_safety_box(
            np.stack([next_positions[0], next_positions[1]])
        )
        next_position1 = next_position[0]
        next_position2 = next_position[1]

        if not self.config.is_dummy:
            self._controller.move_arm(
                next_position1.tolist(), next_position2.tolist()
            ).wait()

        self._num_steps += 1
        step_time = time.time() - start_time
        time.sleep(max(0, (1.0 / self.config.step_frequency) - step_time))

        if not self.config.is_dummy:
            self._turtle2_state = self._controller.get_state().wait()[0]
        observation = self._get_observation()
        reward = self._calc_step_reward(observation)
        terminated = False
        truncated = self._num_steps >= self.config.max_num_steps
        return observation, reward, terminated, truncated, {}

    def _calc_step_reward(self, observation) -> float:
        return 0.0
