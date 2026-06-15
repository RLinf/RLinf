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

"""SO101 pick-and-place task — reach a target joint configuration.

The simplest possible real-world SO101 task. Serves as a template for more
complex tasks built on top of :class:`~rlinf.envs.realworld.so101.SO101Env`.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from ..so101_env import (
    _NUM_ARM_JOINTS,
    SO101Env,
    SO101RobotConfig,
    _to_lerobot_action,
)


@dataclass
class SO101PickConfig(SO101RobotConfig):
    """Task config — overrides defaults for a simple reaching task."""

    target_joint_qpos: np.ndarray = field(
        default_factory=lambda: np.array(
            [30.0, -60.0, 120.0, 0.0, 30.0, 60.0], dtype=np.float64
        )
    )
    reset_joint_qpos: list[float] = field(
        default_factory=lambda: [0.0, -45.0, 90.0, 0.0, 0.0, 45.0]
    )
    reward_threshold_deg: float = 8.0
    max_num_steps: int = 150

    enable_random_reset: bool = True
    """Add a small uniform perturbation to the arm reset configuration."""
    random_joint_noise_deg: float = 10.0

    success_hold_steps: int = 3
    """Consecutive steps within ``reward_threshold_deg`` required for success."""


class SO101PickEnv(SO101Env):
    """SO101 reaching task.

    The agent must move the arm from ``reset_joint_qpos`` to
    ``target_joint_qpos``. Reward is dense (exponential falloff) and saturates
    at 1.0 when the arm holds within tolerance for ``success_hold_steps``.
    """

    def __init__(self, override_cfg, worker_info=None, hardware_info=None, env_idx=0):
        config = SO101PickConfig(**override_cfg)
        super().__init__(config, worker_info, hardware_info, env_idx)
        self._base_reset_joint_qpos = list(self.config.reset_joint_qpos)
        self._perturbed_reset_qpos = None
        self._success_hold_counter = 0

    @property
    def task_description(self):
        return "reach target joint configuration"

    def go_to_rest(self, joint_reset: bool = False):
        """Move to the rest configuration.

        When ``joint_reset`` is ``True``, sweep the arm through its full range
        before settling at rest — useful to untwist cables on long sessions.
        """
        if self.config.is_dummy:
            return

        if joint_reset:
            full_high = _to_lerobot_action(
                self._joint_limit_high, self.config.gripper_limit_high
            )
            self._robot.send_action(full_high)
            time.sleep(1.0)
            full_low = _to_lerobot_action(
                self._joint_limit_low, self.config.gripper_limit_low
            )
            self._robot.send_action(full_low)
            time.sleep(1.0)

        rest = (
            self._perturbed_reset_qpos
            if self._perturbed_reset_qpos is not None
            else self.config.reset_joint_qpos
        )
        rest_action = _to_lerobot_action(
            np.asarray(rest[:_NUM_ARM_JOINTS], dtype=np.float64),
            float(rest[_NUM_ARM_JOINTS]),
        )
        self._robot.send_action(rest_action)
        time.sleep(1.0)

    def reset(self, joint_reset=False, seed=None, options=None):
        self._num_steps = 0
        self._success_hold_counter = 0
        for k in self._key_state:
            self._key_state[k] = False

        if self.config.is_dummy:
            return self._get_observation(), {}

        if self.config.enable_random_reset:
            base = np.array(self._base_reset_joint_qpos[:_NUM_ARM_JOINTS])
            noise = np.random.uniform(
                -self.config.random_joint_noise_deg,
                self.config.random_joint_noise_deg,
                size=_NUM_ARM_JOINTS,
            )
            arm = np.clip(base + noise, self._joint_limit_low, self._joint_limit_high)
            self._perturbed_reset_qpos = (
                arm.tolist() + self._base_reset_joint_qpos[_NUM_ARM_JOINTS:]
            )
        else:
            self._perturbed_reset_qpos = None

        self.go_to_rest(joint_reset)
        self._update_state()
        return self._get_observation(), {}

    def _calc_step_reward(self, observation: dict) -> float:
        if self.config.is_dummy or self.config.target_joint_qpos is None:
            return 0.0

        arm_pos = observation["state"]["joint_position"]
        target_arm = self.config.target_joint_qpos[:_NUM_ARM_JOINTS]
        error = float(np.linalg.norm(arm_pos - target_arm))

        if error < self.config.reward_threshold_deg:
            self._success_hold_counter += 1
            if self._success_hold_counter >= self.config.success_hold_steps:
                return 1.0
            return 0.5
        self._success_hold_counter = 0
        return float(np.exp(-error / self.config.reward_threshold_deg))
