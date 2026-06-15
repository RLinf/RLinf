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

"""SO101 pick-and-place task.

A simple task that teaches the SO101 arm to reach a target joint
configuration from a reset pose. This is the simplest possible
real-world robot task and serves as a template for more complex tasks.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from ..so101_env import SO101Env, SO101RobotConfig


@dataclass
class SO101PickConfig(SO101RobotConfig):
    """Configuration for :class:`SO101PickEnv`.

    Extends the base config with task-specific defaults for a pick-and-place
    reaching task.
    """

    target_joint_qpos: np.ndarray = field(
        default_factory=lambda: np.array(
            [30.0, -60.0, 120.0, 0.0, 30.0, 60.0], dtype=np.float64
        )
    )
    """Target joint configuration (degrees) the arm should reach."""

    reset_joint_qpos: list[float] = field(
        default_factory=lambda: [0.0, -45.0, 90.0, 0.0, 0.0, 45.0]
    )
    """Reset joint configuration (degrees)."""

    reward_threshold_deg: float = 8.0
    """Per-joint tolerance in degrees for success."""

    max_num_steps: int = 150
    """Episode truncation horizon."""

    enable_random_reset: bool = True
    """Add small joint-space perturbation to the reset configuration."""

    random_joint_noise: float = 10.0
    """Max joint angle perturbation in degrees when ``enable_random_reset``
    is True."""

    success_hold_steps: int = 3
    """Number of consecutive steps within the target threshold for success."""


class SO101PickEnv(SO101Env):
    """SO101 pick-and-place reaching task.

    The agent must move the arm from the reset joint configuration to a
    target joint configuration. Reward is computed as the L2 distance
    between current and target joint positions.

    This is intentionally simple — it serves as the "hello world" for
    real-world robot RL and a template for building more complex tasks.
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
        """Move to the rest joint configuration.

        If ``joint_reset`` is ``True``, move through a full-range motion
        first (good for preventing cable tangling over long sessions).
        """
        if self.config.is_dummy:
            return

        from ..so101_env import _SO101_MOTOR_NAMES

        if joint_reset:
            # Full-range motion to prevent cable issues.
            full_range = {}
            for i, name in enumerate(_SO101_MOTOR_NAMES):
                if name == "gripper":
                    full_range[name] = 90.0  # open
                else:
                    full_range[name] = float(self.config.joint_limit_high[i])
            self._robot.send_action(full_range)
            time.sleep(1.0)
            full_range_low = {}
            for i, name in enumerate(_SO101_MOTOR_NAMES):
                if name == "gripper":
                    full_range_low[name] = 0.0  # close
                else:
                    full_range_low[name] = float(self.config.joint_limit_low[i])
            self._robot.send_action(full_range_low)
            time.sleep(1.0)

        # Move to reset configuration.
        reset_qpos = (
            self._perturbed_reset_qpos
            if self._perturbed_reset_qpos is not None
            else self.config.reset_joint_qpos
        )
        rest_action = {}
        for i, name in enumerate(_SO101_MOTOR_NAMES):
            rest_action[name] = float(reset_qpos[i])
        self._robot.send_action(rest_action)
        time.sleep(1.0)

    def reset(self, joint_reset=False, seed=None, options=None):
        """Reset with optional random perturbation on joint positions."""
        self._num_steps = 0
        self._success_hold_counter = 0
        for k in self._key_state:
            self._key_state[k] = False

        if self.config.is_dummy:
            return self._get_observation(), {}

        if self.config.enable_random_reset:
            base_qpos = np.array(self._base_reset_joint_qpos)
            noise = np.random.uniform(
                -self.config.random_joint_noise,
                self.config.random_joint_noise,
                size=6,
            )
            self._perturbed_reset_qpos = np.clip(
                base_qpos + noise,
                self._joint_limit_low,
                self._joint_limit_high,
            ).tolist()
        else:
            self._perturbed_reset_qpos = None

        self.go_to_rest(joint_reset)
        self._update_state()
        return self._get_observation(), {}

    def _calc_step_reward(self, observation: dict) -> float:
        """Compute reward: L2 distance to target joint configuration.

        Returns 1.0 when within the target threshold for
        ``success_hold_steps`` consecutive steps.
        """
        if self.config.is_dummy or self.config.target_joint_qpos is None:
            return 0.0

        joint_pos = observation["state"]["joint_position"]
        error = np.linalg.norm(
            joint_pos - self.config.target_joint_qpos[:6]
        )
        if error < self.config.reward_threshold_deg:
            self._success_hold_counter += 1
            if self._success_hold_counter >= self.config.success_hold_steps:
                return 1.0
            return 0.5  # Partial reward for being close
        else:
            self._success_hold_counter = 0
            # Dense exponential reward.
            return float(np.exp(-error / self.config.reward_threshold_deg))
