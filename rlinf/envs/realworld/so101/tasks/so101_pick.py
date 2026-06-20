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

"""SO101 reaching task — reach a target end-effector position.

The default task uses **end-effector space**: ``target_ee_pose`` is a ``(x, y, z)``
tuple in metres, and forward kinematics computes the current gripper position
from the arm's joint angles each step.  Set ``target_ee_pose`` to ``null`` (None)
in YAML to fall back to joint-angle-based reward via ``target_joint_qpos``.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..so101_env import (
    _NUM_ARM_JOINTS,
    SO101Env,
    SO101RobotConfig,
    _to_lerobot_action,
)


_DEFAULT_EE_TARGET = (0.35, 0.00, 0.00)
"""Default EE target — about 35 cm forward from the base at table height."""


@dataclass
class SO101PickConfig(SO101RobotConfig):
    """Task config — reach an end-effector position (or joint angles)."""

    # --- EE-space target (preferred) ---
    target_ee_pose: Optional[tuple[float, ...]] = field(
        default_factory=lambda: _DEFAULT_EE_TARGET
    )
    """End-effector target ``(x, y, z)`` in **metres**.  ``None`` falls
    back to ``target_joint_qpos`` for joint-space reward."""

    reward_threshold_m: float = 0.03
    """Success radius in **metres** (3 cm default)."""

    # --- Joint-space fallback (used only when target_ee_pose is None) ---
    target_joint_qpos: Optional[np.ndarray] = None
    """Fallback joint-angle target ``[q1..q5, gripper]`` in degrees."""

    # --- Common ---
    reset_joint_qpos: list[float] = field(
        default_factory=lambda: [0.0, -45.0, 90.0, 0.0, 0.0, 45.0]
    )
    reward_threshold_deg: float = 8.0
    max_num_steps: int = 150

    enable_random_reset: bool = True
    """Add a small uniform perturbation to the arm reset configuration."""
    random_joint_noise_deg: float = 10.0

    success_hold_steps: int = 3
    """Consecutive steps within the threshold required for success."""


class SO101PickEnv(SO101Env):
    """SO101 reaching task (EE‑space by default).

    The agent must move the arm from the reset configuration so that its
    end-effector reaches within ``reward_threshold_m`` of
    ``target_ee_pose``.  Reward is dense (exponential falloff) and
    saturates at 1.0 when the arm holds within tolerance for
    ``success_hold_steps``.

    When ``target_ee_pose`` is ``None`` the environment falls back to the
    joint-angle reward defined by ``target_joint_qpos`` and
    ``reward_threshold_deg``.
    """

    def __init__(
        self,
        override_cfg,
        worker_info=None,
        hardware_info=None,
        env_idx=0,
        env_cfg=None,
    ):
        del env_cfg
        config = SO101PickConfig(**override_cfg)
        super().__init__(config, worker_info, hardware_info, env_idx)
        self._base_reset_joint_qpos = list(self.config.reset_joint_qpos)
        self._perturbed_reset_qpos = None
        self._success_hold_counter = 0
        self._kinematics = None
        self._task_is_ee: bool = self.config.target_ee_pose is not None
        self._target_ee = np.asarray(
            self.config.target_ee_pose, dtype=np.float64
        ) if self._task_is_ee else np.zeros(3)

    @property
    def task_description(self):
        if self._task_is_ee:
            return (
                f"reach target ee pose (x={self._target_ee[0]:.3f}, "
                f"y={self._target_ee[1]:.3f}, z={self._target_ee[2]:.3f})"
            )
        return "reach target joint configuration"

    def _init_kinematics(self) -> None:
        """Lazily create the forward-kinematics solver.

        Called once on the first real-hardware step that needs FK.
        Skipped entirely in dummy mode and during data collection.
        """
        if self._kinematics is not None or self.config.is_dummy:
            return

        from lerobot.model.kinematics import RobotKinematics

        if self.config.urdf_path is not None:
            urdf_path = self.config.urdf_path
        else:
            urdf_path = "pack://lerobot/robots/so_follower/so101.urdf"

        self._kinematics = RobotKinematics(
            urdf_path=str(urdf_path),
            target_frame_name="gripper_frame_link",
        )

    def _ee_position(self) -> np.ndarray:
        """Return the current end-effector ``(x, y, z)`` in metres.

        Falls back to ``np.zeros(3)`` when kinematics is unavailable
        (dummy mode / data collection).
        """
        self._init_kinematics()
        if self._kinematics is None:
            return np.zeros(3, dtype=np.float64)

        joint_deg = np.asarray(
            self._state.joint_position, dtype=np.float64
        )
        T = self._kinematics.forward_kinematics(joint_deg)
        return T[:3, 3]

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
        if self.config.is_dummy:
            return 0.0

        # --- EE-space reward (primary path) ---
        if self._task_is_ee:
            ee = self._ee_position()
            error = float(np.linalg.norm(ee - self._target_ee))
            if error < self.config.reward_threshold_m:
                self._success_hold_counter += 1
                if self._success_hold_counter >= self.config.success_hold_steps:
                    return 1.0
                return 0.5
            self._success_hold_counter = 0
            return float(np.exp(-error / self.config.reward_threshold_m))

        # --- Joint-angle reward (fallback) ---
        if self.config.target_joint_qpos is None:
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
