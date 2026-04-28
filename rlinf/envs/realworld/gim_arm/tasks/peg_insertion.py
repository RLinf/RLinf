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

"""GimArm peg-insertion task.

Ported from :class:`rlinf.envs.realworld.franka.tasks.PegInsertionEnv`.

Key differences from the Franka version:
- No ``compliance_param`` / ``precision_param`` (no ROS impedance controller).
- Reset and control are in joint space, not Cartesian space.
- ``go_to_rest`` retracts to ``safe_retract_qpos`` (joint-space) instead of
  a Cartesian Z-axis lift.
- Reward is still computed in Cartesian space via FK, identical to Franka.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from ..gim_arm_env import GimArmEnv, GimArmRobotConfig


@dataclass
class GimArmPegInsertionConfig(GimArmRobotConfig):
    """Configuration for :class:`GimArmPegInsertionEnv`."""

    target_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
    """Target EEF pose ``[x, y, z, rx, ry, rz]`` for reward computation."""

    reward_threshold: np.ndarray = field(
        default_factory=lambda: np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2])
    )
    """Per-axis success tolerances ``[x, y, z, rx, ry, rz]``.
    Only XYZ entries are currently consulted (see ``GimArmRobotConfig.reward_threshold``)."""

    clip_x_range: float = 0.05
    clip_y_range: float = 0.05
    clip_z_range_low: float = 0.0
    clip_z_range_high: float = 0.1
    clip_rz_range: float = np.pi / 6

    random_xy_range: float = 0.05
    """Magnitude of uniform noise (kept for API compatibility)."""

    random_rz_range: float = np.pi / 6
    """Magnitude of rotation noise (kept for API compatibility)."""

    enable_random_reset: bool = True
    """Add small joint-space perturbation to the reset configuration."""

    random_joint_noise: float = 0.02
    """Max joint angle perturbation in radians when ``enable_random_reset`` is True."""

    safe_retract_qpos: list[float] = field(
        default_factory=lambda: [0.0, -1.5, 1.5, 0.0, 0.0, 0.0]
    )
    """Joint configuration used during go_to_rest to clear the insertion hole.
    Should be tuned for your specific hardware setup."""

    add_gripper_penalty: bool = False

    def __post_init__(self):
        self.target_ee_pose = np.array(self.target_ee_pose)
        self.reward_threshold = np.array(self.reward_threshold)
        if self.add_gripper_penalty:
            self.enable_gripper_penalty = True


class GimArmPegInsertionEnv(GimArmEnv):
    """GimArm peg insertion task: insert a peg into a hole.

    Actions are 6-DOF absolute joint-position targets plus a binary gripper command.
    Reward is computed in Cartesian space by comparing FK-based TCP pose
    to the target pose.
    """

    def __init__(self, override_cfg, worker_info=None, hardware_info=None, env_idx=0):
        config = GimArmPegInsertionConfig(**override_cfg)
        super().__init__(config, worker_info, hardware_info, env_idx)
        self._base_reset_joint_qpos = list(self.config.reset_joint_qpos)
        self._perturbed_reset_qpos = None

    @property
    def task_description(self):
        return "peg and insertion"

    def go_to_rest(self, joint_reset: bool = False):
        """Close gripper on peg, retract to safe config, then move to reset pose."""
        if not self.config.is_dummy:
            if self.config.enable_gripper:
                # Close gripper to hold the peg during retraction.
                self._controller.close_gripper().wait()
                time.sleep(0.3)

            # Move to safe retracted position (clears the insertion hole).
            self._controller.reset_joint(self.config.safe_retract_qpos).wait()
            time.sleep(0.5)

        if not self.config.is_dummy:
            if joint_reset:
                self._controller.reset_joint(self.config.joint_reset_qpos).wait()
                time.sleep(0.5)

            # Use per-episode perturbed qpos if available, otherwise config default.
            reset_qpos = (
                self._perturbed_reset_qpos
                if self._perturbed_reset_qpos is not None
                else self.config.reset_joint_qpos
            )
            self._controller.reset_joint(reset_qpos).wait()

    def reset(self, joint_reset=False, **kwargs):
        """Reset with optional random perturbation on joint positions."""
        if self.config.enable_random_reset and not self.config.is_dummy:
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
        return super().reset(joint_reset, **kwargs)
