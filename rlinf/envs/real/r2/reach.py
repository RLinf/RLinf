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

"""Robbyant R2 joint reach: reach and hold a target joint configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from rlinf.robotics.discovery import RobotInfo
from rlinf.scheduler import WorkerInfo

from .base import ARM_DOF, R2Env, R2EnvConfig


@dataclass
class R2ReachConfig(R2EnvConfig):
    """Configuration for :class:`R2ReachEnv`."""

    enable_random_reset: bool = False
    """Perturb the rest configuration at the start of each episode."""

    random_joint_noise: float = 0.05
    """Largest per-joint perturbation in radians when randomising."""

    def __post_init__(self) -> None:
        """Validate ``target_joint_qpos`` against the active arms' DoF."""
        super().__post_init__()
        expected = ARM_DOF * len(self.active_arms)
        if len(self.target_joint_qpos) != expected:
            raise ValueError(
                f"R2 with active_arms={self.active_arms} has {expected} arm joints, "
                f"so 'target_joint_qpos' needs {expected} values, "
                f"got {len(self.target_joint_qpos)}."
            )


class R2ReachEnv(R2Env):
    """Reach and hold a target joint configuration on the Robbyant R2 robot."""

    def __init__(
        self,
        override_cfg: Optional[dict[str, Any]] = None,
        worker_info: Optional[WorkerInfo] = None,
        robot_info: Optional[RobotInfo[Any]] = None,
        env_idx: int = 0,
    ) -> None:
        config = R2ReachConfig(**(override_cfg or {}))
        super().__init__(config, worker_info, robot_info, env_idx)
        self._base_reset_qpos = list(self.config.reset_joint_qpos)

    @property
    def task_description(self) -> str:
        """Natural-language task name for language-conditioned policies."""
        return "reach a joint configuration"

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset, optionally starting from a perturbed rest pose."""
        if self.config.enable_random_reset:
            rng = np.random.default_rng(seed)
            num_joints = ARM_DOF * len(self.config.active_arms)
            noise = rng.uniform(
                -self.config.random_joint_noise,
                self.config.random_joint_noise,
                size=num_joints,
            )
            perturbed = np.asarray(self._base_reset_qpos, dtype=float) + noise
            self.config.reset_joint_qpos = list(
                np.clip(perturbed, self._joint_limit_low, self._joint_limit_high)
            )
        return super().reset(seed=seed, options=options)
