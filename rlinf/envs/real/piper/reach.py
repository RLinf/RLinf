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

"""Piper joint reach: hold a target configuration.

The smallest task that exercises the whole path -- compose the robot, read it,
command it, and score the result -- so a new arm can be brought up before any
manipulation task is written for it.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from rlinf.robotics.discovery import RobotInfo
from rlinf.scheduler import WorkerInfo

from .base import _DOF, PiperEnv, PiperRobotConfig


@dataclass
class PiperReachConfig(PiperRobotConfig):
    """Configuration for :class:`PiperReachEnv`."""

    enable_random_reset: bool = False
    """Perturb the rest configuration at the start of each episode."""

    random_joint_noise: float = 0.05
    """Largest per-joint perturbation in radians when randomising."""

    def __post_init__(self) -> None:
        """Validate the target against the arm's own degrees of freedom."""
        super().__post_init__()
        if len(self.target_joint_qpos) != _DOF:
            raise ValueError(
                f"A Piper has {_DOF} arm joints, so 'target_joint_qpos' needs "
                f"{_DOF} values, got {len(self.target_joint_qpos)}."
            )


class PiperReachEnv(PiperEnv):
    """Reach and hold a target joint configuration."""

    def __init__(
        self,
        override_cfg: dict[str, Any],
        worker_info: Optional[WorkerInfo] = None,
        robot_info: "Optional[RobotInfo[Any]]" = None,
        env_idx: int = 0,
    ) -> None:
        config = PiperReachConfig(**override_cfg)
        super().__init__(config, worker_info, robot_info, env_idx)
        self._base_reset_qpos = list(self.config.reset_joint_qpos)

    @property
    def task_description(self) -> str:
        """Natural-language task name, for policies that condition on one."""
        return "reach a joint configuration"

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset, optionally starting from a perturbed rest pose."""
        if self.config.enable_random_reset:
            rng = np.random.default_rng(seed)
            noise = rng.uniform(
                -self.config.random_joint_noise,
                self.config.random_joint_noise,
                size=_DOF,
            )
            perturbed = np.asarray(self._base_reset_qpos, dtype=float) + noise
            self.config.reset_joint_qpos = list(
                np.clip(perturbed, self._joint_limit_low, self._joint_limit_high)
            )
        return super().reset(seed=seed, options=options)
