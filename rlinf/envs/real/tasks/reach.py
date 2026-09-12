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

"""Joint reach: bring one arm to a target configuration and hold it.

The smallest task that exercises the whole path -- compose the robot, read it,
command it, and score the result -- so a new arm can be brought up before any
manipulation task is written for it. It needs nothing but joints, so it runs
on any arm that reports and accepts them.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from .base import Evaluation, ResetContext, Task, TaskConfig
from .requirements import Needs, Parts, Reading

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rlinf.envs.real.policy import Applied


@dataclass
class JointReachConfig(TaskConfig):
    """Settings for :class:`JointReach`."""

    reset_joint_qpos: Optional[Sequence[float]] = None
    """Rest configuration in radians. ``None`` rests at zero."""

    target_joint_qpos: Optional[Sequence[float]] = None
    """Goal configuration the reward measures against, in radians. ``None``
    aims at zero."""

    reward_threshold: float = 0.05
    """Per-joint tolerance in radians. Every joint must be within it."""

    enable_random_reset: bool = False
    """Perturb the rest configuration at the start of each episode."""

    random_joint_noise: float = 0.05
    """Largest per-joint perturbation in radians when randomising."""


class JointReach(Task):
    """Reach and hold a target joint configuration.

    The sparse reward is 1 while every joint is within
    :attr:`JointReachConfig.reward_threshold` of the target; the dense reward
    is the negative distance to it.
    """

    CONFIG = JointReachConfig
    DESCRIPTION = "reach a joint configuration"

    config: JointReachConfig

    def requirements(self) -> Mapping[str, Needs]:
        """An arm that reports its joints."""
        return {"arm": Needs(observes=frozenset({"arm_joint_position"}))}

    def validate(self, dof: Mapping[str, Optional[int]]) -> None:
        """Size the target to the arm, filling unset poses with zeros."""
        joints = dof.get("arm")
        if joints is None:
            return
        for name in ("reset_joint_qpos", "target_joint_qpos"):
            if getattr(self.config, name) is None:
                setattr(self.config, name, [0.0] * joints)
        given = len(self.config.target_joint_qpos)
        if given != joints:
            raise ValueError(
                f"The arm has {joints} arm joints, so 'target_joint_qpos' needs "
                f"{joints} values, got {given}."
            )

    def home(self, parts: Parts, context: ResetContext) -> None:
        """Rest at the configured pose."""
        parts.arm().reset_joint(self.config.reset_joint_qpos)

    def reset(self, parts: Parts, context: ResetContext) -> None:
        """Rest at the configured pose, perturbed when randomising."""
        pose = np.asarray(self.config.reset_joint_qpos, dtype=float)
        if self.config.enable_random_reset:
            noise = self.config.random_joint_noise
            pose = pose + context.rng.uniform(-noise, noise, size=pose.shape)
            limits = context.action.joint_limits()
            if limits is not None:
                pose = np.clip(pose, *limits)
        parts.arm().reset_joint(list(pose))

    def evaluate(self, reading: Reading, applied: "Applied") -> Evaluation:
        """Score the joint distance to the target."""
        measured = np.asarray(reading.arm()["arm_joint_position"], dtype=float)
        distance = np.abs(measured - np.asarray(self.config.target_joint_qpos))
        hit = bool(np.all(distance < self.config.reward_threshold))
        if self.config.use_dense_reward:
            return Evaluation(reward=float(-np.linalg.norm(distance)), in_zone=hit)
        return Evaluation(reward=1.0 if hit else 0.0, in_zone=hit)
