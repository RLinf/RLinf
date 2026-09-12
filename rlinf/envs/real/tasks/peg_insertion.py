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

"""Peg insertion: bring a grasped peg down into its hole.

The target is the peg seated in the hole. Between episodes the gripper closes
on the peg and the arm takes it clear of the hole before going back to where
it waits, so the peg is never dragged sideways out of the slot.

How an arm gets clear is the arm's business: one driven by tool poses rises
the clearance this task asks for, and one driven by joint targets goes to the
configuration named here as clear of the fixture.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import ResetContext
from .cartesian import CartesianTarget, FixtureConfig
from .requirements import Parts


@dataclass
class PegInsertionConfig(FixtureConfig):
    """Settings for :class:`PegInsertion`."""

    random_xy_range: float = 0.05
    random_rz_range: float = np.pi / 6
    clip_x_range: float = 0.05
    clip_y_range: float = 0.05
    clip_z_range_low: float = 0.0
    clip_z_range_high: float = 0.1
    clip_rz_range: float = np.pi / 6

    clearance: float = 0.10
    """Metres of clearance the peg needs to leave the hole."""

    safe_retract_qpos: Optional[Sequence[float]] = None
    """Configuration that clears the hole, for an arm that cannot be sent a
    tool pose. ``None`` leaves such an arm to go straight to rest."""


class PegInsertion(CartesianTarget):
    """Insert a grasped peg into a hole at the target pose."""

    CONFIG = PegInsertionConfig
    DESCRIPTION = "peg and insertion"

    config: PegInsertionConfig

    def validate(self, dof: Mapping[str, Optional[int]]) -> None:
        """Size the rest configuration to the arm that will wait at it."""
        joints = dof.get("arm")
        if joints is None or self.config.reset_joint_qpos is None:
            return
        given = len(self.config.reset_joint_qpos)
        if given != joints:
            raise ValueError(
                f"The arm has {joints} arm joints, so 'reset_joint_qpos' needs "
                f"{joints} values, got {given}."
            )

    def reset(self, parts: Parts, context: ResetContext) -> None:
        """Grip the peg, take it clear of the hole, then go back to rest."""
        arm = parts.arm()
        context.action.grasp(parts)
        arm.hold()
        arm.clear(
            distance=self.config.clearance,
            qpos=self.config.safe_retract_qpos,
            rate_hz=context.rate_hz,
        )
        self.go_to_rest(parts, context)
