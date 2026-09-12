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

"""Pick and place: pick an object up and set it down at the target.

Written for a dexterous hand, and runs with a gripper too. Between episodes
the hand opens to its resting pose, or the gripper opens, and the arm backs
off before returning to rest.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import ResetContext
from .cartesian import CartesianTarget, CartesianTargetConfig, release_and_back_off
from .requirements import Parts


@dataclass
class PickPlaceConfig(CartesianTargetConfig):
    """Settings for :class:`PickPlace`.

    Left unset, the workspace reaches 2 cm from the target along x and y,
    2 cm below and 10 cm above it, and 0.003 rad in each angle, and the arm
    rests 5 cm above the target.
    """

    target_ee_pose: Sequence[float] = (0.0,) * 6
    reset_ee_pose: Optional[Sequence[float]] = None
    reward_threshold: Sequence[float] = (0.01, 0.01, 0.01, 0.2, 0.2, 0.2)
    ee_pose_limit_min: Optional[Sequence[float]] = None
    ee_pose_limit_max: Optional[Sequence[float]] = None
    enable_random_reset: bool = True
    enable_gripper_penalty: bool = False

    def __post_init__(self) -> None:
        target = np.asarray(self.target_ee_pose, dtype=np.float64)
        if self.ee_pose_limit_min is None:
            self.ee_pose_limit_min = target - np.array(
                [0.02, 0.02, 0.02, 0.003, 0.003, 0.003]
            )
        if self.ee_pose_limit_max is None:
            self.ee_pose_limit_max = target + np.array(
                [0.02, 0.02, 0.1, 0.003, 0.003, 0.003]
            )
        if self.reset_ee_pose is None:
            self.reset_ee_pose = target + np.array([0.0, 0.0, 0.05, 0.0, 0.0, 0.0])
        super().__post_init__()


class PickPlace(CartesianTarget):
    """Pick an object up and place it at the target pose."""

    CONFIG = PickPlaceConfig
    DESCRIPTION = "pick up the toy and place it onto the plate"

    config: PickPlaceConfig

    def reset(self, parts: Parts, context: ResetContext) -> None:
        """Let go, back off, then return to rest."""
        release_and_back_off(parts, context)
        self.go_to_rest(parts, context)
