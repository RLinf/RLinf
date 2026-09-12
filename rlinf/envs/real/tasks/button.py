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

"""Press a button with the tool.

A run gives the button's pose and the clearances around it; everything else --
where the arm waits, how far it may travel, what counts as pressed -- is
derived from that one pose, so the two never disagree. Scoring the tool's
orientation as well as its position is what separates a press from a graze,
which is why this task measures all six axes.
"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .bimanual import MultiArmTarget, MultiArmTargetConfig


@dataclass
class ButtonConfig(MultiArmTargetConfig):
    """Where the button is, and how far around it the tool may go."""

    clip_x_range: float = 0.05
    """Metres either side of the button the tool may travel in x."""

    clip_y_range: float = 0.05
    """Metres either side of the button the tool may travel in y."""

    clip_z_range_low: float = -0.005
    """Metres below the button the tool may travel, as a press overshoots."""

    clip_z_range_high: float = 0.1
    """Metres above the button the tool may travel, and where it waits."""

    clip_rz_range: float = np.pi / 9
    """Radians either way each Euler angle may turn from the button's."""

    press_threshold: Sequence[float] = field(
        default_factory=lambda: (0.015, 0.015, 0.01, 0.15, 0.15, 0.15)
    )
    """How close the tool must be on each axis for the button to count as
    pressed: metres in x, y and z, then radians."""

    def __post_init__(self) -> None:
        """Derive where the arm waits and how far it may go from the button.

        A run states the button's pose once. The rest -- the rest pose above
        it, the box around it, and what counts as pressed -- follows, so no
        two of them can drift apart.
        """
        roles = tuple(self.roles)
        target = np.asarray(self.target_ee_pose, dtype=np.float64)
        if target.ndim == 1:
            target = np.tile(target, (len(roles), 1))
        lift = np.zeros_like(target)
        lift[:, 2] = self.clip_z_range_high
        self.reset_ee_pose = target + lift
        self.reward_threshold = np.tile(
            np.asarray(self.press_threshold, dtype=np.float64), (len(roles), 1)
        )
        spread = np.array(
            [
                self.clip_x_range,
                self.clip_y_range,
                self.clip_z_range_low,
                self.clip_rz_range,
                self.clip_rz_range,
                self.clip_rz_range,
            ]
        )
        self.ee_pose_limit_min = target - spread
        above = spread.copy()
        above[2] = self.clip_z_range_high
        self.ee_pose_limit_max = target + above
        super().__post_init__()


class Button(MultiArmTarget):
    """Bring the tool onto a button and hold it there.

    The task is :class:`MultiArmTarget` with the button's geometry worked out
    once: it adds no scoring or reset of its own.
    """

    CONFIG = ButtonConfig

    DESCRIPTION = "Press the button with the end-effector."

    config: ButtonConfig
