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

"""Cap tightening: screw a held cap onto a bottle.

The target is the cap seated on the bottle's thread. Between episodes the
gripper lets go of the cap and the arm backs off in two short lifts, waiting
for the cap to settle, before returning to rest.
"""

from dataclasses import dataclass

import numpy as np

from .base import ResetContext
from .cartesian import CartesianTarget, FixtureConfig, release_and_back_off
from .requirements import Parts


@dataclass
class BottleCapConfig(FixtureConfig):
    """Settings for :class:`BottleCap`."""

    random_xy_range: float = 0.01
    random_rz_range: float = np.pi / 6
    clip_x_range: float = 0.01
    clip_y_range: float = 0.01
    clip_z_range_low: float = 0.001
    clip_z_range_high: float = 0.02
    clip_rz_range: float = np.pi / 6
    enable_gripper_penalty: bool = False


class BottleCap(CartesianTarget):
    """Tighten a cap onto a bottle at the target pose."""

    CONFIG = BottleCapConfig
    DESCRIPTION = "screw the bottle cap onto the bottle"

    config: BottleCapConfig

    def reset(self, parts: Parts, context: ResetContext) -> None:
        """Let go, back off 3 cm and then 2 cm more, then return to rest."""
        release_and_back_off(parts, context)
        self.go_to_rest(parts, context)
