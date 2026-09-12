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

"""Bin relocation: carry an object from one bin into the one beside it.

The two bins sit either side of the target, with a wall between them. The
workspace keeps the tool out of a box over that wall, stopping a motion where
it would enter it, so the object has to be lifted over rather than dragged
through. Each episode starts over one bin, set by :attr:`BinRelocation.task_id`.
"""

import time
from dataclasses import dataclass, replace
from typing import Any, Optional

import numpy as np

from .base import ResetContext
from .cartesian import CartesianTarget, FixtureConfig
from .requirements import Parts
from .workspace import Box, Workspace


@dataclass
class BinRelocationConfig(FixtureConfig):
    """Settings for :class:`BinRelocation`."""

    random_xy_range: float = 0.01
    random_rz_range: float = np.pi / 9
    clip_x_range: float = 0.10
    clip_y_range: float = 0.15
    clip_z_range_low: float = 0.001
    clip_z_range_high: float = 0.1
    clip_rz_range: float = np.pi / 6


class BinRelocation(CartesianTarget):
    """Move an object into the other bin, over the wall between them."""

    CONFIG = BinRelocationConfig
    DESCRIPTION = "Pick up the object and put it into another bin"

    #: Half-extents of the box over the wall, below and above the target.
    WALL_BELOW = np.array([0.07, 0.03, 0.001])
    WALL_ABOVE = np.array([0.07, 0.03, 0.04])

    #: How far to either side of the target each bin's rest pose is, in metres.
    BIN_OFFSET = 0.1

    config: BinRelocationConfig

    def __init__(self, config: Optional[BinRelocationConfig] = None) -> None:
        super().__init__(config)
        #: Which bin the next episode starts over: 0 moves forward, 1 back.
        self.task_id = 0

    @property
    def workspace(self) -> Workspace:
        """The fixture's box, with the wall between the bins kept clear."""
        centre = self.config.target_ee_pose[:3]
        wall = Box(low=centre - self.WALL_BELOW, high=centre + self.WALL_ABOVE)
        return replace(super().workspace, obstacles=(wall,))

    def set_task_id(self, task_id: int) -> None:
        """Choose the bin the next episode starts over."""
        self.task_id = task_id

    def task_graph(self, obs: Optional[dict[str, Any]] = None) -> Optional[int]:
        """The task that follows this one: the other bin."""
        if obs is None:
            return (self.task_id + 1) % 2
        return None

    def reset(self, parts: Parts, context: ResetContext) -> None:
        """Let go, lift clear of the bin, and rest over the starting bin."""
        if self.task_id not in (0, 1):
            raise ValueError(f"Task id {self.task_id} should be 0 or 1")
        pose = self.rest_pose()
        side = self.BIN_OFFSET if self.task_id == 0 else -self.BIN_OFFSET
        pose[1] = self.config.target_ee_pose[1] + side

        arm = parts.arm()
        context.action.release(parts)
        arm.hold()
        time.sleep(0.5)
        arm.clear(distance=0.10, rate_hz=context.rate_hz)
        self.go_to_rest(parts, context, pose)
