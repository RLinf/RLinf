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

"""Where a task lets the tool go.

A Cartesian task is set up around a fixture: a hole, a bottle, a bin. Its
workspace bounds the tool to a box around that fixture, keeps its orientation
within a window around the fixture's, and keeps it out of anything inside the
box it must not enter. A control that commands poses keeps every pose inside
the workspace of the task it runs.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.real.utils.pose import clip_euler_to_target_window


@dataclass(frozen=True)
class Box:
    """An axis-aligned box in metres."""

    low: np.ndarray
    high: np.ndarray

    def contains(self, point: np.ndarray) -> bool:
        """Whether ``point`` lies inside or on the box."""
        return bool(np.all(point >= self.low) and np.all(point <= self.high))

    def entry(self, start: np.ndarray, end: np.ndarray) -> Optional[np.ndarray]:
        """Where the segment from ``start`` to ``end`` first enters the box.

        Returns:
            The entry point, ``start`` itself when it is already inside, or
            ``None`` when the segment misses the box.
        """
        t_enter, t_exit = 0.0, 1.0
        for axis in range(3):
            if start[axis] < self.low[axis] and end[axis] < self.low[axis]:
                return None
            if start[axis] > self.high[axis] and end[axis] > self.high[axis]:
                return None
            span = end[axis] - start[axis]
            if abs(span) > 1e-10:
                t1 = (self.low[axis] - start[axis]) / span
                t2 = (self.high[axis] - start[axis]) / span
                t_enter = max(t_enter, min(t1, t2))
                t_exit = min(t_exit, max(t1, t2))
                if t_enter > t_exit:
                    return None
        return start + t_enter * (end - start)


@dataclass(frozen=True)
class Workspace:
    """The poses a task allows the tool to be commanded to.

    Attributes:
        low: Lowest ``[x, y, z, rx, ry, rz]``, orientation as xyz Euler angles.
        high: Highest ``[x, y, z, rx, ry, rz]``.
        target_euler: Orientation the window is centred on, so a window that
            straddles ``pi`` stays one interval.
        obstacles: Boxes inside the workspace a motion stops at.
        orientation: How the orientation half of the box is applied.
            ``"window"`` treats it as an interval around ``target_euler``, so
            a range straddling ``pi`` stays one interval. ``"box"`` clips each
            Euler angle to its own bounds, which is what a controller that
            thinks in Euler angles was tuned against.
    """

    low: np.ndarray
    high: np.ndarray
    target_euler: np.ndarray
    obstacles: tuple[Box, ...] = ()
    orientation: str = "window"

    def clip(self, pose: np.ndarray, start: np.ndarray) -> np.ndarray:
        """Bring a commanded pose inside the workspace.

        Args:
            pose: The commanded pose, ``xyz`` plus an ``xyzw`` quaternion.
            start: Where the tool is now, for where a motion enters an
                obstacle.

        Returns:
            A new pose inside the box and orientation window, stopped where
            the motion from ``start`` would enter an obstacle.
        """
        clipped = np.array(pose, dtype=np.float64)
        clipped[:3] = np.clip(clipped[:3], self.low[:3], self.high[:3])
        euler = R.from_quat(clipped[3:]).as_euler("xyz")
        if self.orientation == "box":
            euler = np.clip(euler, self.low[3:], self.high[3:])
        else:
            euler = clip_euler_to_target_window(
                euler=euler,
                target_euler=self.target_euler,
                lower_euler=self.low[3:],
                upper_euler=self.high[3:],
            )
        clipped[3:] = R.from_euler("xyz", euler).as_quat()
        for obstacle in self.obstacles:
            if obstacle.contains(clipped[:3]):
                clipped[:3] = obstacle.entry(np.asarray(start[:3]), clipped[:3])
        return clipped
