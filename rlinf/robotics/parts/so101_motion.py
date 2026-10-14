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

"""Pure motion planning shared by the SO-101 leader and follower.

The planner only deals in RLinf's canonical units: five joint angles in
radians followed by one gripper opening in ``0..1``. Hardware adapters decide
how each frame is transported and how torque is managed.
"""

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

SO101_DOF = 5
SO101_POSE_SIZE = SO101_DOF + 1
DEFAULT_SO101_FPS = 30.0
QUINTIC_PEAK_SLOPE = 1.875


@dataclass(frozen=True)
class SO101MotionPlan:
    """A sampled quintic path between two canonical SO-101 poses."""

    start: np.ndarray
    target: np.ndarray
    duration: float
    fps: float
    steps: int

    @property
    def period(self) -> float:
        """Return the interval between consecutive command frames."""
        return self.duration / self.steps

    def frames(self) -> Iterator[np.ndarray]:
        """Yield interpolated canonical poses, including the final target."""
        delta = self.target - self.start
        for index in range(1, self.steps + 1):
            phase = index / self.steps
            blend = phase**3 * (10.0 + phase * (-15.0 + 6.0 * phase))
            frame = self.start + blend * delta
            yield np.clip(frame, np.minimum(self.start, self.target), np.maximum(self.start, self.target))


def plan_so101_motion(
    start: np.ndarray,
    target: np.ndarray,
    *,
    minimum_duration: float,
    max_joint_speed: Optional[float] = None,
    fps: float = DEFAULT_SO101_FPS,
) -> SO101MotionPlan:
    """Build a synchronized six-axis SO-101 trajectory.

    ``max_joint_speed`` limits only the five arm joints. The gripper follows
    the same normalized quintic path so all six goals are sent together.
    """
    start_array = np.asarray(start, dtype=float).reshape(-1)
    target_array = np.asarray(target, dtype=float).reshape(-1)
    if start_array.shape != (SO101_POSE_SIZE,) or target_array.shape != (
        SO101_POSE_SIZE,
    ):
        raise ValueError("An SO-101 motion needs six canonical pose values.")
    if not np.all(np.isfinite(start_array)) or not np.all(np.isfinite(target_array)):
        raise ValueError("SO-101 motion poses must be finite.")
    if not 0.0 <= start_array[-1] <= 1.0 or not 0.0 <= target_array[-1] <= 1.0:
        raise ValueError("SO-101 gripper poses must be between 0 and 1.")
    if not np.isfinite(minimum_duration) or minimum_duration <= 0:
        raise ValueError("SO-101 motion duration must be finite and positive.")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("SO-101 motion fps must be finite and positive.")
    if max_joint_speed is not None and (
        not np.isfinite(max_joint_speed) or max_joint_speed <= 0
    ):
        raise ValueError("SO-101 motion speed must be finite and positive.")

    duration = float(minimum_duration)
    if max_joint_speed is not None:
        joint_distance = float(np.max(np.abs(target_array[:SO101_DOF] - start_array[:SO101_DOF])))
        duration = max(
            duration,
            QUINTIC_PEAK_SLOPE * joint_distance / float(max_joint_speed),
        )
    steps = max(1, int(np.ceil(duration * float(fps))))
    return SO101MotionPlan(
        start=start_array.copy(),
        target=target_array.copy(),
        duration=duration,
        fps=float(fps),
        steps=steps,
    )
