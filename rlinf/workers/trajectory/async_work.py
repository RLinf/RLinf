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

from collections.abc import Sequence
from typing import Any

from rlinf.scheduler.collective.async_work import AsyncRouteWork, AsyncWork

from .compression import CompressedTrajectory, decompress_trajectory
from .storage import merge_trajectories


class AsyncTrajectoryCommWork(AsyncRouteWork):
    """Wait for all trajectory shards needed by one actor rank."""

    def __init__(self, works: Sequence[AsyncWork]) -> None:
        super().__init__(works, self._merge)

    @staticmethod
    def _merge(trajectories: list[Any]) -> Any:
        trajectories = [
            decompress_trajectory(trajectory)
            if isinstance(trajectory, CompressedTrajectory)
            else trajectory
            for trajectory in trajectories
        ]
        if len(trajectories) == 1:
            return trajectories[0]
        return merge_trajectories(trajectories)
