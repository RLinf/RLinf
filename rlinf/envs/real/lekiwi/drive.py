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

"""LeKiwi drive: command the base under bounded body velocities.

The smallest task that exercises the whole path -- compose the robot, read
it, command it, and watch the camera -- so a new base can be brought up
before any navigation task is written for it.
"""

from typing import Any, Optional

from rlinf.robotics.discovery import RobotInfo
from rlinf.scheduler import WorkerInfo

from .base import LeKiwiEnv, LeKiwiEnvConfig


class LeKiwiDriveEnv(LeKiwiEnv):
    """Drive the base under commanded velocities."""

    def __init__(
        self,
        override_cfg: dict[str, Any],
        worker_info: Optional[WorkerInfo] = None,
        robot_info: "Optional[RobotInfo[Any]]" = None,
        env_idx: int = 0,
    ) -> None:
        config = LeKiwiEnvConfig(**override_cfg)
        super().__init__(config, worker_info, robot_info, env_idx)

    @property
    def task_description(self) -> str:
        """Natural-language task name, for policies that condition on one."""
        return "drive the mobile base"
