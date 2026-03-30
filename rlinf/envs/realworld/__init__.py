# Copyright 2025 The RLinf Authors.
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

from .franka import FrankaEnv, FrankaRobotConfig, FrankaRobotState
from .franka import tasks as franka_tasks
from .realworld_env import RealWorldEnv
from .xsquare import Turtle2Env, Turtle2RobotConfig, Turtle2RobotState
from .xsquare import tasks as xsquare_tasks

RealWorldEnv.realworld_setup()

__all__ = [
    "FrankaEnv",
    "FrankaRobotConfig",
    "FrankaRobotState",
    "franka_tasks",
    "Turtle2Env",
    "Turtle2RobotConfig",
    "Turtle2RobotState",
    "xsquare_tasks",
    "RealWorldEnv",
]
