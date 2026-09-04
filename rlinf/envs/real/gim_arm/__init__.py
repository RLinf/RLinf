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

"""GimArm tasks, and the env they are built on."""

from __future__ import annotations

from rlinf.envs.real.registry import register_tasks
from rlinf.robotics.parts.arms.gim_arm import GimArmRobotState

from .base import GimArmEnv, GimArmRobotConfig
from .peg_insertion import GimArmPegInsertionEnv

# Use the shared factory even though this task declares no additional wrappers.
TASKS = {"GimArmPegInsertionEnv-v1": GimArmPegInsertionEnv}

_ENTRY_POINTS = register_tasks(__name__, globals(), TASKS)

__all__ = [
    "TASKS",
    "GimArmEnv",
    "GimArmPegInsertionEnv",
    "GimArmRobotConfig",
    "GimArmRobotState",
    *_ENTRY_POINTS,
]
