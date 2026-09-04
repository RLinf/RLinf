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

"""Turtle2 mobile-manipulation environment and task registrations."""

from __future__ import annotations

from rlinf.envs.real.registry import register_tasks
from rlinf.robotics.parts.arms.turtle2 import Turtle2RobotState

from .base import Turtle2Env, Turtle2RobotConfig
from .button import ButtonEnv

#: Mapping from Gymnasium IDs to Turtle2 environment classes.
TASKS = {"ButtonEnv-v1": ButtonEnv}

_ENTRY_POINTS = register_tasks(__name__, globals(), TASKS)

__all__ = [
    "TASKS",
    "ButtonEnv",
    "Turtle2Env",
    "Turtle2RobotConfig",
    "Turtle2RobotState",
    *_ENTRY_POINTS,
]
