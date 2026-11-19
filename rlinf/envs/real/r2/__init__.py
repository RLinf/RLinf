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

"""Robbyant R2 tasks, and the env they are built on."""

from __future__ import annotations

from rlinf.envs.real.registry import register_tasks
from rlinf.robotics.parts.arms.r2 import R2RobotState

from .base import R2Env, R2EnvConfig
from .reach import R2ReachConfig, R2ReachEnv

TASKS = {"R2ReachEnv-v1": R2ReachEnv}

_ENTRY_POINTS = register_tasks(__name__, globals(), TASKS)

__all__ = [
    "TASKS",
    "R2Env",
    "R2EnvConfig",
    "R2ReachConfig",
    "R2ReachEnv",
    "R2RobotState",
    *_ENTRY_POINTS,
]
