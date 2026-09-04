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

"""Single- and dual-arm Franka environments and task registrations."""

from __future__ import annotations

from rlinf.envs.real.registry import register_tasks
from rlinf.robotics.parts.end_effectors.base import EndEffectorType

from .base import FrankaEnv, FrankaRobotConfig, FrankaRobotState
from .bin_relocation import FrankaBinRelocationEnv
from .bottle import BottleEnv
from .dex_pnp import DexpnpEnv
from .dual_base import DualFrankaEnv, DualFrankaRobotConfig
from .dual_franka_joint import DualFrankaJointEnv, DualFrankaJointRobotConfig
from .dual_franka_tcp import DualFrankaTCPEnv, DualFrankaTCPRobotConfig
from .peg_insertion import PegInsertionEnv

#: Mapping from Gymnasium IDs to registered Franka environment classes.
TASKS: dict[str, type] = {
    "FrankaEnv-v1": FrankaEnv,
    "PegInsertionEnv-v1": PegInsertionEnv,
    "FrankaBinRelocationEnv-v1": FrankaBinRelocationEnv,
    "BottleEnv-v1": BottleEnv,
    "DexpnpEnv-v1": DexpnpEnv,
    "DualFrankaJointEnv-v1": DualFrankaJointEnv,
    "DualFrankaTCPEnv-v1": DualFrankaTCPEnv,
}

_ENTRY_POINTS = register_tasks(__name__, globals(), TASKS)

__all__ = [
    "TASKS",
    "BottleEnv",
    "DexpnpEnv",
    "DualFrankaEnv",
    "DualFrankaJointEnv",
    "DualFrankaJointRobotConfig",
    "DualFrankaRobotConfig",
    "DualFrankaTCPEnv",
    "DualFrankaTCPRobotConfig",
    "EndEffectorType",
    "FrankaBinRelocationEnv",
    "FrankaEnv",
    "FrankaRobotConfig",
    "FrankaRobotState",
    "PegInsertionEnv",
    *_ENTRY_POINTS,
]
