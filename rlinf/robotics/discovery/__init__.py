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

"""Registering a robot type, and finding its hardware on a cluster.

:mod:`.registry` maps a robot's type name to its config, discovery policy, and
builder, so ``build_robot("Franka", ...)`` works from the name alone.
:mod:`.autoconfig` fills a robot config's fields from the environment, which is
how a node describes the hardware plugged into it.
"""

from .autoconfig import RobotAutoConfig
from .registry import (
    RobotConfig,
    RobotDiscovery,
    RobotInfo,
    RobotRegistration,
    build_robot,
    register_robot,
)

__all__ = [
    "RobotAutoConfig",
    "RobotConfig",
    "RobotDiscovery",
    "RobotInfo",
    "RobotRegistration",
    "build_robot",
    "register_robot",
]
