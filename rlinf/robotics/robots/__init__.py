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

"""Built-in robot compositions and their hardware configurations.

Importing this package registers each robot type and its discovery metadata.
"""

from .dosw1 import DOSW1Robot, DOSW1RobotConfig
from .dual_franka import DualFrankaConfig, DualFrankaRobot
from .franka import FrankaConfig, FrankaRobot
from .gim_arm import GimArmConfig, GimArmRobot
from .piper import PiperConfig, PiperRobot
from .so101 import SO101Config, SO101Robot
from .turtle2 import Turtle2Config, Turtle2Robot

__all__ = [
    "DOSW1Robot",
    "DOSW1RobotConfig",
    "DualFrankaConfig",
    "DualFrankaRobot",
    "FrankaConfig",
    "FrankaRobot",
    "GimArmConfig",
    "GimArmRobot",
    "PiperConfig",
    "PiperRobot",
    "SO101Config",
    "SO101Robot",
    "Turtle2Config",
    "Turtle2Robot",
]
