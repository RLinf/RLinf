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

"""Hardware an operator drives, as parts.

A leader arm is an arm with encoders, a glove reports finger angles, a
spacemouse reports a twist. They connect, they report a reading, they
disconnect, and they are plugged into a particular machine -- so they are
:class:`~rlinf.robotics.parts.base.RobotPart` implementations like any other
piece of hardware, and get placement and lifecycle for free.

What a reading *means* for a robot is not their concern. That mapping lives in
:mod:`rlinf.robotics.teleop`.
"""

from .devices import Glove, PicoController, SpaceMouse, TeleopLeaderArm, TeleopPart

__all__ = [
    "Glove",
    "PicoController",
    "SpaceMouse",
    "TeleopLeaderArm",
    "TeleopPart",
]
