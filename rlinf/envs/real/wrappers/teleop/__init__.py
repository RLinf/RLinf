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

"""Letting an operator take over, on the environment's terms.

What the operator is asking for is worked out in
:mod:`rlinf.robotics.teleop`: devices report, bindings say what a reading means
for the robot, and a group merges them. Everything here is what that cannot
answer on its own, because it depends on the environment.

:class:`TeleopIntervention` decides when the operator's action replaces the
policy's and records that it did. :class:`ComposedTeleop` flattens a group's
named parts into this env's action vector, using the layout from
:mod:`.layout`. :mod:`.config` resolves which devices a config asked for, and
:mod:`.builder` constructs them.
"""

from .composed import ComposedTeleop
from .config import LEGACY_FLAGS, NO_DEVICE, resolve_teleop_devices
from .intervention import TeleopDevice, TeleopIntervention, TeleopSample
from .layout import action_layout
from .streaming import TeleopStreamer

__all__ = [
    "LEGACY_FLAGS",
    "NO_DEVICE",
    "ComposedTeleop",
    "TeleopDevice",
    "TeleopIntervention",
    "TeleopSample",
    "TeleopStreamer",
    "action_layout",
    "resolve_teleop_devices",
]
