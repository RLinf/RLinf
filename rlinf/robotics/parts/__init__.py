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

r"""Robot parts: what a component *is* to the policy.

The taxonomy lives in :mod:`.base`; each category's implementations live in the
subpackage named after it::

    parts/
      base.py                     Endpoint, RobotPart, Connection,
                                  ControllablePart, Group, Camera,
                                  EndEffector, MobileBase, LeggedBase
      arms/                       Franky, Franka ROS, GimArm, Turtle2, DOSW1
      cameras/                    RealSense, ZED, Lumos
      end_effectors/
        grippers/                 Franka, Robotiq
        hands/                    Ruiyan
      transports/                 ROS

A part says what a component means to the policy: its observation contract and,
when controllable, its action contract. A link that presents several components
at once -- a dual-arm controller, a two-armed SDK session -- is a
:class:`~.base.Connection` rather
than a part: it lists what rides on it in :attr:`~.base.Endpoint.exports`, and
the robot composes those. Both are :class:`~.base.Endpoint`\ s, so both are
declared, placed, opened and closed the same way.

Subpackages are not imported here: a node needs only the vendor SDKs for the
hardware it actually has. Import ``rlinf.robotics.parts.cameras`` directly.
"""

from .base import (
    Camera,
    Connection,
    ControllablePart,
    EndEffector,
    Endpoint,
    Group,
    LeggedBase,
    MobileBase,
    RobotPart,
    run_parallel,
)

__all__ = [
    "Connection",
    "Endpoint",
    "Group",
    "Camera",
    "ControllablePart",
    "EndEffector",
    "LeggedBase",
    "MobileBase",
    "RobotPart",
    "run_parallel",
]
