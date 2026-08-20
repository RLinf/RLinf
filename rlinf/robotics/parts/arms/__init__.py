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

"""Arms, and the hardware sessions behind them.

Each module owns one vendor connection and the parts it exposes through
:attr:`~rlinf.robotics.parts.base.Endpoint.exports` -- typically the arm
itself plus its end effector, and for coupled hardware several arms at once.

Symbols load lazily so a node without a given vendor SDK can still import this
package.
"""

# ruff: noqa: F822

from importlib import import_module
from typing import Any

#: Canonical arm observation fields shared by the Franka and GimArm families.
#: An arm reports these and nothing else; end-effector values belong to the
#: end-effector part, and camera frames to camera parts.
ARM_STATE_FIELDS: tuple[str, ...] = (
    "tcp_pose",
    "tcp_vel",
    "arm_joint_position",
    "arm_joint_velocity",
    "tcp_force",
    "tcp_torque",
    "arm_jacobian",
)

_MODULE_BY_NAME: dict[str, str] = {
    "DOSW1Arm": ".dosw1",
    "DOSW1RobotState": ".dosw1",
    "DOSW1ConnectionConfig": ".dosw1",
    "DOSW1EndEffector": ".dosw1",
    "DOSW1Connection": ".dosw1",
    "FrankaROSArm": ".franka_ros",
    "FrankaRobotState": ".franka",
    "FrankyArm": ".franky",
    "GimArm": ".gim_arm",
    "GimArmRobotState": ".gim_arm",
    "Turtle2Connection": ".turtle2",
    "Turtle2RobotState": ".turtle2",
}

__all__ = ["ARM_STATE_FIELDS", *sorted(_MODULE_BY_NAME)]


def __getattr__(name: str) -> Any:
    """Load an arm module only when one of its symbols is requested."""
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
