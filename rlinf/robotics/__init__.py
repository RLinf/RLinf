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

"""RLinf's robotics layer: parts, and the robots composed from them.

Four concepts:

* **Endpoint** -- anything the robot opens on a machine and later closes. It
  has a location and a lifecycle, and that is all. Both of the following are
  endpoints.
* **Part** -- an endpoint you can read, which is what makes it a component of
  the robot: an arm, an end effector, a camera. It has a policy-facing
  observation and action contract.
* **Connection** -- an endpoint that backs several parts without being one. A
  coupled dual-arm controller, a two-armed SDK session:
  :attr:`~rlinf.robotics.parts.base.Endpoint.exports` says what rides on it, and
  the robot composes those. Reading it would mean nothing, so it is a sibling
  of ``Part`` rather than a part that declines to answer -- which is what keeps
  it out of the robot's tree.
* **Robot** -- a named composition of parts. See :mod:`rlinf.robotics.robot`.

Any endpoint can be placed on a node with
:meth:`~rlinf.robotics.parts.base.Endpoint.spawn`, so a camera can run on the
machine it is plugged into while the policy runs elsewhere.

The scheduler never imports this package, and parts never import the scheduler
except through ``spawn``, which loads :mod:`rlinf.robotics.placement` lazily.

Symbols load lazily so a node without a given robot's SDK can still import
``rlinf.robotics``.
"""

# ruff: noqa: F822

from importlib import import_module
from typing import Any

#: Symbols grouped by the module that defines them, so adding one is a
#: single-line change in the group it belongs to.
_MODULE_GROUPS: dict[str, tuple[str, ...]] = {
    ".parts": (
        "Group",
        "Camera",
        "Connection",
        "ControllablePart",
        "EndEffector",
        "Endpoint",
        "LeggedBase",
        "MobileBase",
        "RobotPart",
        "run_parallel",
    ),
    ".parts.arms": ("ARM_STATE_FIELDS",),
    ".parts.cameras": ("camera_cls", "declare_cameras"),
    ".robot": ("Robot",),
    ".parts.views": ("MethodArm", "MethodCamera", "MethodGripper"),
    ".placement": (
        "LocalPartHandle",
        "PartHandle",
        "PartSpec",
        "Placement",
        "RemoteCamera",
        "RemoteControllablePart",
        "RemoteEndEffector",
        "RemotePart",
        "RemotePartHandle",
        "SubpartRef",
    ),
    ".robots": (
        "DOSW1Robot",
        "DOSW1RobotConfig",
        "DualFrankaConfig",
        "DualFrankaRobot",
        "FrankaConfig",
        "FrankaRobot",
        "GimArmConfig",
        "GimArmRobot",
        "Turtle2Config",
        "Turtle2Robot",
        "FRANKA_BACKENDS",
    ),
    ".adapters": (
        "LegacyObservationAdapter",
        "VectorActionAdapter",
        "VectorActionBinding",
    ),
    ".discovery": (
        "RobotAutoConfig",
        "RobotConfig",
        "RobotDiscovery",
        "RobotInfo",
        "RobotRegistration",
        "build_robot",
        "register_robot",
    ),
}

_MODULE_BY_NAME: dict[str, str] = {
    name: module for module, names in _MODULE_GROUPS.items() for name in names
}

__all__ = sorted(_MODULE_BY_NAME)

#: Discovery classes only exist once every robot module has registered itself.
_DISCOVERY_NAMES = frozenset(_MODULE_GROUPS[".discovery"])


def __getattr__(name: str) -> Any:
    """Import the module owning ``name`` on first access."""
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name, __name__)
    if name in _DISCOVERY_NAMES:
        import_module(".robots", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
