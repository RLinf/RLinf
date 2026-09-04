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

"""Public API for robot parts, composition, placement, and discovery.

Connections manage deferred hardware sessions. Robot parts add observation and
action contracts, and robots compose those parts into named trees. Symbols are
loaded lazily so importing this package does not require every vendor SDK.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    # Provide static types for names exported through lazy __getattr__.
    # Tests keep this list synchronized with _MODULE_GROUPS.
    from .adapters import (
        LegacyObservationAdapter,
        VectorActionAdapter,
        VectorActionBinding,
    )
    from .discovery import (
        RobotAutoConfig,
        RobotConfig,
        RobotDiscovery,
        RobotInfo,
        RobotRegistration,
        build_robot,
        register_robot,
    )
    from .parts import (
        Action,
        Connection,
        ControllablePart,
        Features,
        Observation,
        PartGroup,
        RobotPart,
    )
    from .parts.arms import ARM_STATE_FIELDS, Arm, ArmState, BaseArm
    from .parts.cameras import BaseCamera, Camera, CameraInfo
    from .parts.end_effectors import EndEffector
    from .parts.mobility import MobileBase
    from .parts.views import MethodArm, MethodCamera, MethodEndEffector
    from .robot import Robot
    from .robots import (
        DOSW1Robot,
        DOSW1RobotConfig,
        DualFrankaConfig,
        DualFrankaRobot,
        FrankaConfig,
        FrankaRobot,
        GimArmConfig,
        GimArmRobot,
        PiperConfig,
        PiperRobot,
        SO101Config,
        SO101Robot,
        Turtle2Config,
        Turtle2Robot,
    )

#: Public symbols grouped by defining module for lazy loading.
_MODULE_GROUPS: dict[str, tuple[str, ...]] = {
    ".parts": (
        "Action",
        "Connection",
        "ControllablePart",
        "Features",
        "Observation",
        "PartGroup",
        "RobotPart",
    ),
    ".parts.arms": ("ARM_STATE_FIELDS", "Arm", "ArmState", "BaseArm"),
    # Load each device category only when requested.
    ".parts.cameras": ("BaseCamera", "Camera", "CameraInfo"),
    ".parts.end_effectors": ("EndEffector",),
    ".parts.mobility": ("MobileBase",),
    ".robot": ("Robot",),
    ".parts.views": ("MethodArm", "MethodCamera", "MethodEndEffector"),
    ".robots": (
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

__all__ = [
    "ARM_STATE_FIELDS",
    "Action",
    "Arm",
    "ArmState",
    "BaseArm",
    "BaseCamera",
    "Camera",
    "CameraInfo",
    "Connection",
    "ControllablePart",
    "DOSW1Robot",
    "DOSW1RobotConfig",
    "DualFrankaConfig",
    "DualFrankaRobot",
    "EndEffector",
    "Features",
    "FrankaConfig",
    "FrankaRobot",
    "GimArmConfig",
    "GimArmRobot",
    "LegacyObservationAdapter",
    "MethodArm",
    "MethodCamera",
    "MethodEndEffector",
    "MobileBase",
    "Observation",
    "PartGroup",
    "PiperConfig",
    "PiperRobot",
    "Robot",
    "RobotAutoConfig",
    "RobotConfig",
    "RobotDiscovery",
    "RobotInfo",
    "RobotPart",
    "RobotRegistration",
    "SO101Config",
    "SO101Robot",
    "Turtle2Config",
    "Turtle2Robot",
    "VectorActionAdapter",
    "VectorActionBinding",
    "build_robot",
    "register_robot",
]

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
