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

"""Arm interfaces and registered hardware backends.

Driver modules load lazily so importing the package does not require every arm
SDK.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    # Static declarations for the names __getattr__ resolves lazily.
    # A test keeps this block synchronized with _MODULE_BY_NAME.
    from .base import (
        ARM_STATE_FIELDS,
        Arm,
        ArmState,
        BaseArm,
    )
    from .dosw1 import (
        DOSW1Arm,
        DOSW1Connection,
        DOSW1EndEffector,
        DOSW1RobotState,
    )
    from .franka import FrankaRobotState
    from .franka_ros import FrankaROSArm
    from .franky import FrankyArm
    from .gim_arm import (
        GimArm,
        GimArmRobotState,
    )
    from .piper import (
        PiperArm,
        PiperRobotState,
    )
    from .so101 import (
        SO101Arm,
        SO101RobotState,
    )
    from .turtle2 import (
        Turtle2Connection,
        Turtle2RobotState,
    )

_MODULE_BY_NAME: dict[str, str] = {
    "ARM_STATE_FIELDS": ".base",
    "Arm": ".base",
    "ArmState": ".base",
    "BaseArm": ".base",
    "DOSW1Arm": ".dosw1",
    "DOSW1RobotState": ".dosw1",
    "DOSW1EndEffector": ".dosw1",
    "DOSW1Connection": ".dosw1",
    "FrankaROSArm": ".franka_ros",
    "FrankaRobotState": ".franka",
    "FrankyArm": ".franky",
    "GimArm": ".gim_arm",
    "GimArmRobotState": ".gim_arm",
    "PiperArm": ".piper",
    "PiperRobotState": ".piper",
    "SO101Arm": ".so101",
    "SO101RobotState": ".so101",
    "Turtle2Connection": ".turtle2",
    "Turtle2RobotState": ".turtle2",
}

__all__ = [
    "ARM_STATE_FIELDS",
    "Arm",
    "ArmState",
    "BaseArm",
    "DOSW1Arm",
    "DOSW1Connection",
    "DOSW1EndEffector",
    "DOSW1RobotState",
    "FrankaROSArm",
    "FrankaRobotState",
    "FrankyArm",
    "GimArm",
    "GimArmRobotState",
    "PiperArm",
    "PiperRobotState",
    "SO101Arm",
    "SO101RobotState",
    "Turtle2Connection",
    "Turtle2RobotState",
]


def load_drivers() -> None:
    """Import all arm modules to populate the backend registry."""
    for module_name in sorted(set(_MODULE_BY_NAME.values())):
        import_module(module_name, __name__)


def __getattr__(name: str) -> Any:
    """Load an arm module only when one of its symbols is requested."""
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
