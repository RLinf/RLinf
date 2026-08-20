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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, Optional, TypeVar

from rlinf.scheduler.hardware import (
    Hardware,
    HardwareConfig,
    HardwareInfo,
    NodeHardwareConfig,
)

from ..robot import Robot

RobotConfigType = TypeVar("RobotConfigType", bound="RobotConfig", covariant=True)


@dataclass
class RobotConfig(HardwareConfig):
    """Base physical configuration for a registered robot."""


@dataclass
class RobotInfo(HardwareInfo, Generic[RobotConfigType]):
    """Scheduler resource information for a registered robot."""

    config: RobotConfigType


@dataclass(frozen=True)
class RobotRegistration:
    """Everything one registered robot type contributes."""

    robot_cls: type[Robot]
    config_cls: type[RobotConfig]
    discovery_cls: type["RobotDiscovery"]
    build: Optional[Callable[..., Robot]] = None
    """Builder that composes the robot's deferred part declarations."""


class RobotDiscovery(Hardware):
    """Scheduler-facing discovery policy kept separate from robot composition."""

    registry: ClassVar[dict[str, RobotRegistration]] = {}


def build_robot(robot_type: str, **kwargs: Any) -> Robot:
    """Build a registered robot by type name.

    Lets a caller compose a robot from configuration alone, without importing
    that robot's builder directly.
    """
    registration = RobotDiscovery.registry.get(robot_type)
    if registration is None:
        raise KeyError(
            f"Unknown robot type {robot_type!r}. "
            f"Registered: {sorted(RobotDiscovery.registry)}."
        )
    if registration.build is None:
        raise NotImplementedError(f"Robot type {robot_type!r} registered no builder.")
    return registration.build(**kwargs)


def register_robot(
    config_cls: type[RobotConfig],
    robot_cls: type[Robot],
    build: Optional[Callable[..., Robot]] = None,
) -> Callable[[type[RobotDiscovery]], type[RobotDiscovery]]:
    """Register composition, config, discovery, and builder for one robot type.

    One call per robot, made from that robot's own module, so adding hardware
    does not edit a central table.
    """

    def decorator(discovery_cls: type[RobotDiscovery]) -> type[RobotDiscovery]:
        if not issubclass(discovery_cls, RobotDiscovery):
            raise TypeError(
                f"{discovery_cls.__name__} must inherit from RobotDiscovery."
            )
        if not issubclass(config_cls, RobotConfig):
            raise TypeError(f"{config_cls.__name__} must inherit from RobotConfig.")
        if not issubclass(robot_cls, Robot):
            raise TypeError(f"{robot_cls.__name__} must inherit from Robot.")

        robot_type = discovery_cls.HW_TYPE
        if not robot_type:
            raise ValueError(f"{discovery_cls.__name__}.HW_TYPE must be set.")
        if robot_cls.ROBOT_TYPE != robot_type:
            raise ValueError(
                f"{robot_cls.__name__}.ROBOT_TYPE must equal {robot_type!r}."
            )
        if (
            robot_type in RobotDiscovery.registry
            or robot_type in Hardware.hw_types
            or robot_type in NodeHardwareConfig._hardware_config_registry
        ):
            raise ValueError(f"Robot type {robot_type!r} is already registered.")

        Hardware.register()(discovery_cls)
        NodeHardwareConfig.register_hardware_config(robot_type)(config_cls)
        RobotDiscovery.registry[robot_type] = RobotRegistration(
            robot_cls=robot_cls,
            config_cls=config_cls,
            discovery_cls=discovery_cls,
            build=build,
        )
        return discovery_cls

    return decorator
