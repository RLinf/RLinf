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

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, Optional, TypeVar

from rlinf.scheduler.hardware import (
    Hardware,
    HardwareConfig,
    HardwareInfo,
    HardwareResource,
    NodeHardwareConfig,
)

from ..robot import Robot

RobotConfigType = TypeVar("RobotConfigType", bound="RobotConfig", covariant=True)


@dataclass
class RobotConfig(HardwareConfig):
    """Base hardware configuration for a registered robot."""

    #: Whether a node carrying this robot must have at least one camera.
    REQUIRES_CAMERA: ClassVar[bool] = False

    disable_validate: bool = False
    """Skip the node-local hardware checks that enumeration would run.

    Enumeration confirms that the cameras a config names are attached, which
    needs the camera SDK installed on the node doing the enumerating. Set this
    when the config is already trusted and the node has nothing to probe with:
    an offline run, a bench check against faked SDKs, or a scheduler node
    writing configuration for hardware it does not itself carry.

    Only cameras are checked at enumeration. An arm address is not reached
    either way: whether it is an address at all is settled by the arm part
    that dials it.
    """

    def model(self, robot_type: str) -> str:
        """Return the hardware model reported to the scheduler."""
        return robot_type


@dataclass
class RobotInfo(HardwareInfo, Generic[RobotConfigType]):
    """Scheduler resource information for a registered robot."""

    config: RobotConfigType


@dataclass(frozen=True)
class RobotRegistration:
    """Classes and builder associated with one robot type."""

    robot_cls: type[Robot]
    config_cls: type[RobotConfig]
    discovery_cls: type["RobotDiscovery"]
    build: Optional[Callable[..., Robot]] = None
    """Builder that composes the robot's deferred part declarations."""


class RobotDiscovery(Hardware):
    """Discover configured robot resources for the scheduler.

    The standard implementation resolves node-local configuration, prepares
    inferred fields, validates attached hardware, and returns ``RobotInfo``.
    """

    registry: ClassVar[dict[str, RobotRegistration]] = {}

    @classmethod
    def config_cls(cls) -> type[RobotConfig]:
        """Return the config class registered for this robot type."""
        registration = cls.registry.get(cls.HW_TYPE)
        if registration is None:
            raise KeyError(
                f"{cls.__name__} is not registered under {cls.HW_TYPE!r}; "
                "register_robot() is what pairs a discovery with its config."
            )
        return registration.config_cls

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list[HardwareConfig]] = None
    ) -> Optional[HardwareResource]:
        """Describe the robots of this type attached to one node.

        Args:
            node_rank: The rank of the node being enumerated.
            configs: Every hardware config written for this node, of any type.

        Returns:
            Hardware resources on the node, or ``None`` when no matching robot
            is attached.
        """
        assert configs is not None, (
            f"{cls.HW_TYPE} hardware requires explicit configurations; "
            "enumeration reads them rather than probing for the robot."
        )
        # Import lazily to avoid a registry/autoconfig import cycle.
        from .autoconfig import RobotAutoConfig

        config_cls = cls.config_cls()
        mine = [
            config
            for config in configs
            if isinstance(config, config_cls) and config.node_rank == node_rank
        ]
        # Resolve unset fields from node-local environment variables.
        mine = RobotAutoConfig.resolve(mine, config_cls=config_cls, node_rank=node_rank)
        if not mine:
            return None

        infos: list[HardwareInfo] = []
        for config in mine:
            cls.prepare(config, node_rank)
            infos.append(
                RobotInfo(
                    type=cls.HW_TYPE,
                    model=config.model(cls.HW_TYPE),
                    config=config,
                )
            )
            if not config.disable_validate:
                cls.validate(config, node_rank)
        return HardwareResource(type=cls.HW_TYPE, infos=infos)

    @classmethod
    def prepare(cls, config: RobotConfig, node_rank: int) -> None:
        """Populate configuration fields that require node-local discovery."""
        if getattr(config, "camera_serials", ...) is None:
            camera_type = getattr(config, "camera_type", None) or "realsense"
            config.camera_serials = sorted(cls.enumerate_cameras(camera_type))

    @classmethod
    def validate(cls, config: RobotConfig, node_rank: int) -> None:
        """Validate node-local hardware referenced by the robot config."""
        serials = getattr(config, "camera_serials", None)
        if not serials and not config.REQUIRES_CAMERA:
            return
        cls.validate_cameras(
            getattr(config, "camera_type", None) or "realsense",
            serials or (),
            node_rank,
            require_one=config.REQUIRES_CAMERA,
        )

    @classmethod
    def enumerate_cameras(cls, camera_type: str = "realsense") -> set[str]:
        """Return camera identifiers discovered by the selected backend."""
        from ..parts.cameras import BaseCamera

        return BaseCamera.backend(camera_type).discover()

    @classmethod
    def validate_cameras(
        cls,
        camera_type: str,
        serials: "Iterable[str]",
        node_rank: int,
        attached: "Optional[set[str]]" = None,
        require_one: bool = True,
    ) -> None:
        """Validate the camera SDK and configured camera identifiers.

        Args:
            camera_type: The backend the config asked for.
            serials: The camera identifiers the config named.
            node_rank: The node being enumerated, for the error messages.
            attached: Previously enumerated camera identifiers, when available.
            require_one: Whether a robot with no camera at all is an error.
        """
        from ..parts.cameras import BaseCamera

        camera_cls = BaseCamera.backend(camera_type)
        camera_cls.require_sdk(f"node rank {node_rank}")

        present = camera_cls.discover() if attached is None else attached
        if require_one and not present:
            raise ValueError(
                f"No {camera_type} cameras are connected to node rank "
                f"{node_rank}, and this robot requires at least one."
            )
        missing = [serial for serial in serials or () if serial not in present]
        if missing:
            raise ValueError(
                f"Cameras {missing} are not connected to node rank {node_rank}. "
                f"Available {camera_type} cameras: {sorted(present)}."
            )


def build_robot(robot_type: str, **kwargs: Any) -> Robot:
    """Build an unconnected robot from a registered type name."""
    return Robot.of_type(robot_type, **kwargs)


def register_robot(
    config_cls: type[RobotConfig],
    robot_cls: type[Robot],
    build: Optional[Callable[..., Robot]] = None,
) -> Callable[[type[RobotDiscovery]], type[RobotDiscovery]]:
    """Register the config, discovery class, and builder for a robot type."""

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
