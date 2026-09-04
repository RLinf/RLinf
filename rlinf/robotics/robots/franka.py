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

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

from ..discovery import (
    RobotConfig,
)
from ..parts.arms.base import Arm, CartesianCompliance
from ..parts.cameras import Camera
from ..parts.end_effectors import (
    EndEffector,
    EndEffectorType,
    normalize_end_effector_type,
)
from ..robot import Robot


class FrankaRobot(Robot):
    """Composable Franka robot.

    Single-arm by default. :class:`~..dual_franka.DualFrankaRobot` inherits the
    declaration logic and only changes the backend and the arm count.
    """

    ROBOT_TYPE = "Franka"

    #: Driver that reaches the built-in Franka Hand, by arm backend.
    #:
    #: The hand is one device with two drivers, and which applies follows the
    #: arm this robot is built on: a ROS stack publishes to it, libfranka opens
    #: its own session. A config that names the hand outright -- as
    #: ``end_effector_type: franky_gripper`` -- is taken at its word instead.
    HAND_BACKENDS: ClassVar[dict[str, str]] = {
        "franka_ros": "franka_gripper",
        "franky": "franky_gripper",
    }

    BACKEND: str = "franka_ros"
    """Registered arm backend used by this robot.

    Subclasses may select another backend while reusing the same composition.
    See ``Arm.backends()`` for the available names.
    """

    @classmethod
    def declare_arm(
        cls,
        robot_ip: Optional[str],
        *,
        node_rank: int,
        name: str,
        backend: Optional[str] = None,
        **settings: Any,
    ) -> "Arm":
        """Declare the arm alone. Its end effector is composed beside it."""
        return Arm.backend(backend or cls.BACKEND).declare(
            cls.resolved_ip(robot_ip, node_rank=node_rank, name=name),
            node_rank=node_rank,
            worker_name=name,
            **settings,
        )

    @classmethod
    def declare_end_effector(
        cls,
        robot_ip: Optional[str],
        *,
        node_rank: int,
        name: str,
        backend: Optional[str] = None,
        gripper_type: Optional[str] = None,
        gripper_connection: Optional[str] = None,
        end_effector_type: Optional[str] = None,
        end_effector_config: Optional[dict[str, Any]] = None,
    ) -> "EndEffector":
        """Declare the end effector this robot carries.

        It opens its own connection, so it is placed and connected on its own,
        and any arm backend composes with any end effector. It is offered every
        attachment one might be reached through -- a ROS session, a serial
        port, the arm's IP -- and each driver takes the one it uses.

        Args:
            robot_ip: Address of the arm the end effector is mounted on.
            node_rank: Node the end effector is wired to.
            name: Worker name when it is hosted remotely.
            backend: Arm backend this robot is built on, which decides how a
                built-in Franka Hand is reached. See :attr:`HAND_BACKENDS`.
            gripper_type: Gripper fitted, when the config names one that way.
            gripper_connection: Serial port, for a gripper reached over one.
            end_effector_type: End effector fitted, naming a driver outright.
            end_effector_config: Settings passed to that driver.
        """
        return EndEffector.of(
            cls.resolved_end_effector_type(
                backend=backend,
                gripper_type=gripper_type,
                end_effector_type=end_effector_type,
            ),
            robot_ip=cls.resolved_ip(robot_ip, node_rank=node_rank, name=name),
            port=gripper_connection,
            node_rank=node_rank,
            worker_name=name,
            **(end_effector_config or {}),
        )

    @classmethod
    def resolved_end_effector_type(
        cls,
        *,
        backend: Optional[str] = None,
        gripper_type: Optional[str] = None,
        end_effector_type: Optional[str] = None,
    ) -> str:
        """Return the driver name for the end effector this robot carries."""
        resolved = normalize_end_effector_type(
            end_effector_type or "franka_gripper", gripper_type
        )
        if resolved is not EndEffectorType.FRANKA_GRIPPER:
            return resolved.value
        # Only the built-in hand is ambiguous: it is the one device this
        # repository reaches two ways.
        return cls.HAND_BACKENDS.get(backend or cls.BACKEND, resolved.value)

    @staticmethod
    def resolved_ip(robot_ip: Optional[str], *, node_rank: int, name: str) -> str:
        """Return the configured arm IP, or the one enumerated on its node."""
        resolved = robot_ip or resolve_robot_ip(node_rank)
        if not resolved:
            raise ValueError(
                f"Franka part {name!r} has no 'robot_ip' and none could be "
                f"resolved from node rank {node_rank}'s hardware infos."
            )
        return resolved

    @classmethod
    def build_arms(
        cls,
        *,
        robot_ip: Optional[str],
        node_rank: int,
        worker_rank: int = 0,
        env_idx: int = 0,
        backend: Optional[str] = None,
        compliance: Optional[CartesianCompliance] = None,
        end_effector_node_rank: Optional[int] = None,
        end_effector_type: Optional[str] = None,
        end_effector_config: Optional[dict] = None,
        gripper_type: Optional[str] = None,
        gripper_connection: Optional[str] = None,
    ) -> dict[str, Any]:
        """Return the robot's named arm and end-effector declarations.

        The two are siblings rather than one inside the other: an end effector
        opens its own connection, so it keeps its own lifecycle and can sit on
        a different node than the arm it is mounted on. Subclasses can override
        this method to compose a different layout.
        """
        return {
            "arm": cls.declare_arm(
                robot_ip,
                node_rank=node_rank,
                name=f"{cls.ROBOT_TYPE}Arm-{worker_rank}-{env_idx}",
                backend=backend,
                compliance=compliance,
            ),
            "end_effector": cls.declare_end_effector(
                robot_ip,
                backend=backend,
                node_rank=node_rank
                if end_effector_node_rank is None
                else end_effector_node_rank,
                name=f"{cls.ROBOT_TYPE}EndEffector-{worker_rank}-{env_idx}",
                gripper_type=gripper_type,
                gripper_connection=gripper_connection,
                end_effector_type=end_effector_type,
                end_effector_config=end_effector_config,
            ),
        }

    @classmethod
    def build_cameras(
        cls,
        cameras: Optional[Mapping[str, Any]] = None,
        *,
        node_rank: Optional[int] = None,
    ) -> dict[str, Any]:
        """Return the robot's named camera declarations."""
        return Camera.declare(cameras, node_rank=node_rank)

    @classmethod
    def build(
        cls,
        *,
        cameras: Optional[Mapping[str, Any]] = None,
        camera_node_rank: Optional[int] = None,
        **config: Any,
    ) -> "FrankaRobot":
        """Compose a Franka robot from its declared arms and cameras."""
        return cls(
            **cls.build_arms(**config),
            **cls.build_cameras(cameras, node_rank=camera_node_rank),
        )


@dataclass
class FrankaConfig(RobotConfig):
    """Configuration for a robotic system."""

    REQUIRES_CAMERA = True

    compliance: CartesianCompliance = field(default_factory=CartesianCompliance)
    """Cartesian impedance settings, ignored by backends that own their gains."""

    backend: Optional[str] = None
    """Arm backend this robot runs, such as ``"franka_ros"`` or ``"franky"``.
    ``None`` leaves the choice to the robot class's own :attr:`BACKEND`."""

    robot_ip: Optional[str] = None
    """IP address of the robotic system.
    When unset in YAML it is auto-detected from the ``ROBOT_IP`` environment
    variable on the node where the arm is enumerated. For a remote
    ``controller_node_rank`` it may stay unset here and be resolved by the
    controller from its node's hardware infos at launch."""

    camera_serials: Optional[list[str]] = None
    """List of camera serial numbers associated with the robot."""

    camera_type: str = "realsense"
    """Camera backend: ``"realsense"``, ``"zed"``, or ``"lumos"``."""

    gripper_type: str = "franka"
    """Gripper backend: ``"franka"`` (ROS-based) or ``"robotiq"`` (Modbus RTU)."""

    gripper_connection: Optional[str] = None
    """Serial port for Robotiq grippers (e.g. ``"/dev/ttyUSB0"``).
    Ignored when *gripper_type* is ``"franka"``."""

    camera_node_rank: Optional[int] = None
    """Node the cameras are plugged into.
    ``None`` (default) co-locates them with the env worker. Set this when the
    cameras hang off a different machine than the one running the policy."""

    controller_node_rank: Optional[int] = None
    """Node rank where the arm part should run.
    When ``None`` (default), the arm is co-located with the env
    worker.  Set this when the arm/gripper and cameras are on different
    machines (e.g. cameras on a GPU server, arm on a NUC)."""

    def __post_init__(self) -> None:
        """Post-initialization to validate the configuration."""
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in franka config must be an integer. But got {type(self.node_rank)}."
        )

        if self.camera_serials:
            self.camera_serials = list(self.camera_serials)

        self.compliance = CartesianCompliance.from_config(self.compliance)


def resolve_robot_ip(node_rank: int) -> Optional[str]:
    """Resolve a robot IP from hardware enumerated on a cluster node."""
    from rlinf.scheduler import Cluster

    try:
        node_info = Cluster().get_node_info(node_rank)
    except Exception:
        return None
    for resource in node_info.hardware_resources:
        for info in resource.infos:
            robot_ip = getattr(getattr(info, "config", None), "robot_ip", None)
            if robot_ip:
                return robot_ip
    return None


FrankaRobot.register_type(FrankaConfig)
