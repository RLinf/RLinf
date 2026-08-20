# Copyright 2025 The RLinf Authors.
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

import importlib
import ipaddress
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

from rlinf.scheduler.hardware import HardwareConfig, HardwareInfo, HardwareResource

from ..discovery import (
    RobotAutoConfig,
    RobotConfig,
    RobotDiscovery,
    RobotInfo,
)
from ..parts.cameras import declare_cameras
from ..robot import Robot


class FrankaRobot(Robot):
    """Composable Franka robot.

    Single-arm by default. :class:`~..dual_franka.DualFrankaRobot` inherits the
    declaration logic and only changes the backend and the arm count.
    """

    ROBOT_TYPE = "Franka"

    BACKEND: str = "franka_ros"
    """Arm implementation this robot drives. See :data:`FRANKA_BACKENDS`."""

    @classmethod
    def declare_arm(
        cls,
        robot_ip: Optional[str],
        *,
        node_rank: int,
        name: str,
        backend: Optional[str] = None,
        gripper_type: str = "franka",
        gripper_connection: Optional[str] = None,
        end_effector_type: Optional[str] = None,
        end_effector_config: Optional[dict] = None,
    ):
        """Declare one whole arm: its motion and the end effector it carries.

        The end effector rides the arm's own connection, so it comes with the
        arm rather than being composed beside it.
        """
        backend = backend or cls.BACKEND
        resolved_ip = robot_ip or resolve_robot_ip(node_rank)
        if not resolved_ip:
            raise ValueError(
                f"Franka arm {name!r} has no 'robot_ip' and none could be "
                f"resolved from node rank {node_rank}'s hardware infos."
            )
        part_cls, spawn_args = franka_arm_cls(backend), FRANKA_BACKENDS[backend][1]
        return part_cls.at(
            *spawn_args(
                resolved_ip,
                gripper_type=gripper_type,
                gripper_connection=gripper_connection,
                end_effector_type=end_effector_type,
                end_effector_config=end_effector_config,
            ),
            node_rank=node_rank,
            name=name,
        )

    @classmethod
    def build_arms(
        cls,
        *,
        robot_ip: Optional[str],
        node_rank: int,
        worker_rank: int = 0,
        env_idx: int = 0,
        end_effector_type: Optional[str] = None,
        end_effector_config: Optional[dict] = None,
        gripper_connection: Optional[str] = None,
    ) -> dict[str, Any]:
        """The arms this robot carries, by name.

        Override this to give a robot a different number of arms; everything
        else about building stays the same.
        """
        connection = cls.declare_arm(
            robot_ip,
            node_rank=node_rank,
            name=f"{cls.ROBOT_TYPE}Arm-{worker_rank}-{env_idx}",
            gripper_connection=gripper_connection,
            end_effector_type=end_effector_type,
            end_effector_config=end_effector_config,
        )
        return {
            "arm": connection.export("arm"),
            "end_effector": connection.export("end_effector"),
        }

    @classmethod
    def build_cameras(
        cls,
        cameras: Optional[Mapping[str, Any]] = None,
        *,
        node_rank: Optional[int] = None,
    ) -> dict[str, Any]:
        """The cameras this robot carries, each placed where it is plugged in."""
        return declare_cameras(cameras, node_rank=node_rank)

    @classmethod
    def build(
        cls,
        *,
        cameras: Optional[Mapping[str, Any]] = None,
        camera_node_rank: Optional[int] = None,
        **config: Any,
    ) -> "FrankaRobot":
        """Compose this robot from the parts it is made of.

        Everything that varies between Franka robots lives in ``build_arms``, so
        a variant with a different number of arms overrides that alone and
        inherits this.
        """
        return cls(
            **cls.build_arms(**config),
            **cls.build_cameras(cameras, node_rank=camera_node_rank),
        )


class FrankaDiscovery(RobotDiscovery):
    """Discover configured Franka robots."""

    HW_TYPE = FrankaRobot.ROBOT_TYPE
    ROBOT_PING_COUNT: int = 2
    ROBOT_PING_TIMEOUT: int = 1  # in seconds

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list[HardwareConfig]] = None
    ) -> Optional[HardwareResource]:
        """Enumerate the robot resources on a node.

        Args:
            node_rank: The rank of the node being enumerated.
            configs: The configurations for the hardware on a node.

        Returns:
            Optional[HardwareResource]: An object representing the hardware resources. None if no hardware is found.
        """
        assert configs is not None, (
            "Robot hardware requires explicit configurations for robot IP and camera serials for its controller nodes."
        )
        robot_configs: list["FrankaConfig"] = []
        for config in configs:
            if isinstance(config, FrankaConfig) and config.node_rank == node_rank:
                robot_configs.append(config)

        # Fill unset fields from env vars (e.g. ``ROBOT_IP``), one value per
        # config when several robots share this node. With no configs given,
        # create one per comma-separated ``ROBOT_IP``. A remote arm's
        # ``robot_ip`` may stay unset here; the controller resolves it from its
        # own node at launch.
        robot_configs = RobotAutoConfig.resolve(
            robot_configs,
            config_cls=FrankaConfig,
            node_rank=node_rank,
            count_fields=("robot_ip",),
        )

        if robot_configs:
            franka_infos: list[HardwareInfo] = []

            for config in robot_configs:
                camera_type = getattr(config, "camera_type", "realsense")
                cameras = cls.enumerate_cameras(camera_type)

                # Use auto-detected cameras when not explicitly specified
                if config.camera_serials is None:
                    config.camera_serials = list(cameras)

                franka_infos.append(
                    RobotInfo(
                        type=cls.HW_TYPE,
                        model=cls.HW_TYPE,
                        config=config,
                    )
                )

                if config.disable_validate:
                    continue

                # Ping only when the IP is known here; a remote arm's IP is
                # resolved later on the controller's node.
                if config.robot_ip is not None:
                    try:
                        from icmplib import ping
                    except ImportError:
                        raise ImportError(
                            f"icmplib is required for Franka robot IP connectivity check, but it is not installed on the node with rank {node_rank}."
                        )
                    try:
                        response = ping(
                            config.robot_ip,
                            count=cls.ROBOT_PING_COUNT,
                            timeout=cls.ROBOT_PING_TIMEOUT,
                        )
                        if not response.is_alive:
                            raise ConnectionError
                    except ConnectionError as e:
                        raise ConnectionError(
                            f"Cannot reach Franka robot at IP {config.robot_ip} from node rank {node_rank}. Error: {e}"
                        )
                    except PermissionError as e:
                        warnings.warn(
                            f"Permission denied when trying to ping Franka robot at IP {config.robot_ip} from node rank {node_rank}. "
                            f"This may be due to insufficient permissions to send ICMP packets. Ignoring the ping test. Error: {e}"
                        )
                    except Exception as e:
                        warnings.warn(
                            f"An unexpected error occurred while pinging Franka robot at IP {config.robot_ip} from node rank {node_rank}. Ignoring the ping test. Error: {e}"
                        )

                # Validate camera SDK and serials
                cls._validate_camera_sdk(camera_type, node_rank)
                if not cameras:
                    raise ValueError(
                        f"No {camera_type} cameras are connected to node rank {node_rank} "
                        f"while Franka robot requires at least one camera."
                    )
                for serial in config.camera_serials:
                    if serial not in cameras:
                        raise ValueError(
                            f"Camera with serial {serial} is not connected to node rank {node_rank}. "
                            f"Available {camera_type} cameras: {cameras}."
                        )

            return HardwareResource(type=cls.HW_TYPE, infos=franka_infos)
        return None

    @classmethod
    def enumerate_cameras(cls, camera_type: str = "realsense") -> set[str]:
        """Enumerate connected camera serial numbers.

        Args:
            camera_type: ``"realsense"``, ``"zed"``, or ``"lumos"``.
        """
        cameras: set[str] = set()
        ct = camera_type.lower()
        if ct == "zed":
            try:
                import pyzed.sl as sl
            except ImportError:
                return cameras
            for dev in sl.Camera.get_device_list():
                cameras.add(str(dev.serial_number))
        elif ct == "lumos":
            from rlinf.robotics.parts.cameras.lumos import LumosCamera

            cameras.update(LumosCamera.get_device_serial_numbers())
        else:
            try:
                import pyrealsense2 as rs
            except ImportError:
                return cameras
            for device in rs.context().devices:
                cameras.add(device.get_info(rs.camera_info.serial_number))
        return cameras

    @staticmethod
    def _validate_camera_sdk(camera_type: str, node_rank: int) -> None:
        ct = camera_type.lower()
        if ct == "zed":
            try:
                importlib.import_module("pyzed.sl")
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    f"pyzed (ZED SDK) is required for ZED cameras, "
                    f"but it is not installed on node rank {node_rank}."
                )
        elif ct == "lumos":
            try:
                importlib.import_module("cv2")
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    f"opencv-python (cv2) is required for Lumos V4L2 cameras, "
                    f"but it is not installed on node rank {node_rank}."
                )
        else:
            try:
                importlib.import_module("pyrealsense2")
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    f"pyrealsense2 is required for RealSense cameras, "
                    f"but it is not installed on node rank {node_rank}."
                )


@dataclass
class FrankaConfig(RobotConfig):
    """Configuration for a robotic system."""

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

    disable_validate: bool = False
    """Whether to disable validation of robot IP connectivity and camera serials."""

    def __post_init__(self):
        """Post-initialization to validate the configuration."""
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in franka config must be an integer. But got {type(self.node_rank)}."
        )

        # ``robot_ip`` may be left unset here and resolved later from an
        # environment variable (during enumeration) or from the controller
        # node's hardware infos (at controller launch); only validate when
        # a value is present.
        if self.robot_ip is not None:
            try:
                ipaddress.ip_address(self.robot_ip)
            except ValueError:
                raise ValueError(
                    f"'robot_ip' in franka config must be a valid IP address. But got {self.robot_ip}."
                )

        if self.camera_serials:
            self.camera_serials = list(self.camera_serials)


def resolve_robot_ip(node_rank: int) -> Optional[str]:
    """Read a robot IP off a node's enumerated hardware.

    A remote arm may leave ``robot_ip`` unset in YAML because only the node
    wired to it knows the address. Any process in the cluster can ask, so this
    resolves before placement rather than inside the hosted part.
    """
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


def _franka_ros_spawn_args(robot_ip: str, **arm) -> tuple:
    """Positional arguments for :class:`FrankaROSArm`."""
    return (
        robot_ip,
        "serl_franka_controllers",
        arm.get("end_effector_type") or "franka_gripper",
        arm.get("end_effector_config") or {},
        None,
        arm.get("gripper_connection"),
    )


def _franky_spawn_args(robot_ip: str, **arm) -> tuple:
    """Positional arguments for :class:`FrankyArm`."""
    return (robot_ip, arm.get("gripper_type", "franka"), arm.get("gripper_connection"))


#: Backend name to the arm part that speaks it. The backend is a per-robot
#: choice, not a separate robot type.
def franka_arm_cls(backend: str) -> type:
    """Return the arm class for a backend, imported lazily."""
    if backend not in FRANKA_BACKENDS:
        raise ValueError(
            f"Unknown Franka backend {backend!r}. Supported: {sorted(FRANKA_BACKENDS)}."
        )
    part_name, _ = FRANKA_BACKENDS[backend]
    if part_name == "FrankaROSArm":
        from ..parts.arms.franka_ros import FrankaROSArm

        return FrankaROSArm
    from ..parts.arms.franky import FrankyArm

    return FrankyArm


FRANKA_BACKENDS: dict[str, tuple[str, Any]] = {
    "franka_ros": ("FrankaROSArm", _franka_ros_spawn_args),
    "franky": ("FrankyArm", _franky_spawn_args),
}


FrankaRobot.register(FrankaConfig, FrankaDiscovery)
