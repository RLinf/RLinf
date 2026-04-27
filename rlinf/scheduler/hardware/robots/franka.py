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
from dataclasses import dataclass
from typing import Optional

from ..hardware import (
    Hardware,
    HardwareConfig,
    HardwareInfo,
    HardwareResource,
    NodeHardwareConfig,
)


@dataclass
class FrankaHWInfo(HardwareInfo):
    """Hardware information for a robotic system."""

    config: "FrankaConfig"


@Hardware.register()
class FrankaRobot(Hardware):
    """Hardware policy for robotic systems."""

    HW_TYPE = "Franka"
    ROBOT_PING_COUNT: int = 2
    ROBOT_PING_TIMEOUT: int = 1  # in seconds

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list["FrankaConfig"]] = None
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

        if robot_configs:
            franka_infos = []

            for config in robot_configs:
                camera_type = getattr(config, "camera_type", "realsense")
                cameras = cls.enumerate_cameras(camera_type)

                # Use auto-detected cameras when not explicitly specified
                if config.camera_serials is None:
                    config.camera_serials = list(cameras)

                franka_infos.append(
                    FrankaHWInfo(
                        type=cls.HW_TYPE,
                        model=cls.HW_TYPE,
                        config=config,
                    )
                )

                if config.disable_validate:
                    continue

                # Validate IP connectivity
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
            camera_type: ``"realsense"`` or ``"zed"``.
        """
        cameras: set[str] = set()
        if camera_type.lower() == "zed":
            try:
                import pyzed.sl as sl
            except ImportError:
                return cameras
            for dev in sl.Camera.get_device_list():
                cameras.add(str(dev.serial_number))
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
        if camera_type.lower() == "zed":
            try:
                importlib.import_module("pyzed.sl")
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    f"pyzed (ZED SDK) is required for ZED cameras, "
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


@NodeHardwareConfig.register_hardware_config(FrankaRobot.HW_TYPE)
@dataclass
class FrankaConfig(HardwareConfig):
    """Configuration for a robotic system."""

    robot_ip: str
    """IP address of the robotic system."""

    camera_serials: Optional[list[str]] = None
    """List of camera serial numbers associated with the robot."""

    camera_type: str = "realsense"
    """Camera backend: ``"realsense"`` or ``"zed"``."""

    gripper_type: str = "franka"
    """Gripper backend: ``"franka"`` (ROS-based) or ``"robotiq"`` (Modbus RTU)."""

    gripper_connection: Optional[str] = None
    """Serial port for Robotiq grippers (e.g. ``"/dev/ttyUSB0"``).
    Ignored when *gripper_type* is ``"franka"``."""

    controller_node_rank: Optional[int] = None
    """Node rank where :class:`FrankaController` should run.
    When ``None`` (default), the controller is co-located with the env
    worker.  Set this when the arm/gripper and cameras are on different
    machines (e.g. cameras on a GPU server, arm on a NUC)."""

    disable_validate: bool = False
    """Whether to disable validation of robot IP connectivity and camera serials."""

    def __post_init__(self):
        """Post-initialization to validate the configuration."""
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in franka config must be an integer. But got {type(self.node_rank)}."
        )

        try:
            ipaddress.ip_address(self.robot_ip)
        except ValueError:
            raise ValueError(
                f"'robot_ip' in franka config must be a valid IP address. But got {self.robot_ip}."
            )

        if self.camera_serials:
            self.camera_serials = list(self.camera_serials)
