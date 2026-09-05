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

import glob
import importlib
import ipaddress
import os
import warnings
from dataclasses import dataclass
from typing import Optional, cast

from ..hardware import (
    Hardware,
    HardwareConfig,
    HardwareInfo,
    HardwareResource,
    NodeHardwareConfig,
)
from .auto_config import RobotAutoConfig
from .embodied_runtime import EmbodiedRuntimeCLI


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
        cls, node_rank: int, configs: Optional[list[HardwareConfig]] = None
    ) -> Optional[HardwareResource]:
        """Enumerate the robot resources on a node.

        Args:
            node_rank: The rank of the node being enumerated.
            configs: The configurations for the hardware on a node.

        Returns:
            Optional[HardwareResource]: An object representing the hardware resources. None if no hardware is found.
        """
        configs = configs or []
        robot_configs: list["FrankaConfig"] = []
        for config in configs:
            if isinstance(config, FrankaConfig) and config.node_rank == node_rank:
                robot_configs.append(config)

        # Fill unset fields from env vars (e.g. ``ROBOT_IP``), one value per
        # config when several robots share this node. With no configs given,
        # create one per comma-separated ``ROBOT_IP``. A remote arm's
        # ``robot_ip`` may stay unset here; the controller resolves it from its
        # own node at launch.
        robot_configs = cast(
            list["FrankaConfig"],
            RobotAutoConfig.resolve(
                cast(list[HardwareConfig], robot_configs),
                config_cls=FrankaConfig,
                node_rank=node_rank,
                count_fields=("robot_ip",),
            ),
        )
        runtime = None
        runtime_robots = []
        has_runtime_camera = EmbodiedRuntimeCLI.is_enabled("camctr")
        if EmbodiedRuntimeCLI.is_enabled("rosctr"):
            runtime = EmbodiedRuntimeCLI("rosctr")
            runtime_robots = runtime.list_robots()
            if not robot_configs:
                robot_configs = [
                    FrankaConfig(
                        node_rank=node_rank,
                        robot_ip=robot.get("params", {}).get("robot_ip"),
                        embodied_runtime_robot_id=robot["robotId"],
                    )
                    for robot in runtime_robots
                ]

        if runtime is not None and not runtime_robots and not robot_configs:
            raise ValueError("No Franka robots are registered with embodied-runtime.")
        if not robot_configs:
            return None

        franka_infos = []

        for config in robot_configs:
            if runtime is not None:
                robot_ids = {robot["robotId"] for robot in runtime_robots}
                if config.embodied_runtime_robot_id is None:
                    config.embodied_runtime_robot_id = runtime.resolve_robot_id(
                        config.robot_ip
                    )
                elif config.embodied_runtime_robot_id not in robot_ids:
                    raise ValueError(
                        "Robot ID "
                        f"{config.embodied_runtime_robot_id!r} is not managed by "
                        f"embodied-runtime. Available robot IDs: {sorted(robot_ids)}."
                    )
                if config.robot_ip is None:
                    robot = next(
                        robot
                        for robot in runtime_robots
                        if robot["robotId"] == config.embodied_runtime_robot_id
                    )
                    config.robot_ip = robot.get("params", {}).get("robot_ip")
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

            if runtime is not None and config.robot_ip is None:
                raise ValueError(
                    f"embodied-runtime robot {config.embodied_runtime_robot_id!r} "
                    "does not expose params.robot_ip."
                )
            if config.disable_validate:
                continue

            # Ping only when the IP is known here; a remote arm's IP is
            # resolved later on the controller's node.
            if config.robot_ip is not None and runtime is None:
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
                    f"No {camera_type} cameras are connected to node rank "
                    f"{node_rank} while Franka robot requires at least one camera."
                )
            for serial in config.camera_serials:
                if has_runtime_camera:
                    EmbodiedRuntimeCLI("camctr").resolve_camera_id(serial)
                elif serial not in cameras:
                    raise ValueError(
                        f"Camera with serial {serial} is not connected to "
                        f"node rank {node_rank}. Available {camera_type} "
                        f"cameras: {cameras}."
                    )

        return HardwareResource(type=cls.HW_TYPE, infos=franka_infos)

    @classmethod
    def enumerate_cameras(cls, camera_type: str = "realsense") -> set[str]:
        """Enumerate connected camera serial numbers.

        Args:
            camera_type: ``"realsense"``, ``"zed"``, ``"lumos"``, or
                ``"embodied_runtime"``.
        """
        cameras: set[str] = set()
        if EmbodiedRuntimeCLI.is_enabled("camctr"):
            return {
                camera.get("serialNumber") or camera["cameraId"]
                for camera in EmbodiedRuntimeCLI("camctr").list_cameras()
            }
        ct = camera_type.lower()
        if ct in ("embodied_runtime", "runtime"):
            return cameras
        if ct == "zed":
            try:
                import pyzed.sl as sl
            except ImportError:
                return cameras
            for dev in sl.Camera.get_device_list():
                cameras.add(str(dev.serial_number))
        elif ct == "lumos":
            devices = glob.glob("/dev/v4l/by-id/*")
            if not devices:
                devices = glob.glob("/dev/video*")
            cameras.update(os.path.basename(device) for device in devices)
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
        if EmbodiedRuntimeCLI.is_enabled("camctr"):
            return
        ct = camera_type.lower()
        if ct in ("embodied_runtime", "runtime"):
            raise RuntimeError(
                "camera_type='embodied_runtime' requires an enabled camctr CLI "
                f"on node rank {node_rank}."
            )
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


@NodeHardwareConfig.register_hardware_config(FrankaRobot.HW_TYPE)
@dataclass
class FrankaConfig(HardwareConfig):
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
    """Camera backend: ``"realsense"``, ``"zed"``, ``"lumos"``, or
    ``"embodied_runtime"``."""

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

    embodied_runtime_robot_id: Optional[str] = None
    """Robot ID managed by embodied-runtime, auto-detected when possible."""

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
