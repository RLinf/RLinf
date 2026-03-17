# Copyright 2026 Shirui Chen
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
class YAMHWInfo(HardwareInfo):
    """Hardware information for a YAM bimanual robot system."""

    config: "YAMConfig"


@Hardware.register()
class YAMRobot(Hardware):
    """Hardware policy for YAM bimanual robotic systems.

    YAM (Yet Another Manipulator) is a bimanual robot platform.
    Each arm is identified by an IP address; optionally one or more camera
    serial numbers can be pinned to the node.
    """

    HW_TYPE = "YAM"

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list["YAMConfig"]] = None
    ) -> Optional[HardwareResource]:
        """Enumerate the YAM robot resources on a node.

        Args:
            node_rank: The rank of the node being enumerated.
            configs: Per-node YAM configurations from the cluster YAML.

        Returns:
            Optional[HardwareResource]: Hardware resource if any YAM configs
                match this node, otherwise None.
        """
        assert configs is not None, (
            "YAM hardware requires explicit configurations (left_ip, right_ip) "
            "for its controller nodes."
        )

        yam_configs: list["YAMConfig"] = []
        for config in configs:
            if isinstance(config, YAMConfig) and config.node_rank == node_rank:
                yam_configs.append(config)

        if not yam_configs:
            return None

        yam_infos = []
        cameras = cls.enumerate_cameras()

        for config in yam_configs:
            # Auto-detect cameras if not specified
            if config.camera_serials is None:
                config.camera_serials = list(cameras)

            yam_infos.append(
                YAMHWInfo(
                    type=cls.HW_TYPE,
                    model=cls.HW_TYPE,
                    config=config,
                )
            )

            if config.disable_validate:
                continue

            # Validate IP connectivity for both arms
            for arm_label, arm_ip in [
                ("left", config.left_ip),
                ("right", config.right_ip),
            ]:
                if arm_ip is None:
                    continue
                try:
                    from icmplib import ping
                except ImportError:
                    raise ImportError(
                        f"icmplib is required for YAM robot IP connectivity check, "
                        f"but it is not installed on node rank {node_rank}."
                    )
                try:
                    response = ping(arm_ip, count=2, timeout=1)
                    if not response.is_alive:
                        raise ConnectionError
                except ConnectionError as e:
                    raise ConnectionError(
                        f"Cannot reach YAM {arm_label} arm at IP {arm_ip} "
                        f"from node rank {node_rank}. Error: {e}"
                    )
                except PermissionError as e:
                    warnings.warn(
                        f"Permission denied when pinging YAM {arm_label} arm at {arm_ip} "
                        f"from node rank {node_rank}. Ignoring ping test. Error: {e}"
                    )
                except Exception as e:
                    warnings.warn(
                        f"Unexpected error pinging YAM {arm_label} arm at {arm_ip} "
                        f"from node rank {node_rank}. Ignoring ping test. Error: {e}"
                    )

            # Validate camera serials if provided
            if config.camera_serials:
                try:
                    importlib.import_module("pyrealsense2")
                except ModuleNotFoundError:
                    raise ModuleNotFoundError(
                        f"pyrealsense2 is required for YAM camera serial check, "
                        f"but it is not installed on node rank {node_rank}."
                    )
                for serial in config.camera_serials:
                    if serial not in cameras:
                        raise ValueError(
                            f"Camera with serial {serial} for YAM robot is not "
                            f"connected to node rank {node_rank}. "
                            f"Available cameras: {cameras}."
                        )

        return HardwareResource(type=cls.HW_TYPE, infos=yam_infos)

    @classmethod
    def enumerate_cameras(cls) -> set[str]:
        """Enumerate connected RealSense camera serial numbers."""
        cameras: set[str] = set()
        try:
            import pyrealsense2 as rs
        except ImportError:
            return cameras
        for device in rs.context().devices:
            cameras.add(device.get_info(rs.camera_info.serial_number))
        return cameras


@NodeHardwareConfig.register_hardware_config(YAMRobot.HW_TYPE)
@dataclass
class YAMConfig(HardwareConfig):
    """Configuration for a YAM bimanual robot system.

    Example YAML fragment::

        hardware:
          type: YAM
          configs:
            - node_rank: 1
              left_ip: 192.168.1.10
              right_ip: 192.168.1.11
    """

    left_ip: str
    """IP address of the left robot arm controller."""

    right_ip: str
    """IP address of the right robot arm controller."""

    camera_serials: Optional[list[str]] = None
    """RealSense camera serial numbers attached to this node.
    When ``None`` all detected cameras are used."""

    disable_validate: bool = False
    """Skip connectivity and camera validation during hardware enumeration."""

    def __post_init__(self):
        """Validate YAM node and controller address settings after init."""
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in YAMConfig must be an integer. Got {type(self.node_rank)}."
        )
        for label, ip in [("left_ip", self.left_ip), ("right_ip", self.right_ip)]:
            try:
                ipaddress.ip_address(ip)
            except ValueError:
                raise ValueError(
                    f"'{label}' in YAMConfig must be a valid IP address. Got {ip!r}."
                )
        if self.camera_serials is not None:
            self.camera_serials = list(self.camera_serials)
