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

from __future__ import annotations

import ipaddress
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
class DualFrankaHWInfo(HardwareInfo):
    """Hardware information for a dual-arm Franka robotic system."""

    config: "DualFrankaConfig"


@Hardware.register()
class DualFrankaRobot(Hardware):
    """Hardware policy for dual-arm Franka robotic systems.

    Both arms are managed by a single :class:`DualFrankaEnv` instance
    running on the ``node_rank`` specified in the config.  Each arm's
    :class:`FrankaController` can optionally be placed on a different
    node via ``left_controller_node_rank`` / ``right_controller_node_rank``.
    """

    HW_TYPE = "DualFranka"

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list["DualFrankaConfig"]] = None
    ) -> Optional[HardwareResource]:
        """Enumerate the dual-arm robot resources on a node.

        Args:
            node_rank: The rank of the node being enumerated.
            configs: The configurations for the hardware on a node.

        Returns:
            Hardware resource descriptor, or ``None`` when the node has
            no matching dual-Franka configuration.
        """
        assert configs is not None, (
            "DualFranka hardware requires explicit configurations "
            "for robot IPs and camera serials."
        )
        robot_configs: list["DualFrankaConfig"] = []
        for config in configs:
            if isinstance(config, DualFrankaConfig) and config.node_rank == node_rank:
                robot_configs.append(config)

        if not robot_configs:
            return None

        dual_infos: list[DualFrankaHWInfo] = []
        for config in robot_configs:
            dual_infos.append(
                DualFrankaHWInfo(
                    type=cls.HW_TYPE,
                    model=cls.HW_TYPE,
                    config=config,
                )
            )

        return HardwareResource(type=cls.HW_TYPE, infos=dual_infos)


@NodeHardwareConfig.register_hardware_config(DualFrankaRobot.HW_TYPE)
@dataclass
class DualFrankaConfig(HardwareConfig):
    """Configuration for a dual-arm Franka robotic system.

    The env process (cameras + teleop) always runs on the node indicated
    by :attr:`node_rank`.  Each arm's low-level controller can be placed
    on a separate node via the ``*_controller_node_rank`` fields — this
    is the key mechanism for *Option D* (main controller + remote arm).
    """

    left_robot_ip: str = "0.0.0.0"
    """IP address of the left Franka arm."""

    right_robot_ip: str = "0.0.0.0"
    """IP address of the right Franka arm."""

    left_camera_serials: Optional[list[str]] = None
    """Camera serial numbers for the left arm's wrist camera(s)."""

    right_camera_serials: Optional[list[str]] = None
    """Camera serial numbers for the right arm's wrist camera(s)."""

    base_camera_serials: Optional[list[str]] = None
    """Camera serial numbers for the base (third-person) camera(s)."""

    camera_type: str = "zed"
    """Camera backend: ``"realsense"`` or ``"zed"``."""

    left_gripper_type: str = "franka"
    """Gripper backend for the left arm."""

    right_gripper_type: str = "franka"
    """Gripper backend for the right arm."""

    left_gripper_connection: Optional[str] = None
    """Serial port for the left arm's Robotiq gripper."""

    right_gripper_connection: Optional[str] = None
    """Serial port for the right arm's Robotiq gripper."""

    left_controller_node_rank: Optional[int] = None
    """Node rank for the left arm's FrankaController.
    ``None`` means co-located with the env worker."""

    right_controller_node_rank: Optional[int] = None
    """Node rank for the right arm's FrankaController.
    ``None`` means co-located with the env worker."""

    def __post_init__(self):  # noqa: D105
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in DualFranka config must be an integer. "
            f"But got {type(self.node_rank)}."
        )
        for label, ip in [
            ("left_robot_ip", self.left_robot_ip),
            ("right_robot_ip", self.right_robot_ip),
        ]:
            try:
                ipaddress.ip_address(ip)
            except ValueError:
                raise ValueError(
                    f"'{label}' in DualFranka config must be a valid IP address. "
                    f"But got {ip}."
                )
        if self.left_camera_serials:
            self.left_camera_serials = list(self.left_camera_serials)
        if self.right_camera_serials:
            self.right_camera_serials = list(self.right_camera_serials)
        if self.base_camera_serials:
            self.base_camera_serials = list(self.base_camera_serials)
