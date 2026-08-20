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
from typing import Any, Mapping, Optional

from rlinf.scheduler.hardware import HardwareConfig, HardwareInfo, HardwareResource

from ..discovery import (
    RobotAutoConfig,
    RobotConfig,
    RobotDiscovery,
    RobotInfo,
)
from ..parts.base import Group
from ..parts.cameras import declare_cameras
from .franka import FrankaRobot


class DualFrankaRobot(FrankaRobot):
    """Composable dual-arm Franka robot."""

    ROBOT_TYPE = "DualFranka"

    BACKEND = "franky"

    @classmethod
    def build_arms(
        cls,
        *,
        left_robot_ip: Optional[str] = None,
        right_robot_ip: Optional[str] = None,
        left_node_rank: int = 0,
        right_node_rank: int = 0,
        worker_rank: int = 0,
        env_idx: int = 0,
        left_gripper_type: str = "robotiq",
        right_gripper_type: str = "robotiq",
        left_gripper_connection: Optional[str] = None,
        right_gripper_connection: Optional[str] = None,
        arm_cameras: Optional[Mapping[str, Mapping[str, Any]]] = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Two arms instead of one, each with whatever rides on its wrist.

        A wrist camera belongs to the arm it is bolted to, so it is named
        inside that arm's group rather than at the top of the robot.
        """
        if not left_robot_ip or not right_robot_ip:
            raise ValueError("Both Franka robot IPs are required for a dual-arm robot.")

        sides = {
            "left": (
                left_robot_ip,
                left_gripper_type,
                left_gripper_connection,
                left_node_rank,
            ),
            "right": (
                right_robot_ip,
                right_gripper_type,
                right_gripper_connection,
                right_node_rank,
            ),
        }
        arms = {}
        for side, (robot_ip, gripper_type, connection, node_rank) in sides.items():
            declared = cls.declare_arm(
                robot_ip,
                node_rank=node_rank,
                name=f"{cls.ROBOT_TYPE}Arm-{side}-{worker_rank}-{env_idx}",
                gripper_type=gripper_type,
                gripper_connection=connection,
            )
            arms[side] = Group(
                arm=declared.export("arm"),
                end_effector=declared.export("end_effector"),
                **declare_cameras((arm_cameras or {}).get(side), node_rank=node_rank),
            )
        return arms


class DualFrankaDiscovery(RobotDiscovery):
    """Hardware policy for dual-arm Franka robotic systems.

    Both arms are managed by a single :class:`DualFrankaEnv` instance
    running on the ``node_rank`` specified in the config.  Each arm's
    Each arm part can optionally be placed on a different
    node via ``left_controller_node_rank`` / ``right_controller_node_rank``.
    """

    HW_TYPE = DualFrankaRobot.ROBOT_TYPE

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list[HardwareConfig]] = None
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

        # Fill unset fields from env vars (e.g. ``LEFT_ROBOT_IP``), one value per
        # config when several robots share this node. With no configs given,
        # create one per comma-separated arm IP. A remote arm's IP may stay
        # unset here; the controller resolves it from its own node at launch.
        robot_configs = RobotAutoConfig.resolve(
            robot_configs,
            config_cls=DualFrankaConfig,
            node_rank=node_rank,
            count_fields=("left_robot_ip", "right_robot_ip"),
        )

        if not robot_configs:
            return None

        dual_infos: list[HardwareInfo] = []
        for config in robot_configs:
            dual_infos.append(
                RobotInfo(
                    type=cls.HW_TYPE,
                    model=cls.HW_TYPE,
                    config=config,
                )
            )

        return HardwareResource(type=cls.HW_TYPE, infos=dual_infos)


@dataclass
class DualFrankaConfig(RobotConfig):
    """Configuration for a dual-arm Franka robotic system.

    The env process (cameras + teleop) always runs on the node indicated
    by :attr:`node_rank`.  Each arm's low-level controller can be placed
    on a separate node via the ``*_controller_node_rank`` fields — this
    is the key mechanism for *Option D* (main controller + remote arm).
    """

    left_robot_ip: Optional[str] = None
    """IP address of the left Franka arm.
    When unset in YAML it is auto-detected from the ``LEFT_ROBOT_IP``
    environment variable on the node where the arm is enumerated."""

    right_robot_ip: Optional[str] = None
    """IP address of the right Franka arm.
    When unset in YAML it is auto-detected from the ``RIGHT_ROBOT_IP``
    environment variable on the node where the arm is enumerated."""

    left_camera_serials: Optional[list[str]] = None
    """Camera serial numbers for the left arm's wrist camera(s)."""

    right_camera_serials: Optional[list[str]] = None
    """Camera serial numbers for the right arm's wrist camera(s)."""

    base_camera_serials: Optional[list[str]] = None
    """Camera serial numbers for the base (third-person) camera(s)."""

    camera_type: str = "realsense"
    """Default camera backend when a per-slot type is not set.
    Supported: ``"realsense"``, ``"zed"``, ``"lumos"``."""

    base_camera_type: Optional[str] = None
    """Camera backend for the base (third-person) camera(s).
    Falls back to :attr:`camera_type` when ``None``."""

    left_camera_type: Optional[str] = None
    """Camera backend for the left wrist camera(s).
    Falls back to :attr:`camera_type` when ``None``."""

    right_camera_type: Optional[str] = None
    """Camera backend for the right wrist camera(s).
    Falls back to :attr:`camera_type` when ``None``."""

    left_gripper_type: str = "franka"
    """Gripper backend for the left arm."""

    right_gripper_type: str = "franka"
    """Gripper backend for the right arm."""

    left_gripper_connection: Optional[str] = None
    """Serial port for the left arm's Robotiq gripper."""

    right_gripper_connection: Optional[str] = None
    """Serial port for the right arm's Robotiq gripper."""

    left_controller_node_rank: Optional[int] = None
    """Node rank for the left arm part.
    ``None`` means co-located with the env worker."""

    right_controller_node_rank: Optional[int] = None
    """Node rank for the right arm part.
    ``None`` means co-located with the env worker."""

    disable_validate: bool = False
    """Whether to skip validation of the arm addresses.

    Unlike :class:`~rlinf.robotics.robots.franka.FrankaConfig`, this covers only
    the address format: enumeration never pings a dual-arm robot, so there is no
    reachability check to switch off. Set it when the addresses in a config are
    placeholders or belong to hardware that is not there -- an offline run, or a
    bench check against faked SDKs."""

    def __post_init__(self):  # noqa: D105
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in DualFranka config must be an integer. "
            f"But got {type(self.node_rank)}."
        )
        # IPs may be left unset here and resolved later from environment
        # variables during enumeration; only validate the ones present.
        if not self.disable_validate:
            for label, ip in [
                ("left_robot_ip", self.left_robot_ip),
                ("right_robot_ip", self.right_robot_ip),
            ]:
                if ip is not None:
                    self._validate_ip(label, ip)
        if self.left_camera_serials:
            self.left_camera_serials = list(self.left_camera_serials)
        if self.right_camera_serials:
            self.right_camera_serials = list(self.right_camera_serials)
        if self.base_camera_serials:
            self.base_camera_serials = list(self.base_camera_serials)

    @staticmethod
    def _validate_ip(label: str, ip: str) -> None:
        """Validate that ``ip`` is a valid IP address."""
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            raise ValueError(
                f"'{label}' in DualFranka config must be a valid IP address. "
                f"But got {ip}."
            )


DualFrankaRobot.register(DualFrankaConfig, DualFrankaDiscovery)
