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

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from ..discovery import (
    RobotConfig,
)
from ..parts.arms.base import CartesianCompliance
from ..parts.base import PartGroup
from ..parts.cameras import Camera
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
        node_rank: Optional[int] = None,
        left_node_rank: Optional[int] = None,
        right_node_rank: Optional[int] = None,
        worker_rank: int = 0,
        env_idx: int = 0,
        left_gripper_type: str = "robotiq",
        right_gripper_type: str = "robotiq",
        left_gripper_connection: Optional[str] = None,
        right_gripper_connection: Optional[str] = None,
        left_compliance: Optional[CartesianCompliance] = None,
        right_compliance: Optional[CartesianCompliance] = None,
        arm_cameras: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Return the left and right arm groups with their wrist cameras."""
        if not left_robot_ip or not right_robot_ip:
            raise ValueError("Both Franka robot IPs are required for a dual-arm robot.")

        # Individual node ranks override the shared arm placement.
        shared = 0 if node_rank is None else node_rank
        sides = {
            "left": (
                left_robot_ip,
                left_gripper_type,
                left_gripper_connection,
                shared if left_node_rank is None else left_node_rank,
                left_compliance,
            ),
            "right": (
                right_robot_ip,
                right_gripper_type,
                right_gripper_connection,
                shared if right_node_rank is None else right_node_rank,
                right_compliance,
            ),
        }
        arms = {}
        for side, (
            robot_ip,
            gripper_type,
            connection,
            node_rank,
            compliance,
        ) in sides.items():
            arms[side] = PartGroup(
                arm=cls.declare_arm(
                    robot_ip,
                    node_rank=node_rank,
                    name=f"{cls.ROBOT_TYPE}Arm-{side}-{worker_rank}-{env_idx}",
                    compliance=compliance,
                ),
                end_effector=cls.declare_end_effector(
                    robot_ip,
                    backend=cls.BACKEND,
                    node_rank=node_rank,
                    name=f"{cls.ROBOT_TYPE}EndEffector-{side}-{worker_rank}-{env_idx}",
                    gripper_type=gripper_type,
                    gripper_connection=connection,
                ),
                **Camera.declare((arm_cameras or {}).get(side), node_rank=node_rank),
            )
        return arms


@dataclass
class DualFrankaConfig(RobotConfig):
    """Configuration for a dual-arm Franka system.

    ``node_rank`` places the environment process. Each arm controller can be
    placed independently with its ``*_controller_node_rank`` field.
    """

    left_robot_ip: Optional[str] = None
    """IP address of the left Franka arm.
    When unset in YAML it is auto-detected from the ``LEFT_ROBOT_IP``
    environment variable on the node where the arm is enumerated."""

    right_robot_ip: Optional[str] = None
    """IP address of the right Franka arm.
    When unset in YAML it is auto-detected from the ``RIGHT_ROBOT_IP``
    environment variable on the node where the arm is enumerated."""

    compliance: CartesianCompliance = field(default_factory=CartesianCompliance)
    """Cartesian impedance settings shared by both arms.
    Ignored by backends that own their gains."""

    left_compliance: Optional[CartesianCompliance] = None
    """Impedance settings for the left arm. Falls back to :attr:`compliance`."""

    right_compliance: Optional[CartesianCompliance] = None
    """Impedance settings for the right arm. Falls back to :attr:`compliance`."""

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

    def __post_init__(self) -> None:  # noqa: D105
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in DualFranka config must be an integer. "
            f"But got {type(self.node_rank)}."
        )
        if self.left_camera_serials:
            self.left_camera_serials = list(self.left_camera_serials)
        if self.right_camera_serials:
            self.right_camera_serials = list(self.right_camera_serials)
        if self.base_camera_serials:
            self.base_camera_serials = list(self.base_camera_serials)

        self.compliance = CartesianCompliance.from_config(self.compliance)
        for side in ("left_compliance", "right_compliance"):
            given = getattr(self, side)
            if given is not None:
                setattr(self, side, CartesianCompliance.from_config(given))


DualFrankaRobot.register_type(DualFrankaConfig)
