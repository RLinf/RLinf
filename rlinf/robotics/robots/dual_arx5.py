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

"""Dual ARX X5 robot composition and scheduler configuration."""

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..discovery import RobotConfig
from ..parts.arms.arx5 import ARX5_PARK_MAX_DEG, ARX5Arm
from ..parts.base import PartGroup
from ..parts.cameras import Camera, CameraInfo
from ..robot import Robot


class DualARX5Robot(Robot):
    """Two X5 arms with SDK grippers and optional named cameras."""

    ROBOT_TYPE = "DualARX5"

    @classmethod
    def build(
        cls,
        *,
        left_can_port: str,
        right_can_port: str,
        arm_type: int,
        node_rank: int | None = None,
        motion_duration: float = 0.1,
        park_max_deg: float = ARX5_PARK_MAX_DEG,
        left_urdf_path: str | None = None,
        right_urdf_path: str | None = None,
        cameras: Mapping[str, CameraInfo] | None = None,
    ) -> "DualARX5Robot":
        if left_can_port == right_can_port:
            raise ValueError("Each ARX5 arm needs a distinct CAN port.")
        arms: dict[str, Any] = {}
        for side, port, urdf in (
            ("left", left_can_port, left_urdf_path),
            ("right", right_can_port, right_urdf_path),
        ):
            arms[side] = PartGroup(
                arm=ARX5Arm(
                    port,
                    arm_type,
                    motion_duration=motion_duration,
                    park_max_deg=park_max_deg,
                    urdf_path=urdf,
                    node_rank=node_rank,
                )
            )
        return cls(**arms, **Camera.declare(cameras, node_rank=node_rank))


@dataclass
class DualARX5Config(RobotConfig):
    """Physical connections for a pair of X5 arms on one host."""

    left_can_port: str = "can1"
    right_can_port: str = "can3"
    arm_type: int = 0
    left_urdf_path: str | None = None
    right_urdf_path: str | None = None
    base_camera_serials: list[str] | None = None
    left_camera_serials: list[str] | None = None
    right_camera_serials: list[str] | None = None
    camera_type: str = "realsense"
    camera_fps: int = 15
    camera_rotations: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.node_rank, int):
            raise TypeError("DualARX5 node_rank must be an integer.")
        if self.left_can_port == self.right_can_port:
            raise ValueError("DualARX5 CAN ports must differ.")
        if self.arm_type not in (0, 2):
            raise ValueError("DualARX5 arm_type must be 0 or 2.")
        if (
            not isinstance(self.camera_fps, int)
            or isinstance(self.camera_fps, bool)
            or self.camera_fps <= 0
        ):
            raise ValueError("DualARX5 camera_fps must be a positive integer.")
        for slot in ("base", "left", "right"):
            value = getattr(self, f"{slot}_camera_serials")
            if value is not None:
                setattr(self, f"{slot}_camera_serials", list(value))
        camera_names = {
            f"{prefix}_{index}_rgb"
            for slot, prefix in (
                ("base", "base"),
                ("left", "left_wrist"),
                ("right", "right_wrist"),
            )
            for index, _ in enumerate(getattr(self, f"{slot}_camera_serials") or [])
        }
        if not isinstance(self.camera_rotations, Mapping):
            raise ValueError(
                "DualARX5 camera_rotations must map camera names to degrees."
            )
        for name, degrees in self.camera_rotations.items():
            if name not in camera_names:
                raise ValueError(f"Unknown DualARX5 camera rotation key: {name!r}.")
            if type(degrees) is not int or degrees not in (0, 90, 180, 270):
                raise ValueError(
                    f"Camera {name!r} rotation must be 0, 90, 180, or 270 degrees."
                )
        self.camera_rotations = dict(self.camera_rotations)

    @property
    def camera_serials(self) -> list[str]:
        """Return configured camera identifiers for node-local validation."""
        return [
            serial
            for slot in ("base", "left", "right")
            for serial in (getattr(self, f"{slot}_camera_serials") or [])
        ]


DualARX5Robot.register_type(DualARX5Config)
