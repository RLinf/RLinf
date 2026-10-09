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

"""UR5 robot: one Ethernet-driven Universal Robots arm and its cameras."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ..discovery import (
    RobotConfig,
)
from ..parts.cameras import Camera
from ..robot import Robot


class UR5Robot(Robot):
    """Composable Universal Robots UR5.

    This driver carries no gripper, so the tree is ``arm`` plus any cameras. A
    grasp task adds a gripper by composing an end-effector part beside the arm.
    """

    ROBOT_TYPE = "UR5"

    BACKEND: str = "ur_rtde"
    """Registered arm backend, named for the SDK rather than the arm so another
    driver for the same hardware registers beside it. :attr:`UR5Config.backend`
    may select another."""

    @classmethod
    def build_arms(
        cls,
        *,
        robot_ip: str,
        node_rank: int,
        backend: Optional[str] = None,
        speed: float = 1.05,
        acceleration: float = 1.4,
        worker_rank: int = 0,
        env_idx: int = 0,
    ) -> dict[str, Any]:
        """Return the arm declaration."""
        from ..parts.arms import Arm

        arm = Arm.backend(backend or cls.BACKEND).declare(
            robot_ip,
            speed=speed,
            acceleration=acceleration,
            node_rank=node_rank,
            worker_name=f"{cls.ROBOT_TYPE}Arm-{worker_rank}-{env_idx}",
        )
        return {"arm": arm}

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
        robot_ip: str,
        node_rank: int,
        backend: Optional[str] = None,
        speed: float = 1.05,
        acceleration: float = 1.4,
        env_idx: int = 0,
        worker_rank: int = 0,
        cameras: Optional[Mapping[str, Any]] = None,
        camera_node_rank: Optional[int] = None,
    ) -> "UR5Robot":
        """Compose a UR5 robot from its arm and cameras."""
        return cls(
            **cls.build_arms(
                robot_ip=robot_ip,
                backend=backend,
                speed=speed,
                acceleration=acceleration,
                node_rank=node_rank,
                worker_rank=worker_rank,
                env_idx=env_idx,
            ),
            **cls.build_cameras(cameras, node_rank=camera_node_rank),
        )


@dataclass
class UR5Config(RobotConfig):
    """Configuration for a Universal Robots UR5."""

    backend: Optional[str] = None
    """Arm backend, such as ``"ur_rtde"``. ``None`` uses
    :attr:`UR5Robot.BACKEND`."""

    robot_ip: str = "192.168.1.10"
    """Controller IP the RTDE interfaces connect to."""

    speed: float = 1.05
    """Joint speed for commanded motion, in radians per second."""

    acceleration: float = 1.4
    """Joint acceleration for commanded motion, in radians per second squared."""

    camera_serials: Optional[list[str]] = None
    """Camera identifiers. ``None`` discovers cameras during enumeration;
    ``[]`` explicitly selects no cameras."""

    camera_type: str = "realsense"
    """Camera backend: ``"realsense"``, ``"zed"``, or ``"lumos"``."""

    controller_node_rank: Optional[int] = None
    """Node rank where the arm part should run.
    When ``None`` (default), co-located with the env worker."""

    def __post_init__(self) -> None:
        """Post-initialization to validate the configuration."""
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in UR5 config must be an integer. "
            f"But got {type(self.node_rank)}."
        )
        if self.camera_serials:
            self.camera_serials = [str(serial) for serial in self.camera_serials]


UR5Robot.register_type(UR5Config)
