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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ..discovery import (
    RobotConfig,
)
from ..parts.base import PartGroup, RobotPart
from ..parts.cameras import Camera
from ..robot import Robot

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..parts.arms.dosw1 import DOSW1Connection


class DOSW1Robot(Robot):
    """Composable DOS-W1 robot."""

    ROBOT_TYPE = "DOSW1"

    @classmethod
    def build_arms(cls, sdk: "DOSW1Connection") -> dict[str, RobotPart]:
        """Return both arm groups exported by the shared SDK connection."""
        return {
            side: PartGroup(
                arm=sdk.part(side), gripper=sdk.part(f"{side}_end_effector")
            )
            for side in ("left", "right")
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
        robot_url: str = "localhost",
        left_arm_port: int = 50051,
        right_arm_port: int = 50053,
        left_lead_port: int = 50050,
        right_lead_port: int = 50052,
        enable_human_in_loop: bool = False,
        gripper_width_max: float = 0.07,
        is_dummy: bool = False,
        node_rank: Optional[int] = None,
        cameras: Optional[Mapping[str, Any]] = None,
        camera_node_rank: Optional[int] = None,
    ) -> "DOSW1Robot":
        """Compose a DOS-W1 robot around one shared SDK connection."""
        from ..parts.arms.dosw1 import DOSW1Connection

        sdk = DOSW1Connection(
            robot_url=robot_url,
            left_arm_port=left_arm_port,
            right_arm_port=right_arm_port,
            left_lead_port=left_lead_port,
            right_lead_port=right_lead_port,
            enable_human_in_loop=enable_human_in_loop,
            gripper_width_max=gripper_width_max,
            is_dummy=is_dummy,
            node_rank=node_rank,
        )
        return cls(
            **cls.build_arms(sdk),
            **cls.build_cameras(
                cameras,
                node_rank=node_rank if camera_node_rank is None else camera_node_rank,
            ),
        )


@dataclass
class DOSW1RobotConfig(RobotConfig):
    """Configuration for a DOS-W1 dual-arm robot.

    The env process runs on the node indicated by :attr:`node_rank`, and
    talks to the AirBot gRPC services over ``robot_url`` and the four
    follower / leader ports.
    """

    robot_url: str = "localhost"
    """Hostname or IP of the AirBot gRPC endpoint."""

    left_arm_port: int = 50051
    """gRPC port of the left follower arm."""

    right_arm_port: int = 50053
    """gRPC port of the right follower arm."""

    left_lead_port: int = 50050
    """gRPC port of the left leader arm."""

    right_lead_port: int = 50052
    """gRPC port of the right leader arm."""

    camera_serials: Optional[list[str]] = None
    """RealSense camera serial numbers used by the env."""

    def __post_init__(self) -> None:  # noqa: D105
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in DOSW1 config must be an integer. "
            f"But got {type(self.node_rank)}."
        )
        if self.camera_serials is not None:
            self.camera_serials = [str(s) for s in self.camera_serials]


DOSW1Robot.register_type(DOSW1RobotConfig)
