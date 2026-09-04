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

"""SO-101 robot: one lerobot-driven arm and its cameras."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ..discovery import (
    RobotConfig,
)
from ..parts.cameras import Camera
from ..robot import Robot


class SO101Robot(Robot):
    """Composable SO-101 robot.

    The arm carries its gripper as one of its own servos, so the part tree is
    ``arm`` with ``arm.end_effector`` beneath it, plus any cameras.
    """

    ROBOT_TYPE = "SO101"

    @classmethod
    def build_arms(
        cls,
        *,
        port: str,
        node_rank: int,
        calibration_id: Optional[str] = None,
        max_relative_target: Optional[int] = None,
        worker_rank: int = 0,
        env_idx: int = 0,
    ) -> dict[str, Any]:
        """Return the arm declaration, including the gripper it exports."""
        from ..parts.arms.so101 import SO101Arm

        arm = SO101Arm.declare(
            port,
            calibration_id=calibration_id,
            max_relative_target=max_relative_target,
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
        port: str,
        node_rank: int,
        calibration_id: Optional[str] = None,
        max_relative_target: Optional[int] = None,
        env_idx: int = 0,
        worker_rank: int = 0,
        cameras: Optional[Mapping[str, Any]] = None,
        camera_node_rank: Optional[int] = None,
    ) -> "SO101Robot":
        """Compose an SO-101 robot from its arm and cameras."""
        return cls(
            **cls.build_arms(
                port=port,
                calibration_id=calibration_id,
                max_relative_target=max_relative_target,
                node_rank=node_rank,
                worker_rank=worker_rank,
                env_idx=env_idx,
            ),
            **cls.build_cameras(cameras, node_rank=camera_node_rank),
        )


@dataclass
class SO101Config(RobotConfig):
    """Configuration for an SO-101 robot."""

    port: str = "/dev/ttyACM0"
    """Serial device the Feetech servo bus is on."""

    calibration_id: Optional[str] = None
    """lerobot calibration identifier for this arm.

    lerobot stores one calibration file per identifier. The file has to exist
    already: calibrating asks the operator to move the arm by hand, which a
    worker cannot do."""

    max_relative_target: Optional[int] = None
    """Per-step joint limit in degrees, applied by lerobot. ``None`` disables
    clamping, which lets a large action step move the arm at full speed."""

    camera_serials: Optional[list[str]] = None
    """Camera identifiers. ``None`` or ``[]`` runs without cameras."""

    camera_type: str = "realsense"
    """Camera backend: ``"realsense"``, ``"zed"``, or ``"lumos"``."""

    controller_node_rank: Optional[int] = None
    """Node rank where the arm part should run.
    When ``None`` (default), co-located with the env worker."""

    def __post_init__(self) -> None:
        """Post-initialization to validate the configuration."""
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in SO101 config must be an integer. "
            f"But got {type(self.node_rank)}."
        )
        if self.camera_serials:
            self.camera_serials = [str(serial) for serial in self.camera_serials]


SO101Robot.register_type(SO101Config)
