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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

from ..discovery import (
    RobotConfig,
)
from ..parts.cameras import Camera
from ..robot import Robot


class GimArmRobot(Robot):
    """Composable GimArm robot."""

    ROBOT_TYPE = "GimArm"

    @classmethod
    def build_arms(
        cls,
        *,
        can_interface: str,
        arm_variant: str,
        enable_gripper: bool,
        gripper_type: str,
        control_mode: str,
        node_rank: int,
        worker_rank: int = 0,
        env_idx: int = 0,
    ) -> dict[str, Any]:
        """Return the arm declaration, including its optional gripper."""
        from ..parts.arms.gim_arm import GimArm

        arm = GimArm(
            can_interface,
            arm_variant,
            enable_gripper,
            gripper_type,
            control_mode,
            node_rank=node_rank,
            worker_name=f"GimArm-{worker_rank}-{env_idx}",
        )
        # The arm connection exports its gripper when one is enabled.
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
        can_interface: str,
        arm_variant: str,
        enable_gripper: bool,
        gripper_type: str,
        control_mode: str,
        env_idx: int,
        node_rank: int,
        worker_rank: int,
        cameras: Optional[Mapping[str, Any]] = None,
        camera_node_rank: Optional[int] = None,
    ) -> "GimArmRobot":
        """Compose a GimArm robot from its arm and cameras."""
        return cls(
            **cls.build_arms(
                can_interface=can_interface,
                arm_variant=arm_variant,
                enable_gripper=enable_gripper,
                gripper_type=gripper_type,
                control_mode=control_mode,
                node_rank=node_rank,
                worker_rank=worker_rank,
                env_idx=env_idx,
            ),
            **cls.build_cameras(cameras, node_rank=camera_node_rank),
        )


@dataclass
class GimArmConfig(RobotConfig):
    """Configuration for a GimArm robot."""

    can_interface: str = "can0"
    """CAN socket interface name (e.g. ``"can0"``)."""

    arm_variant: str = "gim_arm_xl"
    """Arm variant: ``"gim_arm"`` or ``"gim_arm_xl"``."""

    camera_serials: Optional[list[str]] = None
    """Optional list of camera serial numbers.
    Pass ``[]`` or leave ``None`` to run without cameras.
    Camera auto-detection is not currently implemented for GimArm."""

    camera_type: str = "realsense"
    """Camera backend: ``"realsense"`` or ``"zed"``."""

    enable_gripper: bool = True
    """Whether the gripper is attached and should be controlled."""

    gripper_type: str = "parallel"
    """Gripper type: ``"parallel"`` or ``"single_side"``."""

    controller_node_rank: Optional[int] = None
    """Node rank where the arm part should run.
    When ``None`` (default), co-located with the env worker."""

    def model(self, robot_type: str) -> str:
        """Report the arm variant, which changes reach and payload."""
        return f"{robot_type}_{self.arm_variant}"

    def __post_init__(self) -> None:
        """Post-initialization to validate the configuration."""
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in GimArm config must be an integer. "
            f"But got {type(self.node_rank)}."
        )
        assert self.arm_variant in ("gim_arm", "gim_arm_xl"), (
            f"'arm_variant' must be 'gim_arm' or 'gim_arm_xl'. "
            f"But got '{self.arm_variant}'."
        )
        if self.camera_serials:
            self.camera_serials = list(self.camera_serials)


GimArmRobot.register_type(GimArmConfig)
