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

"""Piper robot: one CAN-driven AgileX arm and its cameras."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ..discovery import (
    RobotConfig,
)
from ..parts.cameras import Camera
from ..robot import Robot


class PiperRobot(Robot):
    """Composable AgileX Piper robot.

    The AgxGripper shares the arm's bus, so the tree is ``arm`` with
    ``arm.end_effector`` beneath it, plus any cameras.
    """

    ROBOT_TYPE = "Piper"

    BACKEND: str = "pyagxarm"
    """Registered arm backend, named for the SDK rather than the arm so
    another driver for the same hardware registers beside it. Subclasses and
    :attr:`PiperConfig.backend` may select another."""

    @classmethod
    def build_arms(
        cls,
        *,
        can_channel: str,
        node_rank: int,
        backend: Optional[str] = None,
        can_interface: str = "socketcan",
        model: str = "piper",
        firmware: Optional[str] = None,
        speed_percent: int = 30,
        gripper_force: float = 1.0,
        gripper_max_width: float = 0.07,
        with_gripper: bool = True,
        worker_rank: int = 0,
        env_idx: int = 0,
    ) -> dict[str, Any]:
        """Return the arm declaration, including the gripper it exports."""
        from ..parts.arms import Arm

        arm = Arm.backend(backend or cls.BACKEND).declare(
            can_channel,
            interface=can_interface,
            model=model,
            firmware=firmware,
            speed_percent=speed_percent,
            gripper_force=gripper_force,
            gripper_max_width=gripper_max_width,
            with_gripper=with_gripper,
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
        can_channel: str,
        node_rank: int,
        backend: Optional[str] = None,
        can_interface: str = "socketcan",
        model: str = "piper",
        firmware: Optional[str] = None,
        speed_percent: int = 30,
        gripper_force: float = 1.0,
        gripper_max_width: float = 0.07,
        with_gripper: bool = True,
        env_idx: int = 0,
        worker_rank: int = 0,
        cameras: Optional[Mapping[str, Any]] = None,
        camera_node_rank: Optional[int] = None,
    ) -> "PiperRobot":
        """Compose a Piper robot from its arm and cameras."""
        return cls(
            **cls.build_arms(
                can_channel=can_channel,
                backend=backend,
                can_interface=can_interface,
                model=model,
                firmware=firmware,
                speed_percent=speed_percent,
                gripper_force=gripper_force,
                gripper_max_width=gripper_max_width,
                with_gripper=with_gripper,
                node_rank=node_rank,
                worker_rank=worker_rank,
                env_idx=env_idx,
            ),
            **cls.build_cameras(cameras, node_rank=camera_node_rank),
        )


@dataclass
class PiperConfig(RobotConfig):
    """Configuration for an AgileX Piper robot."""

    backend: Optional[str] = None
    """Arm backend, such as ``"pyagxarm"``. ``None`` uses
    :attr:`PiperRobot.BACKEND`."""

    can_channel: str = "can0"
    """CAN channel the arm is on. Bring it up at 1 Mbit/s before a run starts;
    the SDK does not configure it."""

    can_interface: str = "socketcan"
    """python-can backend: ``"socketcan"``, ``"slcan"``, or ``"agx_cando"``."""

    model: str = "piper"
    """Arm variant: ``"piper"``, ``"piper_h"``, ``"piper_l"``, or
    ``"piper_x"``. Commands are clipped to the travel the variant reports."""

    firmware: Optional[str] = None
    """Firmware profile. ``None`` reads the version off the arm and picks the
    matching one. Pin it -- ``"default"`` for S-V1.8-2 and older, then
    ``"v183"``, ``"v188"``, ``"v189"`` -- only to override that."""

    speed_percent: int = 30
    """Percentage of maximum speed used for commanded motion."""

    gripper_force: float = 1.0
    """Gripping force in newtons, up to 3.0."""

    gripper_max_width: float = 0.07
    """Stroke at full opening, in metres. Set at the factory to 0.07 or 0.1."""

    with_gripper: bool = True
    """Whether an AgxGripper is fitted. When false there is no gripper path."""

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
            f"'node_rank' in Piper config must be an integer. "
            f"But got {type(self.node_rank)}."
        )
        if self.camera_serials:
            self.camera_serials = [str(serial) for serial in self.camera_serials]


PiperRobot.register_type(PiperConfig)
