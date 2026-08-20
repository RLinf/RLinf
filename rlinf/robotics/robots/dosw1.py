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
from typing import Any, Optional

from rlinf.scheduler.hardware import HardwareConfig, HardwareInfo, HardwareResource

from ..discovery import (
    RobotAutoConfig,
    RobotConfig,
    RobotDiscovery,
    RobotInfo,
)
from ..parts.base import Group
from ..parts.cameras import declare_cameras
from ..robot import Robot


class DOSW1Robot(Robot):
    """Composable DOS-W1 robot."""

    ROBOT_TYPE = "DOSW1"

    @classmethod
    def build_arms(cls, sdk) -> dict[str, Any]:
        """Both arms, each whole, from the shared SDK session."""
        return {
            side: Group(
                arm=sdk.export(side), gripper=sdk.export(f"{side}_end_effector")
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
        """The cameras this robot carries."""
        return declare_cameras(cameras, node_rank=node_rank)

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
        cameras: Optional[Mapping[str, Any]] = None,
        **_: Any,
    ) -> "DOSW1Robot":
        """Compose this robot from the parts it is made of.

        Both arms share one SDK session, and it runs wherever the env worker
        runs, so the declaration carries no node.

        The settings are named rather than taken as one config object, the way
        every other robot names them. Handing the whole object over meant the
        session read fields that live on the env's config, so it could only be
        built by an env -- never from a bench script or a test.
        """
        from ..parts.arms.dosw1 import DOSW1Connection

        if "config" in _:
            raise TypeError(
                "DOSW1Robot.build no longer takes a config object; name the "
                "settings instead, as every other robot does. Passing one "
                "would land in **_ and quietly leave every setting at its "
                "default -- including is_dummy, so the session would reach for "
                "an SDK the caller meant to skip."
            )

        sdk = DOSW1Connection.at(
            robot_url=robot_url,
            left_arm_port=left_arm_port,
            right_arm_port=right_arm_port,
            left_lead_port=left_lead_port,
            right_lead_port=right_lead_port,
            enable_human_in_loop=enable_human_in_loop,
            gripper_width_max=gripper_width_max,
            is_dummy=is_dummy,
        )
        return cls(**cls.build_arms(sdk), **cls.build_cameras(cameras))


class DOSW1Discovery(RobotDiscovery):
    """Hardware policy for the DOS-W1 dual-arm robot.

    Connection parameters (gRPC URL / ports) and RealSense camera serials
    are placed here so that they are managed by the scheduler via
    ``cluster.node_groups.hardware`` and injected into the env worker
    through :class:`~rlinf.robotics.RobotInfo`.
    """

    HW_TYPE = DOSW1Robot.ROBOT_TYPE

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list[HardwareConfig]] = None
    ) -> Optional[HardwareResource]:
        """Enumerate the DOS-W1 robot resources on a node.

        Args:
            node_rank: The rank of the node being enumerated.
            configs: The configurations for the hardware on a node.

        Returns:
            Hardware resource descriptor, or ``None`` when the node has no
            matching DOS-W1 configuration.
        """
        assert configs is not None, (
            "DOSW1 hardware requires explicit configurations for robot URL, "
            "gRPC ports and camera serials."
        )
        robot_configs: list["DOSW1RobotConfig"] = []
        for config in configs:
            if isinstance(config, DOSW1RobotConfig) and config.node_rank == node_rank:
                robot_configs.append(config)

        # Fill unset fields from env vars (e.g. ``ROBOT_URL``), one value per
        # config when several robots share this node. With no configs given,
        # create one per comma-separated ``ROBOT_URL``.
        robot_configs = RobotAutoConfig.resolve(
            robot_configs,
            config_cls=DOSW1RobotConfig,
            node_rank=node_rank,
            count_fields=("robot_url",),
        )

        if not robot_configs:
            return None

        infos: list[HardwareInfo] = [
            RobotInfo(type=cls.HW_TYPE, model=cls.HW_TYPE, config=cfg)
            for cfg in robot_configs
        ]
        return HardwareResource(type=cls.HW_TYPE, infos=infos)


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

    def __post_init__(self):  # noqa: D105
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in DOSW1 config must be an integer. "
            f"But got {type(self.node_rank)}."
        )
        if self.camera_serials is not None:
            self.camera_serials = [str(s) for s in self.camera_serials]


DOSW1Robot.register(DOSW1RobotConfig, DOSW1Discovery)
