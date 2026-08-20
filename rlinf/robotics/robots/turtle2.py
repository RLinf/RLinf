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
from ..robot import Robot


class Turtle2Robot(Robot):
    """Composable Turtle2 robot."""

    ROBOT_TYPE = "Turtle2"

    @classmethod
    def declare_connection(
        cls, *, frequency: int, camera_ids: list[int], node_rank: int, name: str
    ):
        """Declare the one connection that drives everything on this robot."""
        from ..parts.arms.turtle2 import Turtle2Connection

        return Turtle2Connection.at(
            frequency, tuple(camera_ids), node_rank=node_rank, name=name
        )

    @classmethod
    def build_arms(cls, connection) -> dict[str, Any]:
        """Both arms, each whole, from the shared connection."""
        return {
            side: Group(
                arm=connection.export(side),
                gripper=connection.export(f"{side}_end_effector"),
            )
            for side in ("left", "right")
        }

    @classmethod
    def build_cameras(cls, connection, *, count: int) -> dict[str, Any]:
        """The wrist cameras, from that same connection."""
        return {
            f"wrist_{index + 1}": connection.export(f"wrist_{index + 1}")
            for index in range(count)
        }

    @classmethod
    def build(
        cls,
        *,
        frequency: int,
        camera_ids: list[int],
        env_idx: int,
        node_rank: int,
        worker_rank: int,
    ) -> "Turtle2Robot":
        """Compose this robot from the parts it is made of.

        Everything hangs off one declaration, so the connection is opened once
        however many parts refer to it.
        """
        connection = cls.declare_connection(
            frequency=frequency,
            camera_ids=camera_ids,
            node_rank=node_rank,
            name=f"Turtle2Connection-{worker_rank}-{env_idx}",
        )
        return cls(
            **cls.build_arms(connection),
            **cls.build_cameras(connection, count=len(camera_ids)),
        )


class Turtle2Discovery(RobotDiscovery):
    """Discover configured Turtle2 robots."""

    HW_TYPE = Turtle2Robot.ROBOT_TYPE

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list[HardwareConfig]] = None
    ) -> Optional[HardwareResource]:
        """Enumerate the robot resources on a node.

        Args:
            node_rank: The rank of the node being enumerated.
            configs: The configurations for the hardware on a node.

        Returns:
            Optional[HardwareResource]: An object representing the hardware resources. None if no hardware is found.
        """
        assert configs is not None, "Robot hardware requires explicit configurations"
        robot_configs: list["Turtle2Config"] = []
        for config in configs:
            if isinstance(config, Turtle2Config) and config.node_rank == node_rank:
                robot_configs.append(config)

        if robot_configs:
            # Auto-detect any unset fields from environment variables.
            RobotAutoConfig.resolve(robot_configs)

            turtle2_infos: list[HardwareInfo] = []
            for config in robot_configs:
                turtle2_infos.append(
                    RobotInfo(
                        type=cls.HW_TYPE,
                        model=cls.HW_TYPE,
                        config=config,
                    )
                )

            return HardwareResource(type=cls.HW_TYPE, infos=turtle2_infos)
        return None


@dataclass
class Turtle2Config(RobotConfig):
    """Configuration for a robotic system."""

    # empty config

    def __post_init__(self):
        """Post-initialization to validate the configuration."""
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in Turtle2 config must be an integer. But got {type(self.node_rank)}."
        )


Turtle2Robot.register(Turtle2Config, Turtle2Discovery)
