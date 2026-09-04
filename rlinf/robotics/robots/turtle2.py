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
from typing import TYPE_CHECKING

from ..discovery import (
    RobotConfig,
)
from ..parts.base import PartGroup, RobotPart
from ..robot import Robot

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..parts.arms.turtle2 import Turtle2Connection


class Turtle2Robot(Robot):
    """Composable Turtle2 robot."""

    ROBOT_TYPE = "Turtle2"

    @classmethod
    def declare_connection(
        cls, *, frequency: int, camera_ids: list[int], node_rank: int, name: str
    ) -> "Turtle2Connection":
        """Declare the shared Turtle2 connection."""
        from ..parts.arms.turtle2 import Turtle2Connection

        return Turtle2Connection(
            frequency, tuple(camera_ids), node_rank=node_rank, worker_name=name
        )

    @classmethod
    def build_arms(cls, connection: "Turtle2Connection") -> dict[str, RobotPart]:
        """Return both arm groups exported by the shared connection."""
        return {
            side: PartGroup(
                arm=connection.part(side),
                gripper=connection.part(f"{side}_end_effector"),
            )
            for side in ("left", "right")
        }

    @classmethod
    def build_cameras(
        cls, connection: "Turtle2Connection", *, count: int
    ) -> dict[str, RobotPart]:
        """Return wrist cameras exported by the shared connection."""
        return {
            f"wrist_{index + 1}": connection.part(f"wrist_{index + 1}")
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
        """Compose a Turtle2 robot around one shared connection."""
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


@dataclass
class Turtle2Config(RobotConfig):
    """Placement configuration for a Turtle2 robot."""

    def __post_init__(self) -> None:
        """Post-initialization to validate the configuration."""
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in Turtle2 config must be an integer. But got {type(self.node_rank)}."
        )


Turtle2Robot.register_type(Turtle2Config)
