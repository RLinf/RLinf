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

"""LeKiwi mobile base composition and hardware configuration."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ..discovery import RobotConfig
from ..parts.cameras import Camera
from ..parts.mobility import MobileBase
from ..robot import Robot


class LeKiwiRobot(Robot):
    """LeKiwi wheel base composed under the single part name ``base``."""

    ROBOT_TYPE = "LeKiwi"

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
        worker_rank: int = 0,
        env_idx: int = 0,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
        max_raw: int = 3000,
        cameras: Optional[Mapping[str, Any]] = None,
        camera_node_rank: Optional[int] = None,
    ) -> "LeKiwiRobot":
        """Compose an unconnected LeKiwi robot from its serial port and cameras."""
        base_cls = MobileBase.backend("lekiwi")
        base = base_cls(
            port,
            wheel_radius=wheel_radius,
            base_radius=base_radius,
            max_raw=max_raw,
            node_rank=node_rank,
            worker_name=f"{base_cls.__name__}-{worker_rank}-{env_idx}",
        )
        return cls(
            base=base,
            **cls.build_cameras(cameras, node_rank=camera_node_rank),
        )


@dataclass
class LeKiwiConfig(RobotConfig):
    """Hardware and placement configuration for a LeKiwi base."""

    port: str | None = None
    """Serial device the three-wheel servo bus is on, such as ``/dev/ttyACM0``."""

    wheel_radius: float = 0.05
    """Wheel radius in metres."""

    base_radius: float = 0.125
    """Centre-to-wheel distance in metres."""

    max_raw: int = 3000
    """Absolute per-wheel servo velocity limit."""

    camera_serials: Optional[list[str]] = None
    """Camera identifiers. ``None`` discovers cameras during enumeration;
    ``[]`` explicitly selects no cameras."""

    camera_type: str = "webcam"
    """Camera backend; the base carries a plain UVC webcam by default."""

    def __post_init__(self) -> None:
        """Coerce camera identifiers to the strings the backend compares."""
        if self.camera_serials:
            self.camera_serials = [str(serial) for serial in self.camera_serials]


LeKiwiRobot.register_type(LeKiwiConfig)
