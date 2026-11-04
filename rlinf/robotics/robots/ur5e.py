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

"""UR5e robot: one RTDE-controlled arm and optional cameras."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

from ..discovery import RobotConfig
from ..parts.cameras import Camera
from ..parts.end_effectors import EndEffector
from ..robot import Robot


class UR5eRobot(Robot):
    """Composable UR5e with an independently placed arm connection."""

    ROBOT_TYPE = "UR5e"
    BACKEND: str = "ur_rtde"

    @classmethod
    def build_arms(
        cls,
        *,
        robot_ip: str,
        node_rank: Optional[int],
        backend: Optional[str] = None,
        frequency: float = 500.0,
        lookahead_time: float = 0.1,
        gain: float = 300.0,
        controller_node_rank: Optional[int] = None,
        gripper_type: Optional[str] = None,
        gripper_connection: Optional[str] = None,
        end_effector_config: Optional[dict[str, Any]] = None,
        worker_rank: int = 0,
        env_idx: int = 0,
    ) -> dict[str, Any]:
        """Declare the arm and an optional independently connected gripper."""
        from ..parts.arms import Arm

        controller_node = (
            node_rank if controller_node_rank is None else controller_node_rank
        )
        arm = Arm.backend(backend or cls.BACKEND).declare(
            robot_ip,
            frequency=frequency,
            lookahead_time=lookahead_time,
            gain=gain,
            node_rank=controller_node,
            worker_name=f"{cls.ROBOT_TYPE}Arm-{worker_rank}-{env_idx}",
        )
        parts: dict[str, Any] = {"arm": arm}
        if gripper_type is not None:
            parts["end_effector"] = EndEffector.backend(gripper_type).declare(
                robot_ip=robot_ip,
                port=gripper_connection,
                **(end_effector_config or {}),
                node_rank=controller_node,
                worker_name=f"{cls.ROBOT_TYPE}EndEffector-{worker_rank}-{env_idx}",
            )
        elif gripper_connection is not None or end_effector_config:
            raise ValueError("UR5e gripper settings require a gripper_type.")
        return parts

    @classmethod
    def build_cameras(
        cls,
        cameras: Optional[Mapping[str, Any]] = None,
        *,
        node_rank: Optional[int] = None,
    ) -> dict[str, Any]:
        """Declare the explicitly configured cameras."""
        return Camera.declare(cameras, node_rank=node_rank)

    @classmethod
    def build(
        cls,
        *,
        robot_ip: str,
        node_rank: Optional[int],
        backend: Optional[str] = None,
        frequency: float = 500.0,
        lookahead_time: float = 0.1,
        gain: float = 300.0,
        controller_node_rank: Optional[int] = None,
        gripper_type: Optional[str] = None,
        gripper_connection: Optional[str] = None,
        end_effector_config: Optional[dict[str, Any]] = None,
        worker_rank: int = 0,
        env_idx: int = 0,
        cameras: Optional[Mapping[str, Any]] = None,
        camera_node_rank: Optional[int] = None,
    ) -> "UR5eRobot":
        """Compose the UR5e arm and any declared cameras."""
        return cls(
            **cls.build_arms(
                robot_ip=robot_ip,
                node_rank=node_rank,
                backend=backend,
                frequency=frequency,
                lookahead_time=lookahead_time,
                gain=gain,
                controller_node_rank=controller_node_rank,
                gripper_type=gripper_type,
                gripper_connection=gripper_connection,
                end_effector_config=end_effector_config,
                worker_rank=worker_rank,
                env_idx=env_idx,
            ),
            **cls.build_cameras(cameras, node_rank=camera_node_rank),
        )


@dataclass
class UR5eConfig(RobotConfig):
    """Hardware settings for a UR5e position-servo connection."""

    robot_ip: Optional[str] = None
    """Controller IP, resolved from ROBOT_IP on the node when unset."""

    backend: Optional[str] = None
    """Arm backend. ``None`` selects :attr:`UR5eRobot.BACKEND`."""

    frequency: float = 500.0
    """Local servo and receive frequency in Hz, at most 500 for UR5e."""

    lookahead_time: float = 0.1
    """UR position-servo lookahead in seconds, from 0.03 to 0.2."""

    gain: float = 300.0
    """UR position-servo proportional gain, from 100 to 2000."""

    controller_node_rank: Optional[int] = None
    """Node hosting the arm connection. ``None`` uses the robot's node."""

    gripper_type: Optional[str] = None
    """Optional end-effector backend, such as ``robotiq`` for USB-RS485."""

    gripper_connection: Optional[str] = None
    """Gripper serial port, preferably a stable ``/dev/serial/by-id/`` path."""

    end_effector_config: dict[str, Any] = field(default_factory=dict)
    """Settings for the selected gripper, such as its maximum opening width."""

    camera_serials: Optional[list[str]] = None
    """Camera identifiers. ``None`` or ``[]`` runs with the arm alone."""

    camera_type: str = "realsense"
    """Camera backend used for explicitly configured camera identifiers."""

    camera_resolution: tuple[int, int] = (640, 480)
    """Requested native stream width and height, supported by the camera SDK."""

    camera_fps: int = 15
    """Native camera frame rate, independent of environment stepping."""

    camera_node_rank: Optional[int] = None
    """Node hosting the cameras. ``None`` uses the calling process."""

    def __post_init__(self) -> None:
        """Validate placement and normalize camera identifiers."""
        assert isinstance(self.node_rank, int), (
            "'node_rank' in UR5e config must be an integer. "
            f"But got {type(self.node_rank)}."
        )
        if self.camera_serials:
            self.camera_serials = [str(serial) for serial in self.camera_serials]
        if len(self.camera_resolution) != 2 or any(
            not isinstance(size, int) or isinstance(size, bool) or size <= 0
            for size in self.camera_resolution
        ):
            raise ValueError(
                "UR camera_resolution needs positive integer width/height."
            )
        self.camera_resolution = tuple(self.camera_resolution)
        if (
            not isinstance(self.camera_fps, int)
            or isinstance(self.camera_fps, bool)
            or self.camera_fps <= 0
        ):
            raise ValueError("UR camera_fps must be a positive integer.")


UR5eRobot.register_type(UR5eConfig)
