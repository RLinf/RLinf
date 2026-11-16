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

"""Dual-arm YAM station: two i2rt followers and the station cameras."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

from ..discovery import RobotConfig
from ..parts.base import PartGroup
from ..parts.cameras import Camera
from ..robot import Robot


class DualYamRobot(Robot):
    """Two YAM followers, each on its own CAN chain, and the station cameras.

    ``left`` and ``right`` each hold an ``arm`` that carries its gripper as
    ``arm.end_effector``. Cameras sit beside them under their own names.
    """

    ROBOT_TYPE = "DualYam"

    #: Arm backend used when the config names none.
    BACKEND = "i2rt"

    @classmethod
    def build_arms(
        cls,
        *,
        left_can_channel: Optional[str],
        right_can_channel: Optional[str],
        node_rank: Optional[int] = None,
        backend: Optional[str] = None,
        arm_type: str = "yam",
        gripper_type: str = "flexible_4310",
        left_arm: Optional[Mapping[str, Any]] = None,
        right_arm: Optional[Mapping[str, Any]] = None,
        feedback_timeout_s: float = 0.25,
        worker_rank: int = 0,
        env_idx: int = 0,
    ) -> dict[str, PartGroup]:
        """Return the left and right arm groups."""
        from ..parts.arms import Arm

        if not left_can_channel or not right_can_channel:
            raise ValueError(
                "A dual YAM station needs both left_can_channel and "
                f"right_can_channel, but got {left_can_channel!r} and "
                f"{right_can_channel!r}."
            )
        if left_can_channel == right_can_channel:
            raise ValueError(
                f"The two YAM followers share CAN channel {left_can_channel!r}; "
                "each arm needs its own."
            )
        arm_cls = Arm.backend(backend or cls.BACKEND)
        sides = {
            "left": (left_can_channel, left_arm),
            "right": (right_can_channel, right_arm),
        }
        return {
            side: PartGroup(
                arm=arm_cls.declare(
                    channel,
                    gripper_type=gripper_type,
                    arm_type=arm_type,
                    feedback_timeout_s=feedback_timeout_s,
                    **dict(settings or {}),
                    node_rank=node_rank,
                    worker_name=f"{cls.ROBOT_TYPE}Arm-{side}-{worker_rank}-{env_idx}",
                )
            )
            for side, (channel, settings) in sides.items()
        }

    @classmethod
    def build_cameras(
        cls,
        cameras: Optional[Mapping[str, Any]] = None,
        *,
        node_rank: Optional[int] = None,
    ) -> dict[str, Camera]:
        """Return the station's named camera declarations."""
        return Camera.declare(cameras, node_rank=node_rank)

    @classmethod
    def build(
        cls,
        *,
        left_can_channel: Optional[str],
        right_can_channel: Optional[str],
        node_rank: Optional[int] = None,
        backend: Optional[str] = None,
        arm_type: str = "yam",
        gripper_type: str = "flexible_4310",
        left_arm: Optional[Mapping[str, Any]] = None,
        right_arm: Optional[Mapping[str, Any]] = None,
        feedback_timeout_s: float = 0.25,
        worker_rank: int = 0,
        env_idx: int = 0,
        cameras: Optional[Mapping[str, Any]] = None,
        camera_node_rank: Optional[int] = None,
    ) -> "DualYamRobot":
        """Compose the station, cameras first so they open before any CAN bus."""
        return cls(
            **cls.build_cameras(cameras, node_rank=camera_node_rank),
            **cls.build_arms(
                left_can_channel=left_can_channel,
                right_can_channel=right_can_channel,
                node_rank=node_rank,
                backend=backend,
                arm_type=arm_type,
                gripper_type=gripper_type,
                left_arm=left_arm,
                right_arm=right_arm,
                feedback_timeout_s=feedback_timeout_s,
                worker_rank=worker_rank,
                env_idx=env_idx,
            ),
        )


@dataclass
class DualYamConfig(RobotConfig):
    """Configuration for a dual-arm YAM station."""

    REQUIRES_CAMERA = True

    backend: Optional[str] = None
    """Arm backend. ``None`` uses :attr:`DualYamRobot.BACKEND`."""

    left_can_channel: Optional[str] = None
    """CAN interface of the left follower. When unset in YAML it is read from
    ``LEFT_CAN_CHANNEL`` on the node the station is enumerated on."""

    right_can_channel: Optional[str] = None
    """CAN interface of the right follower, or ``RIGHT_CAN_CHANNEL``."""

    arm_type: str = "yam"
    """i2rt arm model of both followers."""

    gripper_type: str = "flexible_4310"
    """i2rt gripper model fitted to both followers."""

    left_arm: Optional[dict[str, Any]] = None
    """i2rt settings for the left follower: ``ee_mass``, ``gripper_limits``,
    ``gravity_comp_factor``, ``use_coulomb_friction``, and
    ``enable_auto_recovery``. Unset keys keep the SDK's values."""

    right_arm: Optional[dict[str, Any]] = None
    """i2rt settings for the right follower, as for :attr:`left_arm`."""

    feedback_timeout_s: float = 0.25
    """Joint feedback older than this fails a read and holds both arms."""

    camera_serials: Optional[list[str]] = None
    """Camera serial numbers in the order the env names them, or
    ``CAMERA_SERIALS`` as a comma-separated list."""

    camera_type: str = "realsense"
    """Camera backend."""

    camera_fps: int = 30
    """Frame rate the cameras capture at."""

    controller_node_rank: Optional[int] = None
    """Node the arm parts run on. ``None`` places them with the env worker."""

    camera_node_rank: Optional[int] = None
    """Node the cameras run on. ``None`` opens them in the env worker."""

    def __post_init__(self) -> None:
        """Normalize Hydra values and reject a camera listed twice."""
        assert isinstance(self.node_rank, int), (
            "'node_rank' in DualYam config must be an integer, but got "
            f"{type(self.node_rank)}."
        )
        for name in ("left_arm", "right_arm"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, dict(value))
        if self.camera_serials is not None:
            self.camera_serials = [str(serial) for serial in self.camera_serials]
            if len(set(self.camera_serials)) != len(self.camera_serials):
                raise ValueError(
                    f"YAM camera serials must be unique, but got {self.camera_serials}."
                )


DualYamRobot.register_type(DualYamConfig)

__all__ = ["DualYamConfig", "DualYamRobot"]
