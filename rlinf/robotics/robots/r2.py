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

"""Robbyant R2 dual-arm wheeled humanoid robot composition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from ..discovery import RobotConfig, RobotDiscovery
from ..parts.arms.r2 import R2_MODE_SERVOJ
from ..parts.base import PartGroup, RobotPart
from ..robot import Robot

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..parts.arms.r2 import R2Connection


class R2Robot(Robot):
    """Composable Robbyant R2 dual-arm wheeled humanoid robot."""

    ROBOT_TYPE = "R2"

    @classmethod
    def declare_connection(
        cls,
        *,
        body_dof: Optional[int | str] = None,
        end_effector_type: Optional[str] = "gripper",
        enable_base: bool = True,
        cameras: Sequence[str] = ("head", "wrist_left", "wrist_right"),
        enable_depth: bool = False,
        use_compressed_camera: bool = False,
        controller_mode: int = R2_MODE_SERVOJ,
        enable_fault_recovery: bool = True,
        max_recovery_attempts: int = 3,
        recovery_cooldown_window: float = 10.0,
        motion_profile: Optional[Mapping[str, float]] = None,
        gripper_min_closure: Sequence[float] = (0.0, 0.15),
        node_rank: Optional[int] = None,
        name: Optional[str] = None,
    ) -> R2Connection:
        """Declare the shared RFrame connection for an R2 robot."""
        from ..parts.arms.r2 import R2Connection

        return R2Connection(
            body_dof=body_dof,
            end_effector_type=end_effector_type,
            enable_base=enable_base,
            cameras=tuple(cameras),
            enable_depth=enable_depth,
            use_compressed_camera=use_compressed_camera,
            controller_mode=controller_mode,
            enable_fault_recovery=enable_fault_recovery,
            max_recovery_attempts=max_recovery_attempts,
            recovery_cooldown_window=recovery_cooldown_window,
            motion_profile=motion_profile,
            gripper_min_closure=gripper_min_closure,
            node_rank=node_rank,
            worker_name=name,
        )

    @classmethod
    def build_arms(cls, connection: R2Connection) -> dict[str, RobotPart]:
        """Return both arm groups exported by the shared R2 connection."""
        groups: dict[str, RobotPart] = {}
        for side in ("left", "right"):
            children: dict[str, RobotPart] = {"arm": connection.part(side)}
            if connection.end_effector_type == "gripper":
                children["gripper"] = connection.part(f"{side}_end_effector")
            elif connection.end_effector_type == "hand":
                children["hand"] = connection.part(f"{side}_end_effector")
            groups[side] = PartGroup(**children)
        return groups

    @classmethod
    def build_cameras(cls, connection: R2Connection) -> dict[str, RobotPart]:
        """Return the named camera views exported by the shared R2 connection."""
        return {name: connection.part(name) for name in connection.camera_names}

    @classmethod
    def build(
        cls,
        *,
        body_dof: Optional[int | str] = None,
        end_effector_type: Optional[str] = "gripper",
        enable_base: bool = True,
        cameras: Sequence[str] = ("head", "wrist_left", "wrist_right"),
        enable_depth: bool = False,
        use_compressed_camera: bool = False,
        controller_mode: int = R2_MODE_SERVOJ,
        enable_fault_recovery: bool = True,
        max_recovery_attempts: int = 3,
        recovery_cooldown_window: float = 10.0,
        motion_profile: Optional[Mapping[str, float]] = None,
        gripper_min_closure: Sequence[float] = (0.0, 0.15),
        env_idx: int = 0,
        node_rank: Optional[int] = None,
        worker_rank: int = 0,
        controller_node_rank: Optional[int] = None,
    ) -> R2Robot:
        """Compose an :class:`R2Robot` around one shared :class:`R2Connection`."""
        if end_effector_type == "auto":
            from ..parts.arms.r2 import R2Connection

            probed_dof, probed_ee = R2Connection.probe_hardware()
            if body_dof in (None, "auto"):
                body_dof = probed_dof
            end_effector_type = probed_ee
        elif end_effector_type is None:
            end_effector_type = "gripper"

        ctrl_rank = (
            controller_node_rank if controller_node_rank is not None else node_rank
        )
        worker_name = (
            f"R2Connection-{worker_rank}-{env_idx}" if ctrl_rank is not None else None
        )
        connection = cls.declare_connection(
            body_dof=body_dof,
            end_effector_type=end_effector_type,
            enable_base=enable_base,
            cameras=cameras,
            enable_depth=enable_depth,
            use_compressed_camera=use_compressed_camera,
            controller_mode=controller_mode,
            enable_fault_recovery=enable_fault_recovery,
            max_recovery_attempts=max_recovery_attempts,
            recovery_cooldown_window=recovery_cooldown_window,
            motion_profile=motion_profile,
            gripper_min_closure=gripper_min_closure,
            node_rank=ctrl_rank,
            name=worker_name,
        )
        parts: dict[str, RobotPart] = {
            **cls.build_arms(connection),
            **cls.build_cameras(connection),
        }
        if enable_base:
            parts["base"] = connection.part("base")
        return cls(**parts)


@dataclass
class R2Config(RobotConfig):
    """Hardware and placement configuration for a Robbyant R2 robot."""

    node_rank: Optional[int] = None
    """Optional node rank where the environment process runs. ``None`` runs locally without Ray."""

    body_dof: Optional[int] = None
    """Whole-body joint count (20 or 21). ``None`` auto-detects from ``/robotic/joint_states``."""

    end_effector_type: Optional[str] = None
    """End-effector variant (``"gripper"``, ``"hand"``, or ``"none"``); ``None`` or ``"auto"`` auto-detects."""

    enable_base: bool = True
    """Whether to include the omnidirectional wheeled chassis part."""

    cameras: list[str] = field(
        default_factory=lambda: ["head", "wrist_left", "wrist_right"]
    )
    """RFrame camera topics exposed by the shared R2 connection."""

    enable_depth: bool = False
    """Whether to subscribe to camera depth topics alongside RGB."""

    use_compressed_camera: bool = False
    """Whether to subscribe to ``CompressedImage`` camera topics."""

    controller_mode: int = R2_MODE_SERVOJ
    """Motion controller mode to keep (0: auto, 1: remote, 2: ServoJ, 4: manual drag)."""

    gripper_min_closure: list[float] = field(default_factory=lambda: [0.0, 0.15])
    """Smallest closure the ServoJ stream sends to the (left, right) gripper, 0 fully open."""

    controller_node_rank: Optional[int] = None
    """Optional node rank where the RFrame connection worker should be placed."""

    def __post_init__(self) -> None:
        """Validate the configuration."""
        assert self.node_rank is None or (
            not isinstance(self.node_rank, bool) and isinstance(self.node_rank, int)
        ), (
            f"'node_rank' in R2 config must be an integer or None. "
            f"But got {type(self.node_rank)}."
        )


class R2Discovery(RobotDiscovery):
    """Discover R2 robot resources and auto-probe unset ``body_dof`` / ``end_effector_type``."""

    HW_TYPE = "R2"

    @classmethod
    def prepare(cls, config: RobotConfig, node_rank: int) -> None:
        """Resolve ``body_dof`` and ``end_effector_type`` from ``/robotic/joint_states`` when unset."""
        super().prepare(config, node_rank)
        if not isinstance(config, R2Config):
            return
        needs_dof = config.body_dof in (None, "auto")
        needs_ee = config.end_effector_type in (None, "auto")
        if not (needs_dof or needs_ee):
            return
        if config.disable_validate:
            if needs_dof:
                config.body_dof = 21
            if needs_ee:
                config.end_effector_type = "gripper"
            return

        from ..parts.arms.r2 import R2Connection

        probed_dof, probed_ee = R2Connection.probe_hardware()
        if needs_dof:
            config.body_dof = probed_dof
        if needs_ee:
            config.end_effector_type = probed_ee


R2Robot.register_type(R2Config, R2Discovery)
