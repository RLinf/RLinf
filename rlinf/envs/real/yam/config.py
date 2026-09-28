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

"""Runtime configuration for the RLinf-native dual YAM environment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .types import NUM_ARM_JOINTS


@dataclass
class YamIKConfig:
    """Limits for one synchronous i2rt IK solve; timing is checked on return."""

    site_name: str = "grasp_site"
    solver: str = "quadprog"
    max_iters: int = 30
    dt: float = 0.01
    position_tolerance: float = 0.001
    rotation_tolerance: float = 0.01
    max_solve_s: float = 0.03

    def __post_init__(self) -> None:
        if not self.site_name or not self.solver:
            raise ValueError("IK site_name and solver must be nonempty")
        if (
            isinstance(self.max_iters, bool)
            or not isinstance(self.max_iters, int)
            or self.max_iters < 1
        ):
            raise ValueError("IK max_iters must be a positive integer")
        for name in (
            "dt",
            "position_tolerance",
            "rotation_tolerance",
            "max_solve_s",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"IK {name} must be finite and positive")
            setattr(self, name, value)


@dataclass
class YamPicoConfig:
    """Dual-controller PICO collection settings."""

    hand: str = "dual"
    zmq_addr: str = "ipc:///tmp/vr_data.ipc"
    position_scale: float = 0.3
    rotation_scale: float = 0.3
    rotation_delta_frame: str = "operator"
    control_threshold: float = 0.85
    max_stale_s: float = 0.2
    max_position_delta: float = 0.005
    max_rotation_delta: float = 0.03
    max_tick_s: float = 0.1
    ik_backtrack_attempts: int = 0
    record_button: str = "right_menu_button"
    discard_button: str = "left_menu_button"
    button_debounce_s: float = 0.2
    keyboard_enabled: bool = False
    wait_for_record_button: bool = True
    # Log per-tick control-loop timing (camera wait vs compute vs gap).
    log_control_timing: bool = False
    left: dict[str, Any] = field(default_factory=dict)
    right: dict[str, Any] = field(default_factory=dict)
    ik: YamIKConfig | Mapping[str, Any] = field(default_factory=YamIKConfig)

    def __post_init__(self) -> None:
        if self.hand != "dual":
            raise ValueError("YAM PICO collection requires hand='dual'")
        if (
            isinstance(self.ik_backtrack_attempts, bool)
            or not isinstance(self.ik_backtrack_attempts, int)
            or self.ik_backtrack_attempts < 0
        ):
            raise ValueError("pico.ik_backtrack_attempts must be a nonnegative integer")
        if self.rotation_delta_frame not in ("operator", "controller_local"):
            raise ValueError(
                "pico.rotation_delta_frame must be operator or controller_local"
            )
        for name in (
            "position_scale",
            "rotation_scale",
            "max_stale_s",
            "max_position_delta",
            "max_rotation_delta",
            "max_tick_s",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"pico.{name} must be finite and positive")
            setattr(self, name, value)
        if (
            not np.isfinite(self.control_threshold)
            or not 0 < self.control_threshold <= 1
        ):
            raise ValueError("pico.control_threshold must be within (0, 1]")
        if not np.isfinite(self.button_debounce_s) or self.button_debounce_s < 0:
            raise ValueError("pico.button_debounce_s must be finite and nonnegative")
        if (
            not isinstance(self.keyboard_enabled, bool)
            or not isinstance(self.wait_for_record_button, bool)
            or not isinstance(self.log_control_timing, bool)
        ):
            raise TypeError("PICO keyboard/wait/timing switches must be bools")
        motion_buttons = {"A", "B", "X", "Y", "grip", "trigger"}
        if (
            not self.record_button
            or not self.discard_button
            or self.record_button == self.discard_button
        ):
            raise ValueError("PICO recording buttons must be nonempty and distinct")
        if {self.record_button, self.discard_button} & motion_buttons:
            raise ValueError("PICO recording buttons must not overlap motion buttons")
        for side in ("left", "right"):
            overrides = dict(getattr(self, side))
            if set(overrides) - {"operator_to_robot_yaw"}:
                raise ValueError(
                    f"pico.{side} only supports operator_to_robot_yaw in v1"
                )
            if not np.isfinite(float(overrides.get("operator_to_robot_yaw", 0.0))):
                raise ValueError(f"pico.{side}.operator_to_robot_yaw must be finite")
            setattr(self, side, overrides)
        if isinstance(self.ik, Mapping):
            self.ik = YamIKConfig(**dict(self.ik))
        if not isinstance(self.ik, YamIKConfig):
            raise TypeError("pico.ik must be an IK configuration")

    def expert_kwargs(self, side: str) -> dict[str, Any]:
        """Build a normal Franka-compatible PicoExpert for one hand."""
        if side not in {"left", "right"}:
            raise ValueError("PICO side must be left or right")
        return {
            "hand": side,
            "zmq_addr": self.zmq_addr,
            "position_scale": self.position_scale,
            "rotation_scale": self.rotation_scale,
            "rotation_delta_frame": self.rotation_delta_frame,
            "control_trigger": "grip",
            "control_threshold": self.control_threshold,
            "max_stale_s": self.max_stale_s,
            "gripper_close_button": "X" if side == "left" else "A",
            "gripper_open_button": "Y" if side == "left" else "B",
            "calibration": {
                "button": "trigger",
                "auto_calibrate_on_start": True,
                "required": True,
            },
            **getattr(self, side),
        }


def _default_joint_limit_min() -> list[list[float]]:
    return [[-float(np.pi)] * NUM_ARM_JOINTS for _ in range(2)]


def _default_joint_limit_max() -> list[list[float]]:
    return [[float(np.pi)] * NUM_ARM_JOINTS for _ in range(2)]


@dataclass
class YamResetConfig:
    """Safe follower motion to a configured episode start pose."""

    enabled: bool = False
    mode: str = "startup"
    left_qpos: list[float] | None = None
    right_qpos: list[float] | None = None
    duration_s: float = 4.0
    max_joint_delta: float = 0.05
    tolerance: float = 0.03
    timeout_s: float = 8.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("reset.enabled must be a bool")
        self.mode = str(self.mode).lower()
        if self.mode not in {"startup", "episode", "manual"}:
            raise ValueError("reset.mode must be 'startup', 'episode', or 'manual'")
        self.duration_s = float(self.duration_s)
        self.max_joint_delta = float(self.max_joint_delta)
        self.tolerance = float(self.tolerance)
        self.timeout_s = float(self.timeout_s)
        scalar_values = {
            "reset.duration_s": self.duration_s,
            "reset.max_joint_delta": self.max_joint_delta,
            "reset.tolerance": self.tolerance,
            "reset.timeout_s": self.timeout_s,
        }
        for name, value in scalar_values.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.duration_s < 0:
            raise ValueError("reset.duration_s must be non-negative")
        if self.max_joint_delta <= 0:
            raise ValueError("reset.max_joint_delta must be positive")
        if self.tolerance <= 0:
            raise ValueError("reset.tolerance must be positive")
        if self.timeout_s <= 0:
            raise ValueError("reset.timeout_s must be positive")
        if self.timeout_s < self.duration_s:
            raise ValueError("reset.timeout_s must be at least reset.duration_s")

        if self.left_qpos is None and self.right_qpos is None:
            if self.enabled:
                raise ValueError(
                    "reset.left_qpos and reset.right_qpos are required when "
                    "reset.enabled=true"
                )
            return
        if self.left_qpos is None or self.right_qpos is None:
            raise ValueError(
                "reset.left_qpos and reset.right_qpos must be configured together"
            )
        left = np.asarray(self.left_qpos, dtype=np.float64)
        right = np.asarray(self.right_qpos, dtype=np.float64)
        if left.shape != (NUM_ARM_JOINTS + 1,) or right.shape != (NUM_ARM_JOINTS + 1,):
            raise ValueError("each reset qpos must contain 6 joints and 1 gripper")
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            raise ValueError("reset qpos values must be finite")
        if not 0.0 <= left[-1] <= 1.0 or not 0.0 <= right[-1] <= 1.0:
            raise ValueError("reset gripper values must be within [0, 1]")
        self.left_qpos = left.tolist()
        self.right_qpos = right.tolist()

    def as_vector(self) -> np.ndarray:
        """Return the canonical 14-D reset target."""
        if self.left_qpos is None or self.right_qpos is None:
            raise RuntimeError("YAM reset qpos is not configured")
        return np.concatenate(
            [
                np.asarray(self.left_qpos, dtype=np.float64),
                np.asarray(self.right_qpos, dtype=np.float64),
            ]
        )


@dataclass
class DualYamJointEnvConfig:
    """Task-level settings; physical device calibration lives in hardware config."""

    is_dummy: bool = False
    task_description: str = "dual-arm YAM manipulation task"
    step_frequency: float = 30.0
    max_num_steps: int = 1000
    max_joint_delta: float = 0.08
    # Per-joint step limits (6 values, one per arm joint). None falls back to
    # the scalar max_joint_delta broadcast to all joints.
    max_joint_delta_per_joint: list[float] | None = None
    enforce_runtime_joint_limits: bool = True
    joint_limit_min: list[list[float]] = field(default_factory=_default_joint_limit_min)
    joint_limit_max: list[list[float]] = field(default_factory=_default_joint_limit_max)
    feedback_timeout_s: float = 0.25
    camera_warmup_timeout_s: float = 10.0
    camera_frame_timeout_s: float = 1.0
    camera_stale_timeout_s: float = 2.0
    image_height: int = 128
    image_width: int = 128
    dummy_camera_names: list[str] = field(
        default_factory=lambda: ["top_rgb", "left_rgb", "right_rgb"]
    )
    manual_episode_control_only: bool = False
    reset: YamResetConfig | Mapping[str, Any] = field(default_factory=YamResetConfig)
    # Park the arms at a configured safe pose before releasing torque on
    # close(); the mode field is ignored (only the pose/timing fields apply).
    park_on_close: YamResetConfig | Mapping[str, Any] = field(
        default_factory=YamResetConfig
    )

    def __post_init__(self) -> None:
        if not isinstance(self.is_dummy, bool):
            raise TypeError("is_dummy must be a bool")
        if not isinstance(self.manual_episode_control_only, bool):
            raise TypeError("manual_episode_control_only must be a bool")
        if not isinstance(self.enforce_runtime_joint_limits, bool):
            raise TypeError("enforce_runtime_joint_limits must be a bool")
        self.step_frequency = float(self.step_frequency)
        self.max_joint_delta = float(self.max_joint_delta)
        if self.max_joint_delta_per_joint is not None:
            per_joint = np.asarray(self.max_joint_delta_per_joint, dtype=np.float64)
            if per_joint.shape != (NUM_ARM_JOINTS,):
                raise ValueError(
                    "max_joint_delta_per_joint must be six values, one per arm joint"
                )
            if not np.all(np.isfinite(per_joint)) or not np.all(per_joint > 0):
                raise ValueError(
                    "max_joint_delta_per_joint values must be positive and finite"
                )
            self.max_joint_delta_per_joint = per_joint.tolist()
        self.feedback_timeout_s = float(self.feedback_timeout_s)
        self.camera_warmup_timeout_s = float(self.camera_warmup_timeout_s)
        self.camera_frame_timeout_s = float(self.camera_frame_timeout_s)
        self.camera_stale_timeout_s = float(self.camera_stale_timeout_s)
        self.image_height = int(self.image_height)
        self.image_width = int(self.image_width)
        self.max_num_steps = int(self.max_num_steps)
        self.joint_limit_min = np.asarray(
            self.joint_limit_min, dtype=np.float64
        ).reshape(2, NUM_ARM_JOINTS)
        self.joint_limit_max = np.asarray(
            self.joint_limit_max, dtype=np.float64
        ).reshape(2, NUM_ARM_JOINTS)
        scalar_values = {
            "step_frequency": self.step_frequency,
            "max_joint_delta": self.max_joint_delta,
            "feedback_timeout_s": self.feedback_timeout_s,
            "camera_warmup_timeout_s": self.camera_warmup_timeout_s,
            "camera_frame_timeout_s": self.camera_frame_timeout_s,
            "camera_stale_timeout_s": self.camera_stale_timeout_s,
        }
        for name, value in scalar_values.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.step_frequency <= 0:
            raise ValueError("step_frequency must be positive")
        if self.max_joint_delta <= 0:
            raise ValueError("max_joint_delta must be positive")
        if self.feedback_timeout_s <= 0:
            raise ValueError("feedback_timeout_s must be positive")
        if self.camera_warmup_timeout_s <= 0:
            raise ValueError("camera_warmup_timeout_s must be positive")
        if self.camera_frame_timeout_s <= 0:
            raise ValueError("camera_frame_timeout_s must be positive")
        if self.camera_stale_timeout_s <= 0:
            raise ValueError("camera_stale_timeout_s must be positive")
        if self.max_num_steps <= 0:
            raise ValueError("max_num_steps must be positive")
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("image dimensions must be positive")
        if not np.all(self.joint_limit_min < self.joint_limit_max):
            raise ValueError(
                "every YAM joint lower limit must be below its upper limit"
            )
        if not np.all(np.isfinite(self.joint_limit_min)) or not np.all(
            np.isfinite(self.joint_limit_max)
        ):
            raise ValueError("YAM joint limits must be finite")
        if isinstance(self.reset, Mapping):
            self.reset = YamResetConfig(**dict(self.reset))
        elif not isinstance(self.reset, YamResetConfig):
            raise TypeError("reset must be a mapping or YamResetConfig")
        if isinstance(self.park_on_close, Mapping):
            self.park_on_close = YamResetConfig(**dict(self.park_on_close))
        elif not isinstance(self.park_on_close, YamResetConfig):
            raise TypeError("park_on_close must be a mapping or YamResetConfig")
        for block, block_name in (
            (self.reset, "reset"),
            (self.park_on_close, "park_on_close"),
        ):
            if block.left_qpos is None:
                continue
            qpos_arms = (
                np.asarray(block.left_qpos, dtype=np.float64),
                np.asarray(block.right_qpos, dtype=np.float64),
            )
            for arm_index, target in enumerate(qpos_arms):
                below_limit = target[:NUM_ARM_JOINTS] < self.joint_limit_min[arm_index]
                above_limit = target[:NUM_ARM_JOINTS] > self.joint_limit_max[arm_index]
                if np.any(below_limit) or np.any(above_limit):
                    raise ValueError(
                        f"{block_name} qpos for arm {arm_index} is outside joint limits"
                    )
        self.task_description = str(self.task_description)
        names = [str(name) for name in self.dummy_camera_names]
        if not names or len(set(names)) != len(names):
            raise ValueError("dummy_camera_names must be non-empty and unique")
        self.dummy_camera_names = names

    @property
    def joint_step_limits(self) -> np.ndarray:
        """Per-joint step limits (6,); scalar max_joint_delta when unset."""
        if self.max_joint_delta_per_joint is None:
            return np.full(NUM_ARM_JOINTS, self.max_joint_delta, dtype=np.float64)
        return np.asarray(self.max_joint_delta_per_joint, dtype=np.float64)
