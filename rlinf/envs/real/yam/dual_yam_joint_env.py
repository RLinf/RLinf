# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gymnasium environment for dual YAM joint-space control."""

from __future__ import annotations

import time
from collections.abc import Mapping
from functools import partial
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from rlinf.envs.real.utils.config import get_hardware_config
from rlinf.robotics import DualYamConfig, DualYamRobot, RobotInfo
from rlinf.robotics.actions import ActionKind, ActionPart
from rlinf.robotics.parts.arms.yam import YamArm, YamKinematicsAdapter
from rlinf.robotics.parts.base import PartGroup
from rlinf.robotics.parts.cameras import BaseCamera, Camera, CameraInfo
from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger

from .config import DualYamJointEnvConfig
from .types import (
    DUAL_ARM_ACTION_DIM,
    NUM_ARM_JOINTS,
    YamCommandResult,
    pack_dual_action,
    split_dual_action,
)

#: Arm groups of the station, in action-vector order.
_SIDES = ("left", "right")


class DualYamJointEnv(gym.Env):
    """A 14-D absolute-joint environment for a pair of YAM follower arms.

    The fixed action and state order is
    ``[left_q0..q5, left_gripper, right_q0..q5, right_gripper]``. Grippers use
    the canonical convention ``0=closed, 1=open``.

    The arms and cameras come from :class:`DualYamRobot`. This env holds every
    command to the task's joint limits and to a per-tick joint step around the
    measured pose, and holds both arms when either one fails. Construction is
    side-effect free: the station opens on the first :meth:`reset`, cameras
    before any CAN bus.
    """

    metadata = {"render_modes": []}

    #: Teleop devices the shared stack may drive this environment with. Each
    #: arm takes joint positions, so ``pico`` solves its controller's pose
    #: through :meth:`arm_kinematics`; give one entry per arm with ``drives``.
    TELEOP: tuple[str, ...] = ("pico",)
    TELEOP_DEFAULT = "none"
    TELEOP_MARK_FLAG = True

    def __init__(
        self,
        override_cfg: dict[str, Any] | None,
        worker_info: Optional[WorkerInfo] = None,
        robot_info: "Optional[RobotInfo[DualYamConfig]]" = None,
        env_idx: int = 0,
    ) -> None:
        super().__init__()
        override_values = dict(override_cfg or {})
        self.config = DualYamJointEnvConfig(**override_values)
        self.hardware = get_hardware_config(
            DualYamConfig, robot_info, is_dummy=self.config.is_dummy
        )
        self.env_idx = int(env_idx)
        self.node_rank = 0
        self.env_worker_rank = 0
        if worker_info is not None:
            self.node_rank = worker_info.cluster_node_rank
            self.env_worker_rank = worker_info.rank
        self._logger = get_logger()
        self.robot: Optional[DualYamRobot] = None
        self._arms: Optional[PartGroup] = None
        self._cameras: list[BaseCamera] = []
        self._last_camera_frame: dict[str, np.ndarray] = {}
        self._last_camera_success_s: dict[str, float] = {}
        # Without hardware the measured pose is the last accepted target.
        self._dummy_state = np.zeros(DUAL_ARM_ACTION_DIM, dtype=np.float64)
        self._started = False
        self._start_failed = False
        self._closed = False
        self._num_steps = 0
        # Per-tick control timing (consumed by the VR wrapper's _report_timing
        # and the log_control_timing switch). Values from the last step.
        self._last_dispatch_time_s: float | None = None
        self._last_pace_s = 0.0
        self._last_command_s = 0.0
        self._last_state_read_s = 0.0
        self._last_camera_read_s = 0.0
        self._last_tick_s: float | None = None
        self._reset_qpos_applied = False

        if not self.config.is_dummy and not {
            "joint_limit_min",
            "joint_limit_max",
        }.issubset(override_values):
            raise ValueError(
                "real YAM environments require explicit joint_limit_min and "
                "joint_limit_max values within the installed i2rt limits"
            )
        self._camera_names = list(self.config.camera_names)
        if not self.config.is_dummy:
            serials = self.hardware.camera_serials or []
            if len(serials) != len(self._camera_names):
                raise ValueError(
                    f"The YAM env names {len(self._camera_names)} cameras "
                    f"{self._camera_names}, but the station lists "
                    f"{len(serials)} camera serials {serials}. Give one serial "
                    "per name, in the same order."
                )

        arm_low = np.concatenate([self.config.joint_limit_min[0], np.array([0.0])])
        arm_high = np.concatenate([self.config.joint_limit_max[0], np.array([1.0])])
        right_low = np.concatenate([self.config.joint_limit_min[1], np.array([0.0])])
        right_high = np.concatenate([self.config.joint_limit_max[1], np.array([1.0])])
        self.action_space = gym.spaces.Box(
            low=np.concatenate([arm_low, right_low]).astype(np.float32),
            high=np.concatenate([arm_high, right_high]).astype(np.float32),
            dtype=np.float32,
        )
        frame_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.config.image_height, self.config.image_width, 3),
            dtype=np.uint8,
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "joint_position": gym.spaces.Box(
                            low=self.action_space.low,
                            high=self.action_space.high,
                            shape=(DUAL_ARM_ACTION_DIM,),
                            dtype=np.float32,
                        )
                    }
                ),
                "frames": gym.spaces.Dict(
                    dict.fromkeys(self._camera_names, frame_space)
                ),
            }
        )

    @property
    def task_description(self) -> str:
        """Natural-language task prompt exposed to ``RealWorldEnv``."""
        return self.config.task_description

    def action_parts(self) -> tuple[ActionPart, ...]:
        """Return the named layout of the 14-D action vector.

        The order matches :meth:`step`: each arm's six absolute joint angles
        followed by its gripper, left arm first.
        """
        from rlinf.envs.real.wrappers.teleop.layout import mirrored

        return mirrored(
            (
                ActionPart("arm", 6, ActionKind.JOINT_POSITION),
                ActionPart("end_effector", 1, ActionKind.GRIPPER),
            ),
            _SIDES,
        )

    def arm_kinematics(self, side: Optional[str]) -> YamKinematicsAdapter:
        """Build the kinematic model of one arm, for a device that solves IK.

        Args:
            side: ``"left"`` or ``"right"``, the arm's action branch.
        """
        if side not in _SIDES:
            raise ValueError(
                f"A dual YAM arm is 'left' or 'right', but a device asked for {side!r}."
            )
        index = _SIDES.index(side)
        return YamKinematicsAdapter(
            arm_type=self.hardware.arm_type,
            gripper_type=self.hardware.gripper_type,
            config=self.config.ik,
            joint_lower=self.config.joint_limit_min[index],
            joint_upper=self.config.joint_limit_max[index],
        )

    def episode_wrappers(self, cfg: Mapping[str, Any]) -> list[Any]:
        """Return PICO episode control when teleoperation is configured.

        Raises:
            ValueError: If teleoperation is configured without runtime joint
                limits, which hold each operator target to the task's range.
        """
        if self.config.is_dummy:
            return []
        from rlinf.envs.real.wrappers.teleop.config import resolve_teleop_devices

        if not resolve_teleop_devices(
            cfg, supported=self.TELEOP, default=self.TELEOP_DEFAULT
        ):
            return []
        if not self.config.enforce_runtime_joint_limits:
            raise ValueError(
                "Teleoperating a YAM station needs enforce_runtime_joint_limits "
                "true, but it is false."
            )
        from .pico_episode import YamPicoEpisode, YamPicoEpisodeConfig

        return [
            partial(
                YamPicoEpisode,
                config=YamPicoEpisodeConfig(**dict(cfg.get("yam_episode", {}))),
            )
        ]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start hardware and optionally move to the configured reset qpos."""
        super().reset(seed=seed)
        self._ensure_started()
        reset_applied = False
        if self.will_reset_to_configured_qpos(options):
            self.reset_to_configured_qpos()
            reset_applied = True
        self._num_steps = 0
        self._last_tick_s = None
        info = {"episode_phase": "pre"}
        if self.config.reset.enabled:
            info["reset_qpos_applied"] = reset_applied
        return self._get_observation(), info

    def will_reset_to_configured_qpos(
        self, options: Mapping[str, Any] | None = None
    ) -> bool:
        """Return whether the next reset will command the configured qpos."""
        if bool((options or {}).get("reset_qpos", False)):
            return True
        reset_config = self.config.reset
        if not reset_config.enabled:
            return False
        if reset_config.mode == "episode":
            return True
        return reset_config.mode == "startup" and not self._reset_qpos_applied

    def reset_to_configured_qpos(self) -> np.ndarray:
        """Move followers to the configured reset pose without resetting Gym."""
        reset_config = self.config.reset
        if not reset_config.enabled:
            raise RuntimeError("YAM reset qpos is disabled")
        self._ensure_started()
        measured = self._move_to(
            reset_config.as_vector(),
            duration_s=reset_config.duration_s,
            max_joint_delta=reset_config.max_joint_delta,
            tolerance=reset_config.tolerance,
            timeout_s=reset_config.timeout_s,
        )
        self._reset_qpos_applied = True
        self._last_tick_s = None
        return measured

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply one bounded absolute-joint target and read the next state."""
        self._ensure_started()
        result, interval_s = self._paced_command(action)
        try:
            observation = self._get_observation()
        except Exception:
            self.emergency_hold()
            raise
        self._num_steps += 1
        truncated = bool(
            not self.config.manual_episode_control_only
            and self._num_steps >= self.config.max_num_steps
        )
        info = {
            "accepted_action": result.accepted.astype(np.float32),
            "action_clipped": bool(result.clipped),
            "action_rejected": result.rejection_reason,
            "yam_command_interval_s": interval_s,
            **self._timing_info(),
        }
        return observation, 0.0, False, truncated, info

    def teleop_tick(self, action: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply a teleoperation target without consuming an episode step."""
        self._ensure_started()
        result, interval_s = self._paced_command(action)
        try:
            observation = self._get_observation()
        except Exception:
            self.emergency_hold()
            raise
        return observation, {
            "accepted_action": result.accepted.astype(np.float32),
            "action_clipped": bool(result.clipped),
            "action_rejected": result.rejection_reason,
            "yam_command_interval_s": interval_s,
            **self._timing_info(),
        }

    def get_hold_action(self, fallback_action: Any = None) -> np.ndarray:
        """Return current measured positions, never an all-zero fallback."""
        del fallback_action
        return self.get_joint_positions()

    def get_joint_positions(self) -> np.ndarray:
        """Return the measured joints and grippers in action order.

        Shape ``(14,)``: each arm's six joint angles followed by its gripper,
        left arm first. Teleoperation devices read this to seed inverse
        kinematics and to hold an arm where it stands.
        """
        self._ensure_started()
        return self._read_state().astype(np.float32)

    def observe(self) -> dict[str, Any]:
        """Return a fresh observation without dispatching a motion command."""
        self._ensure_started()
        return self._get_observation()

    def emergency_hold(self) -> None:
        """Hold both arms at their measured pose, logging what fails.

        For failure paths: it reads no feedback first, so it still holds an
        arm whose feedback has gone stale.
        """
        for side, arm in self._arm_parts().items():
            try:
                arm.hold()
            except Exception:  # pragma: no cover - hardware failure path
                self._logger.exception("Failed to hold the %s YAM follower", side)

    def close(self) -> None:
        """Release the station.

        When ``park_on_close`` is enabled and the station is open, both arms
        first move to the configured safe pose, so releasing torque never
        drops them mid-air. Parking is best-effort: a failure still releases
        the station.
        """
        if self._closed:
            return
        self._start_failed = True
        park = self.config.park_on_close
        if park.enabled and self._started:
            try:
                self._move_to(
                    park.as_vector(),
                    duration_s=park.duration_s,
                    max_joint_delta=park.max_joint_delta,
                    tolerance=park.tolerance,
                    timeout_s=park.timeout_s,
                )
            except Exception as error:  # pragma: no cover - hardware cleanup path
                self._logger.error(f"YAM park_on_close failed: {error}")
        try:
            self._disconnect()
        finally:
            self._started = False
            self._closed = True

    # Hardware lifecycle.

    def _ensure_started(self) -> None:
        if self._closed:
            raise RuntimeError("cannot use a closed DualYamJointEnv")
        if self._start_failed:
            raise RuntimeError(
                "DualYamJointEnv startup previously failed; close and rebuild it"
            )
        if self._started:
            return
        try:
            if not self.config.is_dummy:
                self._connect()
            self._started = True
        except Exception as start_error:
            self._start_failed = True
            try:
                self._disconnect()
            except Exception as cleanup_error:  # pragma: no cover - hardware failure
                self._logger.error(
                    "YAM startup failed (%s), and cleanup also failed: %s",
                    start_error,
                    cleanup_error,
                )
            raise

    def _connect(self) -> None:
        """Open the cameras, then the arms, and hold the measured pose."""
        hardware = self.hardware
        controller_node_rank = hardware.controller_node_rank
        if controller_node_rank is None:
            controller_node_rank = self.node_rank
        self.robot = DualYamRobot.build(
            left_can_channel=hardware.left_can_channel,
            right_can_channel=hardware.right_can_channel,
            node_rank=controller_node_rank,
            backend=hardware.backend,
            arm_type=hardware.arm_type,
            gripper_type=hardware.gripper_type,
            left_arm=hardware.left_arm,
            right_arm=hardware.right_arm,
            feedback_timeout_s=hardware.feedback_timeout_s,
            worker_rank=self.env_worker_rank,
            env_idx=self.env_idx,
            cameras={info.name: info for info in self._camera_infos()},
            camera_node_rank=hardware.camera_node_rank,
        )
        # A camera fault must surface before any arm is under power.
        self._cameras = list(self.robot.parts_of_type(Camera).values())
        for camera in self._cameras:
            camera.connect()
        for camera in self._cameras:
            frame = camera.get_frame(timeout=self.config.camera_warmup_timeout_s)
            self._last_camera_frame[camera.name] = self._process_frame(frame)
            self._last_camera_success_s[camera.name] = time.monotonic()
        self.robot.connect()
        self._arms = PartGroup({side: self.robot.child(side) for side in _SIDES})
        self._validate_joint_limits()
        # Fail startup when feedback is absent or stale instead of accepting a
        # first command without knowing the current pose.
        self._read_state()
        self._hold()

    def _disconnect(self) -> None:
        robot, self.robot = self.robot, None
        self._arms = None
        self._cameras = []
        self._last_camera_frame.clear()
        self._last_camera_success_s.clear()
        if robot is not None:
            robot.disconnect()

    def _camera_infos(self) -> list[CameraInfo]:
        """Name the station's cameras in the order of its serials."""
        hardware = self.hardware
        return [
            CameraInfo(
                name=name,
                serial_number=serial,
                camera_type=hardware.camera_type,
                fps=hardware.camera_fps,
            )
            for name, serial in zip(
                self._camera_names, hardware.camera_serials or [], strict=True
            )
        ]

    def _arm_parts(self) -> dict[str, YamArm]:
        if self.robot is None:
            return {}
        return {side: self.robot.child(side).child("arm", YamArm) for side in _SIDES}

    def _validate_joint_limits(self) -> None:
        """Refuse task limits wider than the travel i2rt allows."""
        tolerance = 1e-6
        for arm_index, arm in enumerate(self._arm_parts().values()):
            sdk_limits = arm.joint_limits()
            configured_lower = self.config.joint_limit_min[arm_index]
            configured_upper = self.config.joint_limit_max[arm_index]
            if np.any(configured_lower < sdk_limits[:, 0] - tolerance) or np.any(
                configured_upper > sdk_limits[:, 1] + tolerance
            ):
                raise ValueError(
                    f"configured YAM arm {arm_index} joint limits must be within "
                    f"the i2rt limits {sdk_limits.tolist()}"
                )

    # Commanding the arms.

    def _read_state(self) -> np.ndarray:
        """Read both arms as the 14-D state, refusing stale feedback."""
        if self.config.is_dummy:
            return self._dummy_state.copy()
        reading = self._arms.get_observation()
        return pack_dual_action(
            *(
                np.concatenate(
                    [
                        reading[side]["arm"]["arm_joint_position"],
                        reading[side]["arm"]["end_effector"]["state"],
                    ]
                )
                for side in _SIDES
            )
        )

    def _dispatch(self, target: np.ndarray) -> None:
        if self.config.is_dummy:
            self._dummy_state = target.copy()
            return
        self._arms.send_action(
            {
                side: {
                    "arm": {
                        "joint_position": arm[:NUM_ARM_JOINTS],
                        "end_effector": {"target": arm[NUM_ARM_JOINTS:]},
                    }
                }
                for side, arm in zip(_SIDES, split_dual_action(target), strict=True)
            }
        )

    def _hold(self) -> np.ndarray:
        """Hold the measured pose of both arms and return it."""
        try:
            state = self._read_state()
        except Exception:
            self.emergency_hold()
            raise
        first_error: Exception | None = None
        for arm in self._arm_parts().values():
            try:
                arm.hold()
            except Exception as error:  # pragma: no cover - hardware failure path
                first_error = first_error or error
                self._logger.exception("Failed to hold a YAM follower")
        if first_error is not None:
            raise first_error
        return state

    def _command(self, action: Any) -> YamCommandResult:
        """Validate and dispatch one dual-arm target.

        Shape errors are rejected before reading or writing hardware. Non-finite
        targets hold both arms and are reported without forwarding them. With
        ``enforce_runtime_joint_limits``, each joint is held to the task limits
        and moves at most its step limit from where it was measured.
        """
        left_requested, right_requested = split_dual_action(action)
        requested = pack_dual_action(left_requested, right_requested)

        if not np.all(np.isfinite(requested)):
            held = self._hold()
            return YamCommandResult(
                requested=requested,
                accepted=held,
                rejection_reason="non_finite_action",
            )

        try:
            measured = self._read_state()
        except Exception:
            self.emergency_hold()
            raise
        accepted_arms: list[np.ndarray] = []
        for arm_index, (target, current_arm) in enumerate(
            zip(
                (left_requested, right_requested),
                split_dual_action(measured),
                strict=True,
            )
        ):
            accepted = target.copy()
            if self.config.enforce_runtime_joint_limits:
                current = current_arm[:NUM_ARM_JOINTS]
                # A joint read just past a limit, such as at a mechanical stop,
                # may still move back inside but never further out.
                lower = np.minimum(self.config.joint_limit_min[arm_index], current)
                upper = np.maximum(self.config.joint_limit_max[arm_index], current)
                accepted[:NUM_ARM_JOINTS] = np.clip(
                    accepted[:NUM_ARM_JOINTS], lower, upper
                )
                accepted[:NUM_ARM_JOINTS] = np.clip(
                    accepted[:NUM_ARM_JOINTS],
                    current - self.config.joint_step_limits,
                    current + self.config.joint_step_limits,
                )
            accepted[-1] = np.clip(accepted[-1], 0.0, 1.0)
            accepted_arms.append(accepted)

        accepted_vector = pack_dual_action(*accepted_arms)
        try:
            self._dispatch(accepted_vector)
        except Exception:
            self.emergency_hold()
            raise
        return YamCommandResult(
            requested=requested,
            accepted=accepted_vector,
            clipped=not np.array_equal(requested, accepted_vector),
        )

    def _move_to(
        self,
        target: Any,
        *,
        duration_s: float,
        max_joint_delta: float,
        tolerance: float,
        timeout_s: float,
    ) -> np.ndarray:
        """Move both arms smoothly to an absolute 14-D target and verify it.

        The interpolation bound is independent of normal policy/teleoperation
        limits, so a reset stays gradual even when runtime step limiting is
        off. The configured hard joint limits always apply here.
        """
        left_target, right_target = split_dual_action(target)
        target_vector = pack_dual_action(left_target, right_target)
        if not np.all(np.isfinite(target_vector)):
            raise ValueError("YAM move target must be finite")
        for arm_index, arm_target in enumerate((left_target, right_target)):
            lower = self.config.joint_limit_min[arm_index]
            upper = self.config.joint_limit_max[arm_index]
            if np.any(arm_target[:NUM_ARM_JOINTS] < lower) or np.any(
                arm_target[:NUM_ARM_JOINTS] > upper
            ):
                raise ValueError(
                    f"YAM move target for arm {arm_index} is outside joint limits"
                )
            if not 0.0 <= arm_target[-1] <= 1.0:
                raise ValueError("YAM move gripper targets must be within [0, 1]")

        duration_s = float(duration_s)
        max_joint_delta = float(max_joint_delta)
        tolerance = float(tolerance)
        timeout_s = float(timeout_s)
        if not all(
            np.isfinite(value)
            for value in (duration_s, max_joint_delta, tolerance, timeout_s)
        ):
            raise ValueError("YAM move timing and tolerance must be finite")
        if duration_s < 0 or max_joint_delta <= 0 or tolerance <= 0:
            raise ValueError("invalid YAM move timing or tolerance")
        if timeout_s < duration_s:
            raise ValueError("YAM move timeout must be at least its duration")

        start = self._read_state()
        joint_indices = np.array([*range(6), *range(7, 13)])
        limits = np.full(12, max_joint_delta, dtype=np.float64)
        if self.config.enforce_runtime_joint_limits:
            limits = np.minimum(limits, np.tile(self.config.joint_step_limits, 2))
        demand = np.abs(target_vector[joint_indices] - start[joint_indices]) / limits
        duration_steps = int(np.ceil(duration_s * self.config.step_frequency))
        # smoothstep has a maximum slope of 1.5, so account for it when
        # bounding the difference between consecutive commanded setpoints.
        slew_steps = int(np.ceil(1.5 * np.max(demand)))
        steps = max(1, duration_steps, slew_steps)
        period_s = 1.0 / self.config.step_frequency
        deadline_s = time.monotonic() + timeout_s
        self._logger.info(
            "Moving YAM followers to reset qpos over %d steps (%.2fs nominal)",
            steps,
            steps * period_s,
        )

        try:
            for step in range(1, steps + 1):
                progress = step / steps
                alpha = progress * progress * (3.0 - 2.0 * progress)
                result = self._command(start + alpha * (target_vector - start))
                if result.rejection_reason is not None:
                    raise RuntimeError(
                        "YAM reset command was rejected: " + result.rejection_reason
                    )
                if step < steps:
                    remaining_s = deadline_s - time.monotonic()
                    if remaining_s <= 0:
                        raise TimeoutError("YAM reset timed out during interpolation")
                    time.sleep(min(period_s, remaining_s))

            while True:
                measured = self._read_state()
                max_error = float(np.max(np.abs(target_vector - measured)))
                if max_error <= tolerance:
                    held = self._hold()
                    self._logger.info(
                        "YAM reset qpos reached (max error %.4f)", max_error
                    )
                    return held
                remaining_s = deadline_s - time.monotonic()
                if remaining_s <= 0:
                    raise TimeoutError(
                        "YAM reset did not converge within "
                        f"{timeout_s:.2f}s (max error {max_error:.4f}, "
                        f"tolerance {tolerance:.4f})"
                    )
                result = self._command(target_vector)
                if result.rejection_reason is not None:
                    raise RuntimeError(
                        "YAM reset command was rejected: " + result.rejection_reason
                    )
                time.sleep(min(period_s, remaining_s))
        except Exception:
            self.emergency_hold()
            raise

    def _paced_command(self, action: np.ndarray) -> tuple[YamCommandResult, float]:
        """Pace, dispatch, and time one command; returns (result, interval_s)."""
        pace_started = time.monotonic()
        self._pace()
        self._last_pace_s = time.monotonic() - pace_started
        dispatch_at = time.monotonic()
        interval_s = (
            0.0
            if self._last_dispatch_time_s is None
            else dispatch_at - self._last_dispatch_time_s
        )
        self._last_dispatch_time_s = dispatch_at
        command_started = time.monotonic()
        result = self._command(action)
        self._last_command_s = time.monotonic() - command_started
        return result, interval_s

    def _pace(self) -> None:
        period_s = 1.0 / self.config.step_frequency
        now = time.perf_counter()
        if self._last_tick_s is not None:
            sleep_s = self._last_tick_s + period_s - now
            if sleep_s > 0:
                time.sleep(sleep_s)
        self._last_tick_s = time.perf_counter()

    # Observation.

    def _get_observation(self) -> dict[str, Any]:
        state_started = time.monotonic()
        state = self._read_state().astype(np.float32)
        self._last_state_read_s = time.monotonic() - state_started
        camera_started = time.monotonic()
        if self.config.is_dummy:
            frames = {
                name: np.zeros(
                    (self.config.image_height, self.config.image_width, 3),
                    dtype=np.uint8,
                )
                for name in self._camera_names
            }
        else:
            frames = self._read_camera_frames()
        self._last_camera_read_s = time.monotonic() - camera_started
        return {"state": {"joint_position": state}, "frames": frames}

    def _timing_info(self) -> dict[str, float]:
        """Per-tick control timing for the last dispatched step."""
        return {
            "yam_pace_s": self._last_pace_s,
            "yam_command_s": self._last_command_s,
            "yam_state_read_s": self._last_state_read_s,
            "yam_camera_read_s": self._last_camera_read_s,
        }

    def _read_camera_frames(self) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        for camera in self._cameras:
            try:
                raw = camera.get_frame(timeout=self.config.camera_frame_timeout_s)
                frame = self._process_frame(raw)
                self._last_camera_frame[camera.name] = frame
                self._last_camera_success_s[camera.name] = time.monotonic()
            except Exception:
                if camera.name not in self._last_camera_frame:
                    raise
                stale_age_s = (
                    time.monotonic() - self._last_camera_success_s[camera.name]
                )
                if stale_age_s > self.config.camera_stale_timeout_s:
                    raise RuntimeError(
                        f"YAM camera {camera.name!r} has been stale for "
                        f"{stale_age_s:.3f}s"
                    )
                self._logger.warning(
                    "YAM camera %s missed a frame; reusing the last frame",
                    camera.name,
                )
                frame = self._last_camera_frame[camera.name]
            frames[camera.name] = frame.copy()
        return frames

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] < 3:
            raise ValueError(
                f"YAM RGB frame must have shape (H, W, >=3), got {array.shape}"
            )
        # Camera backends expose BGR; policy and dataset observations are RGB.
        rgb = np.ascontiguousarray(array[..., :3][..., ::-1], dtype=np.uint8)
        target_shape = (self.config.image_height, self.config.image_width)
        if rgb.shape[:2] != target_shape:
            import cv2

            rgb = cv2.resize(
                rgb,
                (self.config.image_width, self.config.image_height),
                interpolation=cv2.INTER_AREA,
            )
        return np.asarray(rgb, dtype=np.uint8)
