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

"""Dual ARX X5 environment with absolute TCP and gripper actions."""

from __future__ import annotations

import queue
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from rlinf.envs.real.utils.config import get_hardware_config
from rlinf.envs.real.utils.seeding import seed_sampled_spaces
from rlinf.robotics import Camera, DualARX5Config, DualARX5Robot, RobotInfo
from rlinf.robotics.actions import ActionKind, ActionPart
from rlinf.robotics.parts.arms.arx5 import (
    ARX5_PARK_MAX_DEG,
    ARX5_REST_TOLERANCE_DEG,
    ARX5Arm,
    ARX5IKUnreachableError,
    ARX5ParkingMonitor,
    ARX5RestMonitor,
    gripper_endpoints,
)
from rlinf.robotics.parts.cameras import CameraInfo
from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger
from rlinf.utils.rot6d import matrix_to_rot6d, rot6d_to_quat_xyzw_safe

_JOINT_MIN_RAD = np.deg2rad([-150, 0, 0, -90, -90, -120])
_JOINT_MAX_RAD = np.deg2rad([180, 210, 180, 90, 90, 120])
_MAX_JOINT_STEP_RAD = 0.05
_TCP_HOLD_TRANSLATION_TOLERANCE_M = 1e-4
_TCP_HOLD_ROTATION_TOLERANCE_RAD = np.deg2rad(0.1)


class ARX5StopRequested(KeyboardInterrupt):
    """An operator requested a controlled stop before releasing the arms."""


class ARX5JointReturnTimeout(TimeoutError):
    """Valid joint feedback did not reach a return target within its budget."""


@dataclass
class DualARX5EnvConfig:
    """Configure control timing, episode resets, and connection cleanup.

    ``joint_reset_qpos`` sets left/right rows of six radians. When null,
    both arms reset to [0, 0.1, 0.1, 0, 0, 0].
    ``max_step`` (meters) and ``max_step_rad`` limit target changes relative
    to measured feedback; zero disables the corresponding limit. Franka's
    corresponding driver settings limit changes relative to preceding targets.
    ``close`` returns all six arm joints to mechanical zero.
    A return timeout may release parked arms regardless of other joint errors.
    If parking cannot be confirmed, ``close`` raises and leaves the SDK connected.
    A reset timeout continues from measured feedback after stopping the motion.
    Returns stream small joint targets at ``step_frequency``, accelerate up to
    ``joint_return_speed_deg_s``, and slow down near the target. Timeout budgets
    cover return motion and final target verification; stopping,
    parking verification, and waiting for the next Pico start are separate.
    """

    is_dummy: bool = False
    step_frequency: float = 10.0
    rotation_repr: str = "rot6d"
    max_num_steps: int = 300
    task_description: str = ""
    joint_reset_qpos: np.ndarray | None = None
    reset_joint_timeout_s: float = 10.0
    joint_return_speed_deg_s: float = 10.0
    joint_return_acceleration_deg_s2: float = 30.0
    joint_return_slowdown_deg: float = 3.0
    joint_return_stall_timeout_s: float = 2.0
    max_step: float = 0.02
    max_step_rad: float = 0.15
    close_joint_timeout_s: float = 30.0
    close_park_max_deg: float = ARX5_PARK_MAX_DEG
    enforce_workspace_bounds: bool = True
    initial_camera_timeout_s: float = 5.0
    require_camera: bool = True
    image_size: int = 224
    ee_pose_limit_min: np.ndarray = field(default_factory=lambda: np.full((2, 3), -1.0))
    ee_pose_limit_max: np.ndarray = field(default_factory=lambda: np.full((2, 3), 1.0))

    def __post_init__(self) -> None:
        for name in (
            "step_frequency",
            "reset_joint_timeout_s",
            "joint_return_speed_deg_s",
            "joint_return_acceleration_deg_s2",
            "joint_return_slowdown_deg",
            "joint_return_stall_timeout_s",
            "close_joint_timeout_s",
            "close_park_max_deg",
            "initial_camera_timeout_s",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("max_step", "max_step_rad"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    f"{name} must be finite and nonnegative; 0 disables it."
                )
        if self.rotation_repr != "rot6d":
            raise ValueError("DualARX5Env supports only rotation_repr='rot6d'.")
        for name in ("max_num_steps", "image_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(self.enforce_workspace_bounds, bool):
            raise ValueError("enforce_workspace_bounds must be a boolean.")
        if self.joint_reset_qpos is not None:
            target = np.asarray(self.joint_reset_qpos, dtype=np.float64)
            if (
                target.shape != (2, 6)
                or not np.all(np.isfinite(target))
                or np.any(target < _JOINT_MIN_RAD)
                or np.any(target > _JOINT_MAX_RAD)
            ):
                raise ValueError(
                    "joint_reset_qpos requires two finite "
                    "six-joint rows in radians within the ARX5 joint limits."
                )
            self.joint_reset_qpos = target
        lower = np.asarray(self.ee_pose_limit_min, dtype=np.float64)
        upper = np.asarray(self.ee_pose_limit_max, dtype=np.float64)
        if (
            lower.shape != (2, 3)
            or upper.shape != (2, 3)
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower >= upper)
        ):
            raise ValueError("TCP bounds must be two [x, y, z] rows with min < max.")
        self.ee_pose_limit_min = lower
        self.ee_pose_limit_max = upper


class DualARX5Env(gym.Env):
    """Two X5 arms controlled by 20-D [xyz, rot6d, gripper] actions.

    Camera images and TCP poses follow the DualFrankaTCPEnv naming convention.
    Camera timeouts close the environment; other read errors propagate.
    Gripper observations use measured SDK positions normalized to [-1, 1]
    (closed/open). SDK sessions run inside the environment worker.
    """

    TELEOP = ("pico",)
    TELEOP_DEFAULT = "none"
    TELEOP_MARK_FLAG = True
    ACTION_WRAPPERS = ()
    TRANSFORMS = ()
    REFUSE_FLAGS = ("no_gripper",)
    REFUSE_DEFAULTS = {"no_gripper": False}

    def __init__(
        self,
        override_cfg: dict[str, Any],
        worker_info: WorkerInfo | None = None,
        robot_info: RobotInfo[DualARX5Config] | None = None,
        env_idx: int = 0,
    ) -> None:
        self.config = DualARX5EnvConfig(**override_cfg)
        self.hardware = get_hardware_config(
            DualARX5Config, robot_info, is_dummy=self.config.is_dummy
        )
        self._task_description = self.config.task_description
        self._gripper_open, self._gripper_closed = gripper_endpoints(
            self.hardware.arm_type
        )
        self._steps = 0
        self._closed = False
        self._logger = get_logger()
        self._closing = False
        self._stop_requested = False
        self._signal_handlers: dict[int, Any] = {}
        self.robot = None
        self._arms: tuple[ARX5Arm, ARX5Arm] | None = None
        self._arm_executors: tuple[ThreadPoolExecutor, ThreadPoolExecutor] | None = None
        self._cameras: dict[str, Camera] = {}
        self._camera_attempted: set[str] = set()
        camera_specs = self._camera_infos()
        if not camera_specs and self.config.require_camera and not self.config.is_dummy:
            raise ValueError(
                "At least one camera is required for ARX5 data collection."
            )

        xyz_low = (
            self.config.ee_pose_limit_min
            if self.config.enforce_workspace_bounds
            else np.full((2, 3), -np.inf)
        )
        xyz_high = (
            self.config.ee_pose_limit_max
            if self.config.enforce_workspace_bounds
            else np.full((2, 3), np.inf)
        )
        per_arm_low = [*xyz_low[0], *([-1.5] * 6), -1]
        per_arm_high = [*xyz_high[0], *([1.5] * 6), 1]
        right_low = [*xyz_low[1], *([-1.5] * 6), -1]
        right_high = [*xyz_high[1], *([1.5] * 6), 1]
        self.action_space = gym.spaces.Box(
            np.asarray(per_arm_low + right_low, dtype=np.float32),
            np.asarray(per_arm_high + right_high, dtype=np.float32),
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose_rot6d": gym.spaces.Box(
                            -np.inf, np.inf, shape=(18,), dtype=np.float32
                        ),
                        "gripper_position": gym.spaces.Box(
                            -1, 1, shape=(2,), dtype=np.float32
                        ),
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        info.name: gym.spaces.Box(
                            0,
                            255,
                            shape=(self.config.image_size, self.config.image_size, 3),
                            dtype=np.uint8,
                        )
                        for info in camera_specs
                    }
                ),
            }
        )
        if self.config.is_dummy:
            return

        rank = worker_info.cluster_node_rank if worker_info else self.hardware.node_rank
        if rank != self.hardware.node_rank:
            raise ValueError("ARX5 CAN arms must be on the environment worker's node.")
        self.robot = DualARX5Robot.build(
            left_can_port=self.hardware.left_can_port,
            right_can_port=self.hardware.right_can_port,
            arm_type=self.hardware.arm_type,
            motion_duration=1 / self.config.step_frequency,
            park_max_deg=self.config.close_park_max_deg,
            left_urdf_path=self.hardware.left_urdf_path,
            right_urdf_path=self.hardware.right_urdf_path,
            cameras={info.name: info for info in camera_specs},
        )
        self._arms = (
            self.robot.child("left").child("arm", ARX5Arm),
            self.robot.child("right").child("arm", ARX5Arm),
        )
        self._cameras = {
            path.rsplit(".", 1)[-1]: camera
            for path, camera in self.robot.parts_of_type(Camera).items()
        }
        self._arm_executors = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="arx5-left"),
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="arx5-right"),
        )
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                self._signal_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._request_stop_signal)
        try:
            # Own partial startup here: PartGroup.connect() otherwise releases
            # the first arm immediately if the second arm or a camera fails.
            for part in (*self._cameras.values(), *self._arms):
                self.check_stop_requested()
                part.connect()
        except BaseException:
            self.close()
            raise

    def _request_stop_signal(self, signum: int, frame: Any) -> None:
        self._stop_requested = True

    def check_stop_requested(self) -> None:
        """Interrupt ARX5 operations after a local stop request."""
        if self._closing or self._stop_requested:
            raise ARX5StopRequested("ARX5 controlled stop requested.")

    def _camera_infos(self) -> list[CameraInfo]:
        infos = []
        for slot, prefix in (
            ("base", "base"),
            ("left", "left_wrist"),
            ("right", "right_wrist"),
        ):
            for index, serial in enumerate(
                getattr(self.hardware, f"{slot}_camera_serials") or []
            ):
                infos.append(
                    CameraInfo(
                        name=f"{prefix}_{index}_rgb",
                        serial_number=serial,
                        camera_type=self.hardware.camera_type,
                        fps=self.hardware.camera_fps,
                    )
                )
        return infos

    @property
    def task_description(self) -> str:
        return self._task_description

    def action_parts(self) -> tuple[ActionPart, ...]:
        from rlinf.envs.real.wrappers.teleop.layout import mirrored

        return mirrored(
            (
                ActionPart("arm", 9, ActionKind.CARTESIAN_POSE),
                ActionPart("end_effector", 1, ActionKind.GRIPPER),
            ),
            ("left", "right"),
        )

    def get_tcp_pose(self) -> np.ndarray:
        """Return two SDK zero-pose-relative poses in [xyz, xyzw] form."""
        self.check_stop_requested()
        if self._arms is None:
            raise RuntimeError("TCP pose requires connected hardware.")
        states = self._run_arm_calls(self._arms[0].get_state, self._arms[1].get_state)
        return np.concatenate([state.tcp_pose for state in states])

    def read_arm_joint_positions(self, side: str) -> np.ndarray:
        """Read one arm's six measured joint angles in radians."""
        if side not in ("left", "right") or self._arms is None:
            raise ValueError(
                "Joint reading requires a connected left or right ARX5 arm."
            )
        return self._arms[0 if side == "left" else 1].read_joint_positions()

    def validate_joint_step(
        self, side: str, target: np.ndarray, *, duration: float = 2.0
    ) -> np.ndarray:
        """Check a small joint target against the measured pose and X5 limits."""
        current = self.read_arm_joint_positions(side)
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (6,) or not np.all(np.isfinite(target)):
            raise ValueError("ARX5 joint target must have six finite radians.")
        if not np.isfinite(duration) or not 0 < duration <= 10.0:
            raise ValueError(
                "ARX5 check joint duration must be above 0 and at most 10 s."
            )
        tolerance = np.deg2rad(ARX5_REST_TOLERANCE_DEG)
        if np.any(target < _JOINT_MIN_RAD - tolerance) or np.any(
            target > _JOINT_MAX_RAD + tolerance
        ):
            raise ValueError("ARX5 joint target exceeds the documented joint limits.")
        deltas = np.abs(target - current)
        if np.any(deltas > _MAX_JOINT_STEP_RAD):
            joint = int(np.argmax(deltas))
            raise ValueError(
                "ARX5 joint step exceeds 0.05 rad on one axis: "
                f"{side} J{joint + 1}, measured {np.rad2deg(current[joint]):.3f} "
                f"degrees, target {np.rad2deg(target[joint]):.3f} degrees, "
                f"delta {np.rad2deg(deltas[joint]):.3f} degrees."
            )
        return target

    def _return_to_zero_positions(self) -> None:
        """Return all six joints to zero, verifying J2/J3 before SDK release."""
        self._closing = True
        if not self._arms or not any(arm.is_connected for arm in self._arms):
            return
        try:
            self._move_to_joint_positions(
                {"left": np.zeros(6), "right": np.zeros(6)},
                timeout_s=self.config.close_joint_timeout_s,
            )
        except ARX5JointReturnTimeout as exc:
            self._verify_parked_positions()
            self._logger.warning(
                "Close return timed out: %s J2/J3 are parked within %.3f "
                "degrees and stable; releasing the SDK despite other joint errors.",
                exc,
                self.config.close_park_max_deg,
            )
        else:
            self._verify_parked_positions()

    def _verify_parked_positions(self) -> None:
        deadline = time.monotonic() + 5.0
        assert self._arms is not None
        arms = [arm for arm in self._arms if arm.is_connected]
        monitors = [ARX5ParkingMonitor(self.config.close_park_max_deg) for _ in arms]
        last_positions = {}
        while True:
            stable = []
            for arm, monitor in zip(arms, monitors):
                positions = arm.read_joint_positions()
                last_positions[arm.can_port] = np.rad2deg(positions).round(3).tolist()
                stable.append(monitor.observe(positions, time.monotonic()))
            if all(stable):
                self._logger.debug(
                    "ARX5 J2/J3 parked within %.3f degrees and stable for one "
                    "second; all measured joint angles: %s degrees",
                    self.config.close_park_max_deg,
                    last_positions,
                )
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "ARX5 J2/J3 parking and rest not confirmed within 5 s; "
                    f"last measured positions: {last_positions} degrees."
                )
            time.sleep(0.05)

    def _move_to_joint_positions(
        self,
        targets: dict[str, np.ndarray],
        *,
        timeout_s: float,
    ) -> None:
        """Stream bounded joint targets, then confirm a stable final pose."""
        if self._arms is None:
            return
        sides = [
            side for side, arm in zip(("left", "right"), self._arms) if arm.is_connected
        ]
        if not sides:
            return
        target = np.stack([targets[side] for side in sides])
        tolerance_rad = np.deg2rad(0.35)
        current = np.stack([self.read_arm_joint_positions(side) for side in sides])
        if np.max(np.abs(target - current)) <= tolerance_rad:
            return
        speed = np.deg2rad(self.config.joint_return_speed_deg_s)
        acceleration = np.deg2rad(self.config.joint_return_acceleration_deg_s2)
        slowdown = np.deg2rad(self.config.joint_return_slowdown_deg)
        period = min(1 / self.config.step_frequency, 0.1, np.deg2rad(2.5) / speed)
        started = time.monotonic()
        deadline = started + timeout_s
        progress_at = started
        progress_errors = np.abs(target - current)
        commanded = current.copy()
        velocity = np.zeros_like(current)
        monitors = [ARX5RestMonitor(row, tolerance_deg=0.35) for row in target]
        while True:
            if not self._closing:
                self.check_stop_requested()
            frame_started = time.monotonic()
            current = np.stack([self.read_arm_joint_positions(side) for side in sides])
            now = time.monotonic()
            error = target - current
            errors = np.abs(error)
            stable = [
                monitor.observe(row, now) for monitor, row in zip(monitors, current)
            ]
            if all(stable):
                return
            if np.any(progress_errors - errors >= np.deg2rad(0.1)):
                progress_at, progress_errors = now, errors.copy()
            stalled = (
                np.max(errors) > tolerance_rad
                and now - progress_at >= self.config.joint_return_stall_timeout_s
            )
            if now >= deadline or stalled:
                # Cancel the last streamed target before degraded-reset/parking checks.
                self._command_joint_positions(sides, current, period)
                reason = (
                    f"made no progress for {self.config.joint_return_stall_timeout_s:g} s"
                    if stalled
                    else f"exceeded its {timeout_s:g} s budget"
                )
                message = (
                    f"ARX5 joint return {reason}; "
                    f"target degrees: {dict(zip(sides, np.rad2deg(target).round(3).tolist()))}; "
                    f"measured degrees: {dict(zip(sides, np.rad2deg(current).round(3).tolist()))}."
                )
                raise ARX5JointReturnTimeout(message)
            # Advance the trajectory from the preceding command. Re-basing every
            # frame on measured feedback cannot overcome SDK lag or a deadband.
            command_error = target - commanded
            command_errors = np.abs(command_error)
            desired = np.clip(command_error * speed / slowdown, -speed, speed)
            braking_speed = np.sqrt(2 * acceleration * command_errors)
            desired = np.clip(desired, -braking_speed, braking_speed)
            velocity += np.clip(
                desired - velocity, -acceleration * period, acceleration * period
            )
            offset = np.sign(command_error) * np.minimum(
                np.maximum(np.sign(command_error) * velocity * period, 0),
                command_errors,
            )
            final_step = min(speed * period, acceleration * period**2 / 2)
            offset = np.where(command_errors <= final_step, command_error, offset)
            velocity = offset / period
            candidate = commanded + offset
            # Cap lead over feedback and stop integrating when that cap is hit.
            # A blocked arm never receives a distant accumulated target.
            commanded = current + np.clip(
                candidate - current, -np.deg2rad(2.5), np.deg2rad(2.5)
            )
            velocity = np.where(np.abs(candidate - commanded) > 1e-9, 0, velocity)
            # A lagging arm gets enough motion time for the accumulated lead,
            # rather than being asked to catch up that whole distance in one frame.
            duration = max(period, float(np.max(np.abs(commanded - current))) / speed)
            self._command_joint_positions(sides, commanded, duration)
            remaining = min(
                period - (time.monotonic() - frame_started),
                deadline - time.monotonic(),
            )
            if remaining > 0:
                time.sleep(remaining)

    def _command_joint_positions(
        self, sides: list[str], targets: np.ndarray, duration: float
    ) -> None:
        """Validate feedback-relative increments before either arm is commanded."""
        targets = np.stack(
            [
                self.validate_joint_step(side, target, duration=duration)
                for side, target in zip(sides, targets)
            ]
        )
        assert self._arms is not None
        arms = [self._arms[0 if side == "left" else 1] for side in sides]
        if len(arms) == 2:
            self._run_arm_calls(
                lambda: arms[0].command_joint_position(targets[0], duration),
                lambda: arms[1].command_joint_position(targets[1], duration),
            )
        else:
            arms[0].command_joint_position(targets[0], duration)

    def _hold_after_reset_timeout(self) -> None:
        """Replace unfinished position targets and confirm six-joint rest."""
        assert self._arms is not None
        positions = [arm.read_joint_positions() for arm in self._arms]
        for side, position in zip(("left", "right"), positions):
            self.validate_joint_step(side, position, duration=1.0)
        self._run_arm_calls(
            lambda: self._arms[0].command_joint_position(positions[0], 1.0),
            lambda: self._arms[1].command_joint_position(positions[1], 1.0),
        )
        monitors = [ARX5RestMonitor(position) for position in positions]
        started = time.monotonic()
        while True:
            self.check_stop_requested()
            settled = [
                monitor.observe(arm.read_joint_positions(), time.monotonic())
                for arm, monitor in zip(self._arms, monitors)
            ]
            if all(settled) and time.monotonic() - started >= 1.0:
                return
            if time.monotonic() - started >= 5.0:
                raise RuntimeError("ARX5 reset hold did not settle within 5 s.")
            time.sleep(0.05)

    def get_action_scale(self) -> np.ndarray:
        return np.ones(3)

    def _run_arm_calls(
        self, left: Callable[[], Any], right: Callable[[], Any]
    ) -> tuple[Any, Any]:
        """Run one ordered call per arm concurrently and surface both failures."""
        assert self._arm_executors is not None
        futures = (
            self._arm_executors[0].submit(left),
            self._arm_executors[1].submit(right),
        )
        wait(futures)
        return futures[0].result(), futures[1].result()

    def _read_frames(self) -> dict[str, np.ndarray]:
        """Read and transform camera frames, closing on capture timeout."""
        if not self._cameras:
            return {}
        with ThreadPoolExecutor(max_workers=len(self._cameras)) as executor:
            readings = {}
            for name, camera in self._cameras.items():
                first_read = name not in self._camera_attempted
                self._camera_attempted.add(name)
                readings[name] = executor.submit(
                    camera.get_observation,
                    timeout=self.config.initial_camera_timeout_s if first_read else 0.0,
                    max_age=0.2,
                )
        frames = {}
        for name, future in readings.items():
            try:
                bgr = future.result()["frame"]
            except queue.Empty as exc:
                self._stop_requested = True
                self.close()
                raise ARX5StopRequested(f"Camera {name!r} timed out.") from exc
            if (
                not isinstance(bgr, np.ndarray)
                or bgr.ndim != 3
                or bgr.shape[2] != 3
                or min(bgr.shape[:2]) == 0
                or bgr.dtype != np.uint8
            ):
                raise ValueError("Camera frame must be a nonempty uint8 BGR image.")
            height, width = bgr.shape[:2]
            side = min(height, width)
            cropped = bgr[
                (height - side) // 2 : (height + side) // 2,
                (width - side) // 2 : (width + side) // 2,
            ]
            frame = cv2.resize(cropped, (self.config.image_size,) * 2)[..., ::-1].copy()
            degrees = self.hardware.camera_rotations.get(name, 0)
            if degrees:
                frame = np.rot90(frame, k=-(degrees // 90)).copy()
            frames[name] = frame
        return frames

    def _observe(self) -> dict[str, Any]:
        if self.config.is_dummy:
            return self.observation_space.sample()
        assert self._arms is not None
        states = self._run_arm_calls(self._arms[0].get_state, self._arms[1].get_state)
        proprio = np.empty(18, dtype=np.float32)
        grippers = np.empty(2, dtype=np.float32)
        for index, state in enumerate(states):
            proprio[index * 9 : index * 9 + 3] = state.tcp_pose[:3]
            proprio[index * 9 + 3 : index * 9 + 9] = matrix_to_rot6d(
                Rotation.from_quat(state.tcp_pose[3:]).as_matrix()
            )
            span = self._gripper_closed - self._gripper_open
            grippers[index] = np.clip(
                1 - 2 * (state.gripper_position - self._gripper_open) / span, -1, 1
            )
        return {
            "state": {"tcp_pose_rot6d": proprio, "gripper_position": grippers},
            "frames": self._read_frames(),
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        self.check_stop_requested()
        seed_sampled_spaces(seed, self.observation_space)
        self._steps = 0
        reset_info: dict[str, Any] = {"arx5_reset_degraded": False}
        if self._arms:
            targets = (
                self.config.joint_reset_qpos
                if self.config.joint_reset_qpos is not None
                else np.array([[0, 0.1, 0.1, 0, 0, 0]] * 2)
            )
            try:
                self._move_to_joint_positions(
                    dict(zip(("left", "right"), targets)),
                    timeout_s=self.config.reset_joint_timeout_s,
                )
            except ARX5JointReturnTimeout:
                self._hold_after_reset_timeout()
                measured = np.stack([arm.read_joint_positions() for arm in self._arms])
                max_error_deg = float(np.rad2deg(np.max(np.abs(measured - targets))))
                reset_info.update(
                    arx5_reset_degraded=True,
                    arx5_reset_max_error_deg=max_error_deg,
                )
                if self.config.enforce_workspace_bounds:
                    xyz = self.get_tcp_pose().reshape(2, 7)[:, :3]
                    if np.any(xyz < self.config.ee_pose_limit_min) or np.any(
                        xyz > self.config.ee_pose_limit_max
                    ):
                        raise ValueError("ARX5 reset pose is outside the workspace.")
                self._logger.warning(
                    "Reset return stopped; arms are holding their measured "
                    "positions. Continuing from actual feedback; maximum "
                    "return error %.3f degrees; target %s, measured %s degrees.",
                    max_error_deg,
                    np.rad2deg(targets).round(3).tolist(),
                    np.rad2deg(measured).round(3).tolist(),
                )
        observation = self._observe()
        return observation, reset_info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        start = time.monotonic()
        self.check_stop_requested()
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (20,) or not np.all(np.isfinite(values)):
            raise ValueError("DualARX5 action must be 20 finite values.")
        values = np.clip(values, self.action_space.low, self.action_space.high).reshape(
            2, 10
        )
        if self._arms:
            # Validate both states and targets before either arm is commanded.
            states = self._run_arm_calls(
                self._arms[0].get_state, self._arms[1].get_state
            )
            targets = []
            for index, state in enumerate(states):
                current = state.tcp_pose
                target = None
                # An all-zero absolute pose is the collector's idle action.
                # Never turn it into a motion toward the robot origin.
                if np.linalg.norm(values[index, 3:9]) > 1e-8:
                    if self.config.enforce_workspace_bounds and (
                        np.any(current[:3] < self.config.ee_pose_limit_min[index])
                        or np.any(current[:3] > self.config.ee_pose_limit_max[index])
                    ):
                        raise ValueError(
                            f"ARX5 arm {index} is outside the configured workspace; "
                            "correct the bounds or reposition it before control."
                        )
                    delta = values[index, :3] - current[:3]
                    distance = np.linalg.norm(delta)
                    if self.config.max_step > 0 and distance > self.config.max_step:
                        delta *= self.config.max_step / distance
                    xyz = current[:3] + delta
                    current_rot = Rotation.from_quat(current[3:])
                    quat = rot6d_to_quat_xyzw_safe(
                        values[index, 3:9], fallback_quat_xyzw=current_rot.as_quat()
                    )
                    target_rot = Rotation.from_quat(quat)
                    relative = target_rot * current_rot.inv()
                    angle = np.linalg.norm(relative.as_rotvec())
                    if (
                        self.config.max_step_rad > 0
                        and angle > self.config.max_step_rad
                    ):
                        target_rot = (
                            Rotation.from_rotvec(
                                relative.as_rotvec() * self.config.max_step_rad / angle
                            )
                            * current_rot
                        )
                    # The SDK may reject IK for its own measured pose at joint zero.
                    if (
                        distance >= _TCP_HOLD_TRANSLATION_TOLERANCE_M
                        or angle >= _TCP_HOLD_ROTATION_TOLERANCE_RAD
                    ):
                        target = np.r_[xyz, target_rot.as_quat()]
                targets.append(target)

            def command_arm(index: int) -> None:
                self.check_stop_requested()
                arm = self._arms[index]
                side = "left" if index == 0 else "right"
                target = targets[index]
                if target is not None:
                    try:
                        arm.send_action({"tcp_pose": target})
                    except ARX5IKUnreachableError as exc:
                        self._logger.warning("ARX5 %s: no IK solution; %s", side, exc)
                # A zero teleop gripper channel means hold the last target.
                if abs(values[index, 9]) >= 0.5:
                    gripper_target = (
                        self._gripper_open
                        if values[index, 9] > 0
                        else self._gripper_closed
                    )
                    arm.command_gripper(np.array([gripper_target]))

            self._run_arm_calls(lambda: command_arm(0), lambda: command_arm(1))
        self._steps += 1
        observation = self._observe()
        remaining = 1 / self.config.step_frequency - (time.monotonic() - start)
        if remaining > 0 and self._arms:
            time.sleep(remaining)
        return (
            observation,
            0.0,
            False,
            self._steps >= self.config.max_num_steps,
            {},
        )

    def close(self) -> None:
        if self._closed:
            self._restore_signal_handlers()
            return
        self._return_to_zero_positions()
        if self._arm_executors is not None:
            for executor in self._arm_executors:
                executor.shutdown(wait=True)
            self._arm_executors = None
        if self.robot is not None:
            self.robot.disconnect()
            self.robot = None
        self._arms = None
        self._cameras = {}
        self._camera_attempted.clear()
        self._restore_signal_handlers()
        self._closed = True

    def _restore_signal_handlers(self) -> None:
        if threading.current_thread() is threading.main_thread():
            for signum, handler in self._signal_handlers.items():
                signal.signal(signum, handler)
            self._signal_handlers.clear()
