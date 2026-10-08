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

"""SO-101 leader arm driving an SO-101 follower, joint for joint."""

import threading
import time
from typing import Any, Mapping, Optional

import numpy as np

from rlinf.utils.logging import get_logger

from ...actions import ActionKind
from ..base import Features, Observation
from ..so101_motion import plan_so101_motion
from .base import TeleopAction, TeleopDevice

#: Arm joints in bus order, matching the follower's.
MOTORS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)

#: The sixth servo, reported on lerobot's own 0..100 gripper scale.
GRIPPER = "gripper"
GRIPPER_SCALE = 100.0


@TeleopDevice.register("so101_leader")
class SO101Leader(TeleopDevice):
    """SO-101 leader arm, reporting joint angles and grip.

    The leader is the same five joints and gripper as the follower, held
    rather than driven, so the operator's pose is the follower's target
    directly. Readings arrive in the follower's own units -- radians and a
    ``0..1`` grip -- so neither half has to convert.

    Args:
        port: Serial device the leader's servo bus is on.
        calibration_id: lerobot calibration identifier for this leader.
        align_fps: Command frequency used while aligning the leader.
        manual_start_hold_seconds: Seconds to hold the reset pose before
            releasing the leader to the operator.
        manual_control_default: Whether direct teleoperation owns control after
            reset when no keyboard collection handover is configured.
        calibrate: Whether to run lerobot's calibration when the arm has none.
            It asks the operator to move the arm through its range, so only a
            caller holding a terminal should turn it on.
    """

    SDK = "scservo_sdk"

    PRODUCES = {
        "arm": ActionKind.JOINT_POSITION,
        "end_effector": ActionKind.GRIPPER,
    }

    NEEDS = ("joint_positions", "gripper_position")

    # The follower supplies these values while the leader moves in the reset
    # thread.  Declaring them here keeps the synchronization protocol out of
    # the common composed wrapper.
    RESET_CONTEXT_KEYS = (
        "reset_joint_positions",
        "reset_gripper_position",
        "reset_duration",
        "reset_joint_speed",
    )
    PARK_CONTEXT_KEYS = (
        "park_joint_positions",
        "park_gripper_position",
        "park_duration",
    )

    #: Joint limits and the gripper range both come from the action space.
    CLIPS_TO_ACTION_SPACE = True
    #: The follower pose remains an explicit action while the leader is idle.
    APPLIES_WHILE_IDLE = True

    SERIAL_CONNECT_RETRIES = 3
    SERIAL_RETRY_DELAY_S = 0.1
    TORQUE_WRITE_RETRIES = 3
    TORQUE_RETRY_DELAY_S = 0.05

    def __init__(
        self,
        port: str,
        calibration_id: Optional[str] = None,
        align_fps: float = 30.0,
        manual_start_hold_seconds: float = 3.0,
        calibrate: bool = False,
        manual_control_default: bool = False,
    ) -> None:
        if not np.isfinite(align_fps) or align_fps <= 0:
            raise ValueError("SO-101 align_fps must be finite and positive")
        if not np.isfinite(manual_start_hold_seconds) or manual_start_hold_seconds < 0:
            raise ValueError(
                "SO-101 manual_start_hold_seconds must be finite and nonnegative"
            )
        self._port = port
        self._logger = get_logger()
        self._calibration_id = calibration_id
        self._align_fps = float(align_fps)
        self._manual_start_hold_seconds = float(manual_start_hold_seconds)
        self._calibrate = calibrate
        self._manual_control_default = bool(manual_control_default)
        self._serial_lock = threading.RLock()
        self._torque_enabled = False
        self._reset_prepared = False
        self._manual_control_enabled = False
        self._last_valid_observation: Observation | None = None

    @classmethod
    def from_config(
        cls, cfg: Mapping[str, Any], options: Mapping[str, Any], facts: Any
    ) -> Any:
        """Take the port and calibration id from options or the env config.

        Calibration stays off for a configured device. It waits on stdin, and
        a scheduler worker has no terminal to answer it from; the standalone
        entry point is where an operator calibrates an arm.
        """
        from .group import TeleopEntry

        port = options.get("port") or cfg.get("so101_leader_port")
        if port is None:
            raise ValueError(
                "teleop device 'so101_leader' requires a 'port', or "
                "'so101_leader_port' in the env config."
            )
        manual_start_hold_seconds = options.get("manual_start_hold_seconds", 3.0)
        return TeleopEntry(
            cls(
                port=port,
                calibration_id=options.get("calibration_id")
                or cfg.get("so101_leader_id"),
                align_fps=float(options.get("align_fps", 30.0)),
                manual_start_hold_seconds=float(manual_start_hold_seconds),
                manual_control_default=cfg.get("keyboard_reward_wrapper")
                != "start_end",
            ),
            drives=options.get("drives"),
        )

    # Hardware.

    def _open(self) -> Any:
        """Open the leader's servo bus and return lerobot's handle for it.

        Calibration is not run unless the caller asked for it. lerobot's
        procedure asks the operator to move the arm through its range and
        waits on stdin, which would hang a worker that has no terminal.
        """
        self._last_valid_observation = None
        try:
            from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
        except ImportError:  # pragma: no cover - older lerobot
            from lerobot.teleoperators.so101_leader import (
                SO101Leader,
                SO101LeaderConfig,
            )

        leader = SO101Leader(
            SO101LeaderConfig(
                port=self._port, id=self._calibration_id, use_degrees=True
            )
        )
        accepted = False
        try:
            for attempt in range(1, self.SERIAL_CONNECT_RETRIES + 1):
                try:
                    leader.connect(calibrate=False)
                    break
                except ConnectionError:
                    if attempt == self.SERIAL_CONNECT_RETRIES:
                        raise
                    self._logger.warning(
                        "SO-101 leader connection did not receive a status packet; "
                        "retrying (%d/%d)",
                        attempt + 1,
                        self.SERIAL_CONNECT_RETRIES,
                    )
                    try:
                        leader.disconnect()
                    except Exception:  # noqa: BLE001 - retry the original connection
                        pass
                    time.sleep(self.SERIAL_RETRY_DELAY_S)
            if not leader.is_calibrated:
                if not self._calibrate:
                    where = leader.calibration_fpath
                    raise RuntimeError(
                        f"The SO-101 leader on {self._port!r} has no calibration at "
                        f"{where}. Calibrating asks the operator to move the arm "
                        "through its range, so it does not run on its own here. "
                        "Calibrate it once from a terminal with:\n\n"
                        "    python -m rlinf.robotics.parts.teleop.so101_leader "
                        f"--port {self._port} "
                        f"--id {self._calibration_id or '<name-for-this-arm>'} "
                        "--calibrate"
                    )
                leader.calibrate()
            accepted = True
            return leader
        finally:
            if not accepted:
                try:
                    leader.disconnect()
                except Exception:  # noqa: BLE001 - preserve startup failure
                    pass

    def _release(self, device: Any) -> None:
        """lerobot spells this ``disconnect``, which the base does not try."""
        with self._serial_lock:
            self._manual_control_enabled = False
            if self._reset_prepared or self._torque_enabled:
                self._set_torque(False)
                self._reset_prepared = False
            try:
                device.disconnect()
            finally:
                self._last_valid_observation = None

    @property
    def observation_features(self) -> Features:
        """The operator's joint angles, and how far the trigger is squeezed."""
        return {
            "joint_position": {"shape": (5,), "dtype": "float32"},
            "grip": {"shape": (1,), "dtype": "float32"},
        }

    def get_observation(self) -> Observation:
        """Read the leader the operator is holding.

        lerobot reports degrees and a ``0..100`` gripper; the follower speaks
        radians and ``0..1``, so the conversion happens here rather than
        leaving both units loose in the action.
        """
        with self._serial_lock:
            reading = self._device.get_action()
        joints = np.deg2rad([reading[f"{motor}.pos"] for motor in MOTORS])
        grip = np.clip(reading[f"{GRIPPER}.pos"] / GRIPPER_SCALE, 0.0, 1.0)
        observation = {
            "joint_position": np.asarray(joints, dtype=np.float32),
            "grip": np.asarray([grip], dtype=np.float32),
        }
        values = np.concatenate((observation["joint_position"], observation["grip"]))
        if not np.all(np.isfinite(values)):
            if self._last_valid_observation is None:
                raise RuntimeError("SO-101 leader returned an invalid initial reading")
            self._logger.warning(
                "SO-101 leader returned an invalid reading; holding the last valid pose"
            )
            return self._last_valid_observation
        self._last_valid_observation = observation
        return observation

    def prepare_reset(self, context: Mapping[str, Any]) -> None:
        """Move to the reset state while the follower executes its reset."""
        self._manual_control_enabled = False
        joints = np.asarray(context["reset_joint_positions"], dtype=float).reshape(-1)
        grip = np.asarray(context["reset_gripper_position"], dtype=float).reshape(-1)
        duration = float(context["reset_duration"])
        max_joint_speed = float(context["reset_joint_speed"])
        self._move_to(
            joints,
            grip,
            duration,
            max_joint_speed=max_joint_speed,
            start_at=context.get("reset_start_at"),
        )
        self._reset_prepared = True

    def on_reset(self, context: Mapping[str, Any]) -> None:
        """Keep the leader at the reset pose until manual handover."""
        self._manual_control_enabled = self._manual_control_default
        if not self._reset_prepared:
            return

    @property
    def manual_start_hold_seconds(self) -> float:
        """Return the operator handover buffer configured for this leader."""
        return self._manual_start_hold_seconds

    def release_for_manual(self, context: Mapping[str, Any]) -> None:
        """Disable leader torque after the operator handover countdown."""
        del context
        with self._serial_lock:
            self._set_torque(False)
            self._reset_prepared = False
            self._manual_control_enabled = True

    def hold_for_reset(self, context: Mapping[str, Any]) -> None:
        """Hold the current leader pose before the robot is reset or parked."""
        del context
        self._manual_control_enabled = False
        if self._reset_prepared:
            return
        names = (*MOTORS, GRIPPER)
        with self._serial_lock:
            reading = self._device.get_action()
            current = {name: float(reading[f"{name}.pos"]) for name in names}
            self._device.bus.sync_write("Goal_Position", current)
            self._set_torque(True)
            self._reset_prepared = True

    def park(self, context: Mapping[str, Any]) -> None:
        """Move the leader to the follower's configured park target.

        Environments without a park target intentionally leave the leader
        untouched; the base device contract defines parking as a no-op there.
        """
        joints = context.get("park_joint_positions")
        grip = context.get("park_gripper_position")
        duration = context.get("park_duration")
        speed = context.get("reset_joint_speed")
        if joints is None or grip is None or duration is None:
            return
        self._manual_control_enabled = False
        self._move_to(
            np.asarray(joints, dtype=float).reshape(-1),
            np.asarray(grip, dtype=float).reshape(-1),
            float(duration),
            max_joint_speed=None if speed is None else float(speed),
        )
        self._reset_prepared = True

    def abort_reset(self, context: Mapping[str, Any]) -> None:
        """Release the leader after an incomplete reset."""
        self._manual_control_enabled = False
        if not self._reset_prepared:
            return
        with self._serial_lock:
            self._set_torque(False)
            self._reset_prepared = False

    def _move_to(
        self,
        joints: np.ndarray,
        grip: np.ndarray,
        minimum_duration: float,
        *,
        max_joint_speed: Optional[float] = None,
        start_at: Optional[float] = None,
    ) -> None:
        """Move every leader servo smoothly and leave torque enabled."""
        if joints.shape != (len(MOTORS),) or grip.shape != (1,):
            raise ValueError(
                "SO-101 movement requires five joints and one gripper "
                f"position, got {joints.shape} and {grip.shape}."
            )
        if not np.all(np.isfinite(joints)) or not np.all(np.isfinite(grip)):
            raise ValueError("SO-101 movement targets must be finite.")
        if not np.isfinite(minimum_duration) or minimum_duration < 0:
            raise ValueError("SO-101 movement duration must be finite and nonnegative.")
        if max_joint_speed is not None and (
            not np.isfinite(max_joint_speed) or max_joint_speed <= 0
        ):
            raise ValueError("SO-101 movement speed must be finite and positive.")
        target = np.concatenate((joints, grip))
        names = (*MOTORS, GRIPPER)
        with self._serial_lock:
            reading = self._device.get_action()
            current = np.asarray(
                [reading[f"{name}.pos"] for name in names], dtype=float
            )
            current = np.concatenate(
                (np.deg2rad(current[: len(MOTORS)]), [current[-1] / GRIPPER_SCALE])
            )
            plan = plan_so101_motion(
                current,
                target,
                minimum_duration=minimum_duration,
                max_joint_speed=max_joint_speed,
                fps=self._align_fps,
            )
            # Set a current-pose goal before enabling torque so the arm does
            # not jump toward a stale servo target.
            self._device.bus.sync_write(
                "Goal_Position",
                dict(
                    zip(
                        names,
                        np.concatenate(
                            (np.rad2deg(current[: len(MOTORS)]), [current[-1] * GRIPPER_SCALE])
                        ).tolist(),
                        strict=True,
                    )
                ),
            )
            torque_enabled = False
            try:
                self._set_torque(True)
                torque_enabled = True
                if start_at is not None:
                    time.sleep(max(0.0, start_at - time.monotonic()))
                deadline = time.monotonic()
                for index, frame in enumerate(plan.frames(), start=1):
                    values = np.concatenate(
                        (
                            np.rad2deg(frame[: len(MOTORS)]),
                            [frame[-1] * GRIPPER_SCALE],
                        )
                    )
                    self._device.bus.sync_write(
                        "Goal_Position",
                        dict(zip(names, values.tolist(), strict=True)),
                    )
                    deadline += plan.period
                    if index < plan.steps:
                        time.sleep(max(0.0, deadline - time.monotonic()))
            except BaseException:
                if torque_enabled:
                    self._set_torque(False)
                raise

    def _set_torque(self, enabled: bool) -> None:
        """Set leader torque with a short retry for transient bus timeouts."""
        operation = (
            self._device.bus.enable_torque
            if enabled
            else self._device.bus.disable_torque
        )
        action = "enable" if enabled else "disable"
        for attempt in range(1, self.TORQUE_WRITE_RETRIES + 1):
            try:
                operation()
                self._torque_enabled = enabled
                return
            except ConnectionError:
                if attempt == self.TORQUE_WRITE_RETRIES:
                    raise
                self._logger.warning(
                    "SO-101 leader torque %s did not receive a status packet; "
                    "retrying (%d/%d)",
                    action,
                    attempt + 1,
                    self.TORQUE_WRITE_RETRIES,
                )
                time.sleep(self.TORQUE_RETRY_DELAY_S)

    # Driving the robot.

    def action(
        self, reading: Mapping[str, Any], context: Mapping[str, Any]
    ) -> TeleopAction:
        """Take the leader's pose as the follower's target.

        The grip stays on the ``0..1`` axis the SO-101 environment opens over,
        rather than the signed axis a GELLO reports, so neither half clips.
        """
        target = np.asarray(reading["joint_position"], dtype=float)
        current = np.asarray(context["joint_positions"])[0]
        grip = np.asarray(reading["grip"], dtype=float).reshape(1)
        current_grip = np.asarray(context["gripper_position"], dtype=float).reshape(-1)
        if not self._manual_control_enabled:
            target = current
            grip = current_grip
        return TeleopAction(
            parts={"arm": target, "end_effector": grip},
            driving=self._manual_control_enabled,
        )

    def hold(self, context: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Return the current follower pose while collection is idle."""
        joints = np.asarray(context["joint_positions"], dtype=np.float32)
        if joints.ndim == 2:
            joints = joints[0]
        if joints.shape != (5,):
            raise ValueError(f"Expected 5 joints for SO-101, got shape {joints.shape}")
        grip = np.asarray(context["gripper_position"], dtype=np.float32).reshape(-1)
        if grip.shape != (1,):
            raise ValueError(
                f"Expected one gripper position for SO-101, got shape {grip.shape}"
            )
        return {"arm": joints, "end_effector": grip}


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Read an SO-101 leader arm.")
    parser.add_argument(
        "--port", type=str, required=True, help="Serial port of the leader arm."
    )
    parser.add_argument(
        "--id", type=str, default=None, help="lerobot calibration id of the leader."
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Calibrate the arm first. It asks you to move it through its range.",
    )
    args = parser.parse_args()

    # This is a terminal, so it can answer the prompts calibration asks.
    leader = SO101Leader(
        port=args.port, calibration_id=args.id, calibrate=args.calibrate
    )
    leader.connect()
    try:
        with np.printoptions(precision=3, suppress=True):
            while True:
                observation = leader.get_observation()
                print(
                    f"joints={np.rad2deg(observation['joint_position'])} deg  "
                    f"grip={float(observation['grip'][0]):.2f}   ",
                    end="\r",
                )
                time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        leader.disconnect()
