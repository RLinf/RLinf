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

from ...actions import ActionKind
from ..base import Features, Observation
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
        movement_epsilon: Radians of leader motion, summed over the joints,
            below which the operator counts as not driving.
        align_duration_s: Seconds spent moving the leader to the follower pose
            before manual control starts.
        align_fps: Command frequency used while aligning the leader.
        reset_hold_seconds: Seconds to hold the reset pose before releasing the
            leader to the operator.
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

    #: Joint limits and the gripper range both come from the action space.
    CLIPS_TO_ACTION_SPACE = True

    MOVEMENT_EPSILON = 0.01

    def __init__(
        self,
        port: str,
        calibration_id: Optional[str] = None,
        movement_epsilon: float = 0.01,
        align_duration_s: float = 3.0,
        align_fps: float = 30.0,
        reset_hold_seconds: float = 3.0,
        calibrate: bool = False,
    ) -> None:
        if not np.isfinite(align_duration_s) or align_duration_s < 0:
            raise ValueError("SO-101 align_duration_s must be finite and nonnegative")
        if not np.isfinite(align_fps) or align_fps <= 0:
            raise ValueError("SO-101 align_fps must be finite and positive")
        if not np.isfinite(reset_hold_seconds) or reset_hold_seconds < 0:
            raise ValueError("SO-101 reset_hold_seconds must be finite and nonnegative")
        self._port = port
        self._calibration_id = calibration_id
        self.MOVEMENT_EPSILON = movement_epsilon
        self._align_duration_s = float(align_duration_s)
        self._align_fps = float(align_fps)
        self._reset_hold_seconds = float(reset_hold_seconds)
        self._calibrate = calibrate
        self._serial_lock = threading.RLock()
        self._reset_prepared = False

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
        return TeleopEntry(
            cls(
                port=port,
                calibration_id=options.get("calibration_id")
                or cfg.get("so101_leader_id"),
                movement_epsilon=float(options.get("movement_epsilon", 0.01)),
                align_duration_s=float(options.get("align_duration_s", 3.0)),
                align_fps=float(options.get("align_fps", 30.0)),
                reset_hold_seconds=float(options.get("reset_hold_seconds", 3.0)),
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
        leader.connect(calibrate=False)
        if not leader.is_calibrated:
            if not self._calibrate:
                where = leader.calibration_fpath
                leader.disconnect()
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
        return leader

    def _release(self, device: Any) -> None:
        """lerobot spells this ``disconnect``, which the base does not try."""
        with self._serial_lock:
            if self._reset_prepared:
                device.bus.disable_torque()
                self._reset_prepared = False
            device.disconnect()

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
        return {
            "joint_position": np.asarray(joints, dtype=np.float32),
            "grip": np.asarray([grip], dtype=np.float32),
        }

    def prepare_intervention(self, context: Mapping[str, Any]) -> None:
        """Align the leader to the stopped follower and keep torque enabled."""
        joints = np.asarray(context["joint_positions"], dtype=float).reshape(-1)
        grip = np.asarray(context["gripper_position"], dtype=float).reshape(-1)
        self._move_to(joints, grip, self._align_duration_s)

    def prepare_reset(self, context: Mapping[str, Any]) -> None:
        """Move to the reset state while the follower executes its reset."""
        joints = np.asarray(context["reset_joint_positions"], dtype=float).reshape(-1)
        grip = np.asarray(context["reset_gripper_position"], dtype=float).reshape(-1)
        duration = float(context["reset_duration"])
        max_joint_speed = float(context["reset_joint_speed"])
        self._move_to(joints, grip, duration, max_joint_speed=max_joint_speed)
        self._reset_prepared = True

    def on_reset(self, context: Mapping[str, Any]) -> None:
        """Hold the reset state briefly, then release the leader."""
        if not self._reset_prepared:
            return
        try:
            time.sleep(self._reset_hold_seconds)
        finally:
            with self._serial_lock:
                self._device.bus.disable_torque()
                self._reset_prepared = False

    def abort_reset(self, context: Mapping[str, Any]) -> None:
        """Release the leader after an incomplete reset."""
        if not self._reset_prepared:
            return
        with self._serial_lock:
            self._device.bus.disable_torque()
            self._reset_prepared = False

    def _move_to(
        self,
        joints: np.ndarray,
        grip: np.ndarray,
        minimum_duration: float,
        *,
        max_joint_speed: Optional[float] = None,
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
        target = np.concatenate((np.rad2deg(joints), grip * GRIPPER_SCALE))
        names = (*MOTORS, GRIPPER)
        with self._serial_lock:
            reading = self._device.get_action()
            current = np.asarray(
                [reading[f"{name}.pos"] for name in names], dtype=float
            )
            duration = minimum_duration
            if max_joint_speed is not None:
                joint_distance = float(
                    np.max(np.abs(joints - np.deg2rad(current[: len(MOTORS)])))
                )
                duration = max(duration, 1.875 * joint_distance / max_joint_speed)
            steps = max(1, int(np.ceil(duration * self._align_fps)))
            period = duration / steps
            # Set a current-pose goal before enabling torque; a previous
            # intervention may have left an obsolete target in the servos.
            self._device.bus.sync_write(
                "Goal_Position", dict(zip(names, current.tolist(), strict=True))
            )
            torque_enabled = False
            try:
                self._device.bus.enable_torque()
                torque_enabled = True
                deadline = time.monotonic()
                for index in range(1, steps + 1):
                    ratio = index / steps
                    alpha = ratio**3 * (10.0 + ratio * (-15.0 + 6.0 * ratio))
                    values = (1.0 - alpha) * current + alpha * target
                    self._device.bus.sync_write(
                        "Goal_Position",
                        dict(zip(names, values.tolist(), strict=True)),
                    )
                    deadline += period
                    if period:
                        time.sleep(max(0.0, deadline - time.monotonic()))
            except BaseException:
                if torque_enabled:
                    self._device.bus.disable_torque()
                raise

    def on_intervention_start(self, context: Mapping[str, Any]) -> None:
        """Release torque only after the operator has had time to grasp the arm."""
        with self._serial_lock:
            self._device.bus.disable_torque()

    def on_intervention_end(self, context: Mapping[str, Any]) -> None:
        """Hold the leader where the operator returned policy control."""
        names = (*MOTORS, GRIPPER)
        with self._serial_lock:
            reading = self._device.get_action()
            current = {name: float(reading[f"{name}.pos"]) for name in names}
            self._device.bus.sync_write("Goal_Position", current)
            self._device.bus.enable_torque()

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
        # Idle until the operator actually moves, so the policy keeps control
        # while the leader is just resting in its holder.
        moved = (
            float(np.linalg.norm(target - current)) > self.MOVEMENT_EPSILON
            or float(np.linalg.norm(grip - current_grip)) > self.MOVEMENT_EPSILON
        )
        return TeleopAction(parts={"arm": target, "end_effector": grip}, driving=moved)

    def hold(self, context: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Return current follower pose as hold action for smooth intervene.

        Used during explicit-mode intervention buffer periods to keep the robot
        stationary while the operator grabs or releases the leader arm.
        """
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
    # No follower to read, so the arm is measured against where it last was.
    # That is the same comparison action() makes, and it is what decides
    # whether the operator has taken control.
    previous_observation = leader.get_observation()
    previous = previous_observation["joint_position"]
    previous_gripper = previous_observation["grip"]
    try:
        with np.printoptions(precision=3, suppress=True):
            while True:
                observation = leader.get_observation()
                action = leader.action(
                    observation,
                    {
                        "joint_positions": previous[None, :],
                        "gripper_position": previous_gripper[None, :],
                    },
                )
                arm = action.parts["arm"]
                grip = float(action.parts["end_effector"][0])
                print(
                    f"joints={np.rad2deg(arm)} deg  grip={grip:.2f}  "
                    f"driving={action.driving}   ",
                    end="\r",
                )
                previous = arm
                previous_gripper = observation["grip"]
                time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        leader.disconnect()
