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

"""SO-101 follower arm, driven through lerobot's ``SO101Follower``.

The SO-101 is a 5-DOF arm with a parallel gripper, all six joints driven by
Feetech STS3215 serial servos on one bus. lerobot owns the servo protocol and
the calibration file; this module adapts it to the arm contract.

Two things differ from the lerobot API and are converted here:

* lerobot reports and accepts joint values in degrees, and the gripper on its
  own ``0..100`` scale. Every other RLinf arm reports ``arm_joint_position`` in
  radians, so joints are converted on the way out and back on the way in, and
  the gripper is carried as a fraction in ``0..1``.
* lerobot names each value ``"<motor>.pos"``; the canonical observation is one
  vector ordered by :pyattr:`SO101Arm.MOTORS`.
"""

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np

from rlinf.robotics.parts.base import Action, Features, Observation, RobotPart
from rlinf.robotics.parts.views import MethodEndEffector
from rlinf.utils.logging import get_logger

from .base import Arm, BaseArm

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lerobot.robots.so_follower import SO101Follower


@dataclass
class SO101RobotState:
    """State snapshot for the SO-101 follower arm."""

    arm_joint_position: np.ndarray = field(default_factory=lambda: np.zeros(5))
    """Arm joint positions ``[shoulder_pan, shoulder_lift, elbow_flex,
    wrist_flex, wrist_roll]`` in radians."""

    gripper_position: np.ndarray = field(default_factory=lambda: np.zeros(1))
    """Gripper opening as a fraction, ``0.0`` closed to ``1.0`` open."""

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass to a serializable dictionary."""
        return asdict(self)


@Arm.register("so101")
class SO101Arm(BaseArm):
    """SO-101 follower arm on a Feetech serial bus, via lerobot.

    Args:
        port: Serial device the servo bus is on, such as ``/dev/ttyACM0``.
        calibration_id: lerobot calibration identifier. The calibration file it
            names must already exist: see :meth:`_open` for why.
        max_relative_target: Per-step joint limit in degrees that lerobot
            clamps commands to. ``None`` disables clamping.
        cameras: Cameras for lerobot to own. Leave empty and declare RLinf
            :class:`~rlinf.robotics.parts.cameras.Camera` parts instead, so
            they can be placed on their own node.
    """

    SDK = "scservo_sdk"

    #: Arm joints reported as ``arm_joint_position``, in bus order.
    MOTORS: tuple[str, ...] = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    )

    #: The gripper rides on the same bus and is exported as an end effector.
    GRIPPER: str = "gripper"

    #: lerobot's gripper scale. Its own normalisation, not a servo unit.
    GRIPPER_SCALE: float = 100.0

    #: Bus rate the STS3215 servos run at, matching lerobot's default.
    BAUDRATE: int = 1_000_000

    #: Target arrival and stall-release margin, on the gripper's 0..100 scale.
    GRIPPER_TOLERANCE: float = 2.0

    #: Maximum seconds without progress before relieving a pending target.
    GRIPPER_SETTLE_TIMEOUT: float = 3.0

    #: Consecutive still polls that mean the jaws have stopped rather than
    #: not yet started. One is not enough: a servo barely moves in the first
    #: poll after a command, and treating that as a stall freezes the jaws
    #: where they stand.
    GRIPPER_STALL_POLLS: int = 3

    #: The SO-101 reports joints only; it carries no pose or force sensing.
    STATE_FIELDS = ("arm_joint_position",)

    def __init__(
        self,
        port: str,
        *,
        calibration_id: Optional[str] = None,
        max_relative_target: Optional[int] = None,
        cameras: Optional[dict[str, Any]] = None,
    ) -> None:
        self._logger = get_logger()
        self._port = port
        self._calibration_id = calibration_id
        self._max_relative_target = max_relative_target
        self._cameras = dict(cameras or {})
        self._robot: "Optional[SO101Follower]" = None
        self._gripper_condition = threading.Condition(threading.RLock())
        self._gripper_stop = threading.Event()
        self._gripper_thread: Optional[threading.Thread] = None
        self._gripper_target: Optional[float] = None
        self._gripper_requested: Optional[float] = None
        self._gripper_limit: Optional[tuple[float, float]] = None
        self._gripper_error: Optional[Exception] = None

    @classmethod
    def declare(
        cls,
        address: str,
        *,
        gripper_type: Optional[str] = None,
        gripper_connection: Optional[str] = None,
        end_effector_type: Optional[str] = None,
        end_effector_config: Optional[dict] = None,
        **placement: Any,
    ) -> "SO101Arm":
        """Declare an SO-101 on the serial port named by ``address``.

        The gripper is one of the arm's own six servos, so it is neither
        chosen nor wired separately.
        """
        offered = {
            "gripper_type": gripper_type,
            "gripper_connection": gripper_connection,
            "end_effector_type": end_effector_type,
            "end_effector_config": end_effector_config,
        }
        named = sorted(name for name, value in offered.items() if value is not None)
        if named:
            raise TypeError(
                f"The SO-101 gripper is servo {cls.GRIPPER!r} on the arm's own "
                f"bus, so it cannot be fitted or wired separately: drop "
                f"{named} from the config."
            )
        settings = {
            key: placement.pop(key) for key in ("calibration_id",) if key in placement
        }
        return cls(address, **settings, **placement)

    @property
    def action_features(self) -> Features:
        """Describe the absolute joint target, in radians."""
        return {"joint_position": {}}

    @property
    def parts(self) -> dict[str, RobotPart]:
        """Return the gripper exported by this arm connection.

        The servo takes any opening in range, so the view drives it through
        :meth:`move_gripper` rather than falling back to open/close.
        """
        return {
            "end_effector": MethodEndEffector(
                self,
                state_field="gripper_position",
                command="move_gripper",
                is_gripper=True,
            )
        }

    def _open(self) -> "SO101Follower":
        """Open the servo bus through lerobot and return the follower.

        Calibration is not run here. lerobot's :meth:`calibrate` prompts on
        stdin when the servos disagree with the calibration file, which would
        hang a worker that has no terminal, so a missing calibration is
        reported instead. Run lerobot's own calibration once per arm first.
        """
        try:
            # lerobot 0.4 merged the SO-family followers into one module.
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        except ImportError:  # pragma: no cover - older lerobot
            from lerobot.robots.so101_follower import (
                SO101Follower,
                SO101FollowerConfig,
            )

        robot = SO101Follower(
            SO101FollowerConfig(
                port=self._port,
                id=self._calibration_id,
                cameras=self._cameras,
                max_relative_target=self._max_relative_target,
                # Ask for degrees so the conversion here is a plain factor
                # rather than lerobot's normalised -100..100 range.
                use_degrees=True,
            )
        )
        try:
            robot.connect(calibrate=False)
        except RuntimeError as error:
            faulted = self._faulted_motors()
            if not faulted:
                raise
            raise RuntimeError(
                f"The SO-101 on {self._port!r} cannot start: motor(s) "
                f"{faulted} report a latched fault, which lerobot reports as "
                "a missing motor. The gripper reaches this by being held "
                "shut against something until its overload protection trips. "
                "Power-cycle the arm's supply to clear it"
            ) from error
        if not robot.is_calibrated:
            robot.disconnect()
            raise RuntimeError(
                f"The SO-101 on {self._port!r} is not calibrated, and "
                "calibrating it asks the operator to move the arm, which "
                "cannot be done from here. Run lerobot's calibration for "
                f"id={self._calibration_id!r} once, then start again."
            )
        self._logger.info("SO-101 connected on %s", self._port)
        self._robot = robot
        return robot

    def _faulted_motors(self) -> dict[int, int]:
        """Return ``{motor id: error byte}`` for servos answering with a fault.

        lerobot drops a motor that replies with an error set, so its check
        cannot tell a faulted servo from an absent one. Reading the bus
        directly separates the two.
        """
        try:
            import scservo_sdk as scs
        except ImportError:  # pragma: no cover - lerobot ships this
            return {}

        port = scs.PortHandler(self._port)
        try:
            if not port.openPort():
                return {}
            port.setBaudRate(self.BAUDRATE)
            packets = scs.PacketHandler(0)
            faults = {}
            for motor_id in range(1, len(self.MOTORS) + 2):
                _, result, error = packets.ping(port, motor_id)
                if result == scs.COMM_SUCCESS and error:
                    faults[motor_id] = error
            return faults
        except Exception:  # noqa: BLE001 - a diagnostic must not mask the error
            return {}
        finally:
            port.closePort()

    def _opened(self) -> None:
        """Start gripper monitoring on this connection's node."""
        self._gripper_target = self._gripper_requested = None
        self._gripper_limit = None
        self._gripper_error = None
        self._gripper_stop.clear()
        self._gripper_thread = threading.Thread(
            target=self._watch_gripper, name="so101-gripper", daemon=True
        )
        self._gripper_thread.start()

    def _closing(self) -> None:
        """Stop the monitor before releasing its shared servo bus."""
        self._gripper_stop.set()
        if self._gripper_thread is not None:
            self._gripper_thread.join()
            self._gripper_thread = None
        with self._gripper_condition:
            self._gripper_target = None
            self._gripper_condition.notify_all()

    def _release(self, device: "SO101Follower") -> None:
        """Close the servo bus, letting the gripper go slack first.

        lerobot disables torque in motor order, so the gripper goes last and
        stays energised while the arm is released around it. Freeing it first
        means a session never ends with the jaws straining.
        """
        try:
            device.bus.disable_torque(self.GRIPPER)
        except Exception as error:  # noqa: BLE001 - the bus may already be gone
            self._logger.debug("Could not release the SO-101 gripper: %s", error)
        try:
            device.disconnect()
        finally:
            self._robot = None

    def get_state(self) -> SO101RobotState:
        """Read every servo and convert it to canonical units."""
        with self._gripper_condition:
            self._check_gripper_monitor()
            reading = self._robot.get_observation()
        joints = [reading[f"{motor}.pos"] for motor in self.MOTORS]
        grip = reading[f"{self.GRIPPER}.pos"] / self.GRIPPER_SCALE
        return SO101RobotState(
            arm_joint_position=np.deg2rad(np.asarray(joints, dtype=float)),
            gripper_position=np.asarray([grip], dtype=float),
        )

    def send_action(self, action: Action) -> Observation:
        """Move the arm joints to an absolute target in radians."""
        if set(action) != {"joint_position"}:
            raise KeyError(
                "An SO-101 arm action holds only 'joint_position'; the "
                "gripper is commanded through its own end-effector part."
            )
        sent = self.move_joints(action["joint_position"])
        return {"joint_position": sent}

    def move_joints(self, q_target: "Sequence[float]") -> np.ndarray:
        """Command absolute joint positions in radians.

        Returns:
            The target lerobot actually sent, in radians. It differs from the
            request when ``max_relative_target`` clamps the step.
        """
        target = np.asarray(q_target, dtype=float).reshape(-1)
        if target.shape != (len(self.MOTORS),):
            raise ValueError(
                f"Expected {len(self.MOTORS)} joint targets for an SO-101, "
                f"got shape {target.shape}."
            )
        degrees = np.rad2deg(target)
        with self._gripper_condition:
            self._check_gripper_monitor()
            sent = self._robot.send_action(
                {
                    f"{motor}.pos": float(value)
                    for motor, value in zip(self.MOTORS, degrees)
                }
            )
        return np.deg2rad(
            np.asarray([sent[f"{motor}.pos"] for motor in self.MOTORS], dtype=float)
        )

    def move_gripper(self, target: "Sequence[float]") -> None:
        """Command an opening fraction in ``0..1`` without waiting for travel.

        A background monitor relieves stalled jaws. After a stall, command
        away from the obstruction before trying to move through it again.
        """
        value = float(np.asarray(target, dtype=float).reshape(-1)[0])
        opening = float(np.clip(value, 0.0, 1.0)) * self.GRIPPER_SCALE
        with self._gripper_condition:
            self._check_gripper_monitor()
            if self._gripper_limit is not None:
                direction, held = self._gripper_limit
                if (opening - held) * direction >= -self.GRIPPER_TOLERANCE:
                    return
                self._gripper_limit = None
            if opening == self._gripper_requested and self._max_relative_target is None:
                return
            sent = self._robot.send_action({f"{self.GRIPPER}.pos": opening})
            self._gripper_requested = opening
            self._gripper_target = float(sent[f"{self.GRIPPER}.pos"])

    def _check_gripper_monitor(self) -> None:
        """Surface a failed monitor before further reads or commands."""
        if self._gripper_error is not None:
            raise RuntimeError(
                "SO-101 gripper monitoring failed; reconnect the arm."
            ) from self._gripper_error

    def _watch_gripper(self) -> None:
        """Relieve stalled targets while serializing access to the servo bus."""
        previous = None
        direction = 0.0
        still = 0
        deadline = 0.0
        try:
            while not self._gripper_stop.wait(self.SETTLE_POLL_INTERVAL):
                with self._gripper_condition:
                    target = self._gripper_target
                    if target is None:
                        previous = None
                        continue
                    current = float(
                        self._robot.get_observation()[f"{self.GRIPPER}.pos"]
                    )
                    if abs(current - target) <= self.GRIPPER_TOLERANCE:
                        self._gripper_target = None
                        previous = None
                        self._gripper_condition.notify_all()
                        continue
                    next_direction = float(np.sign(target - current))
                    if previous is None or next_direction != direction:
                        previous = current
                        still = 0
                        direction = next_direction
                        deadline = time.monotonic() + self.GRIPPER_SETTLE_TIMEOUT
                        continue
                    # Track the furthest measured position toward the target.
                    # Small advances count; jitter around one position does not.
                    if (current - previous) * direction > 0:
                        previous = current
                        still = 0
                        deadline = time.monotonic() + self.GRIPPER_SETTLE_TIMEOUT
                    else:
                        still += 1
                    if (
                        still >= self.GRIPPER_STALL_POLLS
                        or time.monotonic() >= deadline
                    ):
                        self._robot.send_action({f"{self.GRIPPER}.pos": current})
                        self._gripper_limit = (direction, current)
                        self._gripper_target = None
                        previous = None
                        self._gripper_condition.notify_all()
        except Exception as error:
            with self._gripper_condition:
                self._gripper_error = error
                self._gripper_condition.notify_all()

    def _wait_for_gripper(self) -> None:
        """Wait for a discrete open or close to finish or relieve a stall."""
        with self._gripper_condition:
            while self._gripper_target is not None:
                self._check_gripper_monitor()
                self._gripper_condition.wait()
            self._check_gripper_monitor()

    def open_gripper(self) -> None:
        """Open the gripper fully."""
        self.move_gripper([1.0])
        self._wait_for_gripper()

    def close_gripper(self) -> None:
        """Close the gripper fully."""
        self.move_gripper([0.0])
        self._wait_for_gripper()

    def reset_joint(
        self,
        positions: "Sequence[float]",
        duration: float = 3.0,
        *,
        max_velocity: float = np.deg2rad(30.0),
    ) -> None:
        """Follow a smooth joint trajectory from the measured pose, then settle.

        Args:
            positions: Five target joint angles in radians.
            duration: Minimum trajectory duration in seconds. Longer moves
                take more time to respect ``max_velocity``.
            max_velocity: Maximum commanded speed of any joint, in radians
                per second. Defaults to 30 degrees per second.
        """
        target = np.asarray(positions, dtype=float).reshape(-1)
        if target.shape != (len(self.MOTORS),) or not np.all(np.isfinite(target)):
            raise ValueError("An SO-101 reset needs five finite joint targets.")
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Reset duration must be finite and positive.")
        if not np.isfinite(max_velocity) or max_velocity <= 0:
            raise ValueError("Reset max_velocity must be finite and positive.")
        start = self.get_state().arm_joint_position
        if not np.all(np.isfinite(start)):
            raise RuntimeError("Cannot reset an SO-101 with non-finite joint feedback.")
        delta = target - start
        distance = float(np.max(np.abs(delta)))
        if distance > 1e-8:
            # The quintic blend has peak slope 15/8 and zero endpoint velocity
            # and acceleration. Stretch its duration to bound every joint's speed.
            travel_time = max(duration, 1.875 * distance / max_velocity)
            period = 1.0 / 30.0
            steps = max(2, int(np.ceil(travel_time / period)))
            for index in range(1, steps + 1):
                # Never catch up with a burst of writes after a slow bus call.
                time.sleep(period)
                phase = index / steps
                blend = phase**3 * (10 + phase * (-15 + 6 * phase))
                self.move_joints(start + blend * delta)
        else:
            self.move_joints(target)
        self.wait_until_still()

    def is_robot_up(self) -> bool:
        """Report whether the servo bus and any lerobot cameras are live."""
        return bool(self._robot is not None and self._robot.is_connected)
