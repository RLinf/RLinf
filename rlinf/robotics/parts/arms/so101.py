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
                self, state_field="gripper_position", command="move_gripper"
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
        robot.connect(calibrate=False)
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

    def _release(self, device: "SO101Follower") -> None:
        """Close the servo bus."""
        try:
            device.disconnect()
        finally:
            self._robot = None

    def get_state(self) -> SO101RobotState:
        """Read every servo and convert it to canonical units."""
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
        sent = self._robot.send_action(
            {f"{motor}.pos": float(value) for motor, value in zip(self.MOTORS, degrees)}
        )
        return np.deg2rad(
            np.asarray([sent[f"{motor}.pos"] for motor in self.MOTORS], dtype=float)
        )

    def move_gripper(self, target: "Sequence[float]") -> None:
        """Command the gripper to an opening fraction in ``0..1``."""
        value = float(np.asarray(target, dtype=float).reshape(-1)[0])
        opening = float(np.clip(value, 0.0, 1.0)) * self.GRIPPER_SCALE
        self._robot.send_action({f"{self.GRIPPER}.pos": opening})

    def open_gripper(self) -> None:
        """Open the gripper fully."""
        self.move_gripper([1.0])

    def close_gripper(self) -> None:
        """Close the gripper fully."""
        self.move_gripper([0.0])

    def reset_joint(self, positions: "Sequence[float]", duration: float = 3.0) -> None:
        """Move to a rest pose in radians, returning once the arm has stopped.

        lerobot's bus writes a goal position and returns while the servos are
        still travelling, so this waits for them, as the other arm backends
        do. Without it a state read taken straight after a reset reports the
        pose the arm is leaving. Gives up after ``duration`` seconds.
        """
        self.move_joints(positions)
        self.wait_until_still(duration)

    def is_robot_up(self) -> bool:
        """Report whether the servo bus and any lerobot cameras are live."""
        return bool(self._robot is not None and self._robot.is_connected)
