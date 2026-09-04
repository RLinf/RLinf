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

"""AgileX Piper arm, driven through ``pyAgxArm`` over a CAN bus.

Registered as ``pyagxarm``, after the SDK, so a driver for the same arm over
another SDK registers beside it. The AgxGripper shares the bus and is exported
beneath the arm.

``pyAgxArm`` already works in radians, metres, and newtons. Note that its
``disable()`` and ``reset()`` both cut motor power, dropping a raised arm, so
neither is used here.
"""

import time
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.robotics.parts.base import Action, Features, Observation, RobotPart
from rlinf.robotics.parts.claims import DeviceClaim
from rlinf.robotics.parts.views import MethodEndEffector
from rlinf.utils.logging import get_logger

from .base import Arm, BaseArm


@dataclass
class PiperRobotState:
    """State snapshot for the AgileX Piper arm."""

    tcp_pose: np.ndarray = field(default_factory=lambda: np.zeros(7))
    """``[x, y, z, qx, qy, qz, qw]`` in the base frame, metres and quaternion."""

    arm_joint_position: np.ndarray = field(default_factory=lambda: np.zeros(6))
    """Joint positions in radians."""

    gripper_position: np.ndarray = field(default_factory=lambda: np.zeros(1))
    """Opening as a fraction of the stroke, 0 shut to 1 open."""

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass to a serializable dictionary."""
        return asdict(self)


@Arm.register("pyagxarm")
class PiperArm(BaseArm):
    """AgileX Piper arm on a CAN bus, via ``pyAgxArm``.

    Args:
        channel: CAN channel, such as ``"can0"``. Bring the interface up at
            ``bitrate`` before connecting; the SDK does not configure it.
        interface: python-can backend: ``"socketcan"``, ``"slcan"``, or
            ``"agx_cando"``.
        bitrate: CAN bitrate.
        model: Arm variant: ``"piper"``, ``"piper_h"``, ``"piper_l"``, or
            ``"piper_x"``.
        firmware: Firmware profile. ``None`` reads the version off the arm
            and picks the matching one, which is what you want unless the arm
            cannot be reached to ask. Pin it with ``"default"`` for S-V1.8-2
            and older, then ``"v183"``, ``"v188"``, ``"v189"``.
        speed_percent: Percentage of maximum speed for commanded motion.
        gripper_force: Gripping force in newtons, up to 3.0.
        gripper_max_width: Stroke at full opening, in metres. Set at the
            factory to either 0.07 or 0.1.
        with_gripper: Whether an AgxGripper is fitted.
    """

    SDK = "pyAgxArm"

    DOF: ClassVar[int] = 6

    #: Travel of the standard ``piper``, in radians. A connected arm replaces
    #: these with the limits its own variant reports.
    JOINT_LIMITS_LOWER: ClassVar[np.ndarray] = np.array(
        [-2.617994, 0.0, -2.967060, -1.745330, -1.221730, -2.094396]
    )
    JOINT_LIMITS_UPPER: ClassVar[np.ndarray] = np.array(
        [2.617994, 3.141593, 0.0, 1.745330, 1.221730, 2.094396]
    )

    #: Profile used only to ask an arm its version before the real driver is
    #: built. Any profile can carry that query.
    PROBE_FIRMWARE: ClassVar[str] = "default"

    #: Seconds to wait on the arm, and how often to look.
    ENABLE_TIMEOUT_S: ClassVar[float] = 5.0
    FEEDBACK_TIMEOUT_S: ClassVar[float] = 2.0
    POLL_S: ClassVar[float] = 0.01

    #: No Cartesian velocity, external force, or Jacobian is reported.
    STATE_FIELDS = ("tcp_pose", "arm_joint_position")

    def __init__(
        self,
        channel: str = "can0",
        *,
        interface: str = "socketcan",
        bitrate: int = 1000000,
        model: str = "piper",
        firmware: Optional[str] = None,
        speed_percent: int = 30,
        gripper_force: float = 1.0,
        gripper_max_width: float = 0.07,
        with_gripper: bool = True,
    ) -> None:
        self._logger = get_logger()
        self._channel = channel
        self._interface = interface
        self._bitrate = int(bitrate)
        self._model = model
        self._firmware = firmware
        self._speed_percent = int(np.clip(speed_percent, 1, 100))
        self._gripper_force = float(np.clip(gripper_force, 0.0, 3.0))
        self._gripper_max_width = float(gripper_max_width)
        self._with_gripper = with_gripper
        self._robot: Any = None
        self._gripper: Any = None
        self._limits_lower = self.JOINT_LIMITS_LOWER
        self._limits_upper = self.JOINT_LIMITS_UPPER
        # Two arms on one channel would fight for the bus rather than fail.
        self._claim = DeviceClaim(f"piper-arm:{channel}", type(self).__name__)

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
    ) -> "PiperArm":
        """Declare a Piper on the CAN channel named by ``address``.

        The AgxGripper shares that bus, so it is neither chosen nor wired
        separately; ``with_gripper`` says whether one is fitted.
        """
        cls.refuse_unused(
            gripper_type=gripper_type,
            gripper_connection=gripper_connection,
            end_effector_type=end_effector_type,
            end_effector_config=end_effector_config,
        )
        settings = {
            key: placement.pop(key)
            for key in (
                "interface",
                "bitrate",
                "model",
                "firmware",
                "speed_percent",
                "gripper_force",
                "gripper_max_width",
                "with_gripper",
            )
            if key in placement
        }
        return cls(address, **settings, **placement)

    @property
    def action_features(self) -> Features:
        """Describe the absolute joint target, in radians."""
        return {"joint_position": {}}

    @property
    def parts(self) -> dict[str, RobotPart]:
        """Return the gripper this arm carries on its own bus.

        It takes any opening in range, so the view drives it through
        :meth:`move_gripper` rather than binary open and close.
        """
        if not self._with_gripper:
            return {}
        return {
            "end_effector": MethodEndEffector(
                self, state_field="gripper_position", command="move_gripper"
            )
        }

    def _open(self) -> Any:
        """Build the SDK driver, open the bus, and enable the motors."""
        self._claim.acquire()
        try:
            from pyAgxArm import AgxArmFactory

            firmware = self._firmware or self._detect_firmware()
            config = self._config_for(firmware)
            robot = AgxArmFactory.create_arm(config)
            # Once per driver, and before the read thread starts.
            if self._with_gripper:
                self._gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
            robot.connect()
            self._robot = robot
            self._adopt_joint_limits(config)
            self._enable(robot)
            robot.set_speed_percent(self._speed_percent)
            self._await_feedback(robot)
        except BaseException:
            self._release(self._robot)
            raise
        self._logger.info(
            "Piper %s connected on %s (%s)",
            self._model,
            self._channel,
            self._interface,
        )
        return robot

    def _config_for(self, firmware: str) -> dict:
        """Return the SDK config for this arm on a given firmware profile."""
        from pyAgxArm import create_agx_arm_config

        # The vendor spells it "firmeware_version".
        return create_agx_arm_config(
            robot=self._model,
            firmeware_version=firmware,
            interface=self._interface,
            channel=self._channel,
            bitrate=self._bitrate,
        )

    def _detect_firmware(self) -> str:
        """Ask the arm which firmware it runs and return the SDK profile.

        The profile decides how the driver frames CAN messages, and the arm
        only reports its version once something is talking to it. So a short
        session opens, reads, and closes before the real driver is built, as
        the SDK's own documentation describes. Falling back to the oldest
        profile would silently speak the wrong protocol to a current arm, so a
        failed probe says so instead.

        Raises:
            RuntimeError: If the arm does not report a usable version.
        """
        from pyAgxArm import AgxArmFactory, resolve_firmware_profile

        probe = AgxArmFactory.create_arm(self._config_for(self.PROBE_FIRMWARE))
        try:
            probe.connect()
            reported = probe.get_firmware()
        finally:
            probe.disconnect()

        version = (reported or {}).get("software_version")
        if not version:
            raise RuntimeError(
                f"The Piper on {self._channel!r} did not report a firmware "
                "version, so its protocol cannot be chosen for it. Check the "
                "arm is powered and the bus is up, or set 'firmware' to the "
                "profile matching its version."
            )
        profile = resolve_firmware_profile(self._model, version)
        self._logger.info("Piper reports %s, using profile %r", version, profile)
        return profile

    def _adopt_joint_limits(self, config: dict) -> None:
        """Clip against the limits this variant's configuration carries."""
        limits = config.get("joint_limits") or {}
        names = config.get("joint_names") or []
        if len(names) != self.DOF or not all(name in limits for name in names):
            return
        bounds = np.asarray([limits[name] for name in names], dtype=float)
        self._limits_lower, self._limits_upper = bounds[:, 0], bounds[:, 1]

    def _enable(self, robot: Any) -> None:
        """Wait for every joint to report itself enabled.

        Raises:
            RuntimeError: If the motors do not enable in time.
        """
        deadline = time.time() + self.ENABLE_TIMEOUT_S
        while not robot.enable():
            if time.time() >= deadline:
                raise RuntimeError(
                    f"The Piper on {self._channel!r} did not enable within "
                    f"{self.ENABLE_TIMEOUT_S:g}s. Check that the arm is "
                    f"powered, that {self._channel!r} is up at "
                    f"{self._bitrate} bit/s, and that its emergency stop is "
                    "released."
                )
            time.sleep(self.POLL_S)

    def _await_feedback(self, robot: Any) -> None:
        """Wait for the first joint frame, so a read never starts empty.

        Raises:
            RuntimeError: If no joint feedback arrives in time.
        """
        deadline = time.time() + self.FEEDBACK_TIMEOUT_S
        while robot.get_joint_angles() is None:
            if time.time() >= deadline:
                raise RuntimeError(
                    f"The Piper on {self._channel!r} enabled but sent no joint "
                    f"feedback within {self.FEEDBACK_TIMEOUT_S:g}s. Check that "
                    "the arm is in follower mode: a leader arm does not report."
                )
            time.sleep(self.POLL_S)

    def _release(self, device: Any) -> None:
        """Close the CAN session, leaving the motors holding their position.

        Neither disabled nor reset: both cut motor power, and closing a
        connection should not move the arm. Also used to unwind a failed
        open, so ``device`` may be ``None``.
        """
        try:
            if device is not None:
                device.disconnect()
        finally:
            self._robot = None
            self._gripper = None
            self._claim.release()

    def get_state(self) -> PiperRobotState:
        """Read joints, tool pose, and gripper into canonical units."""
        return PiperRobotState(
            tcp_pose=self._tcp_pose(),
            arm_joint_position=self._joint_reading(),
            gripper_position=np.asarray([self._gripper_opening()], dtype=float),
        )

    def _joint_reading(self) -> np.ndarray:
        """Read just the joint angles, skipping the tool pose a poll ignores."""
        return np.asarray(
            self._require("joint angles", self._robot.get_joint_angles()), dtype=float
        )

    def _require(self, what: str, reading: Any) -> Any:
        """Return a reading's payload, or say that the feed has stopped.

        Raises:
            RuntimeError: If the SDK has no message of this kind. Opening
                waited for feedback, so this means it stopped.
        """
        if reading is None:
            raise RuntimeError(
                f"The Piper on {self._channel!r} stopped reporting {what}. "
                "The CAN feed was live when it connected, so check the bus and "
                "that the arm is still powered."
            )
        return reading.msg

    def _tcp_pose(self) -> np.ndarray:
        """Read the tool pose and convert its Euler angles to a quaternion."""
        pose = self._require("a tool pose", self._robot.get_tcp_pose())
        translation = np.asarray(pose[:3], dtype=float)
        # The SDK composes orientation as R = Rz @ Ry @ Rx: an extrinsic xyz.
        quaternion = R.from_euler("xyz", np.asarray(pose[3:6], dtype=float)).as_quat()
        return np.concatenate([translation, quaternion])

    def _gripper_opening(self) -> float:
        """Return the gripper opening as a fraction of its stroke."""
        if self._gripper is None:
            return 0.0
        status = self._require("gripper status", self._gripper.get_gripper_status())
        if status.mode != "width":
            raise RuntimeError(
                f"The Piper gripper on {self._channel!r} is reporting "
                f"{status.mode!r}, but this driver reads and commands a width. "
                "Configure the gripper in width mode."
            )
        return float(np.clip(status.value / self._gripper_max_width, 0.0, 1.0))

    def send_action(self, action: Action) -> Observation:
        """Move the arm joints to an absolute target in radians."""
        if set(action) != {"joint_position"}:
            raise KeyError(
                "A Piper arm action holds only 'joint_position'; the gripper "
                "is commanded through its own end-effector part."
            )
        sent = self.move_joints(action["joint_position"])
        return {"joint_position": sent}

    def move_joints(self, q_target: "Sequence[float]") -> np.ndarray:
        """Command absolute joint positions in radians.

        Targets are held to the arm's travel: a request beyond it faults the
        arm rather than being ignored.

        Returns:
            The target actually sent, in radians.
        """
        target = np.asarray(q_target, dtype=float).reshape(-1)
        if target.shape != (self.DOF,):
            raise ValueError(
                f"Expected {self.DOF} joint targets for a Piper, got shape "
                f"{target.shape}."
            )
        target = np.clip(target, self._limits_lower, self._limits_upper)
        # move_j selects joint mode itself.
        self._robot.move_j([float(value) for value in target])
        return target

    def move_gripper(self, target: "Sequence[float]") -> None:
        """Command the gripper to an opening fraction in ``0..1``.

        Raises:
            RuntimeError: If no gripper is fitted.
        """
        if self._gripper is None:
            raise RuntimeError(
                f"The Piper on {self._channel!r} was declared without a "
                "gripper, so it cannot be commanded. Set with_gripper=True."
            )
        value = float(np.asarray(target, dtype=float).reshape(-1)[0])
        width = float(np.clip(value, 0.0, 1.0)) * self._gripper_max_width
        self._gripper.move_gripper_m(value=width, force=self._gripper_force)

    def open_gripper(self) -> None:
        """Open the gripper fully."""
        self.move_gripper([1.0])

    def close_gripper(self) -> None:
        """Close the gripper fully."""
        self.move_gripper([0.0])

    def reset_joint(self, positions: "Sequence[float]", duration: float = 3.0) -> None:
        """Move to a rest pose in radians, returning once the arm has stopped.

        ``move_j`` returns while the arm is still travelling at
        ``speed_percent``, so this waits for it, as the other arm backends
        do. Without it a state read taken straight after a reset reports the
        pose the arm is leaving. Gives up after ``duration`` seconds.
        """
        self.move_joints(positions)
        self.wait_until_still(duration)

    def clear_errors(self) -> None:
        """Clear latched joint faults, without cutting motor power."""
        if self._robot is not None:
            self._robot.clear_joint_error()

    def is_robot_up(self) -> bool:
        """Report whether the CAN session is open and delivering frames."""
        return bool(
            self._robot is not None
            and self._robot.is_connected()
            and self._robot.is_ok()
        )
