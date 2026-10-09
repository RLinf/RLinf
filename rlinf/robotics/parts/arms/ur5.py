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

"""Universal Robots UR5 arm, driven through ``ur_rtde`` over Ethernet.

Registered as ``ur_rtde``, after the SDK, so a driver for the same arm over
another interface (for example ROS) registers beside it. ``ur_rtde`` already
works in radians, metres, and newtons.

Two interfaces are opened on the same controller IP: ``RTDEControlInterface``
sends motion commands and ``RTDEReceiveInterface`` reads state. The pair shares
one :class:`~rlinf.robotics.parts.claims.DeviceClaim` so two drivers cannot
drive the same controller at once.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.robotics.parts.base import Action, Features, Observation
from rlinf.robotics.parts.claims import DeviceClaim
from rlinf.utils.logging import get_logger

from .base import Arm, BaseArm


@dataclass
class UR5RobotState:
    """State snapshot for the Universal Robots UR5 arm."""

    arm_joint_position: np.ndarray = field(default_factory=lambda: np.zeros(6))
    """Joint positions in radians, base to wrist."""

    tcp_pose: np.ndarray = field(default_factory=lambda: np.zeros(7))
    """``[x, y, z, qx, qy, qz, qw]`` in the base frame, metres and quaternion."""

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass to a serializable dictionary."""
        return asdict(self)


@Arm.register("ur_rtde")
class UR5Arm(BaseArm):
    """Universal Robots UR5 over Ethernet, via ``ur_rtde``.

    Args:
        address: Controller IP, such as ``"192.168.1.10"``.
        speed: Joint speed for commanded motion, in radians per second.
        acceleration: Joint acceleration for commanded motion, in radians per
            second squared.
    """

    SDK = "ur_rtde"

    DOF: ClassVar[int] = 6

    #: Travel of a UR5. The hardware allows +/- 2*pi; these conservative bench
    #: limits keep a reach task clear of self-collision. A task may narrow them
    #: further through its env config.
    JOINT_LIMITS_LOWER: ClassVar[np.ndarray] = np.full(6, -np.pi)
    JOINT_LIMITS_UPPER: ClassVar[np.ndarray] = np.full(6, np.pi)

    #: The UR5 reports joints and tool pose; force sensing is left out here.
    STATE_FIELDS = ("arm_joint_position", "tcp_pose")

    def __init__(
        self,
        address: str,
        *,
        speed: float = 1.05,
        acceleration: float = 1.4,
    ) -> None:
        self._logger = get_logger()
        self._ip = address
        self._speed = float(speed)
        self._acceleration = float(acceleration)
        self._control: Any = None
        self._receive: Any = None
        # Two interfaces on one controller; a second driver would contend for it.
        self._claim = DeviceClaim(f"ur5:{address}", type(self).__name__)

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
    ) -> "UR5Arm":
        """Declare a UR5 on the controller IP named by ``address``.

        This driver carries no gripper, so the end-effector options are
        refused rather than silently dropped. A grasp task adds a gripper as a
        separate end-effector part composed beside the arm.
        """
        cls.refuse_unused(
            gripper_type=gripper_type,
            gripper_connection=gripper_connection,
            end_effector_type=end_effector_type,
            end_effector_config=end_effector_config,
        )
        settings = {
            key: placement.pop(key)
            for key in ("speed", "acceleration")
            if key in placement
        }
        return cls(address, **settings, **placement)

    @property
    def action_features(self) -> Features:
        """Describe the absolute joint target, in radians."""
        return {"joint_position": {}}

    def _open(self) -> Any:
        """Open the control and receive interfaces on the controller IP."""
        self._claim.acquire()
        try:
            from rtde_control import RTDEControlInterface
            from rtde_receive import RTDEReceiveInterface

            self._receive = RTDEReceiveInterface(self._ip)
            self._control = RTDEControlInterface(self._ip)
        except BaseException:
            self._release(self._control)
            raise
        self._logger.info("UR5 connected on %s", self._ip)
        return self._control

    def _release(self, device: Any) -> None:
        """Disconnect both interfaces, leaving the arm holding its position.

        Also used to unwind a failed open, so either interface may be ``None``.
        """
        try:
            if self._control is not None:
                self._control.disconnect()
            if self._receive is not None:
                self._receive.disconnect()
        finally:
            self._control = None
            self._receive = None
            self._claim.release()

    def get_state(self) -> UR5RobotState:
        """Read joints and tool pose into canonical units."""
        return UR5RobotState(
            arm_joint_position=np.asarray(self._receive.getActualQ(), dtype=float),
            tcp_pose=self._tcp_pose(),
        )

    def _tcp_pose(self) -> np.ndarray:
        """Read the tool pose and convert its rotation vector to a quaternion."""
        pose = np.asarray(self._receive.getActualTCPPose(), dtype=float)
        translation = pose[:3]
        # ur_rtde reports orientation as an axis-angle rotation vector.
        quaternion = R.from_rotvec(pose[3:6]).as_quat()
        return np.concatenate([translation, quaternion])

    def send_action(self, action: Action) -> Observation:
        """Move the arm joints to an absolute target in radians."""
        if set(action) != {"joint_position"}:
            raise KeyError(
                "A UR5 arm action holds only 'joint_position'; this driver "
                "carries no gripper."
            )
        sent = self.move_joints(action["joint_position"])
        return {"joint_position": sent}

    def move_joints(self, q_target: "Sequence[float]") -> np.ndarray:
        """Command absolute joint positions in radians.

        Targets are clipped to the configured travel. Uses a blocking ``moveJ``;
        a task that needs smooth fixed-rate streaming should switch to
        ``servoJ`` here.

        Returns:
            The target actually sent, in radians.
        """
        target = np.asarray(q_target, dtype=float).reshape(-1)
        if target.shape != (self.DOF,):
            raise ValueError(
                f"Expected {self.DOF} joint targets for a UR5, got shape "
                f"{target.shape}."
            )
        target = np.clip(target, self.JOINT_LIMITS_LOWER, self.JOINT_LIMITS_UPPER)
        self._control.moveJ(
            [float(value) for value in target], self._speed, self._acceleration
        )
        return target

    def reset_joint(
        self,
        positions: "Sequence[float]",
        duration: float = 3.0,
        *,
        max_velocity: float = np.deg2rad(30.0),
    ) -> None:
        """Move to a rest pose in radians, returning once the arm has stopped.

        ``moveJ`` blocks until the arm arrives, but ``wait_until_still`` guards
        against a backend that returns early, matching the other arm drivers.
        ``duration`` and ``max_velocity`` are accepted for a uniform reset
        signature; ur_rtde paces the move by its own ``speed``.
        """
        self.move_joints(positions)
        self.wait_until_still(duration)

    def is_robot_up(self) -> bool:
        """Report whether both interfaces are open and connected."""
        return bool(
            self._control is not None
            and self._receive is not None
            and self._control.isConnected()
            and self._receive.isConnected()
        )
