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

"""Kuavo dual-arm connection backed by the Kuavo humanoid SDK."""

from dataclasses import asdict, dataclass, field
from threading import RLock
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike

from rlinf.robotics.parts.base import Connection, RobotPart
from rlinf.robotics.parts.views import MethodArm, MethodEndEffector

_ARM_DOF = 7
_ARM_SIDES = ("left", "right")


@dataclass
class KuavoRobotState:
    """Canonical state exported by the Kuavo SDK connection."""

    left_joint_position: np.ndarray = field(
        default_factory=lambda: np.zeros(_ARM_DOF, dtype=np.float32)
    )
    right_joint_position: np.ndarray = field(
        default_factory=lambda: np.zeros(_ARM_DOF, dtype=np.float32)
    )
    left_gripper: np.ndarray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float32)
    )
    right_gripper: np.ndarray = field(
        default_factory=lambda: np.zeros(1, dtype=np.float32)
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert the state to the mapping consumed by method views."""
        return asdict(self)


class KuavoConnection(Connection):
    """One Kuavo SDK session shared by the selected arms and end effectors."""

    SDK: ClassVar[str] = "kuavo_humanoid_sdk"

    def __init__(
        self,
        which_arm: str = "both",
        end_effector_type: str = "leju_claw",
        head_position: tuple[float, float] | None = None,
        joint_limits_min: ArrayLike = (-3.14,) * (2 * _ARM_DOF),
        joint_limits_max: ArrayLike = (3.14,) * (2 * _ARM_DOF),
        end_effector_limits_min: ArrayLike = (0.0, 0.0),
        end_effector_limits_max: ArrayLike = (1.0, 1.0),
    ) -> None:
        if which_arm not in {"left", "right", "both"}:
            raise ValueError(f"Unsupported Kuavo arm selection {which_arm!r}.")
        if end_effector_type not in {"leju_claw", "qiangnao", "rq2f85"}:
            raise ValueError(
                f"Unsupported Kuavo end effector {end_effector_type!r}."
            )
        if head_position is not None:
            head = np.asarray(head_position, dtype=np.float64).reshape(-1)
            if head.shape != (2,) or not np.all(np.isfinite(head)):
                raise ValueError(
                    "Kuavo head_position must contain two finite yaw/pitch values."
                )
        self.which_arm = which_arm
        self.end_effector_type = end_effector_type
        self.head_position = None if head_position is None else tuple(head)
        self._joint_limits_min = self._limits("minimum", joint_limits_min)
        self._joint_limits_max = self._limits("maximum", joint_limits_max)
        if np.any(self._joint_limits_min > self._joint_limits_max):
            raise ValueError("Kuavo joint minimum exceeds its maximum.")
        self._end_effector_limits_min = self._end_effector_limits(
            "minimum", end_effector_limits_min
        )
        self._end_effector_limits_max = self._end_effector_limits(
            "maximum", end_effector_limits_max
        )
        if np.any(self._end_effector_limits_min > self._end_effector_limits_max):
            raise ValueError("Kuavo end-effector minimum exceeds its maximum.")
        self._lock = RLock()
        self._robot: Any = None
        self._robot_state: Any = None
        self._end_effector: Any = None
        self._gripper_publisher: Any = None
        self._gripper_subscriber: Any = None
        self._joint_state_type: Any = None
        self._rq2f85_position = np.zeros(2, dtype=np.float32)
        self._commanded_joint_position: np.ndarray | None = None
        self._commanded_gripper_position: np.ndarray | None = None

    @property
    def parts(self) -> dict[str, RobotPart]:
        """Expose each selected arm and its end effector as standard parts."""
        sides = _ARM_SIDES if self.which_arm == "both" else (self.which_arm,)
        parts: dict[str, RobotPart] = {}
        for side in sides:
            parts[side] = MethodArm(
                self,
                commands={"arm_joint_position": f"move_{side}_arm"},
                state_fields={
                    "arm_joint_position": f"{side}_joint_position",
                },
            )
            end_effector = MethodEndEffector(
                self,
                state_field=f"{side}_gripper",
                command=f"move_{side}_gripper",
                is_gripper=self.end_effector_type != "qiangnao",
            )
            end_effector.is_hand = self.end_effector_type == "qiangnao"
            parts[f"{side}_end_effector"] = end_effector
        return parts

    def _open(self) -> Any:
        """Initialize the SDK and enter external arm-control mode."""
        from kuavo_humanoid_sdk import (
            DexterousHand,
            KuavoRobot,
            KuavoRobotState,
            KuavoSDK,
            LejuClaw,
        )

        self._clear_runtime_handles()
        try:
            if not KuavoSDK().Init():
                raise RuntimeError("Failed to initialize the Kuavo humanoid SDK.")

            self._robot = KuavoRobot()
            self._robot_state = KuavoRobotState()
            if not self._robot.set_external_control_arm_mode():
                raise RuntimeError("Failed to enter Kuavo external arm-control mode.")
            if self.head_position is not None and not self._robot.control_head(
                *self.head_position
            ):
                raise RuntimeError("Failed to command the Kuavo head position.")

            if self.end_effector_type == "leju_claw":
                self._end_effector = LejuClaw()
            elif self.end_effector_type == "qiangnao":
                self._end_effector = DexterousHand()
            else:
                import rospy
                from sensor_msgs.msg import JointState

                self._joint_state_type = JointState
                self._gripper_publisher = rospy.Publisher(
                    "/gripper/command", JointState, queue_size=10
                )
                self._gripper_subscriber = rospy.Subscriber(
                    "/gripper/state",
                    JointState,
                    self._update_rq2f85_state,
                    queue_size=1,
                )
        except BaseException:
            self._clear_runtime_handles()
            raise
        return self._robot

    def _release(self, device: Any) -> None:
        """Drop local SDK handles; the process-wide ROS session stays alive."""
        self._clear_runtime_handles()

    def _clear_runtime_handles(self) -> None:
        """Release per-connection ROS endpoints and forget SDK state."""
        for endpoint in (self._gripper_subscriber, self._gripper_publisher):
            unregister = getattr(endpoint, "unregister", None)
            if callable(unregister):
                unregister()
        self._gripper_subscriber = None
        self._gripper_publisher = None
        self._joint_state_type = None
        self._end_effector = None
        self._robot_state = None
        self._robot = None
        self._commanded_joint_position = None
        self._commanded_gripper_position = None
        self._rq2f85_position = np.zeros(2, dtype=np.float32)

    def _update_rq2f85_state(self, message: Any) -> None:
        """Store the latest two-axis RQ2F85 state received from ROS."""
        position = np.asarray(message.position, dtype=np.float32).reshape(-1)
        if position.size >= 2:
            with self._lock:
                self._rq2f85_position = position[:2].copy()

    def get_state(self) -> KuavoRobotState:
        """Read both arm joints and normalized end-effector positions."""
        if self._robot_state is None:
            raise RuntimeError("Kuavo connection is not open.")
        joint_position = self._read_arm_positions()
        left_gripper, right_gripper = self._read_grippers()
        return KuavoRobotState(
            left_joint_position=joint_position[:_ARM_DOF],
            right_joint_position=joint_position[_ARM_DOF:],
            left_gripper=np.asarray([left_gripper], dtype=np.float32),
            right_gripper=np.asarray([right_gripper], dtype=np.float32),
        )

    def _read_arm_positions(self) -> np.ndarray:
        """Read and validate the SDK's fixed-width dual-arm joint vector."""
        joint_position = np.asarray(
            self._robot_state.arm_joint_state().position, dtype=np.float32
        )
        if joint_position.shape != (2 * _ARM_DOF,):
            raise ValueError(
                "Kuavo arm state must contain 14 joint positions, "
                f"got {joint_position.shape}."
            )
        if not np.all(np.isfinite(joint_position)):
            raise ValueError("Kuavo arm state must contain finite joint positions.")
        return joint_position

    def _read_grippers(self) -> tuple[float, float]:
        """Return one normalized opening value for each end effector."""
        if self._end_effector is not None:
            left, right = self._end_effector.get_position()
            return self._primary_position(left), self._primary_position(right)
        return (
            self._primary_position(self._rq2f85_position[:1], scale=255.0),
            self._primary_position(self._rq2f85_position[1:], scale=255.0),
        )

    @staticmethod
    def _primary_position(value: ArrayLike, scale: float = 100.0) -> float:
        """Reduce a claw or hand state to its first normalized degree of freedom."""
        position = np.asarray(value, dtype=np.float32).reshape(-1)
        if not position.size or not np.isfinite(position[0]):
            raise ValueError("Kuavo end-effector state must contain a finite value.")
        return float(np.clip(position[0] / scale, 0.0, 1.0))

    def move_left_arm(self, target: ArrayLike) -> None:
        """Command the left arm while holding the current right-arm joints."""
        self._move_arm("left", target)

    def move_right_arm(self, target: ArrayLike) -> None:
        """Command the right arm while holding the current left-arm joints."""
        self._move_arm("right", target)

    def _move_arm(self, side: str, target: ArrayLike) -> None:
        target_array = np.asarray(target, dtype=np.float64).reshape(-1)
        if target_array.shape != (_ARM_DOF,):
            raise ValueError(
                f"Kuavo {side} arm target must have shape ({_ARM_DOF},), "
                f"got {target_array.shape}."
            )
        if not np.all(np.isfinite(target_array)):
            raise ValueError(f"Kuavo {side} arm target must contain finite values.")
        if self._robot is None:
            raise RuntimeError("Kuavo connection is not open.")

        offset = 0 if side == "left" else _ARM_DOF
        lower = self._joint_limits_min[offset : offset + _ARM_DOF]
        upper = self._joint_limits_max[offset : offset + _ARM_DOF]
        target_array = np.clip(target_array, lower, upper)
        with self._lock:
            if self._commanded_joint_position is None:
                joint_position = self._read_arm_positions().astype(np.float64)
            else:
                joint_position = self._commanded_joint_position.copy()
            joint_position[offset : offset + _ARM_DOF] = target_array
            command = joint_position.astype(float).tolist()
            if not self._robot.control_arm_joint_positions(command):
                raise RuntimeError(f"Failed to command the Kuavo {side} arm.")
            self._commanded_joint_position = joint_position

    def move_left_gripper(self, target: ArrayLike) -> None:
        """Command the left end effector with a normalized scalar target."""
        self._move_gripper("left", target)

    def move_right_gripper(self, target: ArrayLike) -> None:
        """Command the right end effector with a normalized scalar target."""
        self._move_gripper("right", target)

    def _move_gripper(self, side: str, target: ArrayLike) -> None:
        target_array = np.asarray(target, dtype=np.float64).reshape(-1)
        if target_array.shape != (1,) or not np.all(np.isfinite(target_array)):
            raise ValueError(
                f"Kuavo {side} end-effector target must be one finite value."
            )
        side_index = 0 if side == "left" else 1
        value = float(
            np.clip(
                target_array[0],
                self._end_effector_limits_min[side_index],
                self._end_effector_limits_max[side_index],
            )
        )
        if self._robot is None:
            raise RuntimeError("Kuavo connection is not open.")
        if self.end_effector_type == "rq2f85":
            with self._lock:
                if self._commanded_gripper_position is None:
                    current = np.asarray(self._read_grippers()) * 255.0
                else:
                    current = self._commanded_gripper_position.copy()
                current[0 if side == "left" else 1] = value * 255.0
                message = self._joint_state_type()
                message.name = ["left_gripper_joint", "right_gripper_joint"]
                message.position = current.astype(float).tolist()
                self._gripper_publisher.publish(message)
                self._commanded_gripper_position = current
            return

        method = getattr(self._end_effector, f"control_{side}")
        if self.end_effector_type == "leju_claw":
            positions = [value * 100.0]
        else:
            positions = [value * 100.0, 100.0, *([value * 100.0] * 4)]
        if not method(
            target_positions=positions,
            target_velocities=None,
            target_torques=None,
        ):
            raise RuntimeError(f"Failed to command the Kuavo {side} end effector.")

    @staticmethod
    def _limits(name: str, values: ArrayLike) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64).reshape(-1)
        if result.shape != (2 * _ARM_DOF,) or not np.all(np.isfinite(result)):
            raise ValueError(
                f"Kuavo joint {name} limits must contain 14 finite values."
            )
        return result

    @staticmethod
    def _end_effector_limits(name: str, values: ArrayLike) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64).reshape(-1)
        if (
            result.shape != (2,)
            or not np.all(np.isfinite(result))
            or np.any(result < 0.0)
            or np.any(result > 1.0)
        ):
            raise ValueError(
                f"Kuavo end-effector {name} limits must contain two finite "
                "values within 0..1."
            )
        return result
