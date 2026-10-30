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

import os
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from threading import Condition, Event, Lock, RLock, Thread
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike

from rlinf.robotics.parts.base import Connection, RobotPart
from rlinf.robotics.parts.views import MethodArm, MethodEndEffector

_ARM_DOF = 7
_ARM_SIDES = ("left", "right")
_CLAW_SERVICE = "/control_robot_leju_claw"
_CLAW_CALLS: dict[tuple[str, str], Thread] = {}
_CLAW_CALLS_LOCK = Lock()


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
        feedback_topics: Mapping[str, str] | None = None,
        joint_state_start: int = 12,
        feedback_timeout_s: float = 2.0,
        feedback_stale_timeout_s: float = 0.5,
    ) -> None:
        if which_arm not in {"left", "right", "both"}:
            raise ValueError(f"Unsupported Kuavo arm selection {which_arm!r}.")
        if end_effector_type not in {"leju_claw", "qiangnao", "rq2f85"}:
            raise ValueError(f"Unsupported Kuavo end effector {end_effector_type!r}.")
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
        if (
            not isinstance(joint_state_start, int)
            or isinstance(joint_state_start, bool)
            or joint_state_start < 0
        ):
            raise ValueError("Kuavo joint_state_start must be a nonnegative integer.")
        for name, value in (
            ("feedback_timeout_s", feedback_timeout_s),
            ("feedback_stale_timeout_s", feedback_stale_timeout_s),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"Kuavo {name} must be finite and positive.")
        self._feedback_topics = dict(
            feedback_topics
            if feedback_topics is not None
            else {
                "joint_q": "/sensors_data_raw",
                "gripper": {
                    "leju_claw": "/leju_claw_state",
                    "qiangnao": "/dexhand/state",
                    "rq2f85": "/gripper/state",
                }[end_effector_type],
            }
        )
        if set(self._feedback_topics) != {"joint_q", "gripper"} or any(
            not isinstance(topic, str) or not topic.strip()
            for topic in self._feedback_topics.values()
        ):
            raise ValueError(
                "Kuavo feedback_topics requires joint_q and gripper topics."
            )
        if len(set(self._feedback_topics.values())) != 2:
            raise ValueError("Kuavo joint_q and gripper require separate ROS topics.")
        self._joint_state_start = joint_state_start
        self._feedback_timeout_s = feedback_timeout_s
        self._feedback_stale_timeout_s = feedback_stale_timeout_s
        self._lock = RLock()
        self._feedback_condition = Condition(self._lock)
        self._feedback_generation: object | None = None
        self._feedback_values: dict[str, tuple[float, np.ndarray] | None] = {
            "joint_q": None,
            "gripper": None,
        }
        self._feedback_subscribers: list[Any] = []
        self._robot: Any = None
        self._end_effector: Any = None
        self._gripper_publisher: Any = None
        self._claw_service: Any = None
        self._claw_request_type: Any = None
        self._claw_command_lock = Lock()
        self._claw_call_thread: Thread | None = None
        self._claw_faulted = False
        self._claw_service_key: tuple[str, str] | None = None
        self._joint_state_type: Any = None
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
        if self.end_effector_type == "leju_claw":
            self._claw_service_key = (
                os.environ.get("ROS_MASTER_URI", "http://localhost:11311"),
                _CLAW_SERVICE,
            )
            with _CLAW_CALLS_LOCK:
                pending = _CLAW_CALLS.get(self._claw_service_key)
                if pending is not None and pending.is_alive():
                    raise RuntimeError(
                        "A previous Kuavo claw service call is still pending."
                    )
        from kuavo_humanoid_sdk import (
            DexterousHand,
            KuavoRobot,
            KuavoSDK,
        )

        self._clear_runtime_handles()
        self._claw_faulted = False
        try:
            if not KuavoSDK().Init():
                raise RuntimeError("Failed to initialize the Kuavo humanoid SDK.")

            self._robot = KuavoRobot()
            self._subscribe_feedback()
            self._wait_for_feedback()
            if self.end_effector_type == "leju_claw":
                import rospy
                from kuavo_humanoid_sdk.msg.kuavo_msgs.srv import (
                    controlLejuClaw,
                    controlLejuClawRequest,
                )

                service_name = _CLAW_SERVICE
                rospy.wait_for_service(service_name, timeout=self._feedback_timeout_s)
                self._claw_service = rospy.ServiceProxy(service_name, controlLejuClaw)
                self._claw_request_type = controlLejuClawRequest

            if not self._robot.set_external_control_arm_mode():
                raise RuntimeError("Failed to enter Kuavo external arm-control mode.")
            if self.head_position is not None and not self._robot.control_head(
                *self.head_position
            ):
                raise RuntimeError("Failed to command the Kuavo head position.")

            if self.end_effector_type == "qiangnao":
                self._end_effector = DexterousHand()
            elif self.end_effector_type == "rq2f85":
                import rospy
                from sensor_msgs.msg import JointState

                self._joint_state_type = JointState
                self._gripper_publisher = rospy.Publisher(
                    "/gripper/command", JointState, queue_size=10
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
        with self._feedback_condition:
            self._feedback_generation = None
            self._feedback_values = {"joint_q": None, "gripper": None}
            self._feedback_condition.notify_all()
        subscribers, self._feedback_subscribers = self._feedback_subscribers, []
        for endpoint in (*subscribers, self._gripper_publisher):
            unregister = getattr(endpoint, "unregister", None)
            if callable(unregister):
                unregister()
        self._gripper_publisher = None
        if self._claw_service is not None:
            self._claw_service.close()
        self._claw_service = None
        self._claw_request_type = None
        self._joint_state_type = None
        self._end_effector = None
        self._robot = None
        self._commanded_joint_position = None
        self._commanded_gripper_position = None

    def _subscribe_feedback(self) -> None:
        import rospy
        from kuavo_humanoid_sdk.msg.kuavo_msgs.msg import lejuClawState, sensorsData
        from sensor_msgs.msg import JointState

        generation = object()
        with self._feedback_condition:
            self._feedback_generation = generation
        message_types = {
            "joint_q": sensorsData,
            "gripper": lejuClawState
            if self.end_effector_type == "leju_claw"
            else JointState,
        }
        for name, topic in self._feedback_topics.items():
            self._feedback_subscribers.append(
                rospy.Subscriber(
                    topic,
                    message_types[name],
                    lambda message, key=name: self._update_feedback(
                        key, message, generation
                    ),
                    queue_size=1,
                )
            )

    def _update_feedback(self, name: str, message: Any, generation: object) -> None:
        try:
            if name == "joint_q":
                raw = np.asarray(message.joint_data.joint_q, dtype=np.float64)
                end = self._joint_state_start + 2 * _ARM_DOF
                if raw.ndim != 1 or raw.size < end:
                    raise ValueError("Joint feedback does not contain both arm slices.")
                position = raw[self._joint_state_start : end]
            else:
                raw = np.asarray(
                    message.data.position
                    if self.end_effector_type == "leju_claw"
                    else message.position,
                    dtype=np.float64,
                )
                width = 12 if self.end_effector_type == "qiangnao" else 2
                scale = 255.0 if self.end_effector_type == "rq2f85" else 100.0
                if raw.shape != (width,) or np.any((raw < 0.0) | (raw > scale)):
                    raise ValueError("End-effector feedback shape or range is invalid.")
                if not np.all(np.isfinite(raw)):
                    raise ValueError("End-effector feedback must be finite.")
                position = raw[[0, 6] if width == 12 else [0, 1]] / scale
            if not np.all(np.isfinite(position)):
                raise ValueError("Joint feedback must be finite.")
            value = (time.monotonic(), position.copy())
        except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
            value = None
        with self._feedback_condition:
            if generation is not self._feedback_generation:
                return
            self._feedback_values[name] = value
            if value is None:
                self._commanded_joint_position = None
                self._commanded_gripper_position = None
            self._feedback_condition.notify_all()

    def _read_feedback(self, name: str) -> np.ndarray:
        with self._lock:
            value = self._feedback_values[name]
            topic = self._feedback_topics[name]
            if value is None:
                raise RuntimeError(f"Kuavo requires valid feedback from {topic}.")
            received_at, position = value
            if time.monotonic() - received_at > self._feedback_stale_timeout_s:
                raise TimeoutError(f"Kuavo feedback from {topic} is stale.")
            return position.copy()

    def _wait_for_feedback(self) -> None:
        deadline = time.monotonic() + self._feedback_timeout_s
        with self._feedback_condition:
            while True:
                try:
                    for name in self._feedback_topics:
                        self._read_feedback(name)
                    return
                except (RuntimeError, TimeoutError) as error:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            "Kuavo did not receive valid arm and gripper feedback "
                            f"within {self._feedback_timeout_s}s: {error}"
                        ) from error
                    self._feedback_condition.wait(remaining)

    def get_state(self) -> KuavoRobotState:
        """Read both arm joints and normalized end-effector positions."""
        if self._robot is None:
            raise RuntimeError("Kuavo connection is not open.")
        joint_position = self._read_arm_positions().astype(np.float32)
        left_gripper, right_gripper = self._read_grippers()
        return KuavoRobotState(
            left_joint_position=joint_position[:_ARM_DOF],
            right_joint_position=joint_position[_ARM_DOF:],
            left_gripper=np.asarray([left_gripper], dtype=np.float32),
            right_gripper=np.asarray([right_gripper], dtype=np.float32),
        )

    def _read_arm_positions(self) -> np.ndarray:
        """Read both arm slices only while their ROS feedback is fresh."""
        return self._read_feedback("joint_q")

    def _read_grippers(self) -> tuple[float, float]:
        """Return fresh, normalized positions for both end effectors."""
        position = self._read_feedback("gripper")
        return float(position[0]), float(position[1])

    def begin_step(self) -> None:
        """Seed a coordinated action's other-side targets from fresh feedback."""
        if self._robot is None:
            raise RuntimeError("Kuavo connection is not open.")
        with self._lock:
            joints = self._read_arm_positions()
            grippers = np.asarray(self._read_grippers())
            self._commanded_joint_position = joints
            scale = 100.0 if self.end_effector_type == "leju_claw" else 255.0
            self._commanded_gripper_position = grippers * scale

    def hold_arms(self) -> np.ndarray:
        """Send one measured dual-arm target, refusing missing or stale feedback."""
        if self._robot is None:
            raise RuntimeError("Kuavo connection is not open.")
        with self._lock:
            measured = self._read_arm_positions()
            self._commanded_joint_position = None
            if np.any(measured < self._joint_limits_min) or np.any(
                measured > self._joint_limits_max
            ):
                raise ValueError(
                    "Cannot hold Kuavo arms with measured joints outside the "
                    "configured limits."
                )
            if not self._robot.control_arm_joint_positions(measured.tolist()):
                raise RuntimeError("Failed to hold both Kuavo arms.")
            self._commanded_joint_position = measured.copy()
            return measured

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
            measured = self._read_arm_positions()
            self._read_grippers()
            if self._commanded_joint_position is None:
                joint_position = measured
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
        self._read_arm_positions()
        grippers = self._read_grippers()
        if self.end_effector_type == "leju_claw":
            self._move_claw(side, value)
            return
        if self.end_effector_type == "rq2f85":
            with self._lock:
                observed = np.asarray(grippers) * 255.0
                if self._commanded_gripper_position is None:
                    current = observed
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
        positions = [value * 100.0, 100.0, *([value * 100.0] * 4)]
        result = method(
            target_positions=positions,
            target_velocities=None,
            target_torques=None,
        )
        # The SDK's asynchronous hand queue returns None on success.
        if result is not None and not result:
            raise RuntimeError(f"Failed to command the Kuavo {side} end effector.")

    def _move_claw(self, side: str, value: float) -> None:
        if not self._claw_command_lock.acquire(timeout=self._feedback_timeout_s):
            with self._lock:
                self._claw_faulted = True
            raise TimeoutError("Timed out waiting for the Kuavo claw command lock.")
        try:
            with self._lock:
                if self._claw_faulted:
                    raise RuntimeError(
                        "Kuavo claw has a service fault; close and rebuild it."
                    )
                if self._claw_service is None:
                    raise RuntimeError("Kuavo claw connection is not open.")
                generation = self._feedback_generation
                observed = np.asarray(self._read_grippers()) * 100.0
                current = (
                    observed
                    if self._commanded_gripper_position is None
                    else self._commanded_gripper_position.copy()
                )
                current[0 if side == "left" else 1] = value * 100.0
                request = self._claw_request_type()
                request.data.name = ["left_claw", "right_claw"]
                request.data.position = current.astype(float).tolist()
                request.data.velocity = [90.0, 90.0]
                request.data.effort = [1.0, 1.0]
                service = self._claw_service
            response = self._call_claw_service(service, request)
            with self._lock:
                if generation is not self._feedback_generation or self._claw_faulted:
                    raise RuntimeError(
                        "Kuavo claw connection closed or faulted during command."
                    )
                if response is None or not response.success:
                    raise RuntimeError(
                        f"Failed to command the Kuavo {side} end effector: "
                        f"{getattr(response, 'message', 'empty service response')}"
                    )
                self._commanded_gripper_position = current
        finally:
            self._claw_command_lock.release()

    def _call_claw_service(self, service: Any, request: Any) -> Any:
        """Bound acknowledgement latency without blocking sensor callbacks.

        ROS cannot cancel a request already delivered to the controller. A timed
        out call latches a fault, and reconnect waits for that call to finish.
        """
        finished = Event()
        outcome: dict[str, Any] = {}
        key = self._claw_service_key
        if key is None:
            raise RuntimeError("Kuavo claw connection is not open.")

        def call() -> None:
            try:
                outcome["response"] = service(request)
            except BaseException as error:
                outcome["error"] = error
            finally:
                with _CLAW_CALLS_LOCK:
                    if _CLAW_CALLS.get(key) is worker:
                        del _CLAW_CALLS[key]
                finished.set()

        with _CLAW_CALLS_LOCK:
            pending = _CLAW_CALLS.get(key)
            if pending is not None and pending.is_alive():
                raise RuntimeError(
                    "A previous Kuavo claw service call is still pending."
                )
            worker = Thread(target=call, daemon=True)
            _CLAW_CALLS[key] = worker
            self._claw_call_thread = worker
            worker.start()
        if not finished.wait(self._feedback_timeout_s):
            with self._lock:
                self._claw_faulted = True
            raise TimeoutError("Timed out waiting for the Kuavo claw service response.")
        if "error" in outcome:
            raise outcome["error"]
        return outcome["response"]

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
