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

"""UR e-Series position servo through ``ur_rtde``."""

import ipaddress
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.robotics.parts.base import Action, Features, Observation
from rlinf.robotics.parts.claims import DeviceClaim
from rlinf.utils.logging import get_logger

from .base import Arm, BaseArm, CartesianCompliance


@dataclass
class URRobotState:
    """UR arm state in the robot base frame, in metres and radians."""

    tcp_pose: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float32))
    """Absolute TCP pose ``[x, y, z, qx, qy, qz, qw]``."""

    tcp_vel: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    """TCP linear and angular velocity, in m/s and rad/s."""

    arm_joint_position: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )
    """Six joint positions in radians."""

    arm_joint_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )
    """Six joint velocities in rad/s."""

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass to a serializable dictionary."""
        return asdict(self)


@Arm.register("ur_rtde")
class URRTDEArm(BaseArm):
    """UR e-Series arm accepting absolute Cartesian position targets.

    A local loop holds the latest target through ``servoL``. Connecting
    uploads the SDK's control script; servoing starts with the first action.

    Args:
        robot_ip: IP address of the robot controller.
        frequency: Servo and receive frequency in Hz, at most 500.
        lookahead_time: UR position-servo lookahead in seconds, from 0.03 to 0.2.
        gain: UR position-servo proportional gain, from 100 to 2000.
    """

    SDK = "rtde_control"
    ROBOT_MODE_RUNNING: ClassVar[int] = 7
    SAFETY_MODE_NORMAL: ClassVar[int] = 1
    SAFETY_MODE_REDUCED: ClassVar[int] = 2
    JOIN_WARNING_TIMEOUT_S: ClassVar[float] = 2.0
    STATE_FIELDS: ClassVar[tuple[str, ...]] = (
        "tcp_pose",
        "tcp_vel",
        "arm_joint_position",
        "arm_joint_velocity",
    )

    def __init__(
        self,
        robot_ip: str,
        frequency: float = 500.0,
        lookahead_time: float = 0.1,
        gain: float = 300,
    ) -> None:
        try:
            ipaddress.ip_address(robot_ip)
        except ValueError:
            raise ValueError(
                f"URRTDEArm requires a robot IP address, got {robot_ip!r}."
            ) from None
        for name, value, low, high in (
            ("frequency", frequency, 0.0, 500.0),
            ("lookahead_time", lookahead_time, 0.03, 0.2),
            ("gain", gain, 100.0, 2000.0),
        ):
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"UR {name} must be in [{low}, {high}], got {value}.")
        if frequency == 0:
            raise ValueError("UR frequency must be greater than zero.")
        self._logger = get_logger()
        self._robot_ip = robot_ip
        self._frequency = float(frequency)
        self._lookahead_time = float(lookahead_time)
        self._gain = float(gain)
        self._claim = DeviceClaim(f"ur-rtde-arm:{robot_ip}", type(self).__name__)
        self._control: Any = None
        self._receive: Any = None
        self._condition = threading.Condition()
        # RTDEControlInterface is not thread-safe, including its query methods.
        self._control_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._target: Optional[np.ndarray] = None
        self._error: Optional[Exception] = None
        self._servo_active = False

    @classmethod
    def declare(
        cls,
        address: str,
        *,
        frequency: float = 500.0,
        lookahead_time: float = 0.1,
        gain: float = 300,
        compliance: CartesianCompliance | Mapping[str, float] | None = None,
        gripper_type: Optional[str] = None,
        gripper_connection: Optional[str] = None,
        end_effector_type: Optional[str] = None,
        end_effector_config: Optional[dict] = None,
        **placement: Any,
    ) -> "URRTDEArm":
        """Declare a position-servo arm with independently composed peripherals."""
        cls.refuse_unused(
            gripper_type=gripper_type,
            gripper_connection=gripper_connection,
            end_effector_type=end_effector_type,
            end_effector_config=end_effector_config,
        )
        if compliance:
            raise ValueError(
                "UR position servo does not accept Cartesian impedance settings."
            )
        return cls(
            address,
            frequency=frequency,
            lookahead_time=lookahead_time,
            gain=gain,
            **placement,
        )

    @property
    def observation_features(self) -> Features:
        """Describe the arm's measured pose, velocity and joint state."""
        return {
            name: {"shape": (7 if name == "tcp_pose" else 6,), "dtype": "float32"}
            for name in self.STATE_FIELDS
        }

    @property
    def action_features(self) -> Features:
        """Describe an absolute TCP target in metres and xyzw quaternion order."""
        return {"tcp_pose": {"shape": (7,), "dtype": "float32"}}

    def _open(self) -> Any:
        """Open receive and control sessions without commanding a target."""
        if self._control is not None or self._receive is not None:
            self._release(self._control)
        self._claim.acquire()
        try:
            from rtde_control import RTDEControlInterface
            from rtde_receive import RTDEReceiveInterface

            self._stop.clear()
            self._error = None
            self._target = None
            self._receive = RTDEReceiveInterface(self._robot_ip, self._frequency)
            self._control = RTDEControlInterface(self._robot_ip, self._frequency)
            if not self._receive.isConnected() or not self._control.isConnected():
                raise RuntimeError(
                    f"UR controller at {self._robot_ip} did not connect."
                )
        except BaseException:
            try:
                self._release(self._control)
            except Exception:
                self._logger.exception(
                    "Could not fully close the failed UR connection."
                )
            raise
        return self._control

    def _check_error(self) -> None:
        with self._condition:
            if self._error is not None:
                raise RuntimeError(
                    "UR position servo failed; disconnect and reconnect before continuing."
                ) from self._error

    def _require_receive(self) -> Any:
        self._check_error()
        if (
            self._receive is None
            or not self.is_connected
            or self._stop.is_set()
            or not self._receive.isConnected()
        ):
            raise RuntimeError(
                "UR arm is disconnected; connect it before reading or commanding it."
            )
        return self._receive

    @staticmethod
    def _vector(value: Any, length: int, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (length,) or not np.isfinite(vector).all():
            raise ValueError(
                f"UR {name} must contain {length} finite values, got {value!r}."
            )
        return vector

    def get_state(self) -> URRobotState:
        """Read measured state and convert UR rotation vectors to quaternions."""
        receive = self._require_receive()
        pose = self._vector(receive.getActualTCPPose(), 6, "TCP pose")
        state = URRobotState(
            tcp_pose=np.concatenate(
                [pose[:3], R.from_rotvec(pose[3:]).as_quat()]
            ).astype(np.float32),
            tcp_vel=self._vector(receive.getActualTCPSpeed(), 6, "TCP speed").astype(
                np.float32
            ),
            arm_joint_position=self._vector(
                receive.getActualQ(), 6, "joint position"
            ).astype(np.float32),
            arm_joint_velocity=self._vector(
                receive.getActualQd(), 6, "joint velocity"
            ).astype(np.float32),
        )
        self._check_error()
        return state

    def is_robot_up(self) -> bool:
        """Report a running control program with normal or reduced safety mode."""
        self._check_error()
        with self._control_lock:
            return bool(
                self._control is not None
                and self._receive is not None
                and not self._stop.is_set()
                and self._control.isConnected()
                and self._receive.isConnected()
                and self._control.isProgramRunning()
                and self._receive.getRobotMode() == self.ROBOT_MODE_RUNNING
                and self._receive.getSafetyMode()
                in (self.SAFETY_MODE_NORMAL, self.SAFETY_MODE_REDUCED)
            )

    def send_action(self, action: Action) -> Observation:
        """Hold one absolute TCP target using a local position-servo loop."""
        if set(action) != {"tcp_pose"}:
            raise KeyError("UR action must contain only 'tcp_pose'.")
        pose = self._vector(action["tcp_pose"], 7, "TCP target").copy()
        norm = np.linalg.norm(pose[3:])
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("UR TCP target requires a finite, nonzero quaternion.")
        pose[3:] /= norm
        with np.errstate(over="ignore"):
            accepted = pose.astype(np.float32)
        if not np.isfinite(accepted).all():
            raise ValueError("UR TCP target must be representable as float32.")
        target = np.concatenate([accepted[:3], R.from_quat(accepted[3:]).as_rotvec()])
        with self._condition:
            self._require_receive()
            self._target = target
            if self._thread is None:
                thread = threading.Thread(target=self._servo_loop, daemon=True)
                thread.start()
                self._thread = thread
            self._condition.notify_all()
        return {"tcp_pose": accepted}

    def _servo_loop(self) -> None:
        while not self._stop.is_set():
            with self._condition:
                self._condition.wait_for(
                    lambda: self._target is not None or self._stop.is_set()
                )
            with self._control_lock:
                with self._condition:
                    if self._stop.is_set():
                        return
                    target = self._target
                if target is None:
                    continue
                try:
                    if (
                        not self._control.isConnected()
                        or not self._control.isProgramRunning()
                    ):
                        raise RuntimeError(
                            "UR control program stopped while position servoing."
                        )
                    start = self._control.initPeriod()
                    self._servo_active = True
                    if not self._control.servoL(
                        target.tolist(),
                        0.0,
                        0.0,
                        1.0 / self._frequency,
                        self._lookahead_time,
                        self._gain,
                    ):
                        raise RuntimeError(
                            "UR controller rejected the position-servo target."
                        )
                    self._control.waitPeriod(start)
                except Exception as error:
                    with self._condition:
                        self._error = error
                        self._target = None
                        self._stop.set()
                    if self._servo_active:
                        try:
                            if not self._control.servoStop():
                                raise RuntimeError(
                                    "UR controller rejected the servo stop."
                                )
                            self._servo_active = False
                        except Exception:
                            self._logger.exception(
                                "Could not stop the failed UR servo."
                            )
                    return

    def reconfigure_compliance_params(self, params: Mapping[str, float]) -> None:
        """Reject Cartesian impedance settings for the position-servo backend."""
        if params:
            raise NotImplementedError(
                "UR position servo does not implement Cartesian impedance gains."
            )

    def _release(self, device: Any) -> None:
        """Stop servoing, stop the uploaded script and close both SDK clients."""
        self._stop.set()
        with self._condition:
            self._target = None
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(self.JOIN_WARNING_TIMEOUT_S)
            if self._thread.is_alive():
                self._logger.warning(
                    "Waiting for the in-flight UR control call before disconnecting."
                )
            self._thread.join()
            self._thread = None
        errors: list[tuple[str, Exception]] = []
        with self._control_lock:
            if self._control is not None:
                operations = []
                if self._servo_active:
                    operations.append(("servoStop", self._control.servoStop))
                operations.append(("stopScript", self._control.stopScript))
                for name, operation in operations:
                    try:
                        if operation() is False:
                            raise RuntimeError(f"UR controller rejected {name}.")
                    except Exception as error:
                        errors.append((name, error))
            for name in ("control", "receive"):
                client = getattr(self, f"_{name}")
                if client is None:
                    continue
                try:
                    self._disconnect_client(client, name)
                except Exception as error:
                    errors.append((f"{name} disconnect", error))
                else:
                    setattr(self, f"_{name}", None)
            if self._control is None:
                self._servo_active = False
            if self._control is None and self._receive is None:
                self._claim.release()
        if errors:
            details = "; ".join(f"{name}: {error}" for name, error in errors)
            raise RuntimeError(
                f"Could not fully close the UR connection: {details}"
            ) from errors[0][1]

    @staticmethod
    def _disconnect_client(client: Any, name: str) -> None:
        """Retry a failed SDK disconnect once before reporting unfinished cleanup."""
        for attempt in range(2):
            try:
                client.disconnect()
                if client.isConnected():
                    raise RuntimeError(f"UR {name} client is still connected.")
                return
            except Exception as error:
                if attempt == 1:
                    raise RuntimeError(
                        f"UR {name} client did not disconnect after two attempts."
                    ) from error
