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

"""ARX X5 arm driver backed by the bimanual SDK."""

import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from rlinf.robotics.parts.arms.base import Arm, BaseArm
from rlinf.robotics.parts.base import Action, Features, RobotPart
from rlinf.robotics.parts.views import MethodEndEffector
from rlinf.utils.logging import get_logger

ARX5_REST_TOLERANCE_DEG = 1.25
ARX5_PARK_MAX_DEG = 0.5
_SDK_OUTPUT_LOCK = threading.Lock()


class _SDKIKOutput:
    """Filter known IK explanations only in the calling SDK command thread."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._local = threading.local()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def _write_line(self, line: str) -> None:
        text = line.lstrip()
        # The SDK surrounds its IK explanation with blank lines and prefixes
        # the first line with a warning symbol (sometimes an emoji variant).
        if not text:
            return
        text = text.removeprefix("⚠").lstrip("\ufe0f \t")
        explanation = (
            text.startswith("set_ee_pose_xyzrpy(") and "IK 无解" in text
        ) or text.startswith(
            (
                "xyz 是**相对零位**",
                "X5-2023 的零位在工作空间边界上",
                "先 set_joint_positions 离开零位再试",
            )
        )
        if not explanation:
            self._stream.write(line)

    def write(self, text: str) -> int:
        if not hasattr(self._local, "pending"):
            return self._stream.write(text)
        lines = (self._local.pending + text).split("\n")
        self._local.pending = lines.pop()
        for line in lines:
            self._write_line(line + "\n")
        return len(text)

    def flush(self) -> None:
        if hasattr(self._local, "pending") and self._local.pending:
            self._write_line(self._local.pending)
            self._local.pending = ""
        self._stream.flush()


@contextmanager
def _quiet_sdk_ik_output() -> Iterator[None]:
    # This SDK prints from its controller; SingleArm ignores verbose=False.
    # A thread-local route avoids silencing the other arm or background logs.
    with _SDK_OUTPUT_LOCK:
        if not isinstance(sys.stdout, _SDKIKOutput):
            sys.stdout = _SDKIKOutput(sys.stdout)
        output = sys.stdout
    previous = getattr(output._local, "pending", None)
    output._local.pending = ""
    try:
        yield
    finally:
        try:
            output.flush()
        finally:
            if previous is None:
                del output._local.pending
            else:
                output._local.pending = previous


class ARX5RestMonitor:
    """Confirm a six-joint target and one second of stable feedback."""

    def __init__(
        self,
        target: np.ndarray,
        *,
        axes: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
        tolerance_deg: float = ARX5_REST_TOLERANCE_DEG,
    ) -> None:
        self.target = np.asarray(target, dtype=np.float64).copy()
        if self.target.shape != (6,) or not np.all(np.isfinite(self.target)):
            raise ValueError("ARX5 rest target must contain six finite radians.")
        if not np.isfinite(tolerance_deg) or tolerance_deg <= 0:
            raise ValueError("ARX5 rest tolerance must be finite and positive.")
        self.axes = list(axes)
        self.tolerance_rad = np.deg2rad(tolerance_deg)
        self.reset()

    def reset(self) -> None:
        self._start: float | None = None
        self._minimum: np.ndarray | None = None
        self._maximum: np.ndarray | None = None
        self._samples = 0

    def observe(self, joints: np.ndarray, now: float) -> bool:
        """Confirm all six joints are at the target throughout a stable window."""
        joints = np.asarray(joints, dtype=np.float64)
        if (
            joints.shape != (6,)
            or not np.all(np.isfinite(joints))
            or np.max(np.abs(joints[self.axes] - self.target[self.axes]))
            > self.tolerance_rad
        ):
            self.reset()
            return False
        joints = joints[self.axes]
        if self._start is not None:
            minimum = np.minimum(self._minimum, joints)
            maximum = np.maximum(self._maximum, joints)
            if np.max(maximum - minimum) > np.deg2rad(0.1):
                self.reset()
            else:
                self._minimum, self._maximum = minimum, maximum
        if self._start is None:
            self._start = now
            self._minimum = joints.copy()
            self._maximum = joints.copy()
        self._samples += 1
        return self._samples >= 3 and now - self._start >= 1.0


class ARX5ParkingMonitor(ARX5RestMonitor):
    """Require J2/J3 near mechanical zero and stable for one second.

    Other joint targets do not gate release. The installation must support
    disconnecting at this parking pose with its actual end-effector payload.
    """

    def __init__(self, maximum_deg: float = ARX5_PARK_MAX_DEG) -> None:
        super().__init__(np.zeros(6), axes=(1, 2), tolerance_deg=maximum_deg)


class ARX5IKUnreachableError(RuntimeError):
    """The SDK rejected a Cartesian target without sending a motion command."""


def gripper_endpoints(arm_type: int) -> tuple[float, float]:
    """Return open and closed motor positions, in SDK radians.

    On the 2023 gripper, increasing the motor position opens the jaws.
    The 2025 model opens at 0.1 and closes toward negative positions.
    """
    if arm_type == 0:
        return 5.0, 0.0
    if arm_type == 2:
        return 0.1, -3.14
    raise ValueError("ARX5 arm_type must be 0 (2023) or 2 (2025).")


@dataclass
class ARX5State:
    """Measured state, with TCP [xyz, qx, qy, qz, qw] relative to SDK zero pose.

    Joint and gripper positions are in radians. Currents retain SDK units.
    """

    tcp_pose: np.ndarray
    arm_joint_position: np.ndarray
    arm_joint_velocity: np.ndarray
    arm_joint_current: np.ndarray
    gripper_position: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@Arm.register("arx5")
class ARX5Arm(BaseArm):
    """One X5 arm; the gripper shares its SDK connection."""

    STATE_FIELDS = (
        "tcp_pose",
        "arm_joint_position",
        "arm_joint_velocity",
        "arm_joint_current",
    )

    def __init__(
        self,
        can_port: str,
        arm_type: int = 0,
        *,
        motion_duration: float = 0.1,
        urdf_path: str | None = None,
        park_max_deg: float = ARX5_PARK_MAX_DEG,
    ) -> None:
        self.gripper_open, self.gripper_closed = gripper_endpoints(arm_type)
        if not can_port:
            raise ValueError("ARX5 can_port must be nonempty.")
        if not np.isfinite(motion_duration) or motion_duration <= 0:
            raise ValueError("ARX5 motion_duration must be finite and positive.")
        self.can_port = can_port
        self.arm_type = arm_type
        self.motion_duration = motion_duration
        self.urdf_path = urdf_path
        if not np.isfinite(park_max_deg) or park_max_deg <= 0:
            raise ValueError("ARX5 park_max_deg must be finite and positive.")
        self.park_max_deg = park_max_deg
        self._stop_requested = False

    @property
    def parts(self) -> dict[str, RobotPart]:
        return {
            "end_effector": MethodEndEffector(
                self, "gripper_position", command="command_gripper", is_gripper=True
            )
        }

    @property
    def action_features(self) -> Features:
        return {
            "joint_position": {"shape": (6,), "dtype": "float64"},
            "tcp_pose": {"shape": (7,), "dtype": "float64"},
        }

    def _open(self) -> Any:
        self._stop_requested = False
        try:
            from bimanual import SingleArm
        except ModuleNotFoundError as exc:
            if exc.name != "bimanual":
                raise
            raise RuntimeError(
                "ARX5 SDK 'bimanual' is not importable in this process. "
                "Add the ARX5_beta directory (the parent of bimanual) to "
                "PYTHONPATH before starting the RLinf collector."
            ) from exc

        config = {"can_port": self.can_port, "type": self.arm_type}
        if self.urdf_path is not None:
            config["urdf_path"] = self.urdf_path
        return SingleArm(config)

    def _release(self, device: Any) -> None:
        # PartGroup rollback can bypass env cleanup. Check the actual parking
        # pose again before release; other joints need not match their targets.
        monitor = ARX5ParkingMonitor(self.park_max_deg)
        settled = False
        last_warning = time.monotonic()
        try:
            while not settled and not self._stop_requested:
                try:
                    self._check_ready()
                    joints = self.read_joint_positions()
                    settled = monitor.observe(joints, time.monotonic())
                except Exception:
                    monitor.reset()
                if not settled:
                    if time.monotonic() - last_warning >= 10:
                        get_logger().critical(
                            "ARX5 %s: SDK close blocked; J2/J3 parking and rest are not "
                            "confirmed. Keeping the connection open.",
                            self.can_port,
                        )
                        last_warning = time.monotonic()
                    time.sleep(0.1)
        finally:
            device.close()

    def request_stop(self) -> None:
        """Bypass parking when an interrupted collector disconnects."""
        self._stop_requested = True

    def _check_ready(self) -> None:
        if self._device is None:
            raise RuntimeError(f"ARX5 {self.can_port} is not connected.")
        if self._device.fault is not None:
            raise RuntimeError(f"ARX5 {self.can_port} fault: {self._device.fault}")

    def is_robot_up(self) -> bool:
        return self.is_connected and self._device.fault is None

    @staticmethod
    def _vector(value: Any, size: int, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (size,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"ARX5 {name} must contain {size} finite values.")
        return vector

    @staticmethod
    def _joint_vector(value: Any, name: str) -> np.ndarray:
        """Keep six arm axes when an SDK reading also includes the gripper."""
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape not in ((6,), (7,)) or not np.all(np.isfinite(vector)):
            raise ValueError(
                f"ARX5 {name} must contain six arm axes and an optional gripper axis."
            )
        return vector[:6].copy()

    def get_state(self) -> ARX5State:
        self._check_ready()
        xyzrpy = self._vector(self._device.get_ee_pose_xyzrpy(), 6, "SDK pose")
        pose = np.r_[xyzrpy[:3], Rotation.from_euler("xyz", xyzrpy[3:]).as_quat()]
        joints = self.read_joint_positions()
        velocities = self.read_joint_velocities()
        currents = self._joint_vector(
            self._device.get_joint_currents(), "joint currents"
        )
        gripper = float(self._device.get_gripper_pos())
        if not np.isfinite(gripper):
            raise ValueError("ARX5 gripper feedback must be finite.")
        return ARX5State(pose, joints, velocities, currents, gripper)

    def read_joint_positions(self) -> np.ndarray:
        """Read joint feedback without depending on TCP or gripper telemetry."""
        self._check_ready()
        return self._joint_vector(self._device.get_joint_positions(), "joint positions")

    def read_joint_velocities(self) -> np.ndarray:
        """Read measured joint velocities in radians per second."""
        self._check_ready()
        return self._joint_vector(
            self._device.get_joint_velocities(), "joint velocities"
        )

    def send_action(self, action: Action) -> dict[str, Any]:
        self._check_ready()
        if len(action) != 1:
            raise ValueError("ARX5 action must have exactly one command field.")
        if "joint_position" in action:
            self.command_joint_position(action["joint_position"], self.motion_duration)
        elif "tcp_pose" in action:
            target = self._vector(action["tcp_pose"], 7, "TCP target")
            xyzrpy = np.r_[target[:3], Rotation.from_quat(target[3:]).as_euler("xyz")]
            with _quiet_sdk_ik_output():
                accepted = self._device.set_ee_pose_xyzrpy(
                    xyzrpy.tolist(), duration=self.motion_duration
                )
            if isinstance(accepted, (bool, np.bool_)) and not accepted:
                raise ARX5IKUnreachableError(
                    f"ARX5 {self.can_port} rejected the TCP target; "
                    "the SDK may have found no IK solution."
                )
        else:
            raise KeyError(f"Unknown ARX5 command: {list(action)}")
        return dict(action)

    def command_joint_position(self, target: Any, duration: float) -> None:
        """Command a validated joint target with an explicit motion duration."""
        self._check_ready()
        joints = self._vector(target, 6, "joint target")
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("ARX5 joint duration must be finite and positive.")
        accepted = self._device.set_joint_positions(
            positions=joints.tolist(), duration=float(duration)
        )
        if isinstance(accepted, (bool, np.bool_)) and not accepted:
            raise RuntimeError(f"ARX5 {self.can_port} rejected the joint target.")

    def command_gripper(self, target: np.ndarray) -> None:
        self._check_ready()
        value = float(np.asarray(target).reshape(1)[0])
        if not np.isfinite(value):
            raise ValueError("Gripper target must be finite.")
        value = float(
            np.clip(
                value,
                min(self.gripper_open, self.gripper_closed),
                max(self.gripper_open, self.gripper_closed),
            )
        )
        self._device.set_gripper_pos(value)
