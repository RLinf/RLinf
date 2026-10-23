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

"""Robbyant R2 robot connection and part views over the RFrame middleware."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.robotics.parts.base import Connection, Features, RobotPart
from rlinf.robotics.parts.end_effectors import BaseHand
from rlinf.robotics.parts.mobility.base import MobileBase
from rlinf.robotics.parts.views import (
    MethodArm,
    MethodCamera,
    MethodEndEffector,
    host_state,
)

R2_MODE_HOLD = 0
R2_MODE_MOVEJ = 1
R2_MODE_SERVOJ = 2
R2_MODE_GRAVITY_COMP = 3

R2_LEFT_ARM_JOINTS: tuple[str, ...] = (
    "J_ArmL_1",
    "J_ArmL_2",
    "J_ArmL_3",
    "J_ArmL_4",
    "J_ArmL_5",
    "J_ArmL_6",
    "J_ArmL_7",
)
R2_RIGHT_ARM_JOINTS: tuple[str, ...] = (
    "J_ArmR_1",
    "J_ArmR_2",
    "J_ArmR_3",
    "J_ArmR_4",
    "J_ArmR_5",
    "J_ArmR_6",
    "J_ArmR_7",
)
R2_NECK_JOINTS: tuple[str, ...] = ("J_NeckYaw", "J_NeckPitch")
R2_WAIST_JOINTS_21DOF: tuple[str, ...] = (
    "J_WaistPitch_1",
    "J_WaistPitch_2",
    "J_WaistPitch_3",
    "J_WaistRoll",
    "J_WaistYaw",
)
R2_WAIST_JOINTS_20DOF: tuple[str, ...] = (
    "J_WaistLift",
    "J_WaistPitch",
    "J_WaistRoll",
    "J_WaistYaw",
)
R2_GRIPPER_JOINTS: tuple[str, ...] = ("GripperL", "GripperR")
R2_HAND_FINGER_ORDER: tuple[str, ...] = (
    "little",
    "ring",
    "middle",
    "index",
    "thumb_bend",
    "thumb_rotation",
)
R2_HAND_JOINTS_LEFT: tuple[str, ...] = (
    "HandL_Little",
    "HandL_Ring",
    "HandL_Middle",
    "HandL_Index",
    "HandL_ThumbBend",
    "HandL_ThumbRot",
)
R2_HAND_JOINTS_RIGHT: tuple[str, ...] = (
    "HandR_Little",
    "HandR_Ring",
    "HandR_Middle",
    "HandR_Index",
    "HandR_ThumbBend",
    "HandR_ThumbRot",
)

R2_CAMERA_TOPICS: dict[str, dict[str, Any]] = {
    "head": {
        "color": "/drivers/camera/zed/image_raw",
        "compressed": "/drivers/camera/zed/image_raw/compressed",
        "depth": "/drivers/camera/zed/depth",
        "resolution": (1280, 720),
    },
    "wrist_left": {
        "color": "/drivers/camera/wrist_left/color/image_raw",
        "compressed": "/drivers/camera/wrist_left/color/image_raw/compressed",
        "depth": "/drivers/camera/wrist_left/depth/image_rect_raw",
        "resolution": (640, 480),
    },
    "wrist_right": {
        "color": "/drivers/camera/wrist_right/color/image_raw",
        "compressed": "/drivers/camera/wrist_right/color/image_raw/compressed",
        "depth": "/drivers/camera/wrist_right/depth/image_rect_raw",
        "resolution": (640, 480),
    },
    "navigation_front": {
        "color": "/drivers/camera/navigation_front/color/image_raw",
        "compressed": "/drivers/camera/navigation_front/color/image_raw/compressed",
        "depth": "/drivers/camera/navigation_front/depth/image_rect_raw",
        "resolution": (640, 480),
    },
    "navigation_rear": {
        "color": "/drivers/camera/navigation_rear/color/image_raw",
        "compressed": "/drivers/camera/navigation_rear/color/image_raw/compressed",
        "depth": "/drivers/camera/navigation_rear/depth/image_rect_raw",
        "resolution": (640, 480),
    },
}


def _pose_to_xyz_rpy(pose: Any) -> np.ndarray:
    """Convert a ROS geometry_msgs/Pose into a 6-element [x, y, z, r, p, y] array."""
    pos = pose.position
    ori = pose.orientation
    quat = np.array([ori.x, ori.y, ori.z, ori.w], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        rpy = np.zeros(3, dtype=np.float64)
    else:
        rpy = R.from_quat(quat / norm).as_euler("xyz", degrees=False)
    return np.array([pos.x, pos.y, pos.z, rpy[0], rpy[1], rpy[2]], dtype=np.float32)


def _xyz_rpy_to_pose(pose_vec: np.ndarray, pose_cls: Any) -> Any:
    """Populate a ROS geometry_msgs/Pose from a 6-element [x, y, z, r, p, y] vector."""
    arr = np.asarray(pose_vec, dtype=np.float64).reshape(6)
    quat = R.from_euler("xyz", arr[3:6], degrees=False).as_quat()
    msg = pose_cls()
    msg.position.x = float(arr[0])
    msg.position.y = float(arr[1])
    msg.position.z = float(arr[2])
    msg.orientation.x = float(quat[0])
    msg.orientation.y = float(quat[1])
    msg.orientation.z = float(quat[2])
    msg.orientation.w = float(quat[3])
    return msg


def _decode_image_message(
    msg: Any,
    *,
    compressed: bool = False,
    depth: bool = False,
    fallback_resolution: tuple[int, int] = (640, 480),
) -> np.ndarray:
    """Decode a sensor_msgs/Image or CompressedImage message into a NumPy array."""
    width_default, height_default = fallback_resolution
    if compressed:
        raw = bytes(getattr(msg, "data", b""))
        if not raw:
            return np.zeros((height_default, width_default, 3), dtype=np.uint8)
        try:
            import cv2

            buf = np.frombuffer(raw, dtype=np.uint8)
            decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if decoded is not None:
                return np.ascontiguousarray(decoded[:, :, ::-1], dtype=np.uint8)
        except ImportError:
            pass
        expected = height_default * width_default * 3
        if len(raw) >= expected:
            return (
                np.frombuffer(raw[:expected], dtype=np.uint8)
                .reshape((height_default, width_default, 3))
                .copy()
            )
        return np.zeros((height_default, width_default, 3), dtype=np.uint8)

    height = int(getattr(msg, "height", 0) or height_default)
    width = int(getattr(msg, "width", 0) or width_default)
    encoding = str(getattr(msg, "encoding", "rgb8")).lower()
    raw = bytes(getattr(msg, "data", b""))

    if depth:
        if "16uc1" in encoding or "mono16" in encoding:
            count = height * width
            if len(raw) >= count * 2:
                arr = np.frombuffer(raw[: count * 2], dtype=np.uint16).reshape(
                    (height, width)
                )
                return arr.astype(np.float32) * 0.001
        count = height * width
        if len(raw) >= count * 4:
            return (
                np.frombuffer(raw[: count * 4], dtype=np.float32)
                .reshape((height, width))
                .copy()
            )
        return np.zeros((height, width), dtype=np.float32)

    channels = 4 if ("rgba" in encoding or "bgra" in encoding) else 3
    count = height * width * channels
    if len(raw) < count:
        return np.zeros((height, width, 3), dtype=np.uint8)
    frame = np.frombuffer(raw[:count], dtype=np.uint8).reshape(
        (height, width, channels)
    )
    if "bgr" in encoding:
        frame = frame[:, :, :3][:, :, ::-1]
    elif channels == 4:
        frame = frame[:, :, :3]
    return np.ascontiguousarray(frame, dtype=np.uint8)


def infer_r2_hardware(
    names: Sequence[str] = (),
    positions: Sequence[float] = (),
) -> tuple[Optional[int], Optional[str]]:
    """Infer ``(body_dof, end_effector_type)`` from a ``/Motion/JointState`` sample.

    Returns ``None`` for either field when the message does not carry enough
    information to distinguish the hardware variant.
    """
    name_set = set(names)
    body_dof: Optional[int] = None
    end_effector_type: Optional[str] = None

    if name_set:
        has_21 = any(
            j in name_set
            for j in ("J_WaistPitch_1", "J_WaistPitch_2", "J_WaistPitch_3")
        )
        has_20 = any(j in name_set for j in ("J_WaistLift", "J_WaistPitch"))
        if has_21 and not has_20:
            body_dof = 21
        elif has_20 and not has_21:
            body_dof = 20
        elif has_21:
            body_dof = 21

        has_hand = any(j.startswith(("HandL_", "HandR_")) for j in name_set)
        has_gripper = "GripperL" in name_set or "GripperR" in name_set
        if has_hand and not has_gripper:
            end_effector_type = "hand"
        elif has_gripper and not has_hand:
            end_effector_type = "gripper"
        elif has_gripper:
            end_effector_type = "gripper"
        elif any(j in name_set for j in R2_LEFT_ARM_JOINTS + R2_RIGHT_ARM_JOINTS):
            end_effector_type = "none"
    elif positions:
        n = len(positions)
        if n in (20, 22, 32):
            body_dof = 20
        elif n in (21, 23, 33):
            body_dof = 21
        if n in (32, 33):
            end_effector_type = "hand"
        elif n in (22, 23):
            end_effector_type = "gripper"
        elif n in (20, 21):
            end_effector_type = "none"

    return body_dof, end_effector_type


@dataclass
class R2RobotState:
    """Snapshot of the R2 dual-arm, torso, end-effector, and chassis state."""

    left_tcp_pose: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )
    left_tcp_vel: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )
    left_joint_positions: np.ndarray = field(
        default_factory=lambda: np.zeros(7, dtype=np.float32)
    )
    left_joint_velocities: np.ndarray = field(
        default_factory=lambda: np.zeros(7, dtype=np.float32)
    )
    left_joint_torques: np.ndarray = field(
        default_factory=lambda: np.zeros(7, dtype=np.float32)
    )
    left_wrench: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )
    left_gripper_position: np.ndarray = field(
        default_factory=lambda: np.ones(1, dtype=np.float32)
    )
    left_hand_position: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )

    right_tcp_pose: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )
    right_tcp_vel: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )
    right_joint_positions: np.ndarray = field(
        default_factory=lambda: np.zeros(7, dtype=np.float32)
    )
    right_joint_velocities: np.ndarray = field(
        default_factory=lambda: np.zeros(7, dtype=np.float32)
    )
    right_joint_torques: np.ndarray = field(
        default_factory=lambda: np.zeros(7, dtype=np.float32)
    )
    right_wrench: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )
    right_gripper_position: np.ndarray = field(
        default_factory=lambda: np.ones(1, dtype=np.float32)
    )
    right_hand_position: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32)
    )

    waist_positions: np.ndarray = field(
        default_factory=lambda: np.zeros(5, dtype=np.float32)
    )
    waist_velocities: np.ndarray = field(
        default_factory=lambda: np.zeros(5, dtype=np.float32)
    )
    neck_positions: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)
    )
    neck_velocities: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)
    )

    base_pose: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    base_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )

    imu_orientation: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    )
    imu_angular_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    imu_linear_acceleration: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )

    battery_percentage: float = 100.0
    is_healthy: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize the state snapshot into a plain dictionary."""
        return asdict(self)


class R2Arm(MethodArm):
    """Single 7-DoF R2 arm view backed by :class:`R2Connection`."""

    def __init__(self, host: Any, side: str) -> None:
        if side not in ("left", "right"):
            raise ValueError(f"Expected R2 arm side 'left' or 'right', got {side!r}.")
        self.side = side
        super().__init__(
            host,
            commands={
                "joint_position": f"move_{side}_arm_joint",
                "joint_positions": f"move_{side}_arm_joint",
                "move_joint": f"move_{side}_arm_joint",
                "tcp_pose": f"move_{side}_arm_cartesian",
                "move_cartesian": f"move_{side}_arm_cartesian",
                "move_Cartesian": f"move_{side}_arm_cartesian",
                "tcp_velocity": f"move_{side}_arm_velocity",
                "move_cartesian_velocity": f"move_{side}_arm_velocity",
                "joint_velocity": f"move_{side}_arm_joint_velocity",
                "move_joint_velocity": f"move_{side}_arm_joint_velocity",
                "joint_torque": f"apply_{side}_arm_torque",
                "apply_torque": f"apply_{side}_arm_torque",
            },
            state_fields={
                "tcp_pose": f"{side}_tcp_pose",
                "tcp_vel": f"{side}_tcp_vel",
                "joint_position": f"{side}_joint_positions",
                "joint_positions": f"{side}_joint_positions",
                "joint_velocity": f"{side}_joint_velocities",
                "joint_velocities": f"{side}_joint_velocities",
                "joint_torque": f"{side}_joint_torques",
                "joint_torques": f"{side}_joint_torques",
                "wrench": f"{side}_wrench",
            },
        )
        self.num_joints = 7

    @property
    def observation_features(self) -> Features:
        """Declare shapes and dtypes of the 7-DoF arm state fields."""
        return {
            "tcp_pose": {"shape": (6,), "dtype": "float32"},
            "tcp_vel": {"shape": (6,), "dtype": "float32"},
            "joint_position": {"shape": (7,), "dtype": "float32"},
            "joint_positions": {"shape": (7,), "dtype": "float32"},
            "joint_velocity": {"shape": (7,), "dtype": "float32"},
            "joint_velocities": {"shape": (7,), "dtype": "float32"},
            "joint_torque": {"shape": (7,), "dtype": "float32"},
            "joint_torques": {"shape": (7,), "dtype": "float32"},
            "wrench": {"shape": (6,), "dtype": "float32"},
        }

    def reset_joint(
        self,
        positions: Sequence[float] | np.ndarray,
        duration: float = 3.0,
        *,
        max_velocity: Optional[float | Sequence[float] | np.ndarray] = None,
    ) -> bool:
        """Move the arm smoothly to ``positions`` using the MoveJ service."""
        fn = getattr(self._host, f"move_{self.side}_arm_movej")
        return bool(fn(positions, duration=duration, max_velocity=max_velocity))

    def clear_errors(self) -> bool:
        """Clear motion controller faults on the shared R2 connection."""
        return bool(self._host.clear_errors())


class R2HandView(MethodEndEffector, BaseHand):
    """Six-DoF R2 dexterous hand view backed by :class:`R2Connection`."""

    FINGER_NAMES = R2_HAND_FINGER_ORDER

    def __init__(self, host: Any, side: str) -> None:
        if side not in ("left", "right"):
            raise ValueError(f"Expected R2 hand side 'left' or 'right', got {side!r}.")
        self.side = side
        super().__init__(
            host,
            f"{side}_hand_position",
            dims=len(R2_HAND_FINGER_ORDER),
            command=f"move_{side}_hand",
            is_gripper=False,
        )


class R2MobileBase(MobileBase):
    """Omnidirectional wheeled chassis view backed by :class:`R2Connection`."""

    def __init__(self, host: Any) -> None:
        self._host = self._owner = host

    @property
    def observation_features(self) -> dict[str, dict[str, Any]]:
        """planar pose ``[x, y, yaw]`` and velocity ``[vx, vy, wz]``."""
        return {
            "base_pose": {"shape": (3,), "dtype": "float32"},
            "base_velocity": {"shape": (3,), "dtype": "float32"},
        }

    @property
    def action_features(self) -> dict[str, dict[str, Any]]:
        """Planar chassis velocity command ``[vx, vy, wz]``."""
        return {
            "base_velocity": {"shape": (3,), "dtype": "float32"},
        }

    def get_observation(self) -> dict[str, np.ndarray]:
        """Return the latest planar pose and velocity from the host state."""
        state = host_state(self._host)
        return {
            "base_pose": np.asarray(state["base_pose"], dtype=np.float32).reshape(3),
            "base_velocity": np.asarray(
                state["base_velocity"], dtype=np.float32
            ).reshape(3),
        }

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Publish a planar chassis velocity command ``[vx, vy, wz]``."""
        if set(action) != {"base_velocity"}:
            raise ValueError(
                f"R2MobileBase expects action key 'base_velocity', got {sorted(action)}."
            )
        vel = np.asarray(action["base_velocity"], dtype=np.float32).reshape(3)
        self._host.move_base(vel)
        return {"base_velocity": vel}

    def get_pose(self) -> np.ndarray:
        """Return the current planar pose ``[x, y, yaw]``."""
        return self.get_observation()["base_pose"]

    def get_velocity(self) -> np.ndarray:
        """Return the current planar velocity ``[vx, vy, wz]``."""
        return self.get_observation()["base_velocity"]

    def move_velocity(self, vx: float, vy: float, wz: float) -> None:
        """Command planar chassis velocity ``(vx, vy, wz)``."""
        self._host.move_base(np.array([vx, vy, wz], dtype=np.float32))

    def navigate_to(
        self, x: float, y: float, yaw: float = 0.0, timeout: float = 30.0
    ) -> bool:
        """Dispatch a waypoint navigation goal to the R2 navigation stack."""
        return bool(self._host.navigate_to(x, y, yaw=yaw, timeout=timeout))

    def stop(self) -> None:
        """Zero chassis velocity and stop active navigation."""
        self._host.stop_base()


class R2Camera(MethodCamera):
    """Camera view over an RFrame image topic on :class:`R2Connection`."""

    def __init__(
        self,
        host: Any,
        camera_name: str,
        resolution: tuple[int, int] = (640, 480),
        *,
        enable_depth: bool = False,
    ) -> None:
        super().__init__(host, "get_camera", camera_name, check_method="check_camera")
        self.name = camera_name
        self.camera_name = camera_name
        self.resolution = resolution
        self.enable_depth = enable_depth

    @property
    def observation_features(self) -> dict[str, dict[str, Any]]:
        """Declare RGB frame and optional depth image shapes."""
        width, height = self.resolution
        features: dict[str, dict[str, Any]] = {
            "frame": {"shape": (height, width, 3), "dtype": "uint8"},
        }
        if self.enable_depth:
            features["depth"] = {"shape": (height, width), "dtype": "float32"}
        return features

    def get_observation(self) -> dict[str, Any]:
        """Return the latest RGB frame and optional depth image."""
        if not self.enable_depth:
            return {"frame": self._host.get_camera(self.camera_name)}
        return self._host.get_camera_observation(self.camera_name)

    def get_frame(self, **_kwargs: Any) -> np.ndarray:
        """Return the latest RGB image array."""
        return self._host.get_camera(self.camera_name)

    def get_depth(self) -> Optional[np.ndarray]:
        """Return the latest depth image in meters when depth is enabled."""
        return self._host.get_camera_depth(self.camera_name)

    def is_ready(self, timeout: float = 0.5) -> bool:
        """Return True once the underlying camera topic has delivered a frame."""
        return bool(self._host.check_camera(self.camera_name, timeout))

    def reopen(self) -> None:
        """No-op because camera subscriptions share the RFrame session."""
        return None


class R2Connection(Connection):
    """Shared RFrame session managing R2 arms, end-effectors, chassis, and cameras."""

    def __init__(
        self,
        *,
        body_dof: Optional[int | str] = None,
        end_effector_type: Optional[str] = "gripper",
        enable_base: bool = False,
        cameras: Sequence[str] = ("head", "wrist_left", "wrist_right"),
        enable_depth: bool = False,
        use_compressed_camera: bool = False,
        enable_shm: bool = False,
        controller_mode: int = R2_MODE_SERVOJ,
        node_name: str = "rlinf_r2_controller",
    ) -> None:
        if body_dof in (None, "auto"):
            self._auto_body_dof = True
            resolved_body_dof = 21
        elif body_dof in (20, 21, "20", "21"):
            self._auto_body_dof = False
            resolved_body_dof = int(body_dof)
        else:
            raise ValueError(
                f"R2 body_dof must be 20, 21, or 'auto'/None, got {body_dof!r}."
            )

        if end_effector_type in (None, "auto"):
            self._auto_end_effector_type = True
            resolved_ee_type = "gripper"
        elif end_effector_type in ("gripper", "hand", "none"):
            self._auto_end_effector_type = False
            resolved_ee_type = str(end_effector_type)
        else:
            raise ValueError(
                "R2 end_effector_type must be 'gripper', 'hand', 'none', or 'auto'/None, "
                f"got {end_effector_type!r}."
            )

        if controller_mode not in (
            R2_MODE_HOLD,
            R2_MODE_MOVEJ,
            R2_MODE_SERVOJ,
            R2_MODE_GRAVITY_COMP,
        ):
            raise ValueError(
                f"R2 controller_mode must be one of 0..3, got {controller_mode!r}."
            )
        unknown_cameras = [c for c in cameras if c not in R2_CAMERA_TOPICS]
        if unknown_cameras:
            raise ValueError(
                f"Unknown R2 camera names {unknown_cameras}; "
                f"valid names are {sorted(R2_CAMERA_TOPICS)}."
            )

        self.body_dof = resolved_body_dof
        self.end_effector_type = resolved_ee_type
        self.detected_body_dof: Optional[int] = None
        self.detected_end_effector_type: Optional[str] = None
        self.enable_base = enable_base
        self.camera_names = tuple(cameras)
        self.enable_depth = enable_depth
        self.use_compressed_camera = use_compressed_camera
        self.enable_shm = enable_shm
        self.controller_mode = controller_mode
        self.node_name = node_name

        self.waist_joint_names = (
            R2_WAIST_JOINTS_21DOF if resolved_body_dof == 21 else R2_WAIST_JOINTS_20DOF
        )
        self.body_joint_names: tuple[str, ...] = (
            self.waist_joint_names
            + R2_NECK_JOINTS
            + R2_LEFT_ARM_JOINTS
            + R2_RIGHT_ARM_JOINTS
        )

        self._lock = threading.Lock()
        self._state = R2RobotState(
            waist_positions=np.zeros(len(self.waist_joint_names), dtype=np.float32),
            waist_velocities=np.zeros(len(self.waist_joint_names), dtype=np.float32),
        )
        self._joint_map_positions: dict[str, float] = {}
        self._joint_map_velocities: dict[str, float] = {}
        self._joint_map_efforts: dict[str, float] = {}
        self._state_ready = threading.Event()

        self._camera_frames: dict[str, np.ndarray] = {}
        self._camera_depths: dict[str, np.ndarray] = {}
        self._camera_events: dict[str, threading.Event] = {
            name: threading.Event() for name in self.camera_names
        }

        self._session: Any = None
        self._readers: list[Any] = []
        self._writers: dict[str, Any] = {}
        self._services: dict[str, Any] = {}
        self._msg_types: dict[str, Any] = {}

    def _set_body_dof(self, body_dof: int) -> None:
        """Update the waist and body joint tables to match ``body_dof`` (20 or 21)."""
        self.body_dof = body_dof
        self.waist_joint_names = (
            R2_WAIST_JOINTS_21DOF if body_dof == 21 else R2_WAIST_JOINTS_20DOF
        )
        self.body_joint_names = (
            self.waist_joint_names
            + R2_NECK_JOINTS
            + R2_LEFT_ARM_JOINTS
            + R2_RIGHT_ARM_JOINTS
        )
        target_len = len(self.waist_joint_names)
        if self._state.waist_positions.size != target_len:
            self._state.waist_positions = np.zeros(target_len, dtype=np.float32)
            self._state.waist_velocities = np.zeros(target_len, dtype=np.float32)

    @classmethod
    def probe_hardware(
        cls,
        *,
        timeout: float = 1.0,
        enable_shm: bool = False,
        default_body_dof: int = 21,
        default_end_effector_type: str = "gripper",
        node_name: str = "rlinf_r2_probe",
    ) -> tuple[int, str]:
        """Probe ``/Motion/JointState`` over RFrame to detect ``(body_dof, end_effector_type)``.

        Opens a temporary RFrame reader on ``/Motion/JointState``, inspects the
        first message received within ``timeout`` seconds, and falls back to
        ``(default_body_dof, default_end_effector_type)`` when no message arrives
        or the RFrame SDK is unavailable.
        """
        try:
            import rframe
            from sensor_msgs.msg import JointState
        except ImportError:
            return default_body_dof, default_end_effector_type

        detected_dof: Optional[int] = None
        detected_ee: Optional[str] = None
        ready = threading.Event()

        def _on_msg(msg: Any) -> None:
            nonlocal detected_dof, detected_ee
            names = list(getattr(msg, "name", []))
            positions = list(getattr(msg, "position", []))
            dof, ee = infer_r2_hardware(names, positions)
            if dof is not None:
                detected_dof = dof
            if ee is not None:
                detected_ee = ee
            ready.set()

        session = (
            rframe.Session(node_name) if hasattr(rframe, "Session") else rframe.App()
        )
        if not hasattr(rframe, "Session") and hasattr(session, "init"):
            session.init()
        channel: Any = None
        try:
            for topic in ("/Motion/JointState", "/robotic/joint_states"):
                cfg = rframe.ChannelConfig()
                if hasattr(cfg, "topic_name"):
                    cfg.topic_name = topic
                if hasattr(cfg, "topic"):
                    cfg.topic = topic
                cfg.channel_type = getattr(
                    rframe.ChannelType,
                    "READER",
                    getattr(rframe.ChannelType, "HYBRID", 0),
                )
                if hasattr(rframe, "HandlerType") and hasattr(cfg, "handler_type"):
                    cfg.handler_type = rframe.HandlerType.READER
                if hasattr(cfg, "enable_shm"):
                    cfg.enable_shm = enable_shm
                channel = (
                    session.create_hybrid_channel(cfg)
                    if hasattr(session, "create_hybrid_channel")
                    else rframe.HybridChannel(cfg)
                )
                channel.register_typed_callback(JointState, _on_msg)
                if ready.is_set():
                    break
            ready.wait(timeout=timeout)
        finally:
            if channel is not None and hasattr(channel, "close"):
                channel.close()
            if hasattr(session, "shutdown"):
                session.shutdown()
            elif hasattr(session, "close"):
                session.close()

        return (
            detected_dof if detected_dof in (20, 21) else default_body_dof,
            detected_ee
            if detected_ee in ("gripper", "hand", "none")
            else default_end_effector_type,
        )

    @property
    def parts(self) -> dict[str, RobotPart]:
        """Decompose the shared R2 connection into arms, end effectors, base, and cameras."""
        parts: dict[str, RobotPart] = {
            "left": R2Arm(self, "left"),
            "right": R2Arm(self, "right"),
        }
        if self.end_effector_type == "gripper":
            parts["left_end_effector"] = MethodEndEffector(
                self,
                "left_gripper_position",
                dims=1,
                command="move_left_gripper",
                is_gripper=True,
            )
            parts["right_end_effector"] = MethodEndEffector(
                self,
                "right_gripper_position",
                dims=1,
                command="move_right_gripper",
                is_gripper=True,
            )
        elif self.end_effector_type == "hand":
            parts["left_end_effector"] = R2HandView(self, "left")
            parts["right_end_effector"] = R2HandView(self, "right")
        if self.enable_base:
            parts["base"] = R2MobileBase(self)
        for name in self.camera_names:
            parts[name] = R2Camera(
                self,
                name,
                resolution=R2_CAMERA_TOPICS[name]["resolution"],
                enable_depth=self.enable_depth,
            )
        return parts

    def _create_channel(self, config: Any) -> Any:
        """Create a HybridChannel from ``self._session`` or ``rframe``."""
        rframe = self._msg_types["rframe"]
        if hasattr(self._session, "create_hybrid_channel"):
            return self._session.create_hybrid_channel(config)
        return rframe.HybridChannel(self._session, config)

    def _subscribe(self, topic: str, msg_cls: Any, callback: Any) -> Any:
        """Register a typed reader channel on ``topic``."""
        rframe = self._msg_types["rframe"]
        cfg = rframe.ChannelConfig()
        cfg.topic_name = topic
        cfg.channel_type = rframe.ChannelType.READER
        if hasattr(cfg, "enable_shm"):
            cfg.enable_shm = self.enable_shm
        channel = self._create_channel(cfg)
        channel.register_typed_callback(msg_cls, callback)
        self._readers.append(channel)
        return channel

    def _advertise(self, topic: str, msg_cls: Any) -> Any:
        """Create a writer channel on ``topic``."""
        rframe = self._msg_types["rframe"]
        cfg = rframe.ChannelConfig()
        cfg.topic_name = topic
        cfg.channel_type = rframe.ChannelType.WRITER
        if hasattr(cfg, "enable_shm"):
            cfg.enable_shm = self.enable_shm
        channel = self._create_channel(cfg)
        self._writers[topic] = (channel, msg_cls)
        return channel

    def _client(self, service_name: str, srv_cls: Any) -> Any:
        """Create an RPC client channel on ``service_name``."""
        rframe = self._msg_types["rframe"]
        cfg = rframe.ChannelConfig()
        cfg.topic_name = service_name
        cfg.channel_type = getattr(
            rframe.ChannelType, "RPC_CLIENT", getattr(rframe.ChannelType, "CLIENT", 2)
        )
        channel = self._create_channel(cfg)
        self._services[service_name] = (channel, srv_cls)
        return channel

    def _open(self) -> Any:
        """Initialize the RFrame session, topic subscriptions, writers, and services."""
        import rframe
        from geometry_msgs.msg import Pose, Twist, WrenchStamped
        from navigation_msgs.msg import NaviState
        from navigation_msgs.srv import NavigationControl
        from r2_motion_msgs.msg import TCP
        from r2_motion_msgs.srv import (
            ForwardKine,
            InverseKine,
            MoveC,
            MoveJ,
            SetLoad,
            SwitchController,
        )
        from rdiagnostic_msgs.msg import HealthState
        from sensor_msgs.msg import (
            BatteryState,
            CompressedImage,
            Image,
            Imu,
            JointState,
        )
        from state_machine_msgs.srv import SwitchMode
        from std_srvs.srv import Trigger
        from task_msgs.srv import AssignTask

        self._msg_types = {
            "rframe": rframe,
            "Pose": Pose,
            "Twist": Twist,
            "WrenchStamped": WrenchStamped,
            "NaviState": NaviState,
            "NavigationControl": NavigationControl,
            "TCP": TCP,
            "ForwardKine": ForwardKine,
            "InverseKine": InverseKine,
            "MoveC": MoveC,
            "MoveJ": MoveJ,
            "SetLoad": SetLoad,
            "SwitchController": SwitchController,
            "HealthState": HealthState,
            "BatteryState": BatteryState,
            "CompressedImage": CompressedImage,
            "Image": Image,
            "Imu": Imu,
            "JointState": JointState,
            "SwitchMode": SwitchMode,
            "Trigger": Trigger,
            "AssignTask": AssignTask,
        }

        if hasattr(rframe, "init"):
            rframe.init()
        self._session = rframe.Session(self.node_name)
        self._readers.clear()
        self._writers.clear()
        self._services.clear()

        self._subscribe("/Motion/JointState", JointState, self._on_joint_state)
        self._subscribe("/Motion/TCP", TCP, self._on_tcp)
        self._subscribe("/drivers/ftsensor/fts_l", WrenchStamped, self._on_left_wrench)
        self._subscribe("/drivers/ftsensor/fts_r", WrenchStamped, self._on_right_wrench)
        self._subscribe("/drivers/imu/body_imu", Imu, self._on_imu)
        self._subscribe("/drivers/bms/battery_state", BatteryState, self._on_battery)
        self._subscribe(
            "/safety/health_monitor/state", HealthState, self._on_health_state
        )

        if self.enable_base:
            self._subscribe("/navigation/state", NaviState, self._on_navi_state)

        for cam_name in self.camera_names:
            spec = R2_CAMERA_TOPICS[cam_name]
            resolution = spec["resolution"]
            if self.use_compressed_camera:
                self._subscribe(
                    spec["compressed"],
                    CompressedImage,
                    self._make_camera_callback(
                        cam_name, compressed=True, depth=False, resolution=resolution
                    ),
                )
            else:
                self._subscribe(
                    spec["color"],
                    Image,
                    self._make_camera_callback(
                        cam_name, compressed=False, depth=False, resolution=resolution
                    ),
                )
            if self.enable_depth:
                self._subscribe(
                    spec["depth"],
                    Image,
                    self._make_camera_callback(
                        cam_name, compressed=False, depth=True, resolution=resolution
                    ),
                )

        self._advertise("/Motion/ServoJ", JointState)
        if self.enable_base:
            self._advertise("/teleop/chassis_Twist", Twist)

        self._client("/Motion/SwitchController", SwitchController)
        self._client("/Motion/MoveJ", MoveJ)
        self._client("/Motion/MoveC", MoveC)
        self._client("/Motion/ClearError", Trigger)
        self._client("/Motion/SetLoad", SetLoad)
        self._client("/Motion/ForwardKine", ForwardKine)
        self._client("/Motion/InverseKine", InverseKine)
        self._client("/state_machine/switch_mode", SwitchMode)
        if self.enable_base:
            self._client("/task_manager/assign_task", AssignTask)
            self._client("/navigation/control", NavigationControl)

        self._state_ready.wait(timeout=1.0)
        self.switch_controller(self.controller_mode)
        return self._session

    def _release(self, device: Any) -> None:
        """Close all channels and shut down the RFrame session."""
        if self.enable_base and "/teleop/chassis_Twist" in self._writers:
            try:
                self.stop_base()
            except Exception:
                pass
        for channel in self._readers:
            if hasattr(channel, "close"):
                channel.close()
        for channel, _ in self._writers.values():
            if hasattr(channel, "close"):
                channel.close()
        for channel, _ in self._services.values():
            if hasattr(channel, "close"):
                channel.close()
        self._readers.clear()
        self._writers.clear()
        self._services.clear()
        if hasattr(device, "close"):
            device.close()
        self._session = None

    def _make_camera_callback(
        self,
        camera_name: str,
        *,
        compressed: bool,
        depth: bool,
        resolution: tuple[int, int],
    ) -> Any:
        def _callback(msg: Any) -> None:
            decoded = _decode_image_message(
                msg,
                compressed=compressed,
                depth=depth,
                fallback_resolution=resolution,
            )
            with self._lock:
                if depth:
                    self._camera_depths[camera_name] = decoded
                else:
                    self._camera_frames[camera_name] = decoded
                    self._camera_events[camera_name].set()

        return _callback

    def _on_joint_state(self, msg: Any) -> None:
        names = list(getattr(msg, "name", []))
        positions = list(getattr(msg, "position", []))
        velocities = list(getattr(msg, "velocity", []))
        efforts = list(getattr(msg, "effort", []))

        detected_dof, detected_ee = infer_r2_hardware(names, positions)

        with self._lock:
            if detected_dof is not None:
                self.detected_body_dof = detected_dof
                if self._auto_body_dof and detected_dof != self.body_dof:
                    self._set_body_dof(detected_dof)
            if detected_ee is not None:
                self.detected_end_effector_type = detected_ee
                if self._auto_end_effector_type:
                    self.end_effector_type = detected_ee

            for idx, name in enumerate(names):
                if idx < len(positions):
                    self._joint_map_positions[name] = float(positions[idx])
                if idx < len(velocities):
                    self._joint_map_velocities[name] = float(velocities[idx])
                if idx < len(efforts):
                    self._joint_map_efforts[name] = float(efforts[idx])

            if names:
                self._state.left_joint_positions = np.array(
                    [self._joint_map_positions.get(j, 0.0) for j in R2_LEFT_ARM_JOINTS],
                    dtype=np.float32,
                )
                self._state.left_joint_velocities = np.array(
                    [
                        self._joint_map_velocities.get(j, 0.0)
                        for j in R2_LEFT_ARM_JOINTS
                    ],
                    dtype=np.float32,
                )
                self._state.left_joint_torques = np.array(
                    [self._joint_map_efforts.get(j, 0.0) for j in R2_LEFT_ARM_JOINTS],
                    dtype=np.float32,
                )
                self._state.right_joint_positions = np.array(
                    [
                        self._joint_map_positions.get(j, 0.0)
                        for j in R2_RIGHT_ARM_JOINTS
                    ],
                    dtype=np.float32,
                )
                self._state.right_joint_velocities = np.array(
                    [
                        self._joint_map_velocities.get(j, 0.0)
                        for j in R2_RIGHT_ARM_JOINTS
                    ],
                    dtype=np.float32,
                )
                self._state.right_joint_torques = np.array(
                    [self._joint_map_efforts.get(j, 0.0) for j in R2_RIGHT_ARM_JOINTS],
                    dtype=np.float32,
                )
                self._state.waist_positions = np.array(
                    [
                        self._joint_map_positions.get(j, 0.0)
                        for j in self.waist_joint_names
                    ],
                    dtype=np.float32,
                )
                self._state.waist_velocities = np.array(
                    [
                        self._joint_map_velocities.get(j, 0.0)
                        for j in self.waist_joint_names
                    ],
                    dtype=np.float32,
                )
                self._state.neck_positions = np.array(
                    [self._joint_map_positions.get(j, 0.0) for j in R2_NECK_JOINTS],
                    dtype=np.float32,
                )
                self._state.neck_velocities = np.array(
                    [self._joint_map_velocities.get(j, 0.0) for j in R2_NECK_JOINTS],
                    dtype=np.float32,
                )
                if "GripperL" in self._joint_map_positions:
                    self._state.left_gripper_position = np.array(
                        [self._joint_map_positions["GripperL"]], dtype=np.float32
                    )
                if "GripperR" in self._joint_map_positions:
                    self._state.right_gripper_position = np.array(
                        [self._joint_map_positions["GripperR"]], dtype=np.float32
                    )
                if any(j in self._joint_map_positions for j in R2_HAND_JOINTS_LEFT):
                    self._state.left_hand_position = np.array(
                        [
                            self._joint_map_positions.get(j, 0.0)
                            for j in R2_HAND_JOINTS_LEFT
                        ],
                        dtype=np.float32,
                    )
                if any(j in self._joint_map_positions for j in R2_HAND_JOINTS_RIGHT):
                    self._state.right_hand_position = np.array(
                        [
                            self._joint_map_positions.get(j, 0.0)
                            for j in R2_HAND_JOINTS_RIGHT
                        ],
                        dtype=np.float32,
                    )
            elif len(positions) in (20, 21):
                arr = np.asarray(positions, dtype=np.float32)
                w = len(positions) - 16
                self._state.waist_positions = arr[:w].copy()
                self._state.neck_positions = arr[w : w + 2].copy()
                self._state.left_joint_positions = arr[w + 2 : w + 9].copy()
                self._state.right_joint_positions = arr[w + 9 : w + 16].copy()
            elif len(positions) >= 14:
                arr = np.asarray(positions, dtype=np.float32)
                self._state.left_joint_positions = arr[:7].copy()
                self._state.right_joint_positions = arr[7:14].copy()

            self._state_ready.set()

    def _on_tcp(self, msg: Any) -> None:
        poses = list(getattr(msg, "tcp_pose", []))
        twists = list(getattr(msg, "tcp_twist", []))
        with self._lock:
            if hasattr(msg, "pose_l"):
                self._state.left_tcp_pose = _pose_to_xyz_rpy(msg.pose_l)
            elif len(poses) >= 1:
                self._state.left_tcp_pose = _pose_to_xyz_rpy(poses[0])
            if hasattr(msg, "pose_r"):
                self._state.right_tcp_pose = _pose_to_xyz_rpy(msg.pose_r)
            elif len(poses) >= 2:
                self._state.right_tcp_pose = _pose_to_xyz_rpy(poses[1])
            if len(twists) >= 1:
                t0 = twists[0]
                self._state.left_tcp_vel = np.array(
                    [
                        t0.linear.x,
                        t0.linear.y,
                        t0.linear.z,
                        t0.angular.x,
                        t0.angular.y,
                        t0.angular.z,
                    ],
                    dtype=np.float32,
                )
            if len(twists) >= 2:
                t1 = twists[1]
                self._state.right_tcp_vel = np.array(
                    [
                        t1.linear.x,
                        t1.linear.y,
                        t1.linear.z,
                        t1.angular.x,
                        t1.angular.y,
                        t1.angular.z,
                    ],
                    dtype=np.float32,
                )

    def _on_left_wrench(self, msg: Any) -> None:
        w = getattr(msg, "wrench", msg)
        with self._lock:
            self._state.left_wrench = np.array(
                [
                    w.force.x,
                    w.force.y,
                    w.force.z,
                    w.torque.x,
                    w.torque.y,
                    w.torque.z,
                ],
                dtype=np.float32,
            )

    def _on_right_wrench(self, msg: Any) -> None:
        w = getattr(msg, "wrench", msg)
        with self._lock:
            self._state.right_wrench = np.array(
                [
                    w.force.x,
                    w.force.y,
                    w.force.z,
                    w.torque.x,
                    w.torque.y,
                    w.torque.z,
                ],
                dtype=np.float32,
            )

    def _on_imu(self, msg: Any) -> None:
        with self._lock:
            ori = msg.orientation
            ang = msg.angular_velocity
            acc = msg.linear_acceleration
            self._state.imu_orientation = np.array(
                [ori.x, ori.y, ori.z, ori.w], dtype=np.float32
            )
            self._state.imu_angular_velocity = np.array(
                [ang.x, ang.y, ang.z], dtype=np.float32
            )
            self._state.imu_linear_acceleration = np.array(
                [acc.x, acc.y, acc.z], dtype=np.float32
            )

    def _on_battery(self, msg: Any) -> None:
        with self._lock:
            self._state.battery_percentage = float(getattr(msg, "percentage", 100.0))

    def _on_health_state(self, msg: Any) -> None:
        with self._lock:
            if hasattr(msg, "is_healthy"):
                self._state.is_healthy = bool(msg.is_healthy)
            elif hasattr(msg, "fault_level"):
                self._state.is_healthy = int(msg.fault_level) == 0

    def _on_navi_state(self, msg: Any) -> None:
        with self._lock:
            if hasattr(msg, "pose"):
                pose_vec = _pose_to_xyz_rpy(msg.pose)
                self._state.base_pose = np.array(
                    [pose_vec[0], pose_vec[1], pose_vec[5]], dtype=np.float32
                )
            elif hasattr(msg, "x"):
                self._state.base_pose = np.array(
                    [
                        float(getattr(msg, "x", 0.0)),
                        float(getattr(msg, "y", 0.0)),
                        float(getattr(msg, "yaw", 0.0)),
                    ],
                    dtype=np.float32,
                )
            if hasattr(msg, "velocity"):
                vel = msg.velocity
                self._state.base_velocity = np.array(
                    [vel.linear.x, vel.linear.y, vel.angular.z], dtype=np.float32
                )

    def _require_connected(self) -> Any:
        """Ensure the RFrame session is active before reading or commanding."""
        if self._session is None:
            raise RuntimeError("R2Connection is not connected. Call connect() first.")
        return self._session

    def get_state(self) -> R2RobotState:
        """Return a thread-safe copy of the latest :class:`R2RobotState`."""
        self._require_connected()
        with self._lock:
            return R2RobotState(
                left_tcp_pose=self._state.left_tcp_pose.copy(),
                left_tcp_vel=self._state.left_tcp_vel.copy(),
                left_joint_positions=self._state.left_joint_positions.copy(),
                left_joint_velocities=self._state.left_joint_velocities.copy(),
                left_joint_torques=self._state.left_joint_torques.copy(),
                left_wrench=self._state.left_wrench.copy(),
                left_gripper_position=self._state.left_gripper_position.copy(),
                left_hand_position=self._state.left_hand_position.copy(),
                right_tcp_pose=self._state.right_tcp_pose.copy(),
                right_tcp_vel=self._state.right_tcp_vel.copy(),
                right_joint_positions=self._state.right_joint_positions.copy(),
                right_joint_velocities=self._state.right_joint_velocities.copy(),
                right_joint_torques=self._state.right_joint_torques.copy(),
                right_wrench=self._state.right_wrench.copy(),
                right_gripper_position=self._state.right_gripper_position.copy(),
                right_hand_position=self._state.right_hand_position.copy(),
                waist_positions=self._state.waist_positions.copy(),
                waist_velocities=self._state.waist_velocities.copy(),
                neck_positions=self._state.neck_positions.copy(),
                neck_velocities=self._state.neck_velocities.copy(),
                base_pose=self._state.base_pose.copy(),
                base_velocity=self._state.base_velocity.copy(),
                imu_orientation=self._state.imu_orientation.copy(),
                imu_angular_velocity=self._state.imu_angular_velocity.copy(),
                imu_linear_acceleration=self._state.imu_linear_acceleration.copy(),
                battery_percentage=self._state.battery_percentage,
                is_healthy=self._state.is_healthy,
            )

    def read_state(self) -> dict[str, Any]:
        """Return the latest state snapshot as a plain dictionary for part views."""
        return self.get_state().to_dict()

    def _publish_servoj(
        self,
        joint_names: Sequence[str],
        positions: Sequence[float],
        velocities: Optional[Sequence[float]] = None,
        efforts: Optional[Sequence[float]] = None,
    ) -> None:
        self._require_connected()
        channel, msg_cls = self._writers["/Motion/ServoJ"]
        msg = msg_cls()
        msg.name = list(joint_names)
        msg.position = [float(x) for x in positions]
        msg.velocity = [float(x) for x in velocities] if velocities is not None else []
        msg.effort = [float(x) for x in efforts] if efforts is not None else []
        channel.send(msg)

    def _call_service(
        self, service_name: str, configure_request: Any, timeout: float = 5.0
    ) -> Any:
        self._require_connected()
        if service_name not in self._services:
            raise RuntimeError(f"R2 service {service_name!r} is not initialized.")
        channel, srv_cls = self._services[service_name]
        req_cls = getattr(srv_cls, "Request", srv_cls)
        req = req_cls()
        configure_request(req)
        if hasattr(channel, "call"):
            return channel.call(req, timeout=timeout)
        return None

    def switch_controller(self, mode: int) -> bool:
        """Switch the R2 motion controller mode (0: Hold, 1: MoveJ, 2: ServoJ, 3: GravityComp)."""
        if mode not in (
            R2_MODE_HOLD,
            R2_MODE_MOVEJ,
            R2_MODE_SERVOJ,
            R2_MODE_GRAVITY_COMP,
        ):
            raise ValueError(f"Unsupported R2 controller mode {mode!r}.")

        def _fill(req: Any) -> None:
            req.mode = int(mode)

        resp = self._call_service("/Motion/SwitchController", _fill)
        self.controller_mode = int(mode)
        return bool(getattr(resp, "success", True))

    def switch_system_mode(self, mode: int) -> bool:
        """Switch the R2 system state machine mode via ``/state_machine/switch_mode``."""

        def _fill(req: Any) -> None:
            req.mode = int(mode)

        resp = self._call_service("/state_machine/switch_mode", _fill)
        return bool(getattr(resp, "success", True))

    def clear_errors(self) -> bool:
        """Clear motion controller faults via ``/Motion/ClearError``."""
        resp = self._call_service("/Motion/ClearError", lambda _req: None)
        return bool(getattr(resp, "success", True))

    def set_payload(
        self,
        side: str,
        mass: float,
        center_of_mass: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> bool:
        """Configure the end-effector payload mass and center of mass on ``/Motion/SetLoad``."""
        if side not in ("left", "right"):
            raise ValueError(f"Expected side 'left' or 'right', got {side!r}.")
        com = np.asarray(center_of_mass, dtype=np.float32).reshape(3)

        def _fill(req: Any) -> None:
            req.arm = 0 if side == "left" else 1
            req.mass = float(mass)
            req.com = [float(com[0]), float(com[1]), float(com[2])]

        resp = self._call_service("/Motion/SetLoad", _fill)
        return bool(getattr(resp, "success", True))

    def forward_kinematics(
        self, joint_positions: Sequence[float]
    ) -> dict[str, np.ndarray]:
        """Compute forward kinematics for whole-body or dual-arm joint positions."""
        q = [float(x) for x in joint_positions]

        def _fill(req: Any) -> None:
            req.joint_positions = q

        resp = self._call_service("/Motion/ForwardKine", _fill)
        poses = list(getattr(resp, "tcp_pose", []))
        if len(poses) >= 2:
            return {
                "left_tcp_pose": _pose_to_xyz_rpy(poses[0]),
                "right_tcp_pose": _pose_to_xyz_rpy(poses[1]),
            }
        state = self.get_state()
        return {
            "left_tcp_pose": state.left_tcp_pose,
            "right_tcp_pose": state.right_tcp_pose,
        }

    def inverse_kinematics(
        self,
        *,
        left_tcp_pose: Optional[np.ndarray] = None,
        right_tcp_pose: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """Compute inverse kinematics via ``/Motion/InverseKine``."""
        state = self.get_state()
        left_target = (
            np.asarray(left_tcp_pose, dtype=np.float32).reshape(6)
            if left_tcp_pose is not None
            else state.left_tcp_pose
        )
        right_target = (
            np.asarray(right_tcp_pose, dtype=np.float32).reshape(6)
            if right_tcp_pose is not None
            else state.right_tcp_pose
        )
        pose_cls = self._msg_types["Pose"]

        def _fill(req: Any) -> None:
            req.tcp_pose = [
                _xyz_rpy_to_pose(left_target, pose_cls),
                _xyz_rpy_to_pose(right_target, pose_cls),
            ]
            req.reference_joint_positions = list(
                np.concatenate(
                    [state.left_joint_positions, state.right_joint_positions]
                ).astype(float)
            )

        resp = self._call_service("/Motion/InverseKine", _fill)
        sol = np.asarray(getattr(resp, "joint_positions", []), dtype=np.float32)
        if sol.size >= 14:
            return {
                "left_joint_positions": sol[:7].copy(),
                "right_joint_positions": sol[7:14].copy(),
            }
        return {
            "left_joint_positions": state.left_joint_positions,
            "right_joint_positions": state.right_joint_positions,
        }

    def move_left_arm_joint(self, positions: np.ndarray) -> None:
        """Stream 7-DoF left arm joint targets over ``/Motion/ServoJ``."""
        q = np.asarray(positions, dtype=np.float32).reshape(7)
        self._publish_servoj(R2_LEFT_ARM_JOINTS, q)

    def move_right_arm_joint(self, positions: np.ndarray) -> None:
        """Stream 7-DoF right arm joint targets over ``/Motion/ServoJ``."""
        q = np.asarray(positions, dtype=np.float32).reshape(7)
        self._publish_servoj(R2_RIGHT_ARM_JOINTS, q)

    def move_dual_arm_joint(
        self, left_positions: np.ndarray, right_positions: np.ndarray
    ) -> None:
        """Stream synchronized 14-DoF dual-arm joint targets over ``/Motion/ServoJ``."""
        ql = np.asarray(left_positions, dtype=np.float32).reshape(7)
        qr = np.asarray(right_positions, dtype=np.float32).reshape(7)
        self._publish_servoj(
            R2_LEFT_ARM_JOINTS + R2_RIGHT_ARM_JOINTS,
            np.concatenate([ql, qr]),
        )

    def move_waist_joint(self, positions: np.ndarray) -> None:
        """Stream waist joint targets over ``/Motion/ServoJ``."""
        expected = len(self.waist_joint_names)
        q = np.asarray(positions, dtype=np.float32).reshape(expected)
        self._publish_servoj(self.waist_joint_names, q)

    def move_neck_joint(self, positions: np.ndarray) -> None:
        """Stream 2-DoF neck joint targets over ``/Motion/ServoJ``."""
        q = np.asarray(positions, dtype=np.float32).reshape(2)
        self._publish_servoj(R2_NECK_JOINTS, q)

    def move_left_arm_joint_velocity(self, velocities: np.ndarray) -> None:
        """Command 7-DoF left arm joint velocities via ``/Motion/ServoJ``."""
        dq = np.asarray(velocities, dtype=np.float32).reshape(7)
        state = self.get_state()
        self._publish_servoj(
            R2_LEFT_ARM_JOINTS, state.left_joint_positions, velocities=dq
        )

    def move_right_arm_joint_velocity(self, velocities: np.ndarray) -> None:
        """Command 7-DoF right arm joint velocities via ``/Motion/ServoJ``."""
        dq = np.asarray(velocities, dtype=np.float32).reshape(7)
        state = self.get_state()
        self._publish_servoj(
            R2_RIGHT_ARM_JOINTS, state.right_joint_positions, velocities=dq
        )

    def apply_left_arm_torque(self, torques: np.ndarray) -> None:
        """Command 7-DoF left arm joint torques via ``/Motion/ServoJ``."""
        tau = np.asarray(torques, dtype=np.float32).reshape(7)
        state = self.get_state()
        self._publish_servoj(
            R2_LEFT_ARM_JOINTS, state.left_joint_positions, efforts=tau
        )

    def apply_right_arm_torque(self, torques: np.ndarray) -> None:
        """Command 7-DoF right arm joint torques via ``/Motion/ServoJ``."""
        tau = np.asarray(torques, dtype=np.float32).reshape(7)
        state = self.get_state()
        self._publish_servoj(
            R2_RIGHT_ARM_JOINTS, state.right_joint_positions, efforts=tau
        )

    def move_left_arm_movej(
        self,
        positions: np.ndarray,
        duration: float = 3.0,
        *,
        max_velocity: Optional[np.ndarray] = None,
    ) -> bool:
        """Execute a point-to-point joint trajectory on the left arm via ``/Motion/MoveJ``."""
        q = np.asarray(positions, dtype=np.float32).reshape(7)

        def _fill(req: Any) -> None:
            req.joint_names = list(R2_LEFT_ARM_JOINTS)
            req.target_positions = [float(x) for x in q]
            req.duration = float(duration)
            if max_velocity is not None:
                req.max_velocity = [
                    float(x)
                    for x in np.asarray(max_velocity, dtype=np.float32).reshape(-1)
                ]

        resp = self._call_service(
            "/Motion/MoveJ", _fill, timeout=max(duration + 5.0, 5.0)
        )
        return bool(getattr(resp, "success", True))

    def move_right_arm_movej(
        self,
        positions: np.ndarray,
        duration: float = 3.0,
        *,
        max_velocity: Optional[np.ndarray] = None,
    ) -> bool:
        """Execute a point-to-point joint trajectory on the right arm via ``/Motion/MoveJ``."""
        q = np.asarray(positions, dtype=np.float32).reshape(7)

        def _fill(req: Any) -> None:
            req.joint_names = list(R2_RIGHT_ARM_JOINTS)
            req.target_positions = [float(x) for x in q]
            req.duration = float(duration)
            if max_velocity is not None:
                req.max_velocity = [
                    float(x)
                    for x in np.asarray(max_velocity, dtype=np.float32).reshape(-1)
                ]

        resp = self._call_service(
            "/Motion/MoveJ", _fill, timeout=max(duration + 5.0, 5.0)
        )
        return bool(getattr(resp, "success", True))

    def move_left_arm_cartesian(self, pose: np.ndarray, duration: float = 1.0) -> bool:
        """Move the left arm TCP to ``pose`` ``[x, y, z, r, p, y]`` via ``/Motion/MoveC``."""
        target = np.asarray(pose, dtype=np.float32).reshape(6)
        pose_cls = self._msg_types["Pose"]

        def _fill(req: Any) -> None:
            req.arm = 0
            req.target_pose = _xyz_rpy_to_pose(target, pose_cls)
            req.duration = float(duration)

        resp = self._call_service(
            "/Motion/MoveC", _fill, timeout=max(duration + 5.0, 5.0)
        )
        return bool(getattr(resp, "success", True))

    def move_right_arm_cartesian(self, pose: np.ndarray, duration: float = 1.0) -> bool:
        """Move the right arm TCP to ``pose`` ``[x, y, z, r, p, y]`` via ``/Motion/MoveC``."""
        target = np.asarray(pose, dtype=np.float32).reshape(6)
        pose_cls = self._msg_types["Pose"]

        def _fill(req: Any) -> None:
            req.arm = 1
            req.target_pose = _xyz_rpy_to_pose(target, pose_cls)
            req.duration = float(duration)

        resp = self._call_service(
            "/Motion/MoveC", _fill, timeout=max(duration + 5.0, 5.0)
        )
        return bool(getattr(resp, "success", True))

    def move_left_arm_velocity(self, velocity: np.ndarray, dt: float = 0.02) -> None:
        """Step the left arm TCP by integrating a 6D Cartesian velocity twist."""
        vel = np.asarray(velocity, dtype=np.float32).reshape(6)
        state = self.get_state()
        target_pose = state.left_tcp_pose + vel * float(dt)
        self.move_left_arm_cartesian(target_pose, duration=dt)

    def move_right_arm_velocity(self, velocity: np.ndarray, dt: float = 0.02) -> None:
        """Step the right arm TCP by integrating a 6D Cartesian velocity twist."""
        vel = np.asarray(velocity, dtype=np.float32).reshape(6)
        state = self.get_state()
        target_pose = state.right_tcp_pose + vel * float(dt)
        self.move_right_arm_cartesian(target_pose, duration=dt)

    def move_left_gripper(self, position: Any) -> None:
        """Command the left parallel gripper position in ``[0.0, 1.0]``."""
        val = float(
            np.clip(np.asarray(position, dtype=np.float32).reshape(-1)[0], 0.0, 1.0)
        )
        self._publish_servoj(("GripperL",), (val,))
        with self._lock:
            self._state.left_gripper_position = np.array([val], dtype=np.float32)

    def move_right_gripper(self, position: Any) -> None:
        """Command the right parallel gripper position in ``[0.0, 1.0]``."""
        val = float(
            np.clip(np.asarray(position, dtype=np.float32).reshape(-1)[0], 0.0, 1.0)
        )
        self._publish_servoj(("GripperR",), (val,))
        with self._lock:
            self._state.right_gripper_position = np.array([val], dtype=np.float32)

    def move_left_hand(self, positions: Any) -> None:
        """Command the 6-DoF left dexterous hand ``(little, ring, middle, index, thumb_bend, thumb_rotation)``."""
        arr = np.asarray(positions, dtype=np.float32).reshape(6)
        self._publish_servoj(R2_HAND_JOINTS_LEFT, arr)
        with self._lock:
            self._state.left_hand_position = arr.copy()

    def move_right_hand(self, positions: Any) -> None:
        """Command the 6-DoF right dexterous hand ``(little, ring, middle, index, thumb_bend, thumb_rotation)``."""
        arr = np.asarray(positions, dtype=np.float32).reshape(6)
        self._publish_servoj(R2_HAND_JOINTS_RIGHT, arr)
        with self._lock:
            self._state.right_hand_position = arr.copy()

    def move_base(self, velocity: np.ndarray) -> None:
        """Publish planar chassis velocity ``[vx, vy, wz]`` on ``/teleop/chassis_Twist``."""
        self._require_connected()
        if "/teleop/chassis_Twist" not in self._writers:
            raise RuntimeError(
                "R2 mobile base is not enabled on this connection (enable_base=False)."
            )
        vel = np.asarray(velocity, dtype=np.float32).reshape(3)
        channel, msg_cls = self._writers["/teleop/chassis_Twist"]
        msg = msg_cls()
        msg.linear.x = float(vel[0])
        msg.linear.y = float(vel[1])
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(vel[2])
        channel.send(msg)
        with self._lock:
            self._state.base_velocity = vel.copy()

    def stop_base(self) -> None:
        """Zero the chassis velocity command and cancel active navigation if supported."""
        if "/teleop/chassis_Twist" in self._writers:
            self.move_base(np.zeros(3, dtype=np.float32))
        if "/navigation/control" in self._services:
            self._call_service(
                "/navigation/control",
                lambda req: setattr(req, "command", 0),
            )

    def navigate_to(
        self, x: float, y: float, yaw: float = 0.0, timeout: float = 30.0
    ) -> bool:
        """Assign a navigation goal ``(x, y, yaw)`` via ``/task_manager/assign_task``."""
        if "/task_manager/assign_task" not in self._services:
            raise RuntimeError(
                "R2 mobile base is not enabled on this connection (enable_base=False)."
            )

        def _fill(req: Any) -> None:
            req.task_type = "navigation"
            req.target_x = float(x)
            req.target_y = float(y)
            req.target_yaw = float(yaw)

        resp = self._call_service("/task_manager/assign_task", _fill, timeout=timeout)
        return bool(getattr(resp, "success", True))

    def check_camera(self, camera_name: str, timeout: float = 0.5) -> bool:
        """Return True if ``camera_name`` has delivered at least one RGB frame."""
        self._require_connected()
        event = self._camera_events.get(camera_name)
        if event is None:
            raise KeyError(f"Unknown R2 camera {camera_name!r}.")
        return bool(event.wait(timeout=max(timeout, 0.0)))

    def get_camera(self, camera_name: str) -> np.ndarray:
        """Return the latest RGB frame for ``camera_name``."""
        self._require_connected()
        if camera_name not in R2_CAMERA_TOPICS:
            raise KeyError(f"Unknown R2 camera {camera_name!r}.")
        event = self._camera_events.get(camera_name)
        if event is not None and not event.is_set():
            event.wait(timeout=0.5)
        with self._lock:
            frame = self._camera_frames.get(camera_name)
            if frame is not None:
                return frame.copy()
        width, height = R2_CAMERA_TOPICS[camera_name]["resolution"]
        return np.zeros((height, width, 3), dtype=np.uint8)

    def get_camera_depth(self, camera_name: str) -> Optional[np.ndarray]:
        """Return the latest depth image for ``camera_name`` when ``enable_depth=True``."""
        self._require_connected()
        if not self.enable_depth:
            return None
        with self._lock:
            depth = self._camera_depths.get(camera_name)
            if depth is not None:
                return depth.copy()
        width, height = R2_CAMERA_TOPICS[camera_name]["resolution"]
        return np.zeros((height, width), dtype=np.float32)

    def get_camera_observation(self, camera_name: str) -> dict[str, Any]:
        """Return ``{'frame': rgb}`` plus ``'depth'`` when depth is enabled."""
        obs: dict[str, Any] = {"frame": self.get_camera(camera_name)}
        if self.enable_depth:
            obs["depth"] = self.get_camera_depth(camera_name)
        return obs
