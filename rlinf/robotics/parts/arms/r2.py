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

"""Robbyant R2 robot connection and part views over the RFrame middleware.

``R2Connection`` speaks the robot's ``/robotic/*`` interface: joint states and
diagnostics arrive on RFrame topics, services go through rclpy, and joint
setpoints stream to ``/robotic/servoj_data`` from a :class:`JointStream`.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.robotics.parts.base import (
    Connection,
    FaultLevel,
    Features,
    HealthStatus,
    RobotPart,
)
from rlinf.robotics.parts.end_effectors import BaseHand
from rlinf.robotics.parts.mobility.base import MobileBase
from rlinf.robotics.parts.motion import JointGroupSpec, JointStream, MotionProfile
from rlinf.robotics.parts.views import (
    MethodArm,
    MethodCamera,
    MethodEndEffector,
    host_state,
)
from rlinf.utils.logging import get_logger

#: Controller run modes, as ``r2_motion_msgs/srv/SwitchMode`` defines them.
R2_MODE_AUTO = 0
R2_MODE_REMOTE = 1
R2_MODE_SERVOJ = 2
R2_MODE_MANUAL_DRAG = 4
R2_CONTROLLER_MODES: frozenset[int] = frozenset(
    {R2_MODE_AUTO, R2_MODE_REMOTE, R2_MODE_SERVOJ, R2_MODE_MANUAL_DRAG}
)

R2_SLAVE_ADDR_TO_JOINT: dict[int, str] = {
    1002: "J_LeftArm1",
    1003: "J_LeftArm2",
    1004: "J_LeftArm3",
    1005: "J_LeftArm4",
    1006: "J_LeftArm5",
    1007: "J_LeftArm6",
    1008: "J_LeftArm7",
    1010: "J_LeftGripper",
    1012: "J_RightArm1",
    1013: "J_RightArm2",
    1014: "J_RightArm3",
    1015: "J_RightArm4",
    1016: "J_RightArm5",
    1017: "J_RightArm6",
    1018: "J_RightArm7",
    1020: "J_RightGripper",
    1022: "J_NeckYaw",
    1023: "J_NeckPitch",
    1024: "J_WaistRoll",
    1025: "J_WaistYaw",
    1026: "J_Hip",
    1027: "J_Knee",
    1028: "J_Ankle",
}

#: EtherCAT slaves re-enabled by clear_errors(); the grippers enable themselves.
R2_BODY_SLAVE_ADDRS: tuple[int, ...] = tuple(
    addr for addr in R2_SLAVE_ADDR_TO_JOINT if addr not in (1010, 1020)
)

#: Motion controller codes, as ``r2_motion_msgs/msg/ErrorCode`` defines them.
R2_ERROR_CODES: dict[int, str] = {
    0: "SUCCESS",
    10: "SYSTEM_POWEROFF",
    11: "SYSTEM_READY",
    12: "SYSTEM_IDLE",
    13: "SYSTEM_RUNNING",
    14: "SYSTEM_FAULT",
    101: "SOFTWARE_INIT_FAIL",
    102: "SOFTWARE_CALIBRATION_ERROR",
    103: "SOFTWARE_CONFIG_ERROR",
    201: "INPUT_PARAM_IN_VALID",
    202: "INPUT_PATTERN_MISMATCH",
    301: "ALGO_INIT_FAIL",
    302: "ALGO_PLAN_FAIL",
    303: "ALGO_UPDATE_FAIL",
    304: "ALGO_DETECT_IN_COLLISION",
    401: "SAFE_EMERGENCY_STOP",
    501: "SLAVE_NOT_READY",
}

#: Codes that report a status rather than a fault.
R2_STATUS_CODES: frozenset[int] = frozenset({0, 11, 12, 13})

#: Codes that reject one request; the controller stays healthy, so they are
#: returned to the caller and never latched as a hardware fault.
R2_REQUEST_ERROR_CODES: frozenset[int] = frozenset({201, 202, 301, 302, 303})

#: Codes that clearing faults in software cannot fix: an emergency stop cuts
#: drive power until a person releases it, and the rest need a restart or new
#: calibration or configuration.
R2_UNRECOVERABLE_ERROR_CODES: frozenset[int] = frozenset({10, 101, 102, 103, 401})

#: Robot-wide states of the RobbyAnt state manager, as
#: ``state_machine_msgs/msg/SystemState`` defines them.
R2_SYSTEM_STATE_FAULT = 5
R2_SYSTEM_STATE_EMERGENCY_STOP = 7

#: Joint names in ``/robotic/joint_states``. The waist is five joints from
#: the ankle up; the 20-DoF body has no neck yaw.
R2_WAIST_JOINTS: tuple[str, ...] = (
    "J_AnklePitch",
    "J_Knee",
    "J_HipPitch",
    "J_HipRoll",
    "J_HipYaw",
)
R2_LEFT_ARM_JOINTS: tuple[str, ...] = tuple(f"J_LeftArm{i}" for i in range(1, 8))
R2_RIGHT_ARM_JOINTS: tuple[str, ...] = tuple(f"J_RightArm{i}" for i in range(1, 8))
R2_NECK_JOINTS: tuple[str, ...] = ("J_NeckYaw", "J_NeckPitch")
R2_GRIPPER_JOINTS: tuple[str, ...] = ("J_LeftGripper", "J_RightGripper")
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

R2_LEFT_ARM_LIMITS_LOW = np.array(
    [-2.74, 0.0, -2.82, -2.33, -2.82, -1.45, -1.45], dtype=np.float32
)
R2_LEFT_ARM_LIMITS_HIGH = np.array(
    [1.34, 3.14, 2.82, 0.02, 2.82, 1.45, 1.45], dtype=np.float32
)
R2_RIGHT_ARM_LIMITS_LOW = np.array(
    [-2.74, -3.14, -2.82, -0.02, -2.82, -1.45, -1.45], dtype=np.float32
)
R2_RIGHT_ARM_LIMITS_HIGH = np.array(
    [1.34, 0.0, 2.82, 2.33, 2.82, 1.45, 1.45], dtype=np.float32
)
R2_WAIST_LIMITS_LOW = np.array([0.0, 0.0, -0.38, -0.47, -1.95], dtype=np.float32)
R2_WAIST_LIMITS_HIGH = np.array([1.48, 2.39, 1.62, 0.47, 1.95], dtype=np.float32)
R2_NECK_LIMITS_LOW = np.array([-1.57, -0.698], dtype=np.float32)
R2_NECK_LIMITS_HIGH = np.array([1.57, 0.349], dtype=np.float32)

#: Default limits of R2's streamed joint motion, in rad/s and rad/s^2. The
#: torso carries both arms, so the waist moves slower than the arms.
R2_MAX_JOINT_VELOCITY = 2.0
R2_MAX_JOINT_ACCELERATION = 20.0
R2_WAIST_MAX_VELOCITY = 0.5
R2_WAIST_MAX_ACCELERATION = 2.0
R2_NECK_MAX_VELOCITY = 1.0
R2_NECK_MAX_ACCELERATION = 4.0

R2_LEFT_HOME_QPOS = np.array(
    [-0.15, 0.20, 0.0, -0.30, 0.0, -0.15, 0.0], dtype=np.float32
)
R2_RIGHT_HOME_QPOS = np.array(
    [-0.15, -0.20, 0.0, 0.30, 0.0, -0.15, 0.0], dtype=np.float32
)

R2_LEFT_READY_QPOS = np.array(
    [-0.35, 0.20, 1.00, -0.70, 0.0, -0.30, 0.0], dtype=np.float32
)
R2_RIGHT_READY_QPOS = np.array(
    [-0.35, -0.20, 1.00, 0.70, 0.0, -0.30, 0.0], dtype=np.float32
)

_R2_WAIST_URDF_CHAIN: tuple[tuple[np.ndarray, np.ndarray], ...] = (
    (np.array([0.0, 0.073, 0.345]), np.array([0.0, -1.0, 0.0])),
    (np.array([0.0, 0.0, 0.34]), np.array([0.0, 1.0, 0.0])),
    (np.array([0.0, 0.0, 0.3]), np.array([0.0, -1.0, 0.0])),
    (np.array([-0.056, -0.0732, 0.14]), np.array([1.0, 0.0, 0.0])),
    (np.array([0.056, 0.0, 0.065]), np.array([0.0, 0.0, 1.0])),
)

_R2_LEFT_ARM_URDF_CHAIN: tuple[tuple[str, np.ndarray, np.ndarray, np.ndarray], ...] = (
    ("fixed", np.array([0.0, 0.0925, 0.242]), np.zeros(3), np.zeros(3)),
    ("rev", np.array([0.0, 0.0229, 0.0]), np.zeros(3), np.array([0.0, 1.0, 0.0])),
    ("rev", np.array([-0.0594, 0.058, 0.0]), np.zeros(3), np.array([1.0, 0.0, 0.0])),
    ("rev", np.array([0.06, 0.0, -0.09]), np.zeros(3), np.array([0.0, 0.0, 1.0])),
    ("rev", np.array([-0.059, -0.01, -0.21]), np.zeros(3), np.array([1.0, 0.0, 0.0])),
    ("rev", np.array([0.059, 0.01, -0.0956]), np.zeros(3), np.array([0.0, 0.0, 1.0])),
    ("rev", np.array([0.0, 0.034, -0.1374]), np.zeros(3), np.array([0.0, -1.0, 0.0])),
    (
        "rev",
        np.array([-0.0364, -0.034, -0.058]),
        np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
    ),
    ("fixed", np.array([0.0364, 0.0, -0.058]), np.zeros(3), np.zeros(3)),
    (
        "fixed",
        np.array([0.0, 0.0, -0.013]),
        np.array([3.1415926, 0.0, -1.5707963]),
        np.zeros(3),
    ),
    (
        "fixed",
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, -1.571428]),
        np.zeros(3),
    ),
    ("fixed", np.array([0.0, 0.0, 0.122]), np.array([0.0, 0.0, -1.57]), np.zeros(3)),
)

_R2_RIGHT_ARM_URDF_CHAIN: tuple[tuple[str, np.ndarray, np.ndarray, np.ndarray], ...] = (
    ("fixed", np.array([0.0, -0.0925, 0.242]), np.zeros(3), np.zeros(3)),
    ("rev", np.array([0.0, -0.0229, 0.0]), np.zeros(3), np.array([0.0, 1.0, 0.0])),
    ("rev", np.array([-0.0594, -0.058, 0.0]), np.zeros(3), np.array([1.0, 0.0, 0.0])),
    ("rev", np.array([0.06, 0.0, -0.09]), np.zeros(3), np.array([0.0, 0.0, -1.0])),
    ("rev", np.array([-0.059, 0.01, -0.21]), np.zeros(3), np.array([1.0, 0.0, 0.0])),
    ("rev", np.array([0.059, -0.01, -0.0956]), np.zeros(3), np.array([0.0, 0.0, -1.0])),
    ("rev", np.array([0.0, 0.034, -0.1374]), np.zeros(3), np.array([0.0, -1.0, 0.0])),
    (
        "rev",
        np.array([-0.0364, -0.034, -0.058]),
        np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
    ),
    ("fixed", np.array([0.0364, 0.0, -0.058]), np.zeros(3), np.zeros(3)),
    (
        "fixed",
        np.array([0.0, 0.0, -0.013]),
        np.array([3.1415926, 0.0, -1.5707963]),
        np.zeros(3),
    ),
    ("fixed", np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.57]), np.zeros(3)),
    ("fixed", np.array([0.0, 0.0, 0.122]), np.array([0.0, 0.0, -1.57]), np.zeros(3)),
)


def _compile_waist_urdf_chain(
    chain: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> tuple[tuple[np.ndarray, int, float], ...]:
    compiled: list[tuple[np.ndarray, int, float]] = []
    for xyz, axis in chain:
        axis_idx = int(np.argmax(np.abs(axis)))
        sign = float(axis[axis_idx])
        compiled.append((np.asarray(xyz, dtype=np.float64), axis_idx, sign))
    return tuple(compiled)


def _compile_arm_urdf_chain(
    chain: tuple[tuple[str, np.ndarray, np.ndarray, np.ndarray], ...],
) -> tuple[
    tuple[tuple[np.ndarray, np.ndarray, bool, int, float], ...],
    np.ndarray,
    np.ndarray,
]:
    rev_steps: list[tuple[np.ndarray, np.ndarray, bool, int, float]] = []
    T_pending = np.eye(4, dtype=np.float64)
    for jtype, xyz, rpy, axis in chain:
        T_o = np.eye(4, dtype=np.float64)
        if np.any(rpy):
            T_o[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
        T_o[:3, 3] = xyz
        T_pending = T_pending @ T_o
        if jtype == "rev":
            R_off = T_pending[:3, :3].copy()
            p_off = T_pending[:3, 3].copy()
            has_rot = not bool(np.allclose(R_off, np.eye(3)))
            axis_idx = int(np.argmax(np.abs(axis)))
            sign = float(axis[axis_idx])
            rev_steps.append((p_off, R_off, has_rot, axis_idx, sign))
            T_pending = np.eye(4, dtype=np.float64)
    p_tip_off = T_pending[:3, 3].copy()
    R_tip_off = T_pending[:3, :3].copy()
    return tuple(rev_steps), p_tip_off, R_tip_off


_COMPILED_WAIST_CHAIN = _compile_waist_urdf_chain(_R2_WAIST_URDF_CHAIN)
_WAIST_ZERO_OFFSET: np.ndarray = sum(
    (xyz for xyz, _, _ in _COMPILED_WAIST_CHAIN), np.zeros(3, dtype=np.float64)
)
_COMPILED_ARM_CHAINS: dict[
    str,
    tuple[
        tuple[tuple[np.ndarray, np.ndarray, bool, int, float], ...],
        np.ndarray,
        np.ndarray,
    ],
] = {
    "left": _compile_arm_urdf_chain(_R2_LEFT_ARM_URDF_CHAIN),
    "right": _compile_arm_urdf_chain(_R2_RIGHT_ARM_URDF_CHAIN),
}


def _apply_cardinal_rot(R_mat: np.ndarray, axis_idx: int, angle: float) -> None:
    """Right-multiply a 3x3 rotation matrix in-place by a rotation about cardinal axis ``axis_idx``."""
    c = math.cos(angle)
    s = math.sin(angle)
    if axis_idx == 0:
        c1 = R_mat[:, 1].copy()
        c2 = R_mat[:, 2].copy()
        R_mat[:, 1] = c * c1 + s * c2
        R_mat[:, 2] = -s * c1 + c * c2
    elif axis_idx == 1:
        c0 = R_mat[:, 0].copy()
        c2 = R_mat[:, 2].copy()
        R_mat[:, 0] = c * c0 - s * c2
        R_mat[:, 2] = s * c0 + c * c2
    else:
        c0 = R_mat[:, 0].copy()
        c1 = R_mat[:, 1].copy()
        R_mat[:, 0] = c * c0 + s * c1
        R_mat[:, 1] = -s * c0 + c * c1


def _matrix_to_euler_xyz(M: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to extrinsic ``xyz`` Euler angles (roll, pitch, yaw)."""
    sy = -float(M[2, 0])
    if abs(sy) < 0.999999:
        y = math.asin(sy)
        x = math.atan2(float(M[2, 1]), float(M[2, 2]))
        z = math.atan2(float(M[1, 0]), float(M[0, 0]))
        return np.array([x, y, z], dtype=np.float64)
    return R.from_matrix(M).as_euler("xyz", degrees=False)


def _so3_error_rotvec(R_err: np.ndarray) -> np.ndarray:
    """Return the SO(3) rotation vector for ``R_err = R_target @ R_current.T``."""
    rx = float(R_err[2, 1] - R_err[1, 2])
    ry = float(R_err[0, 2] - R_err[2, 0])
    rz = float(R_err[1, 0] - R_err[0, 1])
    s_norm = math.hypot(rx, ry, rz)
    tr = float(R_err[0, 0] + R_err[1, 1] + R_err[2, 2])
    if s_norm < 1e-12:
        if tr >= 1.0:
            return np.array([0.5 * rx, 0.5 * ry, 0.5 * rz], dtype=np.float64)
        return R.from_matrix(R_err).as_rotvec()
    theta = math.atan2(0.5 * s_norm, 0.5 * (tr - 1.0))
    if theta > 3.10:
        return R.from_matrix(R_err).as_rotvec()
    scale = theta / s_norm
    return np.array([scale * rx, scale * ry, scale * rz], dtype=np.float64)


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Convert a 3D rotation vector to a 3x3 rotation matrix via Rodrigues' formula."""
    rx = float(rotvec[0])
    ry = float(rotvec[1])
    rz = float(rotvec[2])
    theta = math.hypot(rx, ry, rz)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    kx = rx / theta
    ky = ry / theta
    kz = rz / theta
    c = math.cos(theta)
    s = math.sin(theta)
    v = 1.0 - c
    return np.array(
        [
            [kx * kx * v + c, kx * ky * v - kz * s, kx * kz * v + ky * s],
            [ky * kx * v + kz * s, ky * ky * v + c, ky * kz * v - kx * s],
            [kz * kx * v - ky * s, kz * ky * v + kx * s, kz * kz * v + c],
        ],
        dtype=np.float64,
    )


def _compute_r2_waist_base_transform(
    q_waist: Optional[Sequence[float] | np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(p_waist, R_waist)`` at the top of the 5-DoF waist chain in ``base_link``."""
    R_cur = np.eye(3, dtype=np.float64)
    if q_waist is None:
        return _WAIST_ZERO_OFFSET.copy(), R_cur
    w = np.asarray(q_waist, dtype=np.float64).reshape(-1)
    if not np.any(w):
        return _WAIST_ZERO_OFFSET.copy(), R_cur
    p_cur = np.zeros(3, dtype=np.float64)
    w_n = w.size
    for idx, (xyz, axis_idx, sign) in enumerate(_COMPILED_WAIST_CHAIN):
        p_cur = p_cur + R_cur @ xyz
        if idx < w_n:
            val = float(w[idx])
            if val != 0.0:
                _apply_cardinal_rot(R_cur, axis_idx, sign * val)
    return p_cur, R_cur


def _compute_r2_arm_fk_and_jacobian_matrix(
    side: str,
    q_arm: Sequence[float] | np.ndarray,
    q_waist: Optional[Sequence[float] | np.ndarray] = None,
    *,
    waist_transform: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute float64 ``(p_tip, R_tip, J_pos, J_rot)`` in ``base_link``."""
    if side not in ("left", "right"):
        raise ValueError(f"Expected side 'left' or 'right', got {side!r}.")
    q_a = np.asarray(q_arm, dtype=np.float64).reshape(7)
    if waist_transform is not None:
        p_cur = waist_transform[0].copy()
        R_cur = waist_transform[1].copy()
    else:
        p_cur, R_cur = _compute_r2_waist_base_transform(q_waist)

    rev_steps, p_tip_off, R_tip_off = _COMPILED_ARM_CHAINS[side]
    origins = np.empty((3, 7), dtype=np.float64)
    J_rot = np.empty((3, 7), dtype=np.float64)
    for i, (p_off, R_off, has_rot, axis_idx, sign) in enumerate(rev_steps):
        p_cur = p_cur + R_cur @ p_off
        if has_rot:
            R_cur = R_cur @ R_off
        origins[:, i] = p_cur
        J_rot[:, i] = sign * R_cur[:, axis_idx]
        val = float(q_a[i])
        if val != 0.0:
            _apply_cardinal_rot(R_cur, axis_idx, sign * val)

    p_tip = p_cur + R_cur @ p_tip_off
    R_tip = R_cur @ R_tip_off
    dx = p_tip[0] - origins[0]
    dy = p_tip[1] - origins[1]
    dz = p_tip[2] - origins[2]
    J_pos = np.empty((3, 7), dtype=np.float64)
    J_pos[0] = J_rot[1] * dz - J_rot[2] * dy
    J_pos[1] = J_rot[2] * dx - J_rot[0] * dz
    J_pos[2] = J_rot[0] * dy - J_rot[1] * dx
    return p_tip, R_tip, J_pos, J_rot


def _pose_matrix(pose: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return the 4x4 transform of ``[x, y, z, roll, pitch, yaw]`` (extrinsic ``xyz``)."""
    arr = np.asarray(pose, dtype=np.float64).reshape(6)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_euler("xyz", arr[3:6]).as_matrix()
    T[:3, 3] = arr[:3]
    return T


#: Re-seeded solves allowed to reach one pose; each moves up to the solver's
#: 10 cm lookahead, and the solver rejects targets beyond 1.2 m.
_R2_IK_SEGMENTS = 12


def compute_r2_arm_fk_and_jacobian(
    side: str,
    q_arm: Sequence[float] | np.ndarray,
    q_waist: Optional[Sequence[float] | np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute TCP position, Euler RPY, and 3x7 position/rotation Jacobians in ``base_link``."""
    p_tip, R_tip, J_pos, J_rot = _compute_r2_arm_fk_and_jacobian_matrix(
        side, q_arm, q_waist
    )
    rpy_tip = _matrix_to_euler_xyz(R_tip)
    return (
        p_tip.astype(np.float32),
        rpy_tip.astype(np.float32),
        J_pos.astype(np.float32),
        J_rot.astype(np.float32),
    )


def compute_r2_arm_cartesian_step(
    side: str,
    q_init: Sequence[float] | np.ndarray,
    delta_xyz: Sequence[float] | np.ndarray,
    *,
    q_waist: Optional[Sequence[float] | np.ndarray] = None,
    max_step_rad: float = 0.35,
) -> np.ndarray:
    """Solve a joint-limit-projected damped-least-squares IK step for a 3D TCP delta ``[dx, dy, dz]``."""
    lo = (R2_LEFT_ARM_LIMITS_LOW if side == "left" else R2_RIGHT_ARM_LIMITS_LOW).astype(
        np.float64
    )
    hi = (
        R2_LEFT_ARM_LIMITS_HIGH if side == "left" else R2_RIGHT_ARM_LIMITS_HIGH
    ).astype(np.float64)
    q0 = np.clip(np.asarray(q_init, dtype=np.float64).reshape(7), lo, hi)
    q = q0.copy()
    d_xyz = np.asarray(delta_xyz, dtype=np.float64).reshape(3)
    waist_tf = _compute_r2_waist_base_transform(q_waist)
    p0, _, Jp0, _ = _compute_r2_arm_fk_and_jacobian_matrix(
        side, q, waist_transform=waist_tf
    )
    target_p = p0 + d_xyz

    if d_xyz[2] > 0.0 and float(np.linalg.norm(Jp0[2, :])) < 0.12:
        out_sign = 1.0 if side == "left" else -1.0
        elbow_sign = -1.0 if side == "left" else 1.0
        q[0] = np.clip(q[0] - 0.15, lo[0], hi[0])
        q[1] = np.clip(q[1] + out_sign * 0.18, lo[1], hi[1])
        q[3] = np.clip(q[3] + elbow_sign * 0.30, lo[3], hi[3])

    weights_vec = np.array([1.0, 1.0, 0.8, 1.0, 0.5, 0.4, 0.4], dtype=np.float64)
    eye3 = np.eye(3, dtype=np.float64)
    damping_sq = 0.03**2
    for _ in range(30):
        p, _, Jp, _ = _compute_r2_arm_fk_and_jacobian_matrix(
            side, q, waist_transform=waist_tf
        )
        err = target_p - p
        if float(np.linalg.norm(err)) < 1e-4:
            break
        grad = Jp.T @ err
        clamped = ((q <= lo + 1e-3) & (grad < 0.0)) | ((q >= hi - 1e-3) & (grad > 0.0))
        if np.any(clamped):
            J_eff = Jp.copy()
            J_eff[:, clamped] = 0.0
        else:
            J_eff = Jp
        JW = J_eff * weights_vec
        JJT = JW @ J_eff.T + damping_sq * eye3
        dq = JW.T @ np.linalg.solve(JJT, err)
        q = np.clip(q + 0.6 * dq, lo, hi)

    dq_total = q - q0
    max_abs = float(np.max(np.abs(dq_total)))
    if max_abs > max_step_rad:
        dq_total = dq_total * (max_step_rad / max_abs)
    return np.clip(q0 + dq_total, lo, hi).astype(np.float32)


@dataclass
class R2IKConfig:
    """Inverse-kinematics parameters for 7-DoF R2 arm teleoperation."""

    max_iters: int = 40
    """Maximum damped-least-squares iterations per solve."""

    damping: float = 0.03
    """Damping factor for the 6-DoF Jacobian pseudo-inverse."""

    step_size: float = 0.65
    """Per-iteration step gain applied to the solved joint delta."""

    nullspace_gain: float = 0.05
    """Null-space posture regularization gain toward the arm's ready pose."""

    position_tolerance: float = 0.002
    """Maximum Euclidean TCP position residual (m) accepted as converged."""

    rotation_tolerance: float = 0.02
    """Maximum rotation-vector TCP orientation residual (rad) accepted as converged."""

    max_solve_s: float = 0.15
    """Maximum wall-clock time (s) allowed for one solve call."""

    def __post_init__(self) -> None:
        """Validate IK solver bounds."""
        if (
            isinstance(self.max_iters, bool)
            or not isinstance(self.max_iters, int)
            or self.max_iters <= 0
        ):
            raise ValueError("R2IKConfig.max_iters must be a positive integer.")
        for name in (
            "damping",
            "step_size",
            "position_tolerance",
            "rotation_tolerance",
            "max_solve_s",
        ):
            val = float(getattr(self, name))
            if not np.isfinite(val) or val <= 0.0:
                raise ValueError(f"R2IKConfig.{name} must be finite and positive.")
            setattr(self, name, val)
        self.nullspace_gain = float(self.nullspace_gain)
        if not np.isfinite(self.nullspace_gain) or self.nullspace_gain < 0.0:
            raise ValueError(
                "R2IKConfig.nullspace_gain must be finite and non-negative."
            )


@dataclass(frozen=True)
class R2IKResult:
    """Result of one 7-DoF R2 inverse-kinematics solve."""

    success: bool
    q_target: np.ndarray
    position_error: float
    rotation_error: float
    elapsed_s: float
    reason: Optional[str] = None


class R2KinematicsAdapter:
    """Analytical FK and 6-DoF damped-least-squares IK solver for one R2 arm.

    Implements the ``fk(joints, gripper)``, ``solve(target, seed, gripper)``,
    and ``max_solve_s`` interface consumed by
    :class:`~rlinf.robotics.parts.teleop.pico.PicoJoint`.
    """

    def __init__(
        self,
        side: str,
        *,
        config: Optional[R2IKConfig | Mapping[str, Any]] = None,
        joint_lower: Optional[Sequence[float] | np.ndarray] = None,
        joint_upper: Optional[Sequence[float] | np.ndarray] = None,
        waist_qpos: Optional[Sequence[float] | np.ndarray] = None,
    ) -> None:
        if side not in ("left", "right"):
            raise ValueError(f"R2 arm side must be 'left' or 'right', got {side!r}.")
        self.side = side
        if config is None:
            self.config = R2IKConfig()
        elif isinstance(config, R2IKConfig):
            self.config = config
        else:
            self.config = R2IKConfig(**dict(config))

        default_lo = (
            R2_LEFT_ARM_LIMITS_LOW if side == "left" else R2_RIGHT_ARM_LIMITS_LOW
        ).astype(np.float64)
        default_hi = (
            R2_LEFT_ARM_LIMITS_HIGH if side == "left" else R2_RIGHT_ARM_LIMITS_HIGH
        ).astype(np.float64)
        self.lower = (
            default_lo.copy()
            if joint_lower is None
            else np.maximum(
                default_lo, np.asarray(joint_lower, dtype=np.float64).reshape(7)
            )
        )
        self.upper = (
            default_hi.copy()
            if joint_upper is None
            else np.minimum(
                default_hi, np.asarray(joint_upper, dtype=np.float64).reshape(7)
            )
        )
        if np.any(self.lower >= self.upper):
            raise ValueError(f"R2 {side} arm joint limits do not overlap.")
        self._nominal = np.clip(
            (R2_LEFT_READY_QPOS if side == "left" else R2_RIGHT_READY_QPOS).astype(
                np.float64
            ),
            self.lower,
            self.upper,
        )
        self._waist_qpos = (
            None
            if waist_qpos is None
            else (
                waist_qpos
                if callable(waist_qpos)
                else np.asarray(waist_qpos, dtype=np.float64).reshape(-1).copy()
            )
        )
        self._weights_vec = np.array(
            [1.0, 1.2, 0.6, 1.0, 0.5, 0.6, 0.6], dtype=np.float64
        )
        self._weights = np.diag(self._weights_vec)

    def _current_waist_qpos(self) -> Optional[np.ndarray]:
        if self._waist_qpos is None:
            return None
        if callable(self._waist_qpos):
            val = self._waist_qpos()
            return (
                None if val is None else np.asarray(val, dtype=np.float64).reshape(-1)
            )
        return self._waist_qpos

    @property
    def max_solve_s(self) -> float:
        """Maximum solve duration in seconds before a result is rejected."""
        return self.config.max_solve_s

    def fk(
        self,
        q: Sequence[float] | np.ndarray,
        gripper_position: Optional[float] = None,
    ) -> np.ndarray:
        """Return the 4x4 TCP homogeneous transform in ``base_link``."""
        del gripper_position
        q_vec = np.asarray(q, dtype=np.float64).reshape(7)
        p_tip, R_tip, _, _ = _compute_r2_arm_fk_and_jacobian_matrix(
            self.side, q_vec, self._current_waist_qpos()
        )
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_tip
        T[:3, 3] = p_tip
        return T

    def solve(
        self,
        target: np.ndarray,
        q_seed: Sequence[float] | np.ndarray,
        gripper_position: Optional[float] = None,
    ) -> R2IKResult:
        """Solve 6-DoF TCP pose IK for a 4x4 target transform from ``q_seed``."""
        del gripper_position
        started = time.monotonic()
        q0 = np.asarray(q_seed, dtype=np.float64).reshape(7)
        if not np.all(np.isfinite(q0)):
            raise ValueError("R2 IK requires 7 finite seed joint angles.")
        target_mat = np.asarray(target, dtype=np.float64)
        if target_mat.shape != (4, 4) or not np.all(np.isfinite(target_mat)):
            raise ValueError("R2 IK target must be a finite 4x4 transform.")

        if np.any(q0 < self.lower - 1e-2) or np.any(q0 > self.upper + 1e-2):
            return R2IKResult(
                success=False,
                q_target=q0.copy(),
                position_error=float("inf"),
                rotation_error=float("inf"),
                elapsed_s=time.monotonic() - started,
                reason="seed_out_of_limits",
            )

        q = np.clip(q0, self.lower, self.upper)
        waist_tf = _compute_r2_waist_base_transform(self._current_waist_qpos())
        raw_target_p = target_mat[:3, 3]
        raw_target_R = target_mat[:3, :3]
        p0, R0, Jp0, Jr0 = _compute_r2_arm_fk_and_jacobian_matrix(
            self.side, q, waist_transform=waist_tf
        )
        init_pos_error = float(np.linalg.norm(raw_target_p - p0))
        if init_pos_error > 1.20:
            return R2IKResult(
                success=False,
                q_target=q0.copy(),
                position_error=init_pos_error,
                rotation_error=float("inf"),
                elapsed_s=time.monotonic() - started,
                reason="residual",
            )

        dp = raw_target_p - p0
        if init_pos_error > 0.10:
            target_p = p0 + dp * (0.10 / init_pos_error)
        else:
            target_p = raw_target_p

        dr = _so3_error_rotvec(raw_target_R @ R0.T)
        init_rot_error = float(np.linalg.norm(dr))
        if init_rot_error > 0.35:
            target_R = _rotvec_to_matrix(dr * (0.35 / init_rot_error)) @ R0
        else:
            target_R = raw_target_R

        pos_error = float("inf")
        rot_error = float("inf")
        damping_sq = self.config.damping**2
        eye6 = np.eye(6, dtype=np.float64)

        fine_pos_tol = min(self.config.position_tolerance, 1e-4)
        fine_rot_tol = min(self.config.rotation_tolerance, 1e-3)
        lower = self.lower.copy()
        upper = self.upper.copy()
        if self.side == "left":
            upper[3] = max(float(q0[3]), -0.08)
        else:
            lower[3] = min(float(q0[3]), 0.08)
        err_scale = min(1.0, (init_pos_error + 0.1 * init_rot_error) / 0.025)
        ns_alpha = 0.30 * self.config.nullspace_gain * err_scale
        q_reg = q0 + ns_alpha * (self._nominal - q0)

        clamped_any = False
        err_6d = np.empty(6, dtype=np.float64)
        J_full = np.empty((6, 7), dtype=np.float64)
        for it in range(self.config.max_iters):
            if it == 0:
                p, curr_R, Jp, Jr = p0, R0, Jp0, Jr0
            else:
                p, curr_R, Jp, Jr = _compute_r2_arm_fk_and_jacobian_matrix(
                    self.side, q, waist_transform=waist_tf
                )
            err_p = target_p - p
            err_r = _so3_error_rotvec(target_R @ curr_R.T)
            pos_error = float(np.linalg.norm(err_p))
            rot_error = float(np.linalg.norm(err_r))
            if pos_error <= fine_pos_tol and rot_error <= fine_rot_tol:
                return R2IKResult(
                    success=True,
                    q_target=q.copy(),
                    position_error=pos_error,
                    rotation_error=rot_error,
                    elapsed_s=time.monotonic() - started,
                    reason=None,
                )

            if clamped_any and pos_error > self.config.position_tolerance:
                rot_w = 0.15
            else:
                rot_w = 0.40
            err_6d[:3] = err_p
            err_6d[3:] = rot_w * err_r
            J_full[:3] = Jp
            J_full[3:] = rot_w * Jr
            grad = J_full.T @ err_6d
            clamped = ((q <= lower + 1e-4) & (grad < 0.0)) | (
                (q >= upper - 1e-4) & (grad > 0.0)
            )
            clamped_any = bool(np.any(clamped))
            if clamped_any:
                J_eff = J_full.copy()
                J_eff[:, clamped] = 0.0
            else:
                J_eff = J_full

            JW_eff = J_eff * self._weights_vec
            JJT = JW_eff @ J_eff.T + damping_sq * eye6
            if self.config.nullspace_gain > 0.0:
                dq_null_pref = q_reg - q
                if clamped_any:
                    dq_null_pref = dq_null_pref.copy()
                    dq_null_pref[clamped] = 0.0
                dq = dq_null_pref + JW_eff.T @ np.linalg.solve(
                    JJT, err_6d - J_eff @ dq_null_pref
                )
            else:
                dq = JW_eff.T @ np.linalg.solve(JJT, err_6d)
            max_dq = float(np.max(np.abs(dq)))
            if max_dq > 0.15:
                dq = dq * (0.15 / max_dq)
            step_gain = (
                max(self.config.step_size, 0.85) if it < 12 else self.config.step_size
            )
            q_next = np.clip(q + step_gain * dq, lower, upper)
            if float(np.max(np.abs(q_next - q))) < 1e-5:
                q = q_next
                break
            q = q_next

        p, curr_R, _, _ = _compute_r2_arm_fk_and_jacobian_matrix(
            self.side, q, waist_transform=waist_tf
        )
        pos_error = float(np.linalg.norm(target_p - p))
        rot_error = float(np.linalg.norm(_so3_error_rotvec(target_R @ curr_R.T)))
        converged = (
            pos_error <= self.config.position_tolerance
            and rot_error <= self.config.rotation_tolerance
        ) or (pos_error <= 0.12 and rot_error <= 0.45)
        return R2IKResult(
            success=converged,
            q_target=q.copy() if converged else q0.copy(),
            position_error=pos_error,
            rotation_error=rot_error,
            elapsed_s=time.monotonic() - started,
            reason=None if converged else "residual",
        )


#: RFrame camera topics by camera name; ``resolution`` is ``(width, height)``.
R2_CAMERA_TOPICS: dict[str, dict[str, Any]] = {
    name: {
        "color": f"/robby/sensors/camera/{topic}/color/raw",
        "compressed": f"/robby/sensors/camera/{topic}/color/compressed",
        "depth": f"/robby/sensors/camera/{topic}/depth/raw",
        "resolution": resolution,
    }
    for name, topic, resolution in (
        ("head", "head", (1280, 720)),
        ("wrist_left", "wrist_left", (640, 480)),
        ("wrist_right", "wrist_right", (640, 480)),
        ("navigation_front", "chassis_front", (640, 480)),
        ("navigation_rear", "chassis_rear", (640, 480)),
    )
}

_SHARED_RFRAME_APP: Any = None
_SHARED_RFRAME_LOCK = threading.Lock()


def _get_shared_rframe_app(rframe_mod: Any) -> Any:
    """Return a process-wide initialized ``rframe.App`` singleton."""
    global _SHARED_RFRAME_APP
    with _SHARED_RFRAME_LOCK:
        if os.path.isfile("/opt/robbyant/dynamic_participant.xml"):
            os.environ.setdefault(
                "FASTRTPS_DEFAULT_PROFILES_FILE",
                "/opt/robbyant/dynamic_participant.xml",
            )
            os.environ.pop("RMW_IMPLEMENTATION", None)
            os.environ.pop("ROS_DOMAIN_ID", None)
        if _SHARED_RFRAME_APP is None:
            app = rframe_mod.App()
            if hasattr(app, "init"):
                app.init()
            _SHARED_RFRAME_APP = app
        return _SHARED_RFRAME_APP


def _joint_values(source: Mapping[str, float], names: Sequence[str]) -> np.ndarray:
    """Read ``names`` from a ``joint -> value`` map, zero where a joint is missing."""
    return np.array([source.get(name, 0.0) for name in names], dtype=np.float32)


def _ros2_type_name(msg_cls: Any) -> str:
    """Derive a ``pkg/msg/Type`` string from a ROS2 Python message class."""
    mod = getattr(msg_cls, "__module__", "")
    parts = mod.split(".")
    if len(parts) >= 2 and parts[1] in ("msg", "srv"):
        return f"{parts[0]}/{parts[1]}/{msg_cls.__name__}"
    return msg_cls.__name__


def _pose_to_xyz_rpy(pose: Any) -> np.ndarray:
    """Convert a ROS geometry_msgs/Pose into a 6-element [x, y, z, r, p, y] array."""
    pos = pose.position
    ori = pose.orientation
    qx = float(ori.x)
    qy = float(ori.y)
    qz = float(ori.z)
    qw = float(ori.w)
    norm_sq = qx * qx + qy * qy + qz * qz + qw * qw
    if norm_sq < 1e-16:
        return np.array([pos.x, pos.y, pos.z, 0.0, 0.0, 0.0], dtype=np.float32)
    inv_norm = 1.0 / math.sqrt(norm_sq)
    qx *= inv_norm
    qy *= inv_norm
    qz *= inv_norm
    qw *= inv_norm
    sy = 2.0 * (qy * qw - qx * qz)
    if abs(sy) < 0.999999:
        roll = math.atan2(2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy))
        pitch = math.asin(sy)
        yaw = math.atan2(2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qy * qy + qz * qz))
        return np.array([pos.x, pos.y, pos.z, roll, pitch, yaw], dtype=np.float32)
    rpy = R.from_quat([qx, qy, qz, qw]).as_euler("xyz", degrees=False)
    return np.array([pos.x, pos.y, pos.z, rpy[0], rpy[1], rpy[2]], dtype=np.float32)


def _as_byte_view(data: Any) -> memoryview | bytes:
    """Return a zero-copy ``memoryview`` when ``data`` supports the buffer protocol."""
    if data is None:
        return b""
    try:
        return memoryview(data).cast("B")
    except TypeError:
        return bytes(data)


def _decode_image_message(
    msg: Any,
    *,
    compressed: bool = False,
    depth: bool = False,
    fallback_resolution: tuple[int, int] = (640, 480),
) -> np.ndarray:
    """Decode a sensor_msgs/Image or CompressedImage message into a NumPy array."""
    width_default, height_default = fallback_resolution
    raw = _as_byte_view(getattr(msg, "data", b""))
    if compressed:
        if len(raw) == 0:
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

    if (
        encoding in ("16uc1", "yuyv", "yuv422", "yuv422_yuy2")
        or len(raw) == height * width * 2
    ):
        count = height * width * 2
        if len(raw) >= count:
            yuyv = np.frombuffer(raw[:count], dtype=np.uint8).reshape(
                (height, width, 2)
            )
            try:
                import cv2

                return np.ascontiguousarray(
                    cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUY2), dtype=np.uint8
                )
            except ImportError:
                y = yuyv[:, :, 0].astype(np.float32)
                u = np.repeat(yuyv[:, 0::2, 1].astype(np.float32) - 128.0, 2, axis=1)
                v = np.repeat(yuyv[:, 1::2, 1].astype(np.float32) - 128.0, 2, axis=1)
                r = np.clip(y + 1.402 * v, 0, 255).astype(np.uint8)
                g = np.clip(y - 0.344136 * u - 0.714136 * v, 0, 255).astype(np.uint8)
                b = np.clip(y + 1.772 * u, 0, 255).astype(np.uint8)
                return np.ascontiguousarray(np.stack([r, g, b], axis=-1))

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
) -> tuple[Optional[int], Optional[str]]:
    """Infer ``(body_dof, end_effector_type)`` from a ``JointState`` name list.

    Returns ``None`` for either field when the names do not distinguish the
    hardware variant.
    """
    name_set = set(names)
    body_dof: Optional[int] = None
    end_effector_type: Optional[str] = None
    if any(j in name_set for j in R2_WAIST_JOINTS + R2_LEFT_ARM_JOINTS):
        body_dof = 21 if set(R2_NECK_JOINTS) <= name_set else 20
        if any(j in name_set for j in R2_GRIPPER_JOINTS):
            end_effector_type = "gripper"
        elif any(j.startswith(("HandL_", "HandR_")) for j in name_set):
            end_effector_type = "hand"
        else:
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
        return {
            k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in self.__dict__.items()
        }


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
                "tcp_pose": f"servo_{side}_arm_cartesian",
                "move_cartesian": f"move_{side}_arm_cartesian",
                "move_Cartesian": f"move_{side}_arm_cartesian",
                "tcp_velocity": f"move_{side}_arm_velocity",
                "move_cartesian_velocity": f"move_{side}_arm_velocity",
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
        """Move the arm smoothly to ``positions`` in at least ``duration`` seconds."""
        return bool(
            self._host.move_body_smooth(
                **{f"{self.side}_arm": positions},
                duration=duration,
                max_velocity=max_velocity,
            )
        )

    def reset_dual_joint(
        self,
        left_positions: Sequence[float] | np.ndarray,
        right_positions: Sequence[float] | np.ndarray,
        duration: float = 3.0,
        *,
        max_velocity: Optional[float | Sequence[float] | np.ndarray] = None,
    ) -> bool:
        """Move both arms together and smoothly to their target positions."""
        return bool(
            self._host.move_body_smooth(
                left_arm=left_positions,
                right_arm=right_positions,
                duration=duration,
                max_velocity=max_velocity,
            )
        )

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

    def stop(self) -> None:
        """Zero the chassis velocity."""
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
    """Shared RFrame session managing R2 arms, end effectors, chassis, and cameras.

    Joint commands only update targets: a :class:`JointStream` streams them
    to ``/robotic/servoj_data`` at the motion profile's rate, within the
    joint limits of :attr:`joint_group_specs`. Hands and the chassis take
    their commands directly on their own topics.
    """

    #: Order of the body joints the stream drives.
    BODY_GROUPS: tuple[str, ...] = ("waist", "neck", "left_arm", "right_arm")

    def __init__(
        self,
        *,
        body_dof: Optional[int | str] = None,
        end_effector_type: Optional[str] = "gripper",
        enable_base: bool = True,
        cameras: Sequence[str] = ("head", "wrist_left", "wrist_right"),
        enable_depth: bool = False,
        use_compressed_camera: bool = False,
        controller_mode: int = R2_MODE_SERVOJ,
        enable_fault_recovery: bool = True,
        max_recovery_attempts: int = 3,
        recovery_cooldown_window: float = 10.0,
        motion_profile: Optional[MotionProfile | Mapping[str, Any]] = None,
        gripper_min_closure: Sequence[float] = (0.0, 0.15),
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
                "R2 end_effector_type must be 'gripper', 'hand', 'none', or "
                f"'auto'/None, got {end_effector_type!r}."
            )
        if controller_mode not in R2_CONTROLLER_MODES:
            raise ValueError(
                f"R2 controller_mode must be one of {sorted(R2_CONTROLLER_MODES)}, "
                f"got {controller_mode!r}."
            )
        unknown_cameras = [c for c in cameras if c not in R2_CAMERA_TOPICS]
        if unknown_cameras:
            raise ValueError(
                f"Unknown R2 camera names {unknown_cameras}; "
                f"valid names are {sorted(R2_CAMERA_TOPICS)}."
            )
        closure = tuple(float(x) for x in gripper_min_closure)
        if len(closure) != 2 or any(not 0.0 <= x < 1.0 for x in closure):
            raise ValueError(
                "R2 gripper_min_closure must be (left, right) closures in [0, 1), "
                f"got {gripper_min_closure!r}."
            )
        self.body_dof = resolved_body_dof
        self.end_effector_type = resolved_ee_type
        self.detected_body_dof: Optional[int] = None
        self.detected_end_effector_type: Optional[str] = None
        self.enable_base = enable_base
        self.camera_names = tuple(cameras)
        self.enable_depth = enable_depth
        self.use_compressed_camera = use_compressed_camera
        self.controller_mode = controller_mode
        self.gripper_min_closure = closure
        self.node_name = node_name
        self._logger = get_logger()
        self.configure_fault_recovery(
            enable_fault_recovery=enable_fault_recovery,
            max_recovery_attempts=max_recovery_attempts,
            recovery_cooldown_window=recovery_cooldown_window,
        )
        self._motion_profile = MotionProfile(
            max_joint_velocity=R2_MAX_JOINT_VELOCITY,
            max_joint_acceleration=R2_MAX_JOINT_ACCELERATION,
        )
        self.configure_motion_profile(motion_profile)

        self._lock = threading.Lock()
        self._state = R2RobotState()
        self._joint_map_positions: dict[str, float] = {}
        self._joint_map_velocities: dict[str, float] = {}
        self._joint_map_efforts: dict[str, float] = {}
        self._state_ready = threading.Event()
        self._camera_frames: dict[str, np.ndarray] = {}
        self._camera_depths: dict[str, np.ndarray] = {}
        self._camera_raw_frames: dict[str, tuple[Any, bool, tuple[int, int]]] = {}
        self._camera_raw_depths: dict[str, tuple[Any, tuple[int, int]]] = {}
        self._camera_events: dict[str, threading.Event] = {
            name: threading.Event() for name in self.camera_names
        }

        self._session: Any = None
        self._rframe: Any = None
        self._rclpy: Any = None
        self._rclpy_node: Any = None
        self._readers: list[Any] = []
        self._writers: dict[str, tuple[Any, Any]] = {}
        self._services: dict[str, tuple[Any, Any]] = {}

        # Commanded targets: body groups stream, grippers ride in the same
        # frame, and hands go out on their own topic.
        self._targets: dict[str, np.ndarray] = {}
        self._gripper_targets: dict[str, float] = {}
        self._hand_targets: dict[str, np.ndarray] = {}
        self._targets_from_joint_state = False
        self._stream = JointStream(
            send=self._send_frame,
            measure=self._measured_body,
            faulted=lambda: self._has_active_fault(check_mode=False),
            limits=self._joint_limits,
            profile=lambda: self.motion_profile,
            on_start=self._ensure_servo_mode,
            stopping=self._ros_stopped,
            name="R2 servo loop",
        )
        self._staged_targets: dict[str, np.ndarray] = {}
        self._staged_grippers: dict[str, float] = {}
        self._staged_move: Optional[dict[str, Any]] = None
        self._flushing_command_batch = False
        self._last_ik_warning = -float("inf")

        self._chassis_cmd_started = False
        self._base_pulse_stop = threading.Event()
        self._last_cmd_base_vel = np.zeros(3, dtype=np.float32)
        self._zero_base_frames_sent = 3
        self._active_mode: Optional[int] = None
        self._system_state_code = 2
        self._system_main_state: Optional[int] = None
        self._last_error_code = 0
        self._last_error_msg = ""
        self._health_fault = ""
        self._recovery_timestamps: list[float] = []

    @property
    def joint_group_specs(self) -> dict[str, JointGroupSpec]:
        """Return the :class:`JointGroupSpec` table for R2 body joint groups."""
        return {
            "waist": JointGroupSpec(
                name="waist",
                dim=len(R2_WAIST_JOINTS),
                low=R2_WAIST_LIMITS_LOW,
                high=R2_WAIST_LIMITS_HIGH,
                max_velocity=R2_WAIST_MAX_VELOCITY,
                max_acceleration=R2_WAIST_MAX_ACCELERATION,
            ),
            "neck": JointGroupSpec(
                name="neck",
                dim=len(R2_NECK_JOINTS),
                low=R2_NECK_LIMITS_LOW,
                high=R2_NECK_LIMITS_HIGH,
                max_velocity=R2_NECK_MAX_VELOCITY,
                max_acceleration=R2_NECK_MAX_ACCELERATION,
            ),
            "left_arm": JointGroupSpec(
                name="left_arm",
                dim=7,
                low=R2_LEFT_ARM_LIMITS_LOW,
                high=R2_LEFT_ARM_LIMITS_HIGH,
            ),
            "right_arm": JointGroupSpec(
                name="right_arm",
                dim=7,
                low=R2_RIGHT_ARM_LIMITS_LOW,
                high=R2_RIGHT_ARM_LIMITS_HIGH,
            ),
        }

    @classmethod
    def probe_hardware(
        cls,
        *,
        timeout: float = 2.5,
        default_body_dof: int = 21,
        default_end_effector_type: str = "gripper",
    ) -> tuple[int, str]:
        """Detect ``(body_dof, end_effector_type)`` from ``/robotic/joint_states``.

        Returns the defaults when the RFrame SDK is missing or no joint state
        arrives within ``timeout`` seconds.
        """
        try:
            import rframe
            from sensor_msgs.msg import JointState
        except ImportError:
            return default_body_dof, default_end_effector_type
        detected: dict[str, Any] = {}
        ready = threading.Event()

        def _on_msg(msg: Any) -> None:
            dof, ee = infer_r2_hardware(list(getattr(msg, "name", [])))
            if dof is not None:
                detected["body_dof"] = dof
            if ee is not None:
                detected["end_effector_type"] = ee
            ready.set()

        _get_shared_rframe_app(rframe)
        channel = rframe.HybridChannel(
            rframe.ChannelConfig(
                topic="/robotic/joint_states",
                handler_type=rframe.HandlerType.READER,
                channel_type=rframe.ChannelType.HYBRID,
                enable_shm=True,
                enable_ros2=True,
                type_name="sensor_msgs/msg/JointState",
                ros2_qos={"reliability": "best_effort"},
            )
        )
        try:
            channel.register_typed_callback(JointState, _on_msg)
            ready.wait(timeout=timeout)
        finally:
            channel.close()
        return (
            detected.get("body_dof", default_body_dof),
            detected.get("end_effector_type", default_end_effector_type),
        )

    @property
    def parts(self) -> dict[str, RobotPart]:
        """Decompose the shared R2 connection into arms, end effectors, base, and cameras."""
        parts: dict[str, RobotPart] = {
            "left": R2Arm(self, "left"),
            "right": R2Arm(self, "right"),
        }
        if self.end_effector_type == "gripper":
            for side in ("left", "right"):
                parts[f"{side}_end_effector"] = MethodEndEffector(
                    self,
                    f"{side}_gripper_position",
                    dims=1,
                    command=f"move_{side}_gripper",
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

    # Command batching

    def _on_command_batch_begin(self) -> None:
        """Start staging targets for one combined command."""
        self._on_command_batch_cancel()

    def _on_command_batch_cancel(self) -> None:
        """Discard staged targets."""
        self._staged_targets.clear()
        self._staged_grippers.clear()
        self._staged_move = None

    def _flush_command_batch(self) -> bool:
        """Send the staged targets as one streamed goal or one timed move."""
        targets = dict(self._staged_targets)
        grippers = dict(self._staged_grippers)
        move = self._staged_move
        self._on_command_batch_cancel()
        self._flushing_command_batch = True
        try:
            if move is not None:
                return self._move_body(targets, grippers, **move)
            if targets or grippers:
                self._command(targets, grippers)
            return True
        finally:
            self._flushing_command_batch = False

    def _staging(self) -> bool:
        return self.in_command_batch and not self._flushing_command_batch

    def _stage(self, targets: Mapping[str, Any], grippers: Mapping[str, float]) -> None:
        for group, values in targets.items():
            self._staged_targets[group] = np.asarray(values, dtype=np.float32).copy()
        self._staged_grippers.update(grippers)

    # Connection lifecycle

    def _subscribe(
        self,
        topic: str,
        msg_cls: Any,
        callback: Any,
        *,
        reliability: str = "reliable",
        memory_size: int = 65536,
    ) -> Any:
        """Register a typed reader channel on ``topic``."""
        rframe = self._rframe
        channel = rframe.HybridChannel(
            rframe.ChannelConfig(
                topic=topic,
                handler_type=rframe.HandlerType.READER,
                channel_type=rframe.ChannelType.HYBRID,
                memory_size=memory_size,
                enable_shm=True,
                enable_ros2=True,
                type_name=_ros2_type_name(msg_cls),
                ros2_qos={"reliability": reliability},
            )
        )
        channel.register_typed_callback(msg_cls, callback)
        self._readers.append(channel)
        return channel

    def _advertise(self, topic: str, msg_cls: Any) -> None:
        """Create a writer channel on ``topic``."""
        rframe = self._rframe
        channel = rframe.HybridChannel(
            rframe.ChannelConfig(
                topic=topic,
                handler_type=rframe.HandlerType.WRITER,
                channel_type=rframe.ChannelType.HYBRID,
                enable_shm=True,
                enable_ros2=True,
                type_name=_ros2_type_name(msg_cls),
            )
        )
        self._writers[topic] = (channel, msg_cls)

    def _client(self, service_name: str, srv_cls: Any) -> None:
        """Create an rclpy client for ``service_name``."""
        client = self._rclpy_node.create_client(srv_cls, service_name)
        self._services[service_name] = (client, srv_cls)

    def _open(self) -> Any:
        """Open the RFrame session, releasing what was opened if any step fails."""
        try:
            return self._open_session()
        except BaseException:
            self._release(self._session)
            raise

    def _open_session(self) -> Any:
        """Subscribe to state, advertise command topics, and bring the controller up."""
        import rclpy
        import rframe
        from geometry_msgs.msg import Twist, WrenchStamped
        from navigation_msgs.msg import NaviState
        from r2_motion_msgs.msg import TCP, ServoJ, ServoRYHandDual
        from r2_motion_msgs.srv import GetSystemState, SwitchMode
        from rdiagnostic_msgs.msg import HealthState
        from sensor_msgs.msg import (
            BatteryState,
            CompressedImage,
            Image,
            Imu,
            JointState,
        )
        from std_srvs.srv import Trigger

        self._rframe = rframe
        self._session = _get_shared_rframe_app(rframe)
        self._subscribe(
            "/robotic/joint_states",
            JointState,
            self._on_joint_state,
            reliability="best_effort",
        )
        discovery_started = time.monotonic()
        self._advertise("/robotic/servoj_data", ServoJ)
        self._advertise("/robotic/servo_ry_hand_data", ServoRYHandDual)
        if self.enable_base:
            self._advertise("/chassis/cmd_vel", Twist)
        self._subscribe("/robotic/tcp_data", TCP, self._on_tcp)
        self._subscribe("/robotic/ft_sensor_left", WrenchStamped, self._on_left_wrench)
        self._subscribe(
            "/robotic/ft_sensor_right", WrenchStamped, self._on_right_wrench
        )
        self._subscribe("/robotic/Imu", Imu, self._on_imu, reliability="best_effort")
        self._subscribe(
            "/battery_state", BatteryState, self._on_battery, reliability="best_effort"
        )
        self._subscribe(
            "/diagnostic",
            HealthState,
            self._on_health_state,
            reliability="best_effort",
        )
        try:
            from state_machine_msgs.msg import SystemState

            self._subscribe(
                "/system_state",
                SystemState,
                self._on_system_state,
                reliability="best_effort",
            )
        except ImportError:
            self._logger.warning(
                "state_machine_msgs.msg.SystemState is unavailable, so an R2 "
                "emergency stop is detected only through motion error code 401."
            )
        if self.enable_base:
            self._subscribe(
                "/navi/state", NaviState, self._on_navi_state, reliability="best_effort"
            )
            try:
                from nav_msgs.msg import Odometry

                self._subscribe(
                    "/chassis/est_odom",
                    Odometry,
                    self._on_navi_state,
                    reliability="best_effort",
                )
            except ImportError:
                pass
        for cam_name in self.camera_names:
            spec = R2_CAMERA_TOPICS[cam_name]
            resolution = spec["resolution"]
            self._subscribe(
                spec["compressed" if self.use_compressed_camera else "color"],
                CompressedImage if self.use_compressed_camera else Image,
                self._make_camera_callback(
                    cam_name,
                    compressed=self.use_compressed_camera,
                    depth=False,
                    resolution=resolution,
                ),
                reliability="best_effort",
                memory_size=(4 if self.use_compressed_camera else 8) * 1024 * 1024,
            )
            if self.enable_depth:
                self._subscribe(
                    spec["depth"],
                    Image,
                    self._make_camera_callback(
                        cam_name, compressed=False, depth=True, resolution=resolution
                    ),
                    reliability="best_effort",
                    memory_size=8 * 1024 * 1024,
                )

        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy
        self._rclpy_node = rclpy.create_node(self.node_name)
        self._client("/robotic/switch_mode_cmd", SwitchMode)
        self._client("/robotic/get_system_state_cmd", GetSystemState)
        self._client("/robotic/release_stop", Trigger)
        for module, name, service in (
            ("r2_motion_msgs.srv", "SetAxisEnable", "/robotic/set_axis_enable_cmd"),
            ("r2_motion_msgs.srv", "SetGripperParam", "/robotic/set_gripper_param_cmd"),
            ("rdiagnostic_msgs.srv", "ClearFaults", "/clear_faults"),
            ("system_msgs.srv", "PowerCtrl", "/power_ctrl"),
            ("r2_sensor_msgs.srv", "SetLEDMode", "/head/set_led_mode"),
            ("r2_sensor_msgs.srv", "SetLEDColor", "/head/set_led_color"),
            ("std_srvs.srv", "SetBool", "/chassis/cmd_start"),
        ):
            if service == "/chassis/cmd_start" and not self.enable_base:
                continue
            try:
                self._client(service, getattr(import_module(module), name))
            except (ImportError, AttributeError):
                pass

        response = self._call_service(
            "/robotic/get_system_state_cmd", lambda _req: None, timeout=2.0
        )
        if response is not None:
            with self._lock:
                self._system_state_code = int(getattr(response, "state", 2))
                if hasattr(response, "run_mode"):
                    self._active_mode = int(response.run_mode)
        if not self._state_ready.wait(timeout=5.0):
            raise RuntimeError(
                "R2 published no /robotic/joint_states within 5 s, so there is "
                "no measured pose to start from."
            )
        for cam_name in self.camera_names:
            self._camera_events[cam_name].wait(timeout=2.0)
        if self._has_active_fault(check_mode=False):
            # Connecting clears only what automatic recovery may clear;
            # recover() refuses an emergency stop.
            if self.enable_fault_recovery:
                self.recover(trigger="connect")
            else:
                self._logger.warning(
                    "R2 is faulted at connect and enable_fault_recovery is "
                    f"False, so it stays faulted: {self.diagnose().error_message}"
                )
        else:
            self._configure_grippers()
            self.switch_controller(self.controller_mode)
        # RFrame routes new writers over RTPS only after discovery settles.
        settle = 1.8 - (time.monotonic() - discovery_started)
        if settle > 0.0:
            time.sleep(settle)
        self._sync_cmd_to_actual()
        self._stream.start()
        return self._session

    def _release(self, device: Any) -> None:
        """Stop streaming, close every channel, and forget the commanded state."""
        self._stream.close()
        if "/chassis/cmd_vel" in self._writers:
            try:
                self.stop_base()
            except Exception:
                pass
            if self._chassis_cmd_started and "/chassis/cmd_start" in self._services:
                try:
                    self._call_service(
                        "/chassis/cmd_start",
                        lambda req: setattr(req, "data", False),
                        timeout=1.0,
                    )
                except Exception:
                    pass
        for channel in self._readers:
            channel.close()
        for channel, _ in self._writers.values():
            channel.close()
        if self._rclpy_node is not None:
            try:
                self._rclpy_node.destroy_node()
            except Exception:
                pass
        self._rclpy_node = None
        self._readers.clear()
        self._writers.clear()
        self._services.clear()
        self._camera_frames.clear()
        self._camera_depths.clear()
        self._camera_raw_frames.clear()
        self._camera_raw_depths.clear()
        self._targets.clear()
        self._gripper_targets.clear()
        self._hand_targets.clear()
        self._targets_from_joint_state = False
        self._chassis_cmd_started = False
        self._active_mode = None
        self._system_state_code = 2
        self._system_main_state = None
        self._last_error_code = 0
        self._last_error_msg = ""
        self._health_fault = ""
        self._recovery_timestamps.clear()
        self._last_recovery_event = None
        self._session = None

    # State callbacks

    def _on_joint_state(self, msg: Any) -> None:
        names = list(getattr(msg, "name", []))
        positions = list(getattr(msg, "position", []))
        velocities = list(getattr(msg, "velocity", []))
        efforts = list(getattr(msg, "effort", []))
        detected_dof, detected_ee = infer_r2_hardware(names)
        with self._lock:
            if detected_dof is not None:
                self.detected_body_dof = detected_dof
                if self._auto_body_dof:
                    self.body_dof = detected_dof
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
            if not names:
                return
            pos, vel, eff = (
                self._joint_map_positions,
                self._joint_map_velocities,
                self._joint_map_efforts,
            )
            state = self._state
            state.left_joint_positions = _joint_values(pos, R2_LEFT_ARM_JOINTS)
            state.left_joint_velocities = _joint_values(vel, R2_LEFT_ARM_JOINTS)
            state.left_joint_torques = _joint_values(eff, R2_LEFT_ARM_JOINTS)
            state.right_joint_positions = _joint_values(pos, R2_RIGHT_ARM_JOINTS)
            state.right_joint_velocities = _joint_values(vel, R2_RIGHT_ARM_JOINTS)
            state.right_joint_torques = _joint_values(eff, R2_RIGHT_ARM_JOINTS)
            state.waist_positions = _joint_values(pos, R2_WAIST_JOINTS)
            state.waist_velocities = _joint_values(vel, R2_WAIST_JOINTS)
            state.neck_positions = _joint_values(pos, R2_NECK_JOINTS)
            state.neck_velocities = _joint_values(vel, R2_NECK_JOINTS)
            # The controller reports closure; RLinf's grippers report opening.
            for side, joint in zip(("left", "right"), R2_GRIPPER_JOINTS, strict=True):
                if joint in pos:
                    opening = np.clip(1.0 - pos[joint], 0.0, 1.0)
                    setattr(
                        state,
                        f"{side}_gripper_position",
                        np.array([opening], dtype=np.float32),
                    )
            for side, joints in (
                ("left", R2_HAND_JOINTS_LEFT),
                ("right", R2_HAND_JOINTS_RIGHT),
            ):
                if any(j in pos for j in joints):
                    setattr(state, f"{side}_hand_position", _joint_values(pos, joints))
            self._state_ready.set()

    def _on_tcp(self, msg: Any) -> None:
        with self._lock:
            if hasattr(msg, "pose_l"):
                self._state.left_tcp_pose = _pose_to_xyz_rpy(msg.pose_l)
            if hasattr(msg, "pose_r"):
                self._state.right_tcp_pose = _pose_to_xyz_rpy(msg.pose_r)

    def _make_camera_callback(
        self,
        camera_name: str,
        *,
        compressed: bool,
        depth: bool,
        resolution: tuple[int, int],
    ) -> Any:
        def _callback(msg: Any) -> None:
            with self._lock:
                if depth:
                    self._camera_raw_depths[camera_name] = (msg, resolution)
                    self._camera_depths.pop(camera_name, None)
                else:
                    self._camera_raw_frames[camera_name] = (msg, compressed, resolution)
                    self._camera_frames.pop(camera_name, None)
                    self._camera_events[camera_name].set()

        return _callback

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

    def _on_navi_state(self, msg: Any) -> None:
        with self._lock:
            if hasattr(msg, "current_pose"):
                cp = msg.current_pose
                self._state.base_pose = np.array(
                    [
                        float(getattr(cp, "x", 0.0)),
                        float(getattr(cp, "y", 0.0)),
                        float(getattr(cp, "theta", 0.0)),
                    ],
                    dtype=np.float32,
                )
            elif hasattr(msg, "pose"):
                inner_pose = msg.pose.pose if hasattr(msg.pose, "pose") else msg.pose
                pose_vec = _pose_to_xyz_rpy(inner_pose)
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
            elif hasattr(msg, "twist"):
                tw = msg.twist.twist if hasattr(msg.twist, "twist") else msg.twist
                self._state.base_velocity = np.array(
                    [tw.linear.x, tw.linear.y, tw.angular.z], dtype=np.float32
                )

    def _on_health_state(self, msg: Any) -> None:
        errors = list(getattr(msg, "errors", []) or [])
        motion_faults = [
            err
            for err in errors
            if int(getattr(err, "severity_level", 0)) >= 4
            and str(getattr(err, "code_str", "")).startswith(("M.", "EC", "C.MC"))
        ]
        with self._lock:
            self._state.is_healthy = not motion_faults
            if not motion_faults:
                return
            code_str = str(getattr(motion_faults[0], "code_str", ""))
            msg_str = str(getattr(motion_faults[0], "message", ""))
            detail = f"{code_str}: {msg_str}" if msg_str else code_str
            # RobbyDiagnostic drops a transient fault from the list within a
            # second, so the first one stays latched until clear_errors().
            if "804" in code_str or "COLLISION" in msg_str.upper():
                if self._last_error_code == 0:
                    self._latch_error_locked(304, detail)
            elif not self._health_fault:
                self._health_fault = detail
            self._active_mode = R2_MODE_AUTO

    def _on_system_state(self, msg: Any) -> None:
        main_state = int(getattr(msg, "current_main_state", 0))
        with self._lock:
            self._system_main_state = main_state
            # Latching means an emergency stop is never left behind silently:
            # after a person releases it, they call clear_errors() to resume.
            if main_state == R2_SYSTEM_STATE_EMERGENCY_STOP:
                self._latch_error_locked(401, "emergency stop (system_state)")

    def _latch_error_locked(self, code: int, message: str) -> None:
        """Latch a motion controller fault code; the caller holds ``self._lock``.

        Status codes are not faults, and a rejected request leaves the
        controller healthy, so neither is latched.
        """
        if code in R2_STATUS_CODES:
            return
        if code in R2_REQUEST_ERROR_CODES:
            self._logger.warning(
                f"R2 motion controller rejected a request with {code} "
                f"({R2_ERROR_CODES[code]}): {message}"
            )
            return
        self._last_error_code = code
        self._last_error_msg = message

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
        return dict(self.get_state().__dict__)

    def consume_recovery_event(self) -> Optional[dict[str, Any]]:
        """Pop and return the most recent automatic recovery event, if any."""
        if self._owner is not None:
            return self._owner.consume_recovery_event()
        with self._lock:
            event = self._last_recovery_event
            self._last_recovery_event = None
            return event

    # Commanded targets

    def _sync_cmd_to_actual(self, state: Optional[R2RobotState] = None) -> None:
        """Set every target to the measured state and stop streaming.

        The stream resumes from the measured joints on the next command, so a
        recovery or mode change leaves the robot holding where it stands.
        """
        snap = state if state is not None else self.get_state()
        with self._lock:
            from_joint_state = bool(self._joint_map_positions)
        self._targets = {
            "waist": snap.waist_positions.copy(),
            "neck": snap.neck_positions.copy(),
            "left_arm": snap.left_joint_positions.copy(),
            "right_arm": snap.right_joint_positions.copy(),
        }
        self._gripper_targets = {
            "left": float(snap.left_gripper_position[0]),
            "right": float(snap.right_gripper_position[0]),
        }
        self._hand_targets = {
            "left": snap.left_hand_position.copy(),
            "right": snap.right_hand_position.copy(),
        }
        self._targets_from_joint_state = from_joint_state
        self._stream.hold()

    def _ensure_targets(self) -> None:
        """Start targets from the measured state, or refresh ones set before it arrived."""
        with self._lock:
            have_joint_state = bool(self._joint_map_positions)
        if not self._targets or (
            have_joint_state and not self._targets_from_joint_state
        ):
            self._sync_cmd_to_actual()

    def _apply_targets(
        self, targets: Mapping[str, Any], grippers: Mapping[str, float]
    ) -> None:
        """Clamp and record body group targets and gripper openings."""
        self._ensure_targets()
        specs = self.joint_group_specs
        for group, values in targets.items():
            self._targets[group] = specs[group].clamp(values)
        for side, opening in grippers.items():
            self._gripper_targets[side] = float(np.clip(float(opening), 0.0, 1.0))

    def _arm_target(self, side: str) -> np.ndarray:
        """Return one arm's commanded joints, the seed for its next IK solve."""
        self._ensure_targets()
        return self._targets[f"{side}_arm"].astype(np.float32).copy()

    # Joint stream adapter

    def _body_goal(self) -> np.ndarray:
        return np.concatenate([self._targets[g] for g in self.BODY_GROUPS]).astype(
            np.float64
        )

    def _measured_body(self) -> np.ndarray:
        with self._lock:
            if not self._joint_map_positions:
                raise RuntimeError(
                    "R2 has reported no joint states, so streaming has no "
                    "measured pose to start from."
                )
        state = self.get_state()
        return np.concatenate(
            [
                state.waist_positions,
                state.neck_positions,
                state.left_joint_positions,
                state.right_joint_positions,
            ]
        ).astype(np.float64)

    def _joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        profile = self.motion_profile
        specs = self.joint_group_specs
        groups = [specs[name] for name in self.BODY_GROUPS]
        return (
            np.concatenate([np.full(g.dim, g.velocity_limit(profile)) for g in groups]),
            np.concatenate(
                [np.full(g.dim, g.acceleration_limit(profile)) for g in groups]
            ),
        )

    def _send_frame(self, body: np.ndarray) -> None:
        """Publish one whole-body setpoint to ``/robotic/servoj_data``.

        The frame packs ``waist (5) + left_arm (7) + right_arm (7) + neck``;
        the 20-DoF body sends only neck pitch. Grippers are openings in RLinf
        and closures in ServoJ.
        """
        channel, msg_cls = self._writers["/robotic/servoj_data"]
        waist, neck, left, right = np.split(body, np.cumsum([5, 2, 7]))
        msg = msg_cls()
        msg.joint = [
            float(x)
            for x in np.concatenate(
                [waist, left, right, neck if self.body_dof == 21 else neck[1:]]
            )
        ]
        min_left, min_right = self.gripper_min_closure
        msg.gripper_l = float(
            np.clip(1.0 - self._gripper_targets.get("left", 1.0), min_left, 1.0)
        )
        msg.gripper_r = float(
            np.clip(1.0 - self._gripper_targets.get("right", 1.0), min_right, 1.0)
        )
        channel.send(msg)

    def _ensure_servo_mode(self) -> None:
        """Put the controller in ServoJ mode before streaming starts."""
        if self._active_mode != R2_MODE_SERVOJ:
            self.switch_controller(R2_MODE_SERVOJ)

    def _ros_stopped(self) -> bool:
        # RFrame's own ROS 2 signal handler tears its transport down on
        # Ctrl-C, which rclpy reports as no longer ok.
        return self._rclpy is not None and not self._rclpy.ok()

    # Faults

    def _has_active_fault(self, *, check_mode: bool = True) -> bool:
        """Fast non-RPC check for latched faults, an emergency stop, or a mode drop."""
        with self._lock:
            if self._cached_health is not None and not self._cached_health.healthy:
                return True
            if self._last_error_code != 0 or self._health_fault:
                return True
            if self._system_main_state == R2_SYSTEM_STATE_EMERGENCY_STOP:
                return True
            if self._system_state_code == 4 or not self._state.is_healthy:
                return True
            return (
                check_mode
                and self.controller_mode == R2_MODE_SERVOJ
                and self._active_mode is not None
                and self._active_mode != R2_MODE_SERVOJ
            )

    def diagnose(self) -> HealthStatus:
        """Return a structured hardware diagnostic snapshot for the R2 connection."""
        if self._session is None:
            return HealthStatus(
                healthy=False,
                level=FaultLevel.UNRECOVERABLE_FAULT,
                error_code=-1,
                error_message="R2Connection is not connected.",
            )
        response = self._call_service(
            "/robotic/get_system_state_cmd", lambda _req: None, timeout=0.5
        )
        if response is not None:
            with self._lock:
                self._system_state_code = int(getattr(response, "state", 2))
                if hasattr(response, "run_mode"):
                    self._active_mode = int(response.run_mode)
        state = self.get_state()
        with self._lock:
            err_code = self._last_error_code
            err_msg = self._last_error_msg
            health_fault = self._health_fault
            main_state = self._system_main_state
            sys_state = self._system_state_code
            active_mode = (
                self._active_mode
                if self._active_mode is not None
                else self.controller_mode
            )
        estop = main_state == R2_SYSTEM_STATE_EMERGENCY_STOP
        mode_drop = (
            self.controller_mode == R2_MODE_SERVOJ and active_mode != R2_MODE_SERVOJ
        )
        if estop or err_code in R2_UNRECOVERABLE_ERROR_CODES:
            level = FaultLevel.UNRECOVERABLE_FAULT
        elif err_code != 0 or health_fault or sys_state == 4 or not state.is_healthy:
            level = FaultLevel.RECOVERABLE_FAULT
        elif mode_drop:
            level = FaultLevel.MODE_DROP
        else:
            level = FaultLevel.OK

        reasons: list[str] = []
        if err_code != 0:
            desc = R2_ERROR_CODES.get(err_code, "motion controller error")
            suffix = f" ({err_msg})" if err_msg and err_msg != desc else ""
            reasons.append(f"error_code={err_code}: {desc}{suffix}")
        if estop:
            reasons.append("system_state=EMERGENCY_STOP")
        if estop or err_code == 401:
            reasons.append(
                "release the emergency stop, then call clear_errors() once the "
                "robot is safe to move"
            )
        if health_fault:
            reasons.append(f"health_monitor={health_fault}")
        elif not state.is_healthy:
            reasons.append("health_monitor=unhealthy")
        if sys_state == 4:
            reasons.append("system_state=4 (Fault)")
        if mode_drop:
            reasons.append(
                f"controller_mode_drop (expected={R2_MODE_SERVOJ}, active={active_mode})"
            )
        return HealthStatus(
            healthy=level == FaultLevel.OK,
            level=level,
            error_code=err_code,
            error_message="; ".join(reasons),
            details={
                "system_state": sys_state,
                "system_main_state": main_state,
                "controller_mode": active_mode,
                "battery_percentage": state.battery_percentage,
            },
        )

    def _recover_mode(self) -> str:
        """Hold at the measured pose and re-assert ServoJ mode after a mode drop."""
        self._sync_cmd_to_actual()
        self.switch_controller(R2_MODE_SERVOJ)
        return "switch_controller(SERVOJ)"

    def clear_errors(self) -> bool:
        """Clear controller faults, re-enable the body axes, and restore the controller mode.

        This is also how a person resumes after an emergency stop they have
        released, which automatic recovery never does on its own.
        """
        target_mode = self.controller_mode
        self._sync_cmd_to_actual()
        self._chassis_cmd_started = False
        self._switch_mode(R2_MODE_AUTO)
        if "/power_ctrl" in self._services:
            self._call_service(
                "/power_ctrl",
                lambda req: setattr(
                    req, "cmd", int(getattr(req, "CMD_ENABLE_DRV_PWR", 4))
                ),
                timeout=2.0,
            )
        if "/clear_faults" in self._services:
            self._call_service("/clear_faults", lambda _req: None, timeout=1.5)
        if "/robotic/set_axis_enable_cmd" in self._services:

            def _fill_enable(req: Any) -> None:
                req.slave_addrs = list(R2_BODY_SLAVE_ADDRS)
                req.enable = True

            self._call_service(
                "/robotic/set_axis_enable_cmd", _fill_enable, timeout=10.0
            )
        self._configure_grippers()
        self._reset_head_led()
        response = self._call_service(
            "/robotic/release_stop", lambda _req: None, timeout=1.5
        )
        with self._lock:
            self._last_error_code = 0
            self._last_error_msg = ""
            self._health_fault = ""
            self._system_state_code = 2
            self._state.is_healthy = True
            self._cached_health = None
        self.switch_controller(target_mode)
        self._sync_cmd_to_actual()
        return bool(getattr(response, "success", True))

    def _reset_head_led(self) -> None:
        """Return the head LED to steady white, its no-fault indication."""
        for service, fields in (
            (
                "/head/set_led_mode",
                {
                    "device_id": 0,
                    "mode": 1,
                    "r": 255,
                    "g": 255,
                    "b": 255,
                    "period_ms": 0,
                    "percentage": 100,
                    "duration_ms": 0,
                },
            ),
            (
                "/head/set_led_color",
                {"device_id": 0, "led_indices": [], "r": 255, "g": 255, "b": 255},
            ),
        ):
            if service not in self._services:
                continue

            def _fill(req: Any, values: Mapping[str, Any] = fields) -> None:
                for key, value in values.items():
                    setattr(req, key, value)

            try:
                self._call_service(service, _fill, timeout=0.5)
            except Exception:
                pass

    # Services and controller modes

    def _call_service(
        self, service_name: str, configure_request: Any, timeout: float = 5.0
    ) -> Any:
        """Call a service, latching any fault code its response carries.

        Returns ``None`` when the service is unavailable or does not answer.
        """
        self._require_connected()
        if service_name not in self._services:
            raise RuntimeError(f"R2 service {service_name!r} is not initialized.")
        client, srv_cls = self._services[service_name]
        request = getattr(srv_cls, "Request", srv_cls)()
        configure_request(request)
        if not client.wait_for_service(timeout_sec=min(timeout, 2.0)):
            return None
        future = client.call_async(request)
        self._rclpy.spin_until_future_complete(
            self._rclpy_node, future, timeout_sec=timeout
        )
        response = future.result() if future.done() else None
        error = getattr(response, "error_code", None) if response is not None else None
        if error is not None:
            code = int(
                getattr(
                    error,
                    "code",
                    getattr(
                        error, "error_code", error if isinstance(error, int) else 0
                    ),
                )
                or 0
            )
            # The system state query keeps reporting a cleared code while the
            # controller is healthy.
            stale_report = (
                service_name == "/robotic/get_system_state_cmd"
                and getattr(response, "state", 2) != 4
            )
            if code != 0 and not stale_report:
                message = str(
                    getattr(
                        error,
                        "msg",
                        getattr(error, "message", getattr(response, "error_msg", "")),
                    )
                )
                with self._lock:
                    self._latch_error_locked(code, f"{service_name}: {message}")
        return response

    def _switch_mode(self, mode: int) -> bool:
        """Send one controller mode switch, leaving :attr:`controller_mode` unchanged."""
        response = self._call_service(
            "/robotic/switch_mode_cmd", lambda req: setattr(req, "mode", int(mode))
        )
        if response is not None and bool(getattr(response, "success", True)):
            self._active_mode = int(mode)
            return True
        return False

    def switch_controller(self, mode: int) -> bool:
        """Switch the motion controller mode and make it the one to keep.

        ServoJ is entered through AUTO, which the controller needs to accept a
        new stream. Any other mode stops streaming first.
        """
        if mode not in R2_CONTROLLER_MODES:
            raise ValueError(
                f"Unsupported R2 controller mode {mode!r}; "
                f"expected one of {sorted(R2_CONTROLLER_MODES)}."
            )
        self.controller_mode = int(mode)
        if mode != R2_MODE_SERVOJ:
            self._stream.hold()
            return self._switch_mode(mode)
        self._switch_mode(R2_MODE_AUTO)
        time.sleep(0.15)
        ok = self._switch_mode(R2_MODE_SERVOJ)
        time.sleep(0.15)
        return ok

    def _configure_grippers(self) -> None:
        """Give the parallel grippers full rated current and speed."""
        if (
            self.end_effector_type != "gripper"
            or "/robotic/set_gripper_param_cmd" not in self._services
        ):
            return

        def _fill(req: Any) -> None:
            req.gripper_l_current = req.gripper_r_current = 0.5
            req.gripper_l_vel = req.gripper_r_vel = 150.0
            req.gripper_l_acc = req.gripper_r_acc = 100.0

        try:
            self._call_service("/robotic/set_gripper_param_cmd", _fill, timeout=1.5)
        except Exception:
            pass

    # Joint commands

    def _command(
        self,
        targets: Optional[Mapping[str, Any]] = None,
        grippers: Optional[Mapping[str, float]] = None,
    ) -> None:
        """Stream body group targets and gripper openings without blocking."""
        self._require_connected()
        targets, grippers = dict(targets or {}), dict(grippers or {})
        if self._staging():
            self._stage(targets, grippers)
            return
        if self._check_and_recover("servoj_fault") is not None:
            # Hold where recovery left the robot; the next command drives again.
            return
        self._apply_targets(targets, grippers)
        self._stream.track(self._body_goal())

    def move_left_arm_joint(self, positions: np.ndarray) -> None:
        """Stream 7-DoF left arm joint targets."""
        self._command({"left_arm": positions})

    def move_right_arm_joint(self, positions: np.ndarray) -> None:
        """Stream 7-DoF right arm joint targets."""
        self._command({"right_arm": positions})

    def move_dual_arm_joint(
        self, left_positions: np.ndarray, right_positions: np.ndarray
    ) -> None:
        """Stream both arms' joint targets as one command."""
        self._command({"left_arm": left_positions, "right_arm": right_positions})

    def move_waist_joint(self, positions: np.ndarray) -> None:
        """Stream 5-DoF waist joint targets."""
        self._command({"waist": positions})

    def move_neck_joint(self, positions: np.ndarray) -> None:
        """Stream ``[yaw, pitch]`` neck targets; the 20-DoF body ignores yaw."""
        self._command({"neck": positions})

    def move_left_gripper(self, position: Any) -> None:
        """Command the left gripper opening in ``[0, 1]``, ``1`` fully open."""
        self._command(grippers={"left": float(np.asarray(position).reshape(-1)[0])})

    def move_right_gripper(self, position: Any) -> None:
        """Command the right gripper opening in ``[0, 1]``, ``1`` fully open."""
        self._command(grippers={"right": float(np.asarray(position).reshape(-1)[0])})

    def move_left_hand(self, positions: Any) -> None:
        """Command the left hand's six finger openings in :data:`R2_HAND_FINGER_ORDER`."""
        self._command_hand("left", positions)

    def move_right_hand(self, positions: Any) -> None:
        """Command the right hand's six finger openings in :data:`R2_HAND_FINGER_ORDER`."""
        self._command_hand("right", positions)

    def _command_hand(self, side: str, positions: Any) -> None:
        self._require_connected()
        self._ensure_targets()
        self._hand_targets[side] = np.asarray(positions, dtype=np.float32).reshape(6)
        channel, msg_cls = self._writers["/robotic/servo_ry_hand_data"]
        msg = msg_cls()
        msg.left_finger_open_ratio = [float(x) for x in self._hand_targets["left"]]
        msg.right_finger_open_ratio = [float(x) for x in self._hand_targets["right"]]
        channel.send(msg)

    def move_body_smooth(
        self,
        *,
        left_arm: Optional[Sequence[float] | np.ndarray] = None,
        right_arm: Optional[Sequence[float] | np.ndarray] = None,
        waist: Optional[Sequence[float] | np.ndarray] = None,
        neck: Optional[Sequence[float] | np.ndarray] = None,
        left_gripper: Optional[float] = None,
        right_gripper: Optional[float] = None,
        duration: Optional[float] = None,
        max_velocity: Optional[float | Sequence[float] | np.ndarray] = None,
    ) -> bool:
        """Move body groups to their targets along one raised-cosine trajectory.

        The call blocks until the move ends. It lasts at least ``duration``
        seconds, longer when the joint limits or ``max_velocity`` (rad/s; the
        smallest of several values applies) require. Groups left as ``None``
        keep their current target. Inside a command batch the move is staged
        and merged with the batch's other moves.

        Returns:
            ``True`` when the move completed and no fault is active.
        """
        self._require_connected()
        targets = {
            group: values
            for group, values in (
                ("left_arm", left_arm),
                ("right_arm", right_arm),
                ("waist", waist),
                ("neck", neck),
            )
            if values is not None
        }
        grippers = {
            side: float(value)
            for side, value in (("left", left_gripper), ("right", right_gripper))
            if value is not None
        }
        if self._staging():
            self._stage(targets, grippers)
            staged = self._staged_move or {"duration": 0.0, "max_velocity": None}
            caps = [c for c in (staged["max_velocity"], max_velocity) if c is not None]
            self._staged_move = {
                "duration": max(staged["duration"], float(duration or 0.0)),
                "max_velocity": (
                    min(float(np.min(np.abs(np.asarray(c)))) for c in caps)
                    if caps
                    else None
                ),
            }
            return True
        return self._move_body(
            targets,
            grippers,
            duration=float(duration or 0.0),
            max_velocity=max_velocity,
        )

    def hold(self) -> None:
        """Stop the body and the chassis where they are, ending any running move.

        The streamed joints come to rest within their acceleration limits, and
        the commanded targets move to that rest point, so the next command
        starts from it instead of resuming the stopped move. Safe to call from
        another thread while a move runs; that move returns ``False``.
        """
        self._require_connected()
        self._base_pulse_stop.set()
        rest = self._stream.brake()
        if rest is not None:
            sizes = [self._targets[group].size for group in self.BODY_GROUPS]
            for group, values in zip(
                self.BODY_GROUPS, np.split(rest, np.cumsum(sizes)[:-1])
            ):
                self._targets[group] = values.astype(np.float32)
        self.stop_base()

    def _move_body(
        self,
        targets: Mapping[str, Any],
        grippers: Mapping[str, float],
        *,
        duration: float,
        max_velocity: Optional[float | Sequence[float] | np.ndarray],
    ) -> bool:
        if self.enable_fault_recovery and self._has_active_fault(check_mode=True):
            self.recover("move_precheck")
        if self._has_active_fault(check_mode=False):
            return False
        self._ensure_targets()
        gripper_changed = any(
            abs(float(np.clip(v, 0.0, 1.0)) - self._gripper_targets[side]) > 1e-3
            for side, v in grippers.items()
        )
        self._apply_targets(targets, grippers)
        if not self._stream.move(
            self._body_goal(), duration=duration, max_velocity=max_velocity
        ):
            if self._has_active_fault(check_mode=False):
                self._sync_cmd_to_actual()
                if self.enable_fault_recovery:
                    self.recover("move_fault")
            return False
        if gripper_changed and self.motion_profile.effector_settle_s > 0.0:
            time.sleep(self.motion_profile.effector_settle_s)
        return not self._has_active_fault(check_mode=True)

    # Cartesian commands

    def forward_kinematics(
        self, joint_positions: Sequence[float]
    ) -> dict[str, np.ndarray]:
        """Return both TCP poses for 14 arm joints, or 19 with the waist first."""
        q = np.asarray(joint_positions, dtype=np.float32).reshape(-1)
        if q.size not in (14, 19):
            raise ValueError(
                f"R2 forward kinematics takes 14 arm joints or 19 with the waist, "
                f"got {q.size}."
            )
        waist = q[:5] if q.size == 19 else self.get_state().waist_positions
        arms = q[-14:]
        l_p, l_rpy, _, _ = compute_r2_arm_fk_and_jacobian("left", arms[:7], waist)
        r_p, r_rpy, _, _ = compute_r2_arm_fk_and_jacobian("right", arms[7:], waist)
        return {
            "left_tcp_pose": np.concatenate([l_p, l_rpy]).astype(np.float32),
            "right_tcp_pose": np.concatenate([r_p, r_rpy]).astype(np.float32),
        }

    def inverse_kinematics(
        self,
        *,
        left_tcp_pose: Optional[np.ndarray] = None,
        right_tcp_pose: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """Solve 6-DoF IK for the given TCP poses, seeded from the commanded joints.

        Raises:
            ValueError: When a pose has no solution within the solver's
                tolerances.
        """
        solved: dict[str, np.ndarray] = {}
        for side, pose in (("left", left_tcp_pose), ("right", right_tcp_pose)):
            seed = self._arm_target(side)
            if pose is None:
                solved[side] = seed
                continue
            target = np.asarray(pose, dtype=np.float32).reshape(6)
            q = self._solve_arm_ik(side, _pose_matrix(target), seed)
            if q is None:
                raise ValueError(
                    f"R2 {side} arm has no IK solution within tolerance for "
                    f"tcp_pose={np.round(target, 4).tolist()} from joints "
                    f"{np.round(seed, 3).tolist()}."
                )
            solved[side] = q
        return {
            "left_joint_positions": solved["left"],
            "right_joint_positions": solved["right"],
        }

    def _solve_arm_ik(
        self, side: str, target: np.ndarray, seed: np.ndarray
    ) -> Optional[np.ndarray]:
        """Solve one arm's 7 joints for a 4x4 TCP ``target`` in ``base_link``.

        :class:`R2KinematicsAdapter` moves at most a short lookahead per solve,
        so a distant target is reached over several re-seeded solves. The
        waist is the commanded one, so the target is in the frame the arm
        will be in. Returns ``None`` when no solution meets the solver's
        tolerances.
        """
        solver = R2KinematicsAdapter(side, waist_qpos=self._targets["waist"])
        q = np.asarray(seed, dtype=np.float64).reshape(7)
        for _ in range(_R2_IK_SEGMENTS):
            result = solver.solve(target, q)
            if not result.success:
                return None
            q = result.q_target
            reached = solver.fk(q)
            position_error = float(np.linalg.norm(reached[:3, 3] - target[:3, 3]))
            rotation_error = float(
                np.linalg.norm(_so3_error_rotvec(target[:3, :3] @ reached[:3, :3].T))
            )
            if (
                position_error <= solver.config.position_tolerance
                and rotation_error <= solver.config.rotation_tolerance
            ):
                return q.astype(np.float32)
        return None

    def _warn_ik_miss(self, side: str, what: str) -> None:
        now = time.monotonic()
        if now - self._last_ik_warning >= 1.0:
            self._last_ik_warning = now
            self._logger.warning(
                f"R2 {side} arm holds its target: no IK solution for {what}."
            )

    def move_left_arm_cartesian(self, pose: np.ndarray, duration: float = 1.0) -> bool:
        """Move the left TCP to ``pose`` ``[x, y, z, r, p, y]`` in at least ``duration`` seconds."""
        return self._move_arm_cartesian("left", pose, duration)

    def move_right_arm_cartesian(self, pose: np.ndarray, duration: float = 1.0) -> bool:
        """Move the right TCP to ``pose`` ``[x, y, z, r, p, y]`` in at least ``duration`` seconds."""
        return self._move_arm_cartesian("right", pose, duration)

    def servo_left_arm_cartesian(self, pose: np.ndarray) -> bool:
        """Stream a left TCP target ``[x, y, z, r, p, y]`` without blocking."""
        return self._servo_arm_cartesian("left", pose)

    def servo_right_arm_cartesian(self, pose: np.ndarray) -> bool:
        """Stream a right TCP target ``[x, y, z, r, p, y]`` without blocking."""
        return self._servo_arm_cartesian("right", pose)

    def move_left_arm_velocity(self, velocity: np.ndarray, dt: float = 0.02) -> None:
        """Step the left TCP by a base-frame twist ``[v, w]`` applied for ``dt`` seconds."""
        self._move_arm_velocity("left", velocity, dt)

    def move_right_arm_velocity(self, velocity: np.ndarray, dt: float = 0.02) -> None:
        """Step the right TCP by a base-frame twist ``[v, w]`` applied for ``dt`` seconds."""
        self._move_arm_velocity("right", velocity, dt)

    def _move_arm_cartesian(self, side: str, pose: np.ndarray, duration: float) -> bool:
        target = np.asarray(pose, dtype=np.float32).reshape(6)
        q = self._solve_arm_ik(side, _pose_matrix(target), self._arm_target(side))
        if q is None:
            self._warn_ik_miss(side, f"tcp_pose={np.round(target, 4).tolist()}")
            return False
        return self.move_body_smooth(**{f"{side}_arm": q}, duration=duration)

    def _servo_arm_cartesian(self, side: str, pose: np.ndarray) -> bool:
        target = np.asarray(pose, dtype=np.float32).reshape(6)
        q = self._solve_arm_ik(side, _pose_matrix(target), self._arm_target(side))
        if q is None:
            self._warn_ik_miss(side, f"tcp_pose={np.round(target, 4).tolist()}")
            return False
        self._command({f"{side}_arm": q})
        return True

    def _move_arm_velocity(self, side: str, velocity: np.ndarray, dt: float) -> None:
        """Apply a twist to the commanded TCP pose and stream the result.

        The twist moves the forward kinematics of the commanded joints rather
        than the measured pose, so successive steps compose without lag.
        """
        vel = np.asarray(velocity, dtype=np.float64).reshape(6)
        seed = self._arm_target(side)
        target = R2KinematicsAdapter(side, waist_qpos=self._targets["waist"]).fk(seed)
        target[:3, 3] += vel[:3] * float(dt)
        target[:3, :3] = _rotvec_to_matrix(vel[3:] * float(dt)) @ target[:3, :3]
        q = self._solve_arm_ik(side, target, seed)
        if q is None:
            self._warn_ik_miss(side, f"twist={np.round(vel, 4).tolist()}")
            return
        self._command({f"{side}_arm": q})

    # Chassis

    def move_base(self, velocity: np.ndarray) -> None:
        """Publish planar chassis velocity ``[vx, vy, wz]``."""
        self._require_connected()
        if "/chassis/cmd_vel" not in self._writers:
            raise RuntimeError(
                "R2 mobile base is not enabled on this connection (enable_base=False)."
            )
        vel = np.asarray(velocity, dtype=np.float32).reshape(3)
        moving = bool(np.any(np.abs(vel) > 1e-6))
        if not moving:
            # A few zero frames stop the base; more only load the bus.
            if (
                not np.any(np.abs(self._last_cmd_base_vel) > 1e-6)
                and self._zero_base_frames_sent >= 3
            ):
                return
            self._zero_base_frames_sent += 1
        else:
            self._zero_base_frames_sent = 0
        if (
            moving
            and not self._chassis_cmd_started
            and "/chassis/cmd_start" in self._services
        ):
            self._call_service(
                "/chassis/cmd_start",
                lambda req: setattr(req, "data", True),
                timeout=1.0,
            )
            self._chassis_cmd_started = True
        channel, msg_cls = self._writers["/chassis/cmd_vel"]
        msg = msg_cls()
        msg.linear.x = float(vel[0])
        msg.linear.y = float(vel[1])
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(vel[2])
        channel.send(msg)
        with self._lock:
            self._last_cmd_base_vel = vel.copy()
            self._state.base_velocity = vel.copy()

    def move_base_pulse(
        self,
        velocity: Sequence[float] | np.ndarray,
        duration: float = 0.5,
        rate_hz: Optional[float] = None,
        *,
        ramp_up: Optional[float] = None,
        ramp_down: Optional[float] = None,
    ) -> dict[str, np.ndarray]:
        """Stream ``[vx, vy, wz]`` for ``duration`` seconds with smooth ramps, then stop.

        :meth:`hold` ends the pulse early.
        """
        self._chassis_cmd_started = False
        self._base_pulse_stop.clear()
        dt, up_frames, cruise_frames, down_frames = (
            self.motion_profile.generate_velocity_pulse(
                np.asarray(velocity, dtype=np.float32).reshape(3),
                max(float(duration), 0.0),
                rate_hz=rate_hz,
                ramp_up=ramp_up,
                ramp_down=ramp_down,
            )
        )
        state_during: Optional[R2RobotState] = None
        for frames in (up_frames, cruise_frames, down_frames):
            for frame_vel in frames:
                if self._base_pulse_stop.is_set():
                    break
                started = time.monotonic()
                self.move_base(frame_vel)
                time.sleep(max(0.0, dt - (time.monotonic() - started)))
            if frames is cruise_frames:
                state_during = self.get_state()
        self.stop_base()
        assert state_during is not None
        return {
            "base_pose": self.get_state().base_pose,
            "base_velocity": state_during.base_velocity,
        }

    def stop_base(self) -> None:
        """Zero the chassis velocity command."""
        if "/chassis/cmd_vel" in self._writers:
            self._zero_base_frames_sent = 0
            self.move_base(np.zeros(3, dtype=np.float32))

    # Cameras

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
        raw_entry: Optional[tuple[Any, bool, tuple[int, int]]] = None
        with self._lock:
            frame = self._camera_frames.get(camera_name)
            if frame is not None:
                return frame.copy()
            raw_entry = self._camera_raw_frames.get(camera_name)
        if raw_entry is not None:
            raw_msg, compressed, resolution = raw_entry
            decoded = _decode_image_message(
                raw_msg,
                compressed=compressed,
                depth=False,
                fallback_resolution=resolution,
            )
            with self._lock:
                if self._camera_raw_frames.get(camera_name) is raw_entry:
                    self._camera_frames[camera_name] = decoded
            return decoded.copy()
        width, height = R2_CAMERA_TOPICS[camera_name]["resolution"]
        return np.zeros((height, width, 3), dtype=np.uint8)

    def get_camera_depth(self, camera_name: str) -> Optional[np.ndarray]:
        """Return the latest depth image for ``camera_name`` when ``enable_depth=True``."""
        self._require_connected()
        if not self.enable_depth:
            return None
        raw_entry: Optional[tuple[Any, tuple[int, int]]] = None
        with self._lock:
            depth = self._camera_depths.get(camera_name)
            if depth is not None:
                return depth.copy()
            raw_entry = self._camera_raw_depths.get(camera_name)
        if raw_entry is not None:
            raw_msg, resolution = raw_entry
            decoded = _decode_image_message(
                raw_msg,
                compressed=False,
                depth=True,
                fallback_resolution=resolution,
            )
            with self._lock:
                if self._camera_raw_depths.get(camera_name) is raw_entry:
                    self._camera_depths[camera_name] = decoded
            return decoded.copy()
        width, height = R2_CAMERA_TOPICS[camera_name]["resolution"]
        return np.zeros((height, width), dtype=np.float32)

    def get_camera_observation(self, camera_name: str) -> dict[str, Any]:
        """Return ``{'frame': rgb}`` plus ``'depth'`` when depth is enabled."""
        obs: dict[str, Any] = {"frame": self.get_camera(camera_name)}
        if self.enable_depth:
            obs["depth"] = self.get_camera_depth(camera_name)
        return obs
