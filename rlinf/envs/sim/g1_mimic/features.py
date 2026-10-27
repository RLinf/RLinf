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

"""Quaternion and heading utilities, and world-model residual integration."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    CRITIC_ROOT_ERR_DIM,
    JOINT_POS,
    JOINT_VEL,
    RES_ANG,
    RES_JOINT,
    RES_JOINT_VEL,
    RES_LIN,
    RES_POS,
    RES_ROT,
    RESIDUAL_DIM,
    ROOT_ANG_VEL,
    ROOT_LIN_VEL,
    ROOT_POS,
    ROOT_QUAT,
    STATE_DIM,
)


def _as_batch(x: torch.Tensor) -> torch.Tensor:
    return x if x.dim() >= 2 else x.unsqueeze(0)


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion (wxyz) along the last axis."""
    return F.normalize(q, dim=-1, eps=1e-8)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product ``q1 * q2`` of wxyz quaternions."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Rotation matrix ``[..., 3, 3]`` of a wxyz quaternion."""
    q = quat_normalize(q)
    w, x, y, z = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    r00 = 1 - 2 * (yy + zz)
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)
    r10 = 2 * (xy + wz)
    r11 = 1 - 2 * (xx + zz)
    r12 = 2 * (yz - wx)
    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = 1 - 2 * (xx + yy)
    return torch.stack(
        (
            torch.stack((r00, r01, r02), dim=-1),
            torch.stack((r10, r11, r12), dim=-1),
            torch.stack((r20, r21, r22), dim=-1),
        ),
        dim=-2,
    )


def rotmat_to_6d(rot: torch.Tensor) -> torch.Tensor:
    """First two columns of a rotation matrix, flattened (Rot6D)."""
    return rot[..., :, :2].reshape(*rot.shape[:-2], 6)


def rotvec_to_quat(rotvec: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion of a rotation vector."""
    angle = rotvec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    axis = rotvec / angle
    half = 0.5 * angle
    return torch.cat((torch.cos(half), axis * torch.sin(half)), dim=-1)


def quat_to_rotvec(q: torch.Tensor) -> torch.Tensor:
    """Rotation vector of the shortest rotation a wxyz quaternion represents."""
    q = quat_normalize(q)
    # Shortest rotation: q and -q are the same attitude.
    q = torch.where(q[..., :1] < 0, -q, q)
    w = q[..., :1].clamp(-1.0, 1.0)
    xyz = q[..., 1:]
    half_sin = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(half_sin, w)
    axis = xyz / half_sin.clamp_min(1e-8)
    return axis * angle


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Inverse of a wxyz quaternion (conjugate of its normalized form)."""
    out = quat_normalize(q).clone()
    out[..., 1:] = -out[..., 1:]
    return out


def so3_relative_rotvec(q_pred: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
    """``Log(R_pred R_target^T)``, the rotation taking the target onto the prediction.

    The map uses the shortest quaternion, so ``q`` and ``-q`` match, and a
    pair that sits on opposite sides of the ±π cut still has a small error.
    ``atan2`` stays finite at a relative rotation of π (``w = 0``).
    """
    relative = quat_mul(quat_normalize(q_pred), quat_inv(q_target))
    return quat_to_rotvec(relative)


def heading_yaw(quat: torch.Tensor) -> torch.Tensor:
    """Yaw of the body x axis projected on the ground plane."""
    rot = quat_to_rotmat(quat)
    return torch.atan2(rot[..., 1, 0], rot[..., 0, 0])


def yaw_rotation(yaw: torch.Tensor) -> torch.Tensor:
    """Rotation matrix about z for each yaw angle."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    row0 = torch.stack((c, -s, z), dim=-1)
    row1 = torch.stack((s, c, z), dim=-1)
    row2 = torch.stack((z, z, o), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def projected_gravity(quat: torch.Tensor) -> torch.Tensor:
    """World gravity ``(0, 0, -1)`` expressed in the body frame."""
    rot = quat_to_rotmat(quat)
    gravity = torch.tensor([0.0, 0.0, -1.0], device=quat.device, dtype=quat.dtype)
    gravity = gravity.expand(*quat.shape[:-1], 3)
    return torch.einsum("...ji,...j->...i", rot, gravity)


def tilt_error(quat: torch.Tensor) -> torch.Tensor:
    """``1 + g_z`` in the body frame: 0 upright, 2 upside down."""
    return 1.0 + projected_gravity(quat)[..., 2]


def heading_invariant_features(state: torch.Tensor) -> torch.Tensor:
    """Map the 71-d raw state to 71-d heading-invariant dynamics features."""
    squeezed = state.dim() == 1
    state = _as_batch(state)
    root_pos = state[..., ROOT_POS]
    quat = state[..., ROOT_QUAT]
    yaw = heading_yaw(quat)
    r_heading = yaw_rotation(yaw)
    rot = quat_to_rotmat(quat)
    rot_local = torch.einsum("...ji,...jk->...ik", r_heading, rot)
    lin_world = state[..., ROOT_LIN_VEL]
    lin_heading = torch.einsum("...ji,...j->...i", r_heading, lin_world)
    feat = torch.cat(
        (
            root_pos[..., 2:3],
            rotmat_to_6d(rot_local),
            lin_heading,
            state[..., ROOT_ANG_VEL],
            state[..., JOINT_POS],
            state[..., JOINT_VEL],
        ),
        dim=-1,
    )
    return feat.squeeze(0) if squeezed else feat


def integrate_residual(state: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Apply a 70-d residual in the heading frame. Zero residual leaves state."""
    squeezed = state.dim() == 1
    state = _as_batch(state)
    residual = _as_batch(residual)
    quat = quat_normalize(state[..., ROOT_QUAT])
    yaw = heading_yaw(quat)
    r_heading = yaw_rotation(yaw)
    dpos_h = residual[..., RES_POS]
    dpos_w = torch.einsum("...ij,...j->...i", r_heading, dpos_h)
    # Linear-velocity residual is heading-frame, same frame as the network input.
    # Angular-velocity residual stays in the body frame (added, not yaw-rotated).
    dlin_w = torch.einsum("...ij,...j->...i", r_heading, residual[..., RES_LIN])
    new_pos = state[..., ROOT_POS] + dpos_w
    new_quat = quat_normalize(quat_mul(quat, rotvec_to_quat(residual[..., RES_ROT])))
    next_state = torch.cat(
        (
            new_pos,
            new_quat,
            state[..., JOINT_POS] + residual[..., RES_JOINT],
            state[..., ROOT_LIN_VEL] + dlin_w,
            state[..., ROOT_ANG_VEL] + residual[..., RES_ANG],
            state[..., JOINT_VEL] + residual[..., RES_JOINT_VEL],
        ),
        dim=-1,
    )
    return next_state.squeeze(0) if squeezed else next_state


def root_error_4d(state: torch.Tensor, ref_state: torch.Tensor) -> torch.Tensor:
    """Critic-only root error: heading-frame position (3) and yaw (1)."""
    squeezed = state.dim() == 1
    state = _as_batch(state)
    ref_state = _as_batch(ref_state)
    yaw = heading_yaw(ref_state[..., ROOT_QUAT])
    r_heading = yaw_rotation(yaw)
    pos_err_w = state[..., ROOT_POS] - ref_state[..., ROOT_POS]
    pos_err_h = torch.einsum("...ji,...j->...i", r_heading, pos_err_w)
    yaw_err = heading_yaw(state[..., ROOT_QUAT]) - yaw
    yaw_err = torch.atan2(torch.sin(yaw_err), torch.cos(yaw_err)).unsqueeze(-1)
    out = torch.cat((pos_err_h, yaw_err), dim=-1)
    return out.squeeze(0) if squeezed else out


def identity_residual(
    batch: int, device: torch.device | str, dtype: torch.dtype
) -> torch.Tensor:
    """Zero dynamics residual ``[batch, 70]``."""
    return torch.zeros(batch, RESIDUAL_DIM, device=device, dtype=dtype)


def zero_action(
    batch: int, device: torch.device | str, dtype: torch.dtype
) -> torch.Tensor:
    """Zero action ``[batch, 29]``."""
    return torch.zeros(batch, ACTION_DIM, device=device, dtype=dtype)


def default_state(
    batch: int, device: torch.device | str, dtype: torch.dtype
) -> torch.Tensor:
    """Upright state ``[batch, 71]`` at 0.75 m with zero joints and velocities."""
    state = torch.zeros(batch, STATE_DIM, device=device, dtype=dtype)
    state[..., 2] = 0.75
    state[..., 3] = 1.0
    return state


assert (
    heading_invariant_features(default_state(1, "cpu", torch.float32)).shape[-1] == 71
)
assert (
    root_error_4d(
        default_state(1, "cpu", torch.float32), default_state(1, "cpu", torch.float32)
    ).shape[-1]
    == CRITIC_ROOT_ERR_DIM
)
