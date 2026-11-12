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

"""Quaternion, rotation-matrix and heading utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F


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


def rot_apply(rot: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """``rot @ vec`` for ``[..., 3, 3]`` and ``[..., 3]``.

    The 3x3 products here are elementwise products and a sum, never a
    matmul: TF32 cannot round them, and inductor fuses them with their
    neighbours instead of launching a batched GEMM.
    """
    return (rot * vec.unsqueeze(-2)).sum(-1)


def rot_apply_t(rot: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """``rot^T @ vec``."""
    return (rot * vec.unsqueeze(-1)).sum(-2)


def rot_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` for ``[..., 3, 3]`` matrices."""
    return (a.unsqueeze(-1) * b.unsqueeze(-3)).sum(-2)


def rot_mul_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a^T @ b``."""
    return (a.unsqueeze(-1) * b.unsqueeze(-2)).sum(-3)


def rot_mul_nt(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b^T``."""
    return (a.unsqueeze(-2) * b.unsqueeze(-3)).sum(-1)


def rotmat_to_6d(rot: torch.Tensor) -> torch.Tensor:
    """First two columns of a rotation matrix, row by row (Rot6D)."""
    return rot[..., :, :2].reshape(*rot.shape[:-2], 6)


def rotvec_to_quat(rotvec: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion of a rotation vector."""
    angle = rotvec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    axis = rotvec / angle
    half = 0.5 * angle
    return torch.cat((torch.cos(half), axis * torch.sin(half)), dim=-1)


def rotvec_from_quat(q: torch.Tensor) -> torch.Tensor:
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


def quat_from_euler_xyz(
    roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor
) -> torch.Tensor:
    """wxyz quaternion of ``R_z(yaw) R_y(pitch) R_x(roll)``."""
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    return torch.stack(
        (
            cy * cr * cp + sy * sr * sp,
            cy * sr * cp - sy * cr * sp,
            cy * cr * sp + sy * sr * cp,
            sy * cr * cp - cy * sr * sp,
        ),
        dim=-1,
    )


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
    return -quat_to_rotmat(quat)[..., 2, :]
