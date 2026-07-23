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
"""6D-rotation + SE(3) helpers, shared by all dual-Franka geometry.
Quaternions are xyzw; rot6d is the first two columns of R flattened
(decoding is Gram-Schmidt on r1, r2); SE(3) body-frame delta is
``T_delta = inv(T_state) @ T_abs``.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

__all__ = [
    "quat_xyzw_to_rot6d",
    "rot6d_to_matrix",
    "matrix_to_rot6d",
    "rot6d_to_quat_xyzw",
    "rot6d_to_quat_xyzw_safe",
    "pose_to_SE3",
    "SE3_to_pose",
    "se3_body_delta",
    "se3_body_compose",
]


def matrix_to_rot6d(R_mat: np.ndarray) -> np.ndarray:
    """Encode a rotation matrix as a 6D vector (first two columns, flattened).

    Supports batched input: trailing ``(3, 3)`` axes → trailing ``(6,)`` axis.
    """
    R_mat = np.asarray(R_mat)
    if R_mat.shape[-2:] != (3, 3):
        raise ValueError(f"expected (..., 3, 3); got {R_mat.shape}")
    col0 = R_mat[..., :, 0]  # (..., 3)
    col1 = R_mat[..., :, 1]  # (..., 3)
    return np.concatenate([col0, col1], axis=-1).astype(np.float32)


def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """Decode a 6D rotation vector to a valid ``SO(3)`` matrix via Gram-Schmidt.

    - ``b1 = r1 / |r1|``
    - ``b2 = (r2 - (b1·r2) b1) / |…|``
    - ``b3 = b1 × b2``
    - ``R = [b1 | b2 | b3]`` (columns)

    Raises if ``|r1|`` or the post-projection ``|r2_perp|`` underflow
    (would produce a degenerate frame).
    """
    r6 = np.asarray(r6, dtype=np.float64)
    if r6.shape[-1] != 6:
        raise ValueError(f"expected trailing dim 6; got {r6.shape}")

    r1 = r6[..., :3]
    r2 = r6[..., 3:]

    n1 = np.linalg.norm(r1, axis=-1, keepdims=True)
    if np.any(n1 < 1e-8):
        raise ValueError(
            f"rot6d_to_matrix: |r1| underflow (min={float(n1.min()):.2e}); "
            "r1 ≈ 0, cannot form orthonormal frame."
        )
    b1 = r1 / n1

    dot = np.sum(b1 * r2, axis=-1, keepdims=True)
    r2_perp = r2 - dot * b1
    n2 = np.linalg.norm(r2_perp, axis=-1, keepdims=True)
    if np.any(n2 < 1e-8):
        raise ValueError(
            f"rot6d_to_matrix: |r2_perp| underflow (min={float(n2.min()):.2e}); "
            "r1 and r2 collinear, cannot form orthonormal frame."
        )
    b2 = r2_perp / n2

    b3 = np.cross(b1, b2, axis=-1)

    R_mat = np.stack([b1, b2, b3], axis=-1)  # (..., 3, 3)
    return R_mat.astype(np.float32)


def quat_xyzw_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternion(s) to 6D rotation via scipy rotation matrix."""
    quat = np.asarray(quat)
    if quat.shape[-1] != 4:
        raise ValueError(f"expected trailing dim 4; got {quat.shape}")
    flat = quat.reshape(-1, 4)
    mats = R.from_quat(flat).as_matrix()  # (N, 3, 3)
    r6 = matrix_to_rot6d(mats)  # (N, 6)
    return r6.reshape(*quat.shape[:-1], 6)


def rot6d_to_quat_xyzw(r6: np.ndarray) -> np.ndarray:
    """Convert 6D rotation(s) back to xyzw quaternion via matrix intermediate."""
    r6 = np.asarray(r6)
    mat = rot6d_to_matrix(r6)
    flat = mat.reshape(-1, 3, 3)
    quat = R.from_matrix(flat).as_quat()  # (N, 4), xyzw
    return quat.reshape(*r6.shape[:-1], 4).astype(np.float32)


def rot6d_to_quat_xyzw_safe(
    r6: np.ndarray, fallback_quat_xyzw: np.ndarray
) -> np.ndarray:
    """Best-effort rot6d → xyzw quat for live-robot dispatch.

    The strict :func:`rot6d_to_quat_xyzw` raises on NaN, ``|r1| → 0``, or
    r1/r2 collinearity — correct during training / offline preprocessing
    where such failures must surface loudly. During eval, a single bad
    model output (divergent training step leaving NaN, or action-space
    clipping pushing rot6d into a degenerate basis) should NOT abort the
    rollout; this helper substitutes ``fallback_quat_xyzw`` for any such
    single-sample failure.

    Args:
        r6: Shape-(6,) rotation encoding.
        fallback_quat_xyzw: Shape-(4,) xyzw quat to return on failure
            (typically the previous commanded quat or the live TCP quat).

    Returns:
        Shape-(4,) xyzw quat, float32.
    """
    r6 = np.asarray(r6, dtype=np.float64).reshape(-1)
    fallback = np.asarray(fallback_quat_xyzw, dtype=np.float32).reshape(-1)
    if fallback.shape != (4,):
        raise ValueError(f"fallback must be (4,); got {fallback.shape}")
    if r6.shape != (6,) or not np.all(np.isfinite(r6)):
        return fallback.copy()
    try:
        return rot6d_to_quat_xyzw(r6)
    except ValueError:
        return fallback.copy()


def pose_to_SE3(xyz: np.ndarray, r6: np.ndarray) -> np.ndarray:
    """Pack (xyz, rot6d) into a 4x4 homogeneous transform.

    Supports batched input: ``xyz (..., 3)`` and ``r6 (..., 6)`` →
    ``T (..., 4, 4)``.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape[-1] != 3:
        raise ValueError(f"xyz expected trailing 3; got {xyz.shape}")

    R_mat = rot6d_to_matrix(r6).astype(np.float64)
    batch_shape = R_mat.shape[:-2]
    if xyz.shape[:-1] != batch_shape:
        raise ValueError(
            f"batch shape mismatch: xyz {xyz.shape[:-1]} vs r6 {batch_shape}"
        )

    T = np.zeros(batch_shape + (4, 4), dtype=np.float64)
    T[..., :3, :3] = R_mat
    T[..., :3, 3] = xyz
    T[..., 3, 3] = 1.0
    return T


def SE3_to_pose(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack a 4x4 homogeneous transform into ``(xyz, rot6d)``."""
    T = np.asarray(T)
    if T.shape[-2:] != (4, 4):
        raise ValueError(f"expected trailing (4, 4); got {T.shape}")
    xyz = T[..., :3, 3].astype(np.float32)
    r6 = matrix_to_rot6d(T[..., :3, :3])
    return xyz, r6


def se3_body_delta(T_state: np.ndarray, T_abs: np.ndarray) -> np.ndarray:
    """Body-frame delta: ``T_delta = inv(T_state) @ T_abs``.

    Broadcast-friendly: if ``T_state`` is ``(..., 4, 4)`` and ``T_abs``
    is ``(..., H, 4, 4)`` (chunk), numpy matmul takes care of the extra
    time axis.
    """
    T_state = np.asarray(T_state, dtype=np.float64)
    T_abs = np.asarray(T_abs, dtype=np.float64)
    T_state_inv = np.linalg.inv(T_state)
    return T_state_inv @ T_abs


def se3_body_compose(T_state: np.ndarray, T_delta: np.ndarray) -> np.ndarray:
    """Body-frame compose: ``T_abs = T_state @ T_delta``."""
    T_state = np.asarray(T_state, dtype=np.float64)
    T_delta = np.asarray(T_delta, dtype=np.float64)
    return T_state @ T_delta
