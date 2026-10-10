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
"""Round-trip and numerical-stability tests for ``rlinf.utils.rot6d``."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from rlinf.utils.rot6d import (
    SE3_to_pose,
    matrix_to_rot6d,
    pose_to_SE3,
    quat_xyzw_to_rot6d,
    rot6d_to_matrix,
    rot6d_to_quat_xyzw,
    se3_body_compose,
    se3_body_delta,
)

RNG = np.random.default_rng(0)
N = 1000
TOL = 1e-5


def _random_rotations(n: int, rng: np.random.Generator = RNG):
    return R.random(n, random_state=rng.integers(1 << 31))


def test_matrix_rot6d_round_trip_batched():
    mats = _random_rotations(N).as_matrix()  # (N, 3, 3)
    r6 = matrix_to_rot6d(mats)  # (N, 6)
    assert r6.shape == (N, 6)
    R_rec = rot6d_to_matrix(r6)  # (N, 3, 3)
    assert R_rec.shape == (N, 3, 3)
    max_err = float(np.max(np.abs(R_rec - mats)))
    assert max_err < TOL, f"max round-trip error {max_err:.2e}"


def test_decoded_matrix_is_valid_SO3():
    r6 = matrix_to_rot6d(_random_rotations(N).as_matrix())
    R_rec = rot6d_to_matrix(r6).astype(np.float64)
    # Determinant ~= 1
    dets = np.linalg.det(R_rec)
    assert np.allclose(dets, 1.0, atol=TOL), f"dets range [{dets.min()}, {dets.max()}]"
    # Orthogonality: R^T R = I
    eye = np.einsum("...ji,...jk->...ik", R_rec, R_rec)
    assert np.allclose(eye, np.eye(3), atol=TOL)


def test_quat_rot6d_round_trip():
    quat = _random_rotations(N).as_quat()  # xyzw
    r6 = quat_xyzw_to_rot6d(quat)
    assert r6.shape == (N, 6)
    quat_rec = rot6d_to_quat_xyzw(r6)
    # Quaternion sign ambiguity: compare via rotation angle between q and q_rec
    rel = R.from_quat(quat_rec) * R.from_quat(quat).inv()
    ang = rel.magnitude()  # radians, in [0, π]
    assert np.all(ang < 1e-4), f"max quat-angle error {float(ang.max()):.2e} rad"


def test_rot6d_to_matrix_rejects_degenerate_r1():
    r6 = np.zeros(6, dtype=np.float32)
    r6[3] = 1.0  # r1 == 0, r2 nonzero
    with pytest.raises(ValueError, match="r1"):
        rot6d_to_matrix(r6)


def test_rot6d_to_matrix_rejects_collinear():
    r6 = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0], dtype=np.float32)  # r2 ∥ r1
    with pytest.raises(ValueError, match="collinear"):
        rot6d_to_matrix(r6)


def test_SE3_pose_round_trip_single():
    rng = np.random.default_rng(1)
    for _ in range(100):
        xyz = rng.normal(size=3).astype(np.float32)
        r6 = matrix_to_rot6d(R.random(random_state=rng.integers(1 << 31)).as_matrix())
        T = pose_to_SE3(xyz, r6)
        assert T.shape == (4, 4)
        assert np.allclose(T[3], [0, 0, 0, 1])
        xyz_rec, r6_rec = SE3_to_pose(T)
        assert np.allclose(xyz_rec, xyz, atol=TOL)
        # r6 round-trip via SE(3) uses matrix columns → exact up to float32
        R_orig = rot6d_to_matrix(r6)
        R_rec = rot6d_to_matrix(r6_rec)
        assert np.allclose(R_rec, R_orig, atol=TOL)


def test_SE3_body_delta_compose_round_trip():
    """For random (T_state, T_abs), compose(T_state, delta) == T_abs."""
    rng = np.random.default_rng(2)
    H = 20  # chunk length; must match action_horizon
    for _ in range(50):
        xyz_s = rng.normal(size=3)
        r6_s = matrix_to_rot6d(R.random(random_state=rng.integers(1 << 31)).as_matrix())
        T_state = pose_to_SE3(xyz_s, r6_s)

        # Chunked absolute targets
        xyz_abs = rng.normal(size=(H, 3))
        mats_abs = R.random(H, random_state=rng.integers(1 << 31)).as_matrix()
        r6_abs = matrix_to_rot6d(mats_abs)
        T_abs = pose_to_SE3(xyz_abs, r6_abs)
        assert T_abs.shape == (H, 4, 4)

        # T_state broadcasts to (H, 4, 4) on the left
        T_delta = se3_body_delta(T_state, T_abs)
        T_rec = se3_body_compose(T_state, T_delta)

        assert np.allclose(T_rec[..., :3, :3], T_abs[..., :3, :3], atol=1e-6)
        assert np.allclose(T_rec[..., :3, 3], T_abs[..., :3, 3], atol=1e-6)
