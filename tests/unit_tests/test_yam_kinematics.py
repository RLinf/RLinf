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

"""Real-model IK tests: no CAN, cameras, Ray or viewer are opened."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rlinf.envs.real.yam.config import YamIKConfig
from rlinf.envs.real.yam.kinematics import YamKinematicsAdapter

pytest.importorskip("i2rt.robots.kinematics")
pytest.importorskip("quadprog")


@pytest.fixture
def kin():
    return YamKinematicsAdapter(config=YamIKConfig(max_solve_s=2.0))


@pytest.mark.parametrize("grip", [0.0, 0.5, 1.0])
def test_real_gripper_fk_ik_and_frozen_fingers(kin, grip):
    q = np.array([0.1, 1.0, 1.0, 0.1, 0.1, 0.1])
    goal_q = q + np.array([0.002, -0.003, 0.004, 0.001, -0.002, 0.003])
    target = kin.fk(goal_q, grip)
    result = kin.solve(target, q, grip)
    assert result.success, result
    assert result.q_target.shape == (6,)
    pose = kin.fk(result.q_target, grip)
    assert np.linalg.norm(pose[:3, 3] - target[:3, 3]) < 0.001
    assert Rotation.from_matrix(target[:3, :3] @ pose[:3, :3].T).magnitude() < 0.01
    full = kin._kin._configuration.q
    np.testing.assert_allclose(full[kin._finger_q], grip * 0.048, atol=1e-8)


def test_unreachable_target_and_seed_outside_limits_are_rejected(kin):
    q = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    target = kin.fk(q, 0.5)
    target[0, 3] += 5.0
    assert not kin.solve(target, q, 0.5).success
    bad_seed = q.copy()
    bad_seed[0] = 100.0
    assert kin.solve(target, bad_seed, 0.5).reason == "seed_out_of_limits"


def test_model_instances_are_independent_and_tcp_is_finger_independent(kin):
    other = YamKinematicsAdapter(config=YamIKConfig(max_solve_s=2.0))
    q = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    first = kin.fk(q, 0.0)
    other.fk(q + 0.1, 1.0)
    np.testing.assert_allclose(kin.fk(q, 1.0), first)
    assert other.model is not kin.model


def test_bad_transforms_and_slow_results(kin):
    q = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="4x4"):
        kin.solve(np.zeros(7), q, 0.5)
    target = kin.fk(q, 0.5)
    target[0, 0] = 10
    with pytest.raises(ValueError, match="rotation"):
        kin.solve(target, q, 0.5)
    slow = YamKinematicsAdapter(config=YamIKConfig(max_solve_s=1e-12))
    assert slow.solve(kin.fk(q, 0.5), q, 0.5).reason == "solve_timeout"


def test_converged_target_beyond_runtime_step_is_not_rejected(kin):
    q = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    goal = q.copy()
    goal[0] += 0.12
    target = kin.fk(goal, 0.5)
    result = kin.solve(target, q, 0.5)
    assert result.success, result
    assert np.max(np.abs(result.q_target - q)) > 0.05
    assert result.position_error < kin.config.position_tolerance
    assert result.rotation_error < kin.config.rotation_tolerance


def test_unverified_model_is_rejected_before_loading():
    with pytest.raises(ValueError, match="requires yam"):
        YamKinematicsAdapter(gripper_type="no_gripper")


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("degrees", [-5.0, 5.0])
def test_fixed_tcp_rotation_targets_are_reachable(kin, axis, degrees):
    q = np.array([0.1, 1.0, 1.0, 0.4, 0.5, 0.3])
    reference = kin.fk(q, 0.5)
    target = reference.copy()
    delta = Rotation.from_rotvec(np.eye(3)[axis] * np.deg2rad(degrees))
    target[:3, :3] = delta.as_matrix() @ reference[:3, :3]
    result = kin.solve(target, q, 0.5)
    assert result.success, result
    actual = kin.fk(result.q_target, 0.5)
    assert np.linalg.norm(actual[:3, 3] - reference[:3, 3]) < 0.001
    assert Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude() < 0.01
