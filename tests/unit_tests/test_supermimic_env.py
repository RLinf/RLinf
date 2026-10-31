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

"""SuperMimic G1 environment: kinematics, reward, observation, velocity task, and MuJoCo Warp."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from _supermimic_support import AnalyticG1Env

from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    EPISODE_CLIP_END,
    EPISODE_CONT,
    EPISODE_FAILURE,
    EPISODE_TIMEOUT,
    G1_JOINT_NAMES,
    JOINT_POS,
    ROOT_LIN_VEL,
    XI_DIM,
    critic_input_dim,
)
from rlinf.envs.sim.g1_mimic.env import compose_end_kind
from rlinf.envs.sim.g1_mimic.features import (
    default_state,
    heading_invariant_features,
    heading_yaw,
    integrate_residual,
    quat_mul,
    quat_normalize,
    rotmat_to_6d,
    rotvec_to_quat,
    yaw_rotation,
)
from rlinf.envs.sim.g1_mimic.g1_fk import (
    TRACKED_BODY_NAMES,
    asset_default_state,
    asset_joint_limits,
    g1_kinematics,
)
from rlinf.envs.sim.g1_mimic.mjlab_env import (
    DECIMATION,
    PHYSICS_DT,
    MjlabG1Env,
    check_default_pose,
)
from rlinf.envs.sim.g1_mimic.motion import MotionLibrary, load_motion_library
from rlinf.envs.sim.g1_mimic.obs import add_sensor_noise, analytic_obs
from rlinf.envs.sim.g1_mimic.reward import (
    analytic_reward,
    differentiable_fk,
    failure_signals,
    motion_critic_input,
    reanchor_bodies,
    zita_critic_features,
)
from rlinf.envs.sim.g1_mimic.velocity import (
    heading_velocity,
    sample_velocity_command,
    velocity_fall,
)

needs_mjlab = pytest.mark.skipif(
    importlib.util.find_spec("mjlab") is None,
    reason="mjlab is an optional SuperMimic extra",
)

needs_warp = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("mjlab") is None,
    reason="MuJoCo Warp tests need CUDA and mjlab",
)


def _write_clip(directory: Path) -> None:
    state = asset_default_state()
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "fps": 50.0,
            "joint_names": list(G1_JOINT_NAMES),
            "q0": state[7:36].clone(),
            "state": state.unsqueeze(0).repeat(40, 1),
        },
        directory / "clip_000.pt",
    )


def _same_instant(env) -> None:
    """Returned 71-d state matches current qpos/qvel after one forward."""
    state = env._read_state()
    indexing = env.robot.data.indexing
    qpos = env.robot.data.data.qpos
    qvel = env.robot.data.data.qvel
    qadr = indexing.free_joint_q_adr
    vadr = indexing.free_joint_v_adr
    assert torch.allclose(state[:, :3] + env._origins, qpos[:, qadr[:3]], atol=1e-4)
    assert torch.allclose(state[:, 3:7], qpos[:, qadr[3:7]], atol=1e-4)
    assert torch.allclose(state[:, 7:36], qpos[:, indexing.joint_q_adr], atol=1e-4)
    assert torch.allclose(state[:, 36:39], qvel[:, vadr[:3]], atol=1e-4)
    assert torch.allclose(state[:, 39:42], qvel[:, vadr[3:6]], atol=1e-4)
    assert torch.allclose(state[:, 42:71], qvel[:, indexing.joint_v_adr], atol=1e-4)
    env.sim.forward()
    again = env._read_state()
    assert torch.allclose(state, again, atol=1e-5)


def test_zero_residual_keeps_unit_quaternion():
    state = default_state(4, "cpu", torch.float32)
    state[:, 3:7] = quat_normalize(torch.randn(4, 4))
    residual = torch.zeros(4, 70)
    next_state = integrate_residual(state, residual)
    assert torch.allclose(next_state, state, atol=1e-6)
    norms = next_state[:, 3:7].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_heading_features_ignore_global_yaw():
    state = default_state(1, "cpu", torch.float32)
    yawed = state.clone()
    yawed[0, 3:7] = torch.tensor([0.7071, 0.0, 0.0, 0.7071])
    f0 = heading_invariant_features(state)
    f1 = heading_invariant_features(yawed)
    assert f0.shape[-1] == 71
    # height and joint features stay aligned after removing yaw
    assert torch.allclose(f0[..., 0], f1[..., 0], atol=1e-5)


def test_episode_end_codes_are_distinct():
    assert len({EPISODE_CONT, EPISODE_FAILURE, EPISODE_TIMEOUT, EPISODE_CLIP_END}) == 4


def test_heading_velocity_residual_rotates_with_yaw():
    state = default_state(1, "cpu", torch.float32)
    state[0, 36] = 0.2
    residual = torch.zeros(1, 70)
    residual[0, 35] = 0.1
    nxt = integrate_residual(state, residual)
    turned = state.clone()
    turned[0, 3:7] = rotvec_to_quat(torch.tensor([[0.0, 0.0, torch.pi / 2]]))[0]
    rotation = yaw_rotation(heading_yaw(turned[0, 3:7]))
    turned[0, 36:39] = rotation @ state[0, 36:39]
    nxt_turned = integrate_residual(turned, residual)
    expected = rotation @ nxt[0, 36:39]
    assert torch.allclose(nxt_turned[0, 36:39], expected, atol=1e-5)


@needs_mjlab
def test_reward_orientation_keeps_the_batch_axis():
    state = default_state(2, "cpu", torch.float32)
    ref = state.clone()
    state[1, 3:7] = rotvec_to_quat(torch.tensor([[0.6, 0.0, 0.0]]))[0]
    action = torch.zeros(2, ACTION_DIM)
    out = analytic_reward(state, ref, action, action)
    assert out["reward"].shape == (2,)
    assert out["root_pos"].shape == (2,)
    assert float(out["reward"][0]) > float(out["reward"][1])
    assert float(out["root_ori"][0]) > float(out["root_ori"][1])


@needs_mjlab
def test_tilt_compares_the_reference_pose():
    state = default_state(1, "cpu", torch.float32)
    state[0, 3:7] = rotvec_to_quat(torch.tensor([[1.5, 0.0, 0.0]]))[0]
    upright = default_state(1, "cpu", torch.float32)
    _, _, failed = failure_signals(state, upright)
    assert bool(failed)
    _, tilt, failed_same = failure_signals(state, state.clone())
    assert abs(float(tilt)) < 1e-4
    assert not bool(failed_same)


@needs_mjlab
def test_reference_clock_holds_the_last_frame():
    clip = default_state(5, "cpu", torch.float32)
    clip[:, 0] = torch.arange(5, dtype=torch.float32)
    library = MotionLibrary([clip])
    held = library.state_at(torch.tensor([0]), torch.tensor([100]))
    last = library.state_at(torch.tensor([0]), torch.tensor([4]))
    assert torch.allclose(held, last)
    assert torch.allclose(
        library.goal_ref(torch.tensor([0]), torch.tensor([4])),
        library.goal_ref(torch.tensor([0]), torch.tensor([80])),
    )
    env = AnalyticG1Env(
        1, library, torch.device("cpu"), max_episode_seconds=10.0, eval_mode=True
    )
    env.reset(torch.tensor([0]), torch.tensor([0]), torch.tensor([3]))
    continued = env.step(torch.zeros(1, ACTION_DIM))
    assert int(continued["end_kind"]) == EPISODE_CONT
    assert int(env.motion_time) == 4
    terminal = env.step(torch.zeros(1, ACTION_DIM))
    assert int(terminal["end_kind"]) == EPISODE_CLIP_END
    assert torch.allclose(terminal["ref_state"], last)
    assert int(env.motion_time) == 5


@needs_mjlab
def test_pelvis_reward_anchor_ignores_a_waist_yaw_the_torso_anchor_sees():
    robot = default_state(1, "cpu", torch.float64)
    ref = robot.clone()
    waist = 7 + G1_JOINT_NAMES.index("waist_yaw_joint")
    robot[:, waist] = 0.5
    action = torch.zeros(1, ACTION_DIM, dtype=torch.float64)
    torso = analytic_reward(robot, ref, action, action, reward_anchor="torso")
    pelvis = analytic_reward(robot, ref, action, action, reward_anchor="pelvis")
    assert float(torso["root_ori"]) < float(pelvis["root_ori"])
    pos, rot, _, _ = differentiable_fk(robot)
    ref_pos, ref_rot, _, _ = differentiable_fk(ref)
    anchored, _ = reanchor_bodies(pos, rot, ref_pos, ref_rot, "pelvis")
    assert torch.allclose(anchored[:, 0, :2], pos[:, 0, :2], atol=1e-12)


@needs_mjlab
def test_body_terms_reanchor_the_reference_at_the_robot_torso_and_keep_root_drift():
    from rlinf.envs.sim.g1_mimic.g1_fk import TRACKED_BODY_NAMES

    torch.manual_seed(0)
    robot = default_state(4, "cpu", torch.float64)
    robot[:, 3:7] = rotvec_to_quat(0.3 * torch.randn(4, 3, dtype=torch.float64))
    robot[:, JOINT_POS] = 0.2 * torch.randn(4, 29, dtype=torch.float64)
    ref = robot.clone()
    ref[:, 0:2] += torch.randn(4, 2, dtype=torch.float64)
    ref[:, 3:7] = rotvec_to_quat(0.5 * torch.randn(4, 3, dtype=torch.float64))
    ref[:, JOINT_POS] += 0.3 * torch.randn(4, 29, dtype=torch.float64)
    pos, rot, _, _ = differentiable_fk(robot)
    ref_pos, ref_rot, _, _ = differentiable_fk(ref)
    apos, arot = reanchor_bodies(pos, rot, ref_pos, ref_rot)
    torso = TRACKED_BODY_NAMES.index("torso_link")
    # The anchored torso takes the robot torso's xy and the reference torso's height.
    assert torch.allclose(apos[:, torso, :2], pos[:, torso, :2], atol=1e-12)
    assert torch.allclose(apos[:, torso, 2], ref_pos[:, torso, 2], atol=1e-12)
    # BeyondMimic turns the reference by the yaw of R_robot_torso R_ref_torso^T.
    relative = rot[:, torso] @ ref_rot[:, torso].transpose(-1, -2)
    yaw = torch.atan2(relative[:, 1, 0], relative[:, 0, 0])
    cos, sin, zero, one = (
        yaw.cos(),
        yaw.sin(),
        torch.zeros_like(yaw),
        torch.ones_like(yaw),
    )
    expected_turn = torch.stack(
        (
            torch.stack((cos, -sin, zero), dim=-1),
            torch.stack((sin, cos, zero), dim=-1),
            torch.stack((zero, zero, one), dim=-1),
        ),
        dim=-2,
    )
    # One yaw turn for every body, so the reference's own pose is kept.
    turn = arot @ ref_rot.transpose(-1, -2)
    assert torch.allclose(turn[:, torso], expected_turn, atol=1e-10)
    assert torch.allclose(turn, turn[:, :1].expand_as(turn), atol=1e-10)
    assert torch.allclose(
        turn[..., 2, :],
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).expand(4, 14, 3),
        atol=1e-10,
    )
    assert torch.allclose(
        apos - apos[:, torso : torso + 1],
        torch.einsum("bkij,bkj->bki", turn, ref_pos - ref_pos[:, torso : torso + 1]),
        atol=1e-10,
    )
    # A reference that turns the pelvis by -0.5 rad and the waist by +0.5 rad
    # keeps the upper body where the robot's is once anchored at the torso.
    upright = default_state(1, "cpu", torch.float64)
    twisted = upright.clone()
    twisted[0, 3:7] = rotvec_to_quat(
        torch.tensor([[0.0, 0.0, -0.5]], dtype=torch.float64)
    )[0]
    twisted[0, 7 + G1_JOINT_NAMES.index("waist_yaw_joint")] = 0.5
    pos, rot, _, _ = differentiable_fk(upright)
    ref_pos, ref_rot, _, _ = differentiable_fk(twisted)
    apos, arot = reanchor_bodies(pos, rot, ref_pos, ref_rot)
    upper = [TRACKED_BODY_NAMES.index(name) for name in TRACKED_BODY_NAMES[7:]]
    lower = [TRACKED_BODY_NAMES.index(name) for name in TRACKED_BODY_NAMES[1:7]]
    assert float((apos[0, upper] - pos[0, upper]).norm(dim=-1).max()) < 1e-9
    assert float((apos[0, lower] - pos[0, lower]).norm(dim=-1).min()) > 1e-3

    robot = default_state(1, "cpu", torch.float32)
    ref = robot.clone()
    ref[0, 0] = 1.5
    ref[0, 1] = -0.4
    ref[0, 3:7] = rotvec_to_quat(torch.tensor([[0.0, 0.0, 1.5707963]]))[0]
    action = torch.zeros(1, ACTION_DIM)
    ref_still = ref.clone()
    ref_still[0, ROOT_LIN_VEL] = 0.0
    drifted = analytic_reward(robot, ref_still, action, action)
    matched = analytic_reward(robot, robot, action, action)
    assert float(drifted["root_pos"]) < float(matched["root_pos"])
    assert float(drifted["body_tracking"]) == pytest.approx(
        float(matched["body_tracking"]), abs=1e-4
    )


def test_sensor_noise_is_uniform_and_skips_previous_action():
    torch.manual_seed(0)
    obs = torch.zeros(20000, 93)
    noisy = add_sensor_noise(obs, True)
    limits = ((0, 29, 0.01), (29, 58, 0.5), (58, 61, 0.2), (61, 64, 0.05))
    for start, end, half in limits:
        block = noisy[:, start:end]
        assert float(block.abs().max()) <= half + 1e-6
        std = float(block.std())
        assert std == pytest.approx(half / math.sqrt(3.0), rel=0.05)
        assert float(block.mean()) == pytest.approx(0.0, abs=half * 0.05)
    assert torch.equal(noisy[:, 64:], torch.zeros_like(noisy[:, 64:]))
    assert torch.equal(add_sensor_noise(obs[:4], False), obs[:4])


@needs_mjlab
def test_fk_fast_paths_match_the_reference_math():
    from rlinf.envs.sim.g1_mimic import g1_fk
    from rlinf.envs.sim.g1_mimic.features import quat_to_rotmat
    from rlinf.envs.sim.g1_mimic.reward import (
        PELVIS,
        TORSO,
        differentiable_fk,
        failure_signals,
    )

    gen = torch.Generator().manual_seed(0)
    q1 = torch.randn(64, 4, generator=gen, dtype=torch.float64)
    q2 = torch.randn(64, 4, generator=gen, dtype=torch.float64)
    vec = torch.randn(64, 3, generator=gen, dtype=torch.float64)
    table = g1_fk._hamilton_table().double()
    assert torch.allclose(g1_fk._quat_mul(q1, q2, table), quat_mul(q1, q2), atol=1e-12)
    rotated = torch.einsum("...ij,...j->...i", quat_to_rotmat(q1), vec)
    assert torch.allclose(g1_fk._rotate(q1, vec), rotated, atol=1e-12)

    base = g1_fk.asset_default_state()
    state = base + 0.2 * torch.randn(16, 71, generator=gen)
    ref = base + 0.2 * torch.randn(16, 71, generator=gen)
    assert g1_fk.g1_kinematics().free_tracked == PELVIS
    pos = differentiable_fk(state)[0]
    ref_pos = differentiable_fk(ref)[0]
    height, _, _ = failure_signals(state, ref)
    assert torch.equal(height, pos[:, PELVIS, 2] - ref_pos[:, PELVIS, 2])
    torso_height, _, _ = failure_signals(state, ref, "torso")
    assert torch.equal(torso_height, pos[:, TORSO, 2] - ref_pos[:, TORSO, 2])


@needs_mjlab
def test_level_kinematics_match_the_body_loop():
    """Depth batching is the same map as the body-by-body walk, gradients too."""
    kin = g1_kinematics()
    gen = torch.Generator().manual_seed(1)
    state = torch.randn(5, 71, generator=gen, dtype=torch.float64)
    state[:, 3:7] = torch.nn.functional.normalize(state[:, 3:7], dim=-1)
    left = state.clone().requires_grad_(True)
    right = state.clone().requires_grad_(True)
    serial = kin._forward_serial(left)
    level = kin._forward_levels(right)
    for got, ref in zip(level, serial, strict=True):
        assert torch.allclose(got, ref, atol=1e-12, rtol=1e-12)
    (sum(part.sum() for part in serial)).backward()
    (sum(part.sum() for part in level)).backward()
    assert torch.allclose(left.grad, right.grad, atol=1e-9, rtol=1e-9)


def test_obs99_relative_heading_is_not_noised_and_is_differentiable():
    state = default_state(1, "cpu", torch.float32)
    ref = rotvec_to_quat(torch.tensor([[0.0, 0.0, 0.4]]))
    obs = analytic_obs(
        state, torch.zeros(1, ACTION_DIM), torch.zeros(1, 33), ref_quat=ref
    )
    assert obs.shape[-1] == 99
    plain = analytic_obs(state, torch.zeros(1, ACTION_DIM), torch.zeros(1, 33))
    assert plain.shape[-1] == 93
    assert torch.allclose(obs[..., :93], plain)
    same = analytic_obs(
        state, torch.zeros(1, ACTION_DIM), torch.zeros(1, 33), ref_quat=state[..., 3:7]
    )
    # First two columns of the identity, row-major, matching zita-ch mat_to_6d.
    assert torch.allclose(
        same[0, 93:], torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]), atol=1e-5
    )
    torch.manual_seed(0)
    noisy = add_sensor_noise(obs.expand(1000, 99).clone(), True)
    assert torch.equal(noisy[:, 93:], obs.expand(1000, 99)[:, 93:])
    leaf = default_state(1, "cpu", torch.float32).requires_grad_(True)
    out = analytic_obs(
        leaf, torch.zeros(1, ACTION_DIM), torch.zeros(1, 33), ref_quat=ref
    )
    out.sum().backward()
    assert leaf.grad is not None and torch.isfinite(leaf.grad).all()


def test_velocity_command_mix_and_fall_threshold():
    torch.manual_seed(0)
    command, from_ref = sample_velocity_command(20000, torch.device("cpu"))
    assert float(from_ref.float().mean()) == pytest.approx(0.5, abs=0.02)
    random = command[~from_ref]
    standing = random.abs().sum(dim=-1) == 0
    assert float(standing.float().mean()) == pytest.approx(0.1, abs=0.02)
    moving = random[~standing]
    assert float(moving[:, 0].min()) >= -1.5
    assert float(moving[:, 0].max()) <= 2.0
    assert float(moving[:, 1].abs().max()) <= 1.32
    assert float(moving[:, 2].abs().max()) <= 3.35
    state = default_state(1, "cpu", torch.float32)
    state[0, 2] = 0.2
    assert bool(velocity_fall(state))
    state[0, 2] = 0.8
    assert not bool(velocity_fall(state))
    planar, yaw = heading_velocity(state)
    assert planar.shape[-1] == 2
    assert yaw.shape == (1,)


def test_velocity_soft_fall_matches_the_hard_threshold_away_from_it():
    from rlinf.envs.sim.g1_mimic.features import default_state
    from rlinf.envs.sim.g1_mimic.velocity import velocity_fall, velocity_fall_soft

    standing = default_state(1, "cpu", torch.float32)
    standing[0, 2] = 0.8
    low = standing.clone()
    low[0, 2] = 0.2
    assert not bool(velocity_fall(standing))
    assert bool(velocity_fall(low))
    assert float(velocity_fall_soft(standing)) < 0.01
    assert float(velocity_fall_soft(low)) > 0.99


def test_end_kind_keeps_timeout_and_clip_end_beside_a_task_failure():
    failure = torch.tensor([True, False, False, True])
    timeout = torch.tensor([True, True, False, False])
    clip_end = torch.tensor([False, False, True, True])
    end = compose_end_kind(failure, timeout, clip_end)
    assert end.tolist() == [
        EPISODE_FAILURE,
        EPISODE_TIMEOUT,
        EPISODE_CLIP_END,
        EPISODE_FAILURE,
    ]


def test_analytic_termination_is_a_soft_version_of_the_hard_failure_test():
    from rlinf.envs.sim.g1_mimic.reward import analytic_failure_prob

    height = torch.tensor([0.0, 0.1, -0.25, 0.4, 0.0, 0.0, -0.5])
    tilt = torch.tensor([0.0, 0.3, 0.0, 0.0, 0.8, -1.2, 0.9])
    prob = analytic_failure_prob(height, tilt)
    assert float(prob[0]) < 1e-5 and float(prob[1]) < 1e-3
    assert float(prob[2]) == pytest.approx(0.5, abs=1e-6)
    assert float(prob[4]) == pytest.approx(0.5, abs=1e-5)
    assert torch.all(prob[[3, 5, 6]] > 0.99)


def test_unknown_env_backend_names_the_registered_ones():
    from rlinf.envs.sim.g1_mimic.env import env_backends, make_env

    assert "mjlab" in env_backends()
    with pytest.raises(ValueError, match="unknown SuperMimic env backend 'nope'"):
        make_env("nope", num_envs=1)


def test_clips_must_use_the_asset_default_pose_as_q0():
    pd_offset = torch.linspace(-0.3, 0.3, ACTION_DIM)
    check_default_pose(pd_offset.double(), pd_offset)
    q0 = pd_offset.clone()
    q0[5] += 0.01
    with pytest.raises(ValueError, match=G1_JOINT_NAMES[5]):
        check_default_pose(q0, pd_offset)


@needs_warp
def test_g1_fk_matches_mujoco_state():
    import mujoco
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

    model = get_spec().compile()
    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    data.qpos[0:3] = np.array([0.15, -0.2, 0.82])
    data.qpos[3:7] = np.array([0.94, 0.12, -0.2, 0.18])
    data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])
    data.qpos[7:] = rng.uniform(-0.4, 0.4, size=model.nq - 7)
    data.qvel[:] = rng.uniform(-0.8, 0.8, size=model.nv)
    mujoco.mj_forward(model, data)
    state = torch.zeros(1, 71)
    state[0, 0:7] = torch.tensor(data.qpos[:7], dtype=torch.float32)
    state[0, 7:36] = torch.tensor(data.qpos[7:], dtype=torch.float32)
    state[0, 36:42] = torch.tensor(data.qvel[:6], dtype=torch.float32)
    state[0, 42:71] = torch.tensor(data.qvel[6:], dtype=torch.float32)
    pos, rot, lin, ang = g1_kinematics().forward(state)
    max_pos = 0.0
    max_rot = 0.0
    max_lin = 0.0
    max_ang = 0.0
    for i, name in enumerate(TRACKED_BODY_NAMES):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        max_pos = max(
            max_pos, float((pos[0, i] - torch.tensor(data.xpos[bid])).abs().max())
        )
        sim_rot = torch.tensor(data.xmat[bid].reshape(3, 3), dtype=torch.float32)
        max_rot = max(max_rot, float((rot[0, i] - sim_rot).abs().max()))
        res = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, res, 0)
        ang_m = torch.tensor(res[:3], dtype=torch.float32)
        lin_com = torch.tensor(res[3:], dtype=torch.float32)
        origin = torch.tensor(data.xpos[bid], dtype=torch.float32)
        com = torch.tensor(data.xipos[bid], dtype=torch.float32)
        lin_origin = lin_com + torch.cross(ang_m, origin - com, dim=-1)
        max_lin = max(max_lin, float((lin[0, i] - lin_origin).abs().max()))
        max_ang = max(max_ang, float((ang[0, i] - ang_m).abs().max()))
    assert max_pos < 1e-4
    assert max_rot < 1e-4
    assert max_lin < 1e-4
    assert max_ang < 1e-4
    action = torch.zeros(1, ACTION_DIM)
    ours = analytic_reward(state, state, action, action)
    assert torch.isfinite(ours["reward"]).all()
    # Identical states: body and root tracking terms are at their positive caps.
    assert float(ours["body_tracking"]) > 0.0
    assert float(ours["root_pos"]) > 0.0
    assert float(ours["root_ori"]) > 0.0
    low, high = asset_joint_limits()
    assert low.shape == (ACTION_DIM,)
    assert torch.all(high > low)
    assert not torch.allclose(high, torch.full_like(high, 2.5))
    over = state.clone()
    over[0, 7] = float(high[0]) + 0.2
    limited = analytic_reward(over, over, action, action, low, high)
    assert float(limited["joint_limit"]) < 0.0


@needs_warp
def test_mjlab_warp_step_keeps_the_terminal_state(tmp_path):
    if not torch.cuda.is_available():
        raise RuntimeError("MuJoCo Warp G1 step needs the GPU")
    _write_clip(tmp_path)
    motion = load_motion_library(str(tmp_path), 1, "cuda")
    env = MjlabG1Env(
        1,
        motion,
        torch.device("cuda"),
        max_episode_seconds=10.0,
        eval_mode=True,
    )
    import mujoco_warp as mjwarp

    assert abs(float(env.sim.mj_model.opt.timestep) - PHYSICS_DT) < 1e-8
    linesearch_warn = int(mjwarp.OverflowType.LS_ITERATIONS)
    assert not int(env.sim.wp_model.opt.warn_overflow) & linesearch_warn
    assert DECIMATION == 4
    assert float(env.action_scale.max() - env.action_scale.min()) > 0.0
    assert tuple(env.robot.joint_names) == G1_JOINT_NAMES
    env.reset(
        torch.tensor([0], device="cuda"),
        torch.tensor([0], device="cuda"),
        torch.tensor([0], device="cuda"),
    )
    pos, _, lin, ang = g1_kinematics().forward(env.state)
    for i, name in enumerate(TRACKED_BODY_NAMES):
        local = env.robot.body_names.index(name)
        sim_pos = env.robot.data.body_link_pos_w[0, local] - env._origins[0]
        sim_lin = env.robot.data.body_link_lin_vel_w[0, local]
        sim_ang = env.robot.data.body_link_ang_vel_w[0, local]
        assert torch.allclose(pos[0, i].cpu(), sim_pos.cpu(), atol=2e-3)
        assert torch.allclose(lin[0, i].cpu(), sim_lin.cpu(), atol=2e-2)
        assert torch.allclose(ang[0, i].cpu(), sim_ang.cpu(), atol=2e-2)
    held = env.step(torch.zeros(1, ACTION_DIM, device="cuda"))
    assert torch.allclose(held["next_state"], env.state)
    assert float(env.state[0, 2]) > 0.5
    fallen = env.state.clone()
    fallen[0, 2] = 0.05
    env._write_state(torch.tensor([0], device="cuda"), fallen)
    env.state = env._read_state()
    result = env.step(torch.zeros(1, ACTION_DIM, device="cuda"))
    assert int(result["end_kind"][0]) == EPISODE_FAILURE
    assert float(result["next_state"][0, 2]) < 0.35
    low = float(env.state[0, 2])
    env.reset(
        torch.tensor([0], device="cuda"),
        torch.tensor([0], device="cuda"),
        torch.tensor([0], device="cuda"),
    )
    assert float(env.state[0, 2]) > low + 0.3


@needs_warp
def test_warp_state_is_one_physical_time(tmp_path):
    """Nonzero root and joint velocity, several 5 ms substeps, one timestamp."""
    _write_clip(tmp_path)
    motion = load_motion_library(str(tmp_path), 1, "cuda")
    env = MjlabG1Env(
        1,
        motion,
        torch.device("cuda"),
        max_episode_seconds=10.0,
        eval_mode=True,
    )
    env.reset(
        torch.tensor([0], device="cuda"),
        torch.tensor([0], device="cuda"),
        torch.tensor([0], device="cuda"),
    )
    _same_instant(env)
    launched = env.state.clone()
    launched[:, 2] = 1.4
    launched[:, 36:39] = torch.tensor([1.0, -0.25, 0.15], device="cuda")
    launched[:, 39:42] = torch.tensor([0.3, -0.2, 0.45], device="cuda")
    launched[:, 42:71] = 0.8
    env._write_state(torch.tensor([0], device="cuda"), launched)
    env.state = env._read_state()
    qvel = env.robot.data.data.qvel
    vadr = env.robot.data.indexing.free_joint_v_adr
    assert float(qvel[0, vadr[0]].abs()) > 0.5
    assert float(qvel[0, vadr[3:6]].abs().max()) > 0.1
    assert float(qvel[0, env.robot.data.indexing.joint_v_adr].abs().max()) > 0.1
    _same_instant(env)
    for _ in range(3):
        result = env.step(torch.zeros(1, ACTION_DIM, device="cuda"))
        env.state = result["next_state"]
        _same_instant(env)
        assert torch.allclose(result["next_state"], env.state, atol=1e-6)
    # The old bug disagreed with qpos.root.x by about one 5 ms step.
    root_x = result["next_state"][0, 0] + env._origins[0, 0]
    qpos_x = env.robot.data.data.qpos[0, env.robot.data.indexing.free_joint_q_adr[0]]
    assert torch.allclose(root_x, qpos_x, atol=1e-4)
    assert torch.isfinite(result["next_state"]).all()
    reward = analytic_reward(
        result["next_state"],
        result["ref_state"],
        torch.zeros(1, ACTION_DIM, device="cuda"),
        torch.zeros(1, ACTION_DIM, device="cuda"),
    )
    assert torch.isfinite(reward["reward"]).all()


@needs_warp
def test_warp_step_scores_against_the_next_reference_frame(tmp_path):
    frames = 12
    state = asset_default_state().unsqueeze(0).repeat(frames, 1)
    state[:, 0] = 0.01 * torch.arange(frames, dtype=state.dtype)
    tmp_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "fps": 50.0,
            "joint_names": list(G1_JOINT_NAMES),
            "q0": state[0, 7:36].clone(),
            "state": state,
        },
        tmp_path / "clip_000.pt",
    )
    motion = load_motion_library(str(tmp_path), 1, "cuda")
    env = MjlabG1Env(2, motion, torch.device("cuda"), eval_mode=True)
    ids = torch.arange(2, device="cuda")
    clip = torch.zeros(2, dtype=torch.long, device="cuda")
    start = torch.tensor([5, frames - 1], device="cuda")
    env.reset(ids, clip, start)
    result = env.step(torch.zeros(2, ACTION_DIM, device="cuda"))
    assert env.motion_time.tolist() == [6, frames]
    expected = motion.state_at(clip, start + 1)
    assert torch.equal(result["ref_state"], expected)
    assert float(result["ref_state"][1, 0]) == pytest.approx(0.01 * (frames - 1))
    assert result["clip_end"].tolist() == [False, True]


@needs_mjlab
def test_zita_critic_features_match_a_perfect_track():
    state = default_state(2, "cpu", torch.float32)
    state[:, JOINT_POS] = 0.2
    ref = state.clone()
    xi = torch.randn(2, XI_DIM)
    feat = zita_critic_features(state, ref, xi)
    assert feat.shape[-1] == 71 + 29 + 14 * 3 + 3 + 6 + XI_DIM
    assert float(feat[:, 71:100].abs().max()) < 1e-5
    assert float(feat[:, 100:142].abs().max()) < 1e-4
    assert float(feat[:, 142:145].abs().max()) < 1e-5
    identity = rotmat_to_6d(torch.eye(3))
    assert torch.allclose(feat[:, 145:151], identity.expand(2, 6), atol=1e-5)
    assert torch.allclose(feat[:, 151:], xi)
    goal = torch.randn(2, 730)
    full = motion_critic_input(state, goal, xi, ref, "zita")
    assert full.shape[-1] == critic_input_dim(730, "zita")
    assert torch.allclose(full[:, -730:], goal)
    moved = state.clone()
    moved[:, JOINT_POS] = state[:, JOINT_POS] + 0.1
    changed = zita_critic_features(moved, ref, xi)
    assert float((changed[:, 71:100] - feat[:, 71:100]).abs().max()) == pytest.approx(
        0.1
    )
