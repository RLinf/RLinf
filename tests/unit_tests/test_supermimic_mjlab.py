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

"""G1 kinematic parity and one MuJoCo Warp step. Not a LAFAN1 score."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mjlab")
if not torch.cuda.is_available():
    pytest.skip("MuJoCo Warp tests need CUDA", allow_module_level=True)

from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    EPISODE_FAILURE,
    G1_JOINT_NAMES,
)
from rlinf.envs.sim.g1_mimic.g1_fk import (
    TRACKED_BODY_NAMES,
    asset_default_state,
    asset_joint_limits,
    g1_kinematics,
)
from rlinf.envs.sim.g1_mimic.mjlab_env import DECIMATION, PHYSICS_DT, MjlabG1Env
from rlinf.envs.sim.g1_mimic.motion import load_motion_library
from rlinf.envs.sim.g1_mimic.reward import analytic_reward


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


def test_g1_fk_matches_mujoco_state():
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
        max_pos = max(max_pos, float((pos[0, i] - torch.tensor(data.xpos[bid])).abs().max()))
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


def test_mjlab_warp_step_keeps_the_terminal_state(tmp_path):
    if not torch.cuda.is_available():
        raise RuntimeError("MuJoCo Warp G1 step needs the GPU")
    _write_clip(tmp_path)
    motion = load_motion_library(str(tmp_path), 1, "cuda", allow_synthetic=False)
    env = MjlabG1Env(
        1,
        motion,
        torch.device("cuda"),
        max_episode_seconds=10.0,
        eval_mode=True,
    )
    assert abs(float(env.sim.mj_model.opt.timestep) - PHYSICS_DT) < 1e-8
    assert DECIMATION == 4
    assert float(env.action_scale.max() - env.action_scale.min()) > 0.0
    assert tuple(env.robot.joint_names) == G1_JOINT_NAMES
    env.reset(torch.tensor([0], device="cuda"), torch.tensor([0], device="cuda"), torch.tensor([0], device="cuda"))
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
