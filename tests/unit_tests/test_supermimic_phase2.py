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

"""730-d goal, 99-d observation, velocity interface, and BONES-SEED filters."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from rlinf.envs.sim.g1_mimic.bones import (
    feasibility_reject,
    load_bones_library,
    name_rejected,
)
from rlinf.envs.sim.g1_mimic.constants import ACTION_DIM, G1_JOINT_NAMES, JOINT_VEL
from rlinf.envs.sim.g1_mimic.features import default_state, rotvec_to_quat
from rlinf.envs.sim.g1_mimic.motion import MotionLibrary
from rlinf.envs.sim.g1_mimic.networks import SuperMimicModels
from rlinf.envs.sim.g1_mimic.obs import add_sensor_noise, analytic_obs
from rlinf.envs.sim.g1_mimic.velocity import (
    heading_velocity,
    sample_velocity_command,
    velocity_fall,
)

needs_mjlab = pytest.mark.skipif(
    importlib.util.find_spec("mjlab") is None,
    reason="mjlab is an optional SuperMimic extra",
)


def test_goal730_uses_future_frames_and_holds_the_last():
    clip = default_state(4, "cpu", torch.float32)
    for i in range(4):
        clip[i, 0] = float(i)
        clip[i, JOINT_VEL] = 10.0
    goal = MotionLibrary([clip]).goal730(torch.tensor([0]), torch.tensor([0]))
    assert goal.shape == (1, 730)
    frames = goal.view(1, 10, 73)
    assert torch.allclose(frames[0, :, 0], torch.tensor([1, 2, 3, 3, 3, 3, 3, 3, 3, 3]).float())
    assert torch.allclose(frames[0, 0, 44:73], torch.ones(29))


def test_obs99_relative_heading_is_not_noised_and_is_differentiable():
    state = default_state(1, "cpu", torch.float32)
    ref = rotvec_to_quat(torch.tensor([[0.0, 0.0, 0.4]]))
    obs = analytic_obs(state, torch.zeros(1, ACTION_DIM), torch.zeros(1, 33), ref_quat=ref)
    assert obs.shape[-1] == 99
    plain = analytic_obs(state, torch.zeros(1, ACTION_DIM), torch.zeros(1, 33))
    assert plain.shape[-1] == 93
    assert torch.allclose(obs[..., :93], plain)
    same = analytic_obs(state, torch.zeros(1, ACTION_DIM), torch.zeros(1, 33), ref_quat=state[..., 3:7])
    # First two columns of the identity, row-major, matching zita-ch mat_to_6d.
    assert torch.allclose(same[0, 93:], torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]), atol=1e-5)
    torch.manual_seed(0)
    noisy = add_sensor_noise(obs.expand(1000, 99).clone(), True)
    assert torch.equal(noisy[:, 93:], obs.expand(1000, 99)[:, 93:])
    leaf = default_state(1, "cpu", torch.float32).requires_grad_(True)
    out = analytic_obs(leaf, torch.zeros(1, ACTION_DIM), torch.zeros(1, 33), ref_quat=ref)
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


def _changed(before: dict[str, torch.Tensor], module: torch.nn.Module) -> bool:
    return any(not torch.equal(before[name], value) for name, value in module.state_dict().items())


def _snap(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def test_motion_and_velocity_policy_steps_touch_disjoint_parameters():
    """One optimizer step each. Critic steps are separate and come after."""
    torch.manual_seed(0)
    models = SuperMimicModels(680, 93, velocity=True)
    actor_ids = {id(param) for param in models.actor_params()}
    velocity_ids = {id(param) for param in models.velocity_actor_params()}
    world_ids = {id(param) for param in models.wm_params()}
    motion_critic_ids = {id(param) for param in models.critic_params()}
    velocity_critic_ids = {id(param) for param in models.velocity_critic_params()}
    assert actor_ids.isdisjoint(velocity_ids)
    assert actor_ids.isdisjoint(world_ids)
    assert velocity_ids.isdisjoint(world_ids)
    assert motion_critic_ids.isdisjoint(velocity_critic_ids)
    assert all(id(param) in actor_ids for param in models.decoder.parameters())

    history = torch.randn(2, 930)
    goal = torch.randn(2, 680)
    command = torch.randn(2, 3)
    opt_actor = torch.optim.Adam(models.actor_params(), lr=1e-2)
    opt_velocity = torch.optim.Adam(models.velocity_actor_params(), lr=1e-2)
    for param in models.parameters():
        param.requires_grad_(False)
    for param in models.actor_params():
        param.requires_grad_(True)
    before = {name: _snap(module) for name, module in models.named_children()}
    models.policy(history, goal).sum().backward()
    opt_actor.step()
    assert _changed(before["encoder"], models.encoder)
    assert _changed(before["decoder"], models.decoder)
    assert not _changed(before["velocity_encoder"], models.velocity_encoder)
    assert not _changed(before["dynamics"], models.dynamics)
    assert not _changed(before["termination"], models.termination)
    assert not _changed(before["critic"], models.critic)
    assert not _changed(before["velocity_critic"], models.velocity_critic)

    models.zero_grad(set_to_none=True)
    for param in models.parameters():
        param.requires_grad_(False)
    for param in models.velocity_actor_params():
        param.requires_grad_(True)
    before = {name: _snap(module) for name, module in models.named_children()}
    action = models.decoder(models.velocity_encoder(command, history), history)
    action.sum().backward()
    assert any(param.grad is not None and float(param.grad.abs().sum()) > 0 for param in models.velocity_encoder.parameters())
    assert all(param.grad is None for param in models.decoder.parameters())
    assert all(param.grad is None for param in models.encoder.parameters())
    opt_velocity.step()
    assert _changed(before["velocity_encoder"], models.velocity_encoder)
    assert not _changed(before["decoder"], models.decoder)
    assert not _changed(before["encoder"], models.encoder)
    assert not _changed(before["dynamics"], models.dynamics)
    assert not _changed(before["critic"], models.critic)
    assert not _changed(before["velocity_critic"], models.velocity_critic)

    state = torch.randn(2, 71, requires_grad=True)
    for param in models.wm_params():
        param.requires_grad_(False)
    with torch.no_grad():
        models.dynamics.net[-1].weight.fill_(0.01)
    models.dynamics(state, torch.randn(2, 29), torch.randn(2, 33)).sum().backward()
    assert state.grad is not None and float(state.grad.abs().sum()) > 0
    assert all(param.grad is None for param in models.wm_params())

    models.zero_grad(set_to_none=True)
    for param in models.velocity_encoder.parameters():
        param.requires_grad_(True)
    z_motion = models.encoder(goal).detach()
    z_vel = models.velocity_encoder(command, history)
    (0.1 * (z_vel - z_motion).pow(2).mean()).backward()
    assert all(param.grad is None for param in models.encoder.parameters())
    opt_critic = torch.optim.Adam(models.critic_params(), lr=1e-2)
    opt_velocity_critic = torch.optim.Adam(models.velocity_critic_params(), lr=1e-2)
    models.zero_grad(set_to_none=True)
    for param in models.parameters():
        param.requires_grad_(False)
    for param in models.critic_params():
        param.requires_grad_(True)
    before = {name: _snap(module) for name, module in models.named_children()}
    models.critic(torch.randn(2, 71), goal, torch.randn(2, 33), torch.randn(2, 4)).sum().backward()
    opt_critic.step()
    assert _changed(before["critic"], models.critic)
    assert not _changed(before["velocity_critic"], models.velocity_critic)
    assert not _changed(before["encoder"], models.encoder)
    models.zero_grad(set_to_none=True)
    for param in models.parameters():
        param.requires_grad_(False)
    for param in models.velocity_critic_params():
        param.requires_grad_(True)
    before = {name: _snap(module) for name, module in models.named_children()}
    models.velocity_critic(torch.randn(2, 71), command, torch.randn(2, 33)).sum().backward()
    opt_velocity_critic.step()
    assert _changed(before["velocity_critic"], models.velocity_critic)
    assert not _changed(before["critic"], models.critic)


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


def test_velocity_encoder_grad_stops_at_the_decoder_and_motion_encoder():
    torch.manual_seed(0)
    models = SuperMimicModels(680, 93, velocity=True)
    for param in models.decoder.parameters():
        param.requires_grad_(False)
    history = torch.randn(2, 930)
    command = torch.randn(2, 3)
    goal = torch.randn(2, 680)
    z_vel = models.velocity_encoder(command, history)
    action = models.decoder(z_vel, history)
    action.sum().backward()
    assert any(param.grad is not None and float(param.grad.abs().sum()) > 0 for param in models.velocity_encoder.parameters())
    assert all(param.grad is None for param in models.decoder.parameters())
    assert all(param.grad is None for param in models.encoder.parameters())
    models.zero_grad()
    z_motion = models.encoder(goal).detach()
    z_vel = models.velocity_encoder(command, history)
    (0.1 * (z_vel - z_motion).pow(2).mean()).backward()
    assert all(param.grad is None for param in models.encoder.parameters())
    assert any(param.grad is not None for param in models.velocity_encoder.parameters())


def test_bones_name_and_feasibility_filters(tmp_path):
    assert not name_rejected("walk_forward")
    assert not name_rejected("visit_friend")
    assert name_rejected("sit_down")
    assert name_rejected("climb_box")
    assert name_rejected("jump off 50cm")
    zmin = torch.zeros(50)
    feet = torch.zeros(50, 2)
    pelvis = torch.full((50,), 0.8)
    upright = torch.ones(50, dtype=torch.bool)
    speed = torch.zeros(50)
    assert feasibility_reject(zmin, feet, pelvis, upright, speed) is None
    high = zmin.clone()
    high[:40] = 0.3
    assert feasibility_reject(high, feet, pelvis, upright, speed) == "elevated support"
    deep = zmin.clone()
    deep[0] = -0.2
    assert feasibility_reject(deep, feet, pelvis, upright, speed) == "ground penetration"
    floating = feet.clone()
    floating[:30] = 0.05
    assert feasibility_reject(zmin, floating, pelvis, upright, speed) == "floating feet"
    with pytest.raises(FileNotFoundError):
        load_bones_library(str(tmp_path), 1, "cpu")
    state = default_state(8, "cpu", torch.float32)
    torch.save(
        {
            "name": "sit_on_chair",
            "fps": 50.0,
            "joint_names": list(G1_JOINT_NAMES),
            "q0": torch.zeros(29),
            "state": state,
        },
        tmp_path / "clip_000.pt",
    )
    with pytest.raises(RuntimeError, match="removed every clip"):
        load_bones_library(str(tmp_path), 1, "cpu")
    torch.save(
        {
            "name": "walk_subject1",
            "fps": 50.0,
            "joint_names": list(G1_JOINT_NAMES),
            "q0": torch.zeros(29),
            "state": state,
            "zmin": torch.zeros(8),
            "foot_clearance": torch.zeros(8, 2),
            "upright": torch.ones(8, dtype=torch.bool),
        },
        tmp_path / "clip_000.pt",
    )
    library = load_bones_library(str(tmp_path), 1, torch.device("cuda"))
    assert library.num_clips == 1
    assert library.clip_names == ["walk_subject1"]
    assert library._packed.device.type == "cpu"
    queried = library.state_at(torch.tensor([0]), torch.tensor([3]))
    assert torch.allclose(queried, state[3])
    bad = tmp_path / "human"
    bad.mkdir()
    human = torch.zeros(8, 20)
    torch.save(
        {
            "name": "human_walk",
            "fps": 50.0,
            "joint_names": ["a"],
            "q0": torch.zeros(20),
            "state": human,
        },
        bad / "clip_000.pt",
    )
    with pytest.raises(ValueError):
        load_bones_library(str(bad), 1, "cpu")
