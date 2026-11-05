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

"""SuperMimic core contracts, data preparation, training and evaluation."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from _supermimic_support import (
    AnalyticG1Env,
    small_config,
    synthetic_clip_dir,
    synthetic_clips,
)
from omegaconf import OmegaConf
from torch import nn

from evaluations.eval_supermimic import resolve_checkpoint
from evaluations.eval_supermimic_push import require_matching_checkpoint
from rlinf.algorithms.supermimic.objectives import (
    actor_objective,
    gae_targets,
    state_prediction_loss,
)
from rlinf.algorithms.supermimic.replay import SuperMimicReplay
from rlinf.algorithms.supermimic.reset_sampler import (
    ADAPTIVE_PROB,
    MEMORY_EMA,
    MEMORY_PROB,
    RETURN_EMA,
    TEMPERATURE,
    UNIFORM_PROB,
    BinResetSampler,
)
from rlinf.algorithms.supermimic.validate import validate_supermimic_cfg
from rlinf.envs.sim.g1_mimic.bones import (
    feasibility_reject,
    load_bones_library,
    name_rejected,
)
from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    EPISODE_CLIP_END,
    EPISODE_CONT,
    EPISODE_FAILURE,
    EPISODE_TIMEOUT,
    G1_JOINT_NAMES,
    JOINT_POS,
    JOINT_VEL,
    MAX_TRACKING_REWARD,
    ROOT_ANG_VEL,
    ROOT_LIN_VEL,
    ROOT_POS,
    ROOT_QUAT,
    STATE_DIM,
    XI_DIM,
    critic_input_dim,
)
from rlinf.envs.sim.g1_mimic.env import compose_end_kind
from rlinf.envs.sim.g1_mimic.features import (
    default_state,
    heading_invariant_features,
    heading_yaw,
    integrate_residual,
    quat_inv,
    quat_mul,
    quat_normalize,
    quat_to_rotvec,
    rotmat_to_6d,
    rotvec_to_quat,
    so3_relative_rotvec,
    yaw_rotation,
)
from rlinf.envs.sim.g1_mimic.g1_fk import (
    TRACKED_BODY_NAMES,
    asset_default_state,
    asset_joint_limits,
    g1_kinematics,
)
from rlinf.envs.sim.g1_mimic.mjlab_env import (
    MjlabG1Env,
    _expand_meaninertia,
    check_default_pose,
)
from rlinf.envs.sim.g1_mimic.motion import (
    MotionLibrary,
    control_time_grid,
    decode_motion_payload,
    load_motion_library,
    resample_states,
)
from rlinf.envs.sim.g1_mimic.obs import add_sensor_noise, analytic_obs
from rlinf.envs.sim.g1_mimic.push_recovery import (
    DT,
    RecoveryCriteria,
    check_push_protocol,
    default_standing_state,
    hold_pose_library,
    is_stable,
    judge_post_push,
    root_tilt,
    summarize_trials,
    trial_plan,
    world_push_xy,
    write_push_outputs,
)
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
from rlinf.models.embodiment.supermimic import (
    DynamicsNet,
    MotionEncoder,
    SuperMimicModels,
)
from rlinf.workers.supermimic import checkpoint as ckpt
from rlinf.workers.supermimic.checkpoint import verify_checkpoint
from rlinf.workers.supermimic.eval_rollout import (
    checked_eval_states,
    checked_policy_action,
    run_tracking_eval,
    summarize_tracking_eval,
)
from rlinf.workers.supermimic.trainer import SuperMimicTrainer
from toolkits.supermimic.convert_bones_seed import (
    CsvError,
    build_payload,
    clip_name_for,
    convert_paths,
    output_filename,
    parse_bones_csv,
)

needs_mjlab = pytest.mark.skipif(
    importlib.util.find_spec("mjlab") is None,
    reason="mjlab is an optional SuperMimic extra",
)


CPU = torch.device("cpu")


def _ema(ret: float, g_max: float = 10.0) -> float:
    return (1.0 - RETURN_EMA) * g_max + RETURN_EMA * ret


def _state_with_quat(quat: torch.Tensor) -> torch.Tensor:
    state = default_state(quat.shape[0], "cpu", torch.float32)
    state[:, ROOT_QUAT] = quat
    return state


def test_reset_bins_take_only_rows_that_log_motion_returns():
    gamma = 0.9
    sampler = BinResetSampler(torch.tensor([100, 100]), 3, 10.0, gamma, 20, CPU)
    # Env 0 and env 2 track motion; env 1 is a velocity env and never logs.
    rows = torch.tensor([True, False, True])
    mid = torch.tensor([0, 0, 1])
    sampler.log_step(torch.tensor([1.0, 5.0, 2.0]), mid, torch.tensor([0, 0, 60]), rows)
    sampler.finish(torch.tensor([2]), torch.tensor([EPISODE_FAILURE]), None)
    for step in (1, 2):
        times = torch.tensor([step, step, 9 + step])
        sampler.log_step(torch.tensor([1.0, 5.0, 3.0]), mid, times, rows)
    assert sampler.ep_len.tolist() == [3, 0, 2]
    # Two motion episodes of different lengths finish on the same step.
    sampler.finish(
        torch.tensor([0, 2]), torch.tensor([EPISODE_FAILURE, EPISODE_FAILURE]), None
    )
    clip0_returns = [1 + gamma + gamma**2, 1 + gamma, 1.0]
    clip1_returns = [3 + gamma * 3, 3.0]
    bins = sampler.bin_return.tolist()
    assert bins[0] == pytest.approx(_ema(sum(clip0_returns) / 3))
    assert bins[1] == 10.0
    assert bins[2] == pytest.approx(_ema(sum(clip1_returns) / 2))
    assert bins[3] == pytest.approx(_ema(2.0))
    before = sampler.bin_return.clone()
    sampler.finish(
        torch.tensor([1]), torch.tensor([EPISODE_TIMEOUT]), torch.tensor([99.0])
    )
    assert torch.equal(sampler.bin_return, before)
    assert int(sampler.ep_len.sum()) == 0


def test_uniform_reset_draws_leave_bin_statistics_alone():
    torch.manual_seed(0)
    sampler = BinResetSampler(torch.tensor([120, 30, 75]), 4, 10.0, 0.99, 20, CPU)
    sampler.bin_return.copy_(torch.linspace(0.0, 9.0, sampler.num_bins))
    returns = sampler.bin_return.clone()
    memory = sampler.bin_memory.clone()
    motion_id, motion_time = sampler.sample_uniform(30000)
    assert torch.equal(sampler.bin_return, returns)
    assert torch.equal(sampler.bin_memory, memory)
    bins = sampler.bin_of(motion_id, motion_time)
    assert torch.all(motion_time >= sampler.bin_start[bins])
    assert torch.all(motion_time < sampler.bin_end[bins])
    counts = torch.bincount(bins, minlength=sampler.num_bins).float()
    assert torch.all((counts - counts.mean()).abs() < 0.1 * counts.mean())
    # The mixture draw, by contrast, advances the memory distribution.
    sampler.sample(4)
    assert not torch.equal(sampler.bin_memory, memory)


def test_replay_window_and_prev_action():
    buf = SuperMimicReplay(1, 16, torch.device("cpu"))
    env_ids = torch.zeros(1, dtype=torch.long)
    policy_actions = []
    for t in range(10):
        state = default_state(1, "cpu", torch.float32)
        state[0, 7] = float(t)
        action_p = torch.full((1, ACTION_DIM), float(t + 1))
        action_e = action_p + 0.5
        policy_actions.append(action_p[0].clone())
        end = torch.tensor([EPISODE_CONT if t < 9 else EPISODE_FAILURE])
        buf.add(
            env_ids,
            state,
            state + 0.01,
            action_p,
            action_e,
            torch.zeros(1),
            end,
            torch.zeros(1, 33),
            torch.zeros(1, dtype=torch.long),
            torch.tensor([t]),
            torch.tensor([t]),
            prev_action=torch.zeros_like(action_p)
            if t == 0
            else policy_actions[t - 1].unsqueeze(0),
        )
    starts = buf.valid_window_starts(8)
    assert starts
    env, physical = starts[0]
    prev = buf.prev_action_at(env, physical)
    # first slot of the episode has previous action 0
    assert torch.equal(prev, torch.zeros(ACTION_DIM))
    # window starting at t=2 should recover action_policy[t=1]
    later = [s for s in starts if int(buf.time_in_episode[s[0], s[1]]) == 2]
    assert later
    prev2 = buf.prev_action_at(*later[0])
    assert torch.allclose(prev2, policy_actions[1])
    batch = buf.sample_windows(8, 4)
    assert batch is not None
    for row, start_time in enumerate(batch["motion_time"].tolist()):
        expected = torch.stack(policy_actions[start_time : start_time + 8]) + 0.5
        torch.testing.assert_close(batch["action_executed"][row], expected)
        previous = (
            torch.zeros(ACTION_DIM)
            if start_time == 0
            else policy_actions[start_time - 1]
        )
        torch.testing.assert_close(batch["prev_action"][row], previous)
    # no window may start so that it crosses the failure at t=9
    for _env, phys in starts:
        kinds = [int(buf.end_kind[0, (phys + j) % 16]) for j in range(7)]
        assert EPISODE_FAILURE not in kinds


def test_actor_objective_three_gradient_paths():
    torch.manual_seed(0)
    batch, horizon, dim = 2, 4, 8

    class ToyWM(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(1))

        def forward(self, state, action):
            return state + 0.1 * action.mean(-1, keepdim=True) * self.scale

    class ToyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(dim, dim)

        def forward(self, state):
            return self.net(state)

    class ToyCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(dim, 1)

        def forward(self, state):
            return self.net(state).squeeze(-1)

    class ToyTerm(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(dim, 1)

        def forward(self, state):
            return self.net(state).squeeze(-1)

    policy = ToyPolicy()
    wm = ToyWM()
    critic = ToyCritic()
    term = ToyTerm()
    state0 = torch.randn(batch, dim)

    def rollout(use_reward, use_critic, use_term, const_reward=False):
        state = state0
        rewards, d_hats = [], []
        for _ in range(horizon):
            action = policy(state)
            state = wm(state, action)
            if const_reward:
                rewards.append(torch.ones(batch))
            elif use_reward:
                rewards.append(state.pow(2).mean(-1))
            else:
                rewards.append(state.new_zeros(batch))
            logit = term(state)
            d_hats.append(torch.sigmoid(logit) if use_term else state.new_zeros(batch))
        rewards_t = torch.stack(rewards, dim=1)
        d_t = torch.stack(d_hats, dim=1)
        v_last = critic(state) if use_critic else state.new_zeros(batch)
        return actor_objective(rewards_t, d_t, v_last, gamma=0.99)

    for module in (wm, critic, term):
        for p in module.parameters():
            p.requires_grad_(False)

    # Reward through frozen world model.
    policy.zero_grad(set_to_none=True)
    rollout(True, False, False).backward()
    assert policy.net.weight.grad is not None
    assert torch.isfinite(policy.net.weight.grad).all()
    assert policy.net.weight.grad.abs().sum() > 0
    assert wm.scale.grad is None

    # Critic bootstrap: grad reaches actor, not critic weights.
    policy.zero_grad(set_to_none=True)
    rollout(False, True, False).backward()
    assert policy.net.weight.grad is not None
    assert policy.net.weight.grad.abs().sum() > 0
    assert critic.net.weight.grad is None

    # Survival only. A constant reward has no path through the state, so a
    # detached termination probability would leave the policy gradient at zero.
    policy.zero_grad(set_to_none=True)
    objective = rollout(False, False, True, const_reward=True)
    objective.backward()
    assert policy.net.weight.grad is not None
    assert policy.net.weight.grad.abs().sum() > 0
    assert term.net.weight.grad is None


def test_gae_matches_one_step_identity():
    rewards = torch.ones(1, 2)
    d_hat = torch.zeros(1, 2)
    values = torch.tensor([[0.0, 0.0, 1.0]])
    out = gae_targets(rewards, d_hat, values, gamma=1.0, lam=1.0)
    assert torch.allclose(out[0], torch.tensor([3.0, 2.0]))


def _sorted_slots(env, phys, capacity):
    return torch.sort(env * capacity + phys).values


def test_replay_index_stays_equal_to_a_full_scan():
    """Writes revise the packed lists. The uniform set of slots stays the same."""
    num_envs, capacity, k = 2, 12, 4
    buf = SuperMimicReplay(num_envs, capacity, torch.device("cpu"))
    env_ids = torch.arange(num_envs)
    state = torch.zeros(num_envs, STATE_DIM)
    action = torch.zeros(num_envs, ACTION_DIM)
    ep_time = torch.zeros(num_envs, dtype=torch.long)

    def matches():
        got_env, got_phys = buf._cached_windows(k)
        exp_env, exp_phys = buf._window_env_phys(k)
        assert torch.equal(
            _sorted_slots(got_env, got_phys, capacity),
            _sorted_slots(exp_env, exp_phys, capacity),
        )
        got_env, got_phys = buf._cached_starts()
        exp_env, exp_phys = buf._consistent_starts()
        assert torch.equal(
            _sorted_slots(got_env, got_phys, capacity),
            _sorted_slots(exp_env, exp_phys, capacity),
        )

    for step in range(20):
        end = torch.full((num_envs,), EPISODE_CONT)
        if step == 6:
            end[0] = EPISODE_FAILURE
        elif step == 14:
            end[1] = EPISODE_FAILURE
        targets = env_ids if step != 11 else env_ids[:1]
        buf.add(
            targets,
            state[targets],
            state[targets],
            action[targets],
            action[targets],
            torch.zeros(targets.shape[0]),
            end[targets],
            torch.zeros(targets.shape[0], XI_DIM),
            torch.zeros(targets.shape[0], dtype=torch.long),
            ep_time[targets],
            ep_time[targets],
        )
        ep_time[targets] = ep_time[targets] + 1
        if step == 6:
            ep_time[0] = 0
        elif step == 14:
            ep_time[1] = 0
        if step == 8:
            assert buf.num_valid_windows(k) > 0
            assert buf.sample_starts(4) is not None
        if step >= 8:
            matches()

    def _forbid(*_args, **_kwargs):
        raise AssertionError("a later write rescanned the ring")

    buf._window_env_phys = _forbid
    buf._consistent_starts = _forbid
    buf.add(
        env_ids,
        state,
        state,
        action,
        action,
        torch.zeros(num_envs),
        torch.full((num_envs,), EPISODE_CONT),
        torch.zeros(num_envs, XI_DIM),
        torch.zeros(num_envs, dtype=torch.long),
        ep_time,
        ep_time,
    )
    assert buf.sample_windows(k, 2) is not None
    assert buf.sample_starts(2) is not None


def test_rotation_loss_respects_quaternion_sign_and_pi_branch():
    quat = rotvec_to_quat(torch.tensor([[0.4, -0.2, 0.3]]))
    same = state_prediction_loss(_state_with_quat(quat), _state_with_quat(-quat))
    assert float(same) < 1e-8
    left = rotvec_to_quat(torch.tensor([[math.pi - 0.05, 0.0, 0.0]]))
    right = rotvec_to_quat(torch.tensor([[-(math.pi - 0.05), 0.0, 0.0]]))
    relative = so3_relative_rotvec(left, right).norm()
    assert float(relative) == pytest.approx(0.1, abs=1e-4)
    branched = state_prediction_loss(_state_with_quat(left), _state_with_quat(right))
    assert float(branched) < 2.0
    small_rot = torch.tensor([[1e-4, 0.0, 0.0]], requires_grad=True)
    small = rotvec_to_quat(small_rot)
    base = default_state(1, "cpu", torch.float32)
    pred = torch.cat((base[:, :3], small, base[:, 7:]), dim=-1)
    loss = state_prediction_loss(
        pred, _state_with_quat(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    )
    loss.backward()
    assert torch.isfinite(small_rot.grad).all()
    near_rot = torch.tensor([[math.pi - 1e-4, 0.0, 0.0]], requires_grad=True)
    near_pi = rotvec_to_quat(near_rot)
    pred_pi = torch.cat((base[:, :3], near_pi, base[:, 7:]), dim=-1)
    loss_pi = state_prediction_loss(
        pred_pi, _state_with_quat(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    )
    loss_pi.backward()
    assert torch.isfinite(loss_pi)
    assert torch.isfinite(near_rot.grad).all()
    assert float(near_rot.grad.abs().max()) < 1e6


def test_history_prev_action_survives_wraps_and_episode_mix():
    buf = SuperMimicReplay(2, 16, torch.device("cpu"))
    for env in range(2):
        for t in range(40):
            state = default_state(1, "cpu", torch.float32)
            time_in = t % 11
            prev = (
                torch.zeros(1, ACTION_DIM)
                if time_in == 0
                else torch.full((1, ACTION_DIM), float(time_in))
            )
            end = EPISODE_FAILURE if t % 11 == 10 else EPISODE_CONT
            obs = analytic_obs(state, prev, torch.zeros(1, 33))
            buf.add(
                torch.tensor([env]),
                state,
                state,
                torch.full((1, ACTION_DIM), float(t + 1)),
                torch.full((1, ACTION_DIM), float(t + 1)),
                torch.zeros(1),
                torch.tensor([end]),
                torch.zeros(1, 33),
                torch.zeros(1, dtype=torch.long),
                torch.tensor([t]),
                torch.tensor([t % 11]),
                prev_action=prev,
                obs_noisy=obs,
            )
    env_ids, phys = buf._consistent_starts()
    assert env_ids.numel() > 0
    picked = torch.arange(0, env_ids.numel(), max(env_ids.numel() // 8, 1))
    for index in picked.tolist():
        env = int(env_ids[index])
        slot = int(phys[index])
        t = int(buf.time_in_episode[env, slot])
        oldest = max(0, t - 9)
        history = buf.history_at(env, slot).view(10, 93)
        expected = 0.0 if oldest == 0 else float(oldest)
        assert torch.allclose(
            history[0, -ACTION_DIM:], torch.full((ACTION_DIM,), expected)
        )


def test_reset_finish_ignores_leftover_rewards():
    def make():
        return BinResetSampler(
            torch.tensor([200]),
            num_envs=2,
            g_max=10.0,
            gamma=0.99,
            max_episode_steps=8,
            device=torch.device("cpu"),
        )

    def arm(sampler, tail: float):
        sampler._reward[:] = tail
        sampler._reward[0, :3] = torch.tensor([1.0, 2.0, 3.0])
        sampler._reward[1, :1] = 4.0
        sampler._bin[:] = 0
        sampler.ep_len[0] = 3
        sampler.ep_len[1] = 1

    clean = make()
    dirty = make()
    arm(clean, 0.0)
    arm(dirty, 50.0)
    kind = torch.tensor([EPISODE_FAILURE, EPISODE_FAILURE])
    clean.finish(torch.tensor([0, 1]), kind, None)
    dirty.finish(torch.tensor([0, 1]), kind, None)
    assert torch.allclose(clean.bin_return, dirty.bin_return)
    assert torch.all(dirty._reward == 0)
    assert torch.all(dirty.ep_len == 0)

    solo = []
    for env, length in ((0, 3), (1, 1)):
        one = make()
        arm(one, 50.0)
        before = one.bin_return.clone()
        one.finish(torch.tensor([env]), kind[env : env + 1], None)
        solo.append((float(one.bin_return[0]) - 0.95 * float(before[0])) / 0.05)
    batch = make()
    arm(batch, 50.0)
    before = batch.bin_return.clone()
    batch.finish(torch.tensor([0, 1]), kind, None)
    batch_mean = (float(batch.bin_return[0]) - 0.95 * float(before[0])) / 0.05
    assert batch_mean == pytest.approx((solo[0] * 3 + solo[1]) / 4, abs=1e-4)

    reused = make()
    reused._reward[0, :5] = 3.0
    reused._bin[0, :5] = 0
    reused.ep_len[0] = 5
    reused.finish(torch.tensor([0]), torch.tensor([EPISODE_FAILURE]), None)
    reused._reward[0, 0] = 0.0
    reused._bin[0, 0] = 0
    reused.ep_len[0] = 1
    reused.finish(torch.tensor([0]), torch.tensor([EPISODE_FAILURE]), None)
    fresh = make()
    fresh._reward[0, 0] = 0.0
    fresh._bin[0, 0] = 0
    fresh.ep_len[0] = 1
    # Match the EMA state after the long episode, then the short return is 0.
    fresh.bin_return.copy_(reused.bin_return)
    # The short episode's return is 0, so another EMA step moves toward 0.
    # Recompute from the post-long value instead of comparing to a fresh 10.
    assert float(reused.bin_return[0]) < 10.0
    probs = reused.mixture(update_memory=False)
    assert torch.isfinite(probs).all()
    assert float(probs.sum()) == pytest.approx(1.0)
    assert torch.all(probs > 0)


def test_bf16_matmul_returns_float32_and_float32_gradients():
    """The linear stack may run in bfloat16. The tensor handed back does not."""
    torch.manual_seed(0)
    encoder = MotionEncoder(8)
    goal = torch.randn(4, 8)
    fp32 = encoder(goal)
    encoder.bf16_matmul = True
    mixed = encoder(goal)
    assert fp32.dtype == torch.float32
    assert mixed.dtype == torch.float32
    assert mixed.shape == fp32.shape
    assert torch.isfinite(mixed).all()
    assert float((mixed - fp32).detach().abs().max()) > 0.0
    mixed.sum().backward()
    weight = encoder.net[0].weight
    assert weight.dtype == torch.float32
    assert weight.grad is not None
    assert weight.grad.dtype == torch.float32
    assert torch.isfinite(weight.grad).all()


@pytest.mark.parametrize("loss_mode", ["paper", "scaled", "running"])
def test_dynamics_window_matches_the_open_loop(loss_mode):
    """The fused k-step loss is the same sum the trainer used to accumulate."""
    from rlinf.algorithms.supermimic.scan import make_dynamics_window

    torch.manual_seed(0)
    from rlinf.models.embodiment.supermimic import running_state_loss

    dynamics = DynamicsNet(output_norm=loss_mode == "running")
    twin = DynamicsNet(output_norm=loss_mode == "running")
    twin.load_state_dict(dynamics.state_dict())
    batch, k = 2, 3
    state0 = torch.randn(batch, 71)
    state0[:, 3:7] = torch.nn.functional.normalize(state0[:, 3:7], dim=-1)
    actions = torch.randn(batch, k, 29)
    xi = torch.randn(batch, 33)
    targets = torch.randn(batch, k, 71)
    targets[:, :, 3:7] = torch.nn.functional.normalize(targets[:, :, 3:7], dim=-1)
    left = state0.detach().clone().requires_grad_(True)
    right = state0.detach().clone().requires_grad_(True)

    pred = left
    manual = left.new_zeros(())
    first = last = None
    for t in range(k):
        pred = integrate_residual(pred, dynamics(pred, actions[:, t], xi))
        if loss_mode == "running":
            step = running_state_loss(pred, targets[:, t], dynamics, "heading")
        else:
            step = state_prediction_loss(
                pred, targets[:, t], scaled=loss_mode == "scaled"
            )
        if t == 0:
            first = step
        last = step
        manual = manual + step
    window = make_dynamics_window(
        k, learned=False, termination_body="pelvis", loss_mode=loss_mode
    )
    loss, one, end = window(
        twin, None, right, actions, xi, targets, targets, torch.zeros(batch, k), 1.0
    )
    assert torch.allclose(loss, manual, atol=1e-6)
    assert torch.allclose(one, first, atol=1e-6)
    assert torch.allclose(end, last, atol=1e-6)
    manual.backward()
    loss.backward()
    assert torch.allclose(left.grad, right.grad, atol=1e-5)
    for src, dst in zip(dynamics.parameters(), twin.parameters(), strict=True):
        assert torch.allclose(src.grad, dst.grad, atol=1e-5)


def test_residual_frames_round_trip_and_a_roll_splits_the_linear_velocity():
    from rlinf.envs.sim.g1_mimic.constants import RES_LIN
    from rlinf.envs.sim.g1_mimic.features import (
        frame_local_error,
        quat_mul,
        residual_target,
        rotvec_to_quat,
    )

    torch.manual_seed(0)
    state = torch.zeros(4, 71)
    state[:, 3] = 1.0
    state[:, 3:7] = rotvec_to_quat(torch.tensor([0.4, 0.0, 0.2]).expand(4, 3))
    nxt = state.clone()
    nxt[:, 0:3] += torch.tensor([0.1, -0.2, 0.05])
    nxt[:, 7:36] += 0.05
    nxt[:, 36:39] += torch.tensor([0.3, -0.1, 0.2])
    nxt[:, 39:42] += 0.1
    nxt[:, 42:71] += 0.02
    nxt[:, 3:7] = quat_mul(
        state[:, 3:7], rotvec_to_quat(torch.tensor([0.0, 0.05, 0.2]).expand(4, 3))
    )
    zero = torch.zeros(4, 70)
    for frame in ("heading", "body"):
        held = integrate_residual(state, zero, frame)
        assert torch.allclose(held, state, atol=1e-5)
        back = integrate_residual(state, residual_target(state, nxt, frame), frame)
        assert torch.allclose(back, nxt, atol=1e-4)
        assert torch.allclose(frame_local_error(nxt, nxt, frame), torch.zeros(4, 70))
    residual = torch.zeros(4, 70)
    residual[:, RES_LIN] = torch.tensor([1.0, 0.0, 0.0])
    heading = integrate_residual(state, residual, "heading")
    body = integrate_residual(state, residual, "body")
    assert not torch.allclose(heading[:, 36:39], body[:, 36:39], atol=1e-3)


def test_running_norm_matches_welford_and_then_freezes():
    from rlinf.models.embodiment.supermimic import RunningNorm

    norm = RunningNorm(2, max_count=2, eps=1e-2)
    norm.update(torch.tensor([[0.0, 2.0], [2.0, 4.0]]))
    assert torch.allclose(norm.mean, torch.tensor([1.0, 3.0]))
    assert torch.allclose(norm.var, torch.ones(2))
    scale = torch.sqrt(torch.tensor(1.0 + 1e-4))
    assert torch.allclose(norm(torch.tensor([1.0, 3.0])), torch.zeros(2), atol=1e-5)
    assert torch.allclose(norm.std(), scale.expand(2))
    frozen = norm.mean.clone()
    norm.update(torch.zeros(4, 2))
    assert torch.equal(norm.mean, frozen)
    assert float(norm.count) == 2
    norm.update(torch.zeros(0, 2))
    assert float(norm.count) == 2
    assert norm._count_cpu == float(norm.count)
    restored = RunningNorm(2, max_count=2, eps=1e-2)
    restored.load_state_dict(norm.state_dict())
    assert restored._count_cpu == float(restored.count)
    assert torch.equal(restored.mean, frozen)
    one = RunningNorm(2)
    one.update(torch.tensor([[4.0, -1.0]]))
    assert torch.equal(one.mean, torch.tensor([4.0, -1.0]))
    assert torch.equal(one.var, torch.zeros(2))


def test_running_state_loss_averages_the_scaled_components():
    from rlinf.envs.sim.g1_mimic.features import error_floor, frame_local_error
    from rlinf.models.embodiment.supermimic import DynamicsNet, running_state_loss

    net = DynamicsNet(output_norm=True)
    with torch.no_grad():
        net.out_norm.var.fill_(0.25)
    pred = torch.zeros(2, 71)
    true = torch.zeros(2, 71)
    pred[:, 3] = 1.0
    true[:, 3] = 1.0
    pred[:, 0] = 0.1
    err = frame_local_error(pred, true, "heading")
    scale = torch.maximum(net.out_norm.std(), error_floor(err))
    expect = (err / scale).square().mean()
    assert torch.allclose(running_state_loss(pred, true, net, "heading"), expect)
    assert float(expect) < float((err / scale).square().sum(-1).mean())


@needs_mjlab
def test_imagination_keeps_the_terminal_value_gradient():
    """Earlier critic features leave the graph. Rewards and V(s_H) stay on it."""
    from rlinf.algorithms.supermimic.scan import make_imagination
    from rlinf.envs.sim.g1_mimic.constants import HISTORY_LEN, OBS_DIM
    from rlinf.envs.sim.g1_mimic.reward import differentiable_fk

    torch.manual_seed(0)
    batch, horizon = 2, 2
    models = SuperMimicModels(680, input_norm=True)
    seen: dict[str, torch.Tensor] = {}

    def _capture(_module, inputs, output):
        seen["feat"] = inputs[0]
        seen["value"] = output

    hook = models.critic.register_forward_hook(_capture)
    state = default_state(batch, "cpu", torch.float32)
    prev = torch.zeros(batch, ACTION_DIM)
    xi = torch.zeros(batch, XI_DIM)
    history = torch.zeros(batch, HISTORY_LEN * OBS_DIM)
    refs = state.unsqueeze(1).expand(batch, horizon + 1, -1).contiguous()
    goals = torch.randn(batch, horizon + 1, 680)
    kin = differentiable_fk(state)
    ref_kin = tuple(
        part.unsqueeze(1).expand(batch, horizon + 1, *part.shape[1:]).contiguous()
        for part in kin
    )
    run = make_imagination(
        horizon, False, "pelvis", "root4", False, OBS_DIM, 0.99, scale_xi=True
    )
    _states, _goals, feats, rewards, d_hat, objective = run(
        models.encoder,
        models.decoder,
        models.dynamics,
        models.termination,
        models.critic,
        state,
        prev,
        xi,
        history,
        refs,
        goals,
        *ref_kin,
        torch.zeros(ACTION_DIM),
        torch.full((ACTION_DIM,), -2.5),
        torch.full((ACTION_DIM,), 2.5),
    )
    hook.remove()
    assert feats.shape == (batch, horizon + 1, models.critic.net[0].in_features)
    assert not feats.requires_grad
    assert rewards.requires_grad
    assert seen["feat"].requires_grad
    with torch.no_grad():
        rebuilt = actor_objective(rewards, d_hat, models.critic(feats[:, -1]), 0.99)
    assert torch.allclose(rebuilt, objective)
    # The dynamics' last layer is zero at init, so the action gets no state
    # gradient yet. The value must still reach the dynamics weights, which
    # only happens through the predicted state. Action rate still reaches
    # the decoder from the reward.
    value_only = actor_objective(rewards.detach(), d_hat.detach(), seen["value"], 0.99)
    dyn_grads = torch.autograd.grad(
        value_only,
        list(models.dynamics.parameters()),
        allow_unused=True,
        retain_graph=True,
    )
    reward_grads = torch.autograd.grad(
        rewards.sum(), list(models.decoder.parameters()), allow_unused=True
    )
    assert any(grad is not None and float(grad.abs().sum()) > 0 for grad in dyn_grads)
    assert any(
        grad is not None and float(grad.abs().sum()) > 0 for grad in reward_grads
    )


@pytest.mark.parametrize("gamma", [0.01, 0.8, 0.99])
@pytest.mark.parametrize("horizon", [1, 7, 500])
def test_reset_returns_match_backward_recurrence(gamma, horizon):
    lengths = [horizon, max(1, horizon // 2), max(1, horizon - 1)]
    sampler = BinResetSampler(torch.tensor([horizon]), 3, 10.0, gamma, horizon, CPU)
    rewards = torch.rand(3, horizon, generator=torch.Generator().manual_seed(17))
    sampler._reward.copy_(rewards)
    sampler._bin[:] = torch.arange(horizon) // sampler.bin_frames
    sampler.ep_len[:] = torch.tensor(lengths)
    kinds = torch.tensor([EPISODE_TIMEOUT, EPISODE_FAILURE, EPISODE_CLIP_END])
    bootstrap = torch.tensor([3.0, 100.0, 100.0])
    sums = torch.zeros(sampler.num_bins, dtype=torch.float64)
    counts = torch.zeros_like(sums)
    for row, length in enumerate(lengths):
        value = float(bootstrap[row]) if row == 0 else 0.0
        for step in range(length - 1, -1, -1):
            value = float(rewards[row, step]) + gamma * value
            bin_id = step // sampler.bin_frames
            sums[bin_id] += value
            counts[bin_id] += 1
    expected = torch.where(
        counts > 0,
        (1 - RETURN_EMA) * 10.0 + RETURN_EMA * sums / counts.clamp_min(1),
        10.0,
    )
    sampler.finish(torch.arange(3), kinds, bootstrap)
    assert torch.isfinite(sampler.bin_return).all()
    torch.testing.assert_close(
        sampler.bin_return.double(), expected, atol=2e-5, rtol=2e-6
    )
    assert torch.isfinite(sampler.mixture(update_memory=False)).all()
    assert sampler.sample(10)[0].numel() == 10


@pytest.mark.parametrize("end_kind", [EPISODE_FAILURE, EPISODE_CLIP_END])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("deferred", [False, True])
def test_reset_ignores_nonfinite_bootstrap_off_timeouts(end_kind, value, deferred):
    from rlinf.workers.supermimic.distributed import DataParallel

    sampler = BinResetSampler(torch.tensor([50]), 1, 10.0, 0.99, 2, CPU)
    sampler.deferred = deferred
    sampler.log_step(torch.tensor([1.0]), torch.tensor([0]), torch.tensor([0]))
    sampler.finish(torch.tensor([0]), torch.tensor([end_kind]), torch.tensor([value]))
    if deferred:
        sampler.sync(DataParallel(device=CPU))
    torch.testing.assert_close(sampler.bin_return, torch.tensor([_ema(1.0)]))
    assert sampler.sample(2)[0].numel() == 2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_reset_rejects_nonfinite_timeout_bootstrap_before_updating(value):
    sampler = BinResetSampler(torch.tensor([50]), 1, 10.0, 0.99, 2, CPU)
    sampler.log_step(torch.tensor([1.0]), torch.tensor([0]), torch.tensor([0]))
    before = sampler.bin_return.clone()
    with pytest.raises(RuntimeError, match="non-finite timeout bootstrap"):
        sampler.finish(
            torch.tensor([0]), torch.tensor([EPISODE_TIMEOUT]), torch.tensor([value])
        )
    assert torch.equal(sampler.bin_return, before)
    assert sampler.ep_len.tolist() == [1]
    assert float(sampler._reward[0, 0]) == 1.0


def test_reset_ignores_nonfinite_bootstrap_for_empty_episode():
    sampler = BinResetSampler(torch.tensor([50]), 1, 10.0, 0.99, 2, CPU)
    sampler.finish(
        torch.tensor([0]),
        torch.tensor([EPISODE_TIMEOUT]),
        torch.tensor([float("nan")]),
    )
    assert torch.equal(sampler.bin_return, torch.tensor([10.0]))
    assert sampler.sample(2)[0].numel() == 2


class _CudaBatchTensor(torch.Tensor):
    """CPU storage with CUDA eligibility for compile-cache tests."""

    @property
    def is_cuda(self):
        return True


class _ScanOwner:
    def __init__(self, value):
        from rlinf.algorithms.supermimic.scan import CompileCache

        self.value = value
        self.cache = CompileCache()

    def goal(self, batch):
        return batch + self.value


def test_compile_cache_keeps_owners_separate_and_reuses_each(monkeypatch):
    from rlinf.algorithms.supermimic.scan import run_compiled

    monkeypatch.delenv("SUPERMIMIC_SCAN_EAGER", raising=False)
    compiled = []

    def compiler(fn, **kwargs):
        compiled.append(fn)
        return fn

    monkeypatch.setattr(torch, "compile", compiler)
    batch = torch.zeros(256).as_subclass(_CudaBatchTensor)
    first, second = _ScanOwner(1.0), _ScanOwner(2.0)
    for owner in (first, second):
        with pytest.warns(RuntimeWarning, match="Compiling"):
            result = run_compiled("collect-goal", owner.goal, batch, cache=owner.cache)
        torch.testing.assert_close(result, batch + owner.value)
        result = run_compiled("collect-goal", owner.goal, batch, cache=owner.cache)
        torch.testing.assert_close(result, batch + owner.value)
    assert len(compiled) == 2


def test_compile_fallback_is_local_to_its_owner(monkeypatch):
    from rlinf.algorithms.supermimic.scan import run_compiled

    class CompilerError(RuntimeError):
        pass

    monkeypatch.delenv("SUPERMIMIC_SCAN_EAGER", raising=False)
    compiled = []
    first, second = _ScanOwner(1.0), _ScanOwner(2.0)

    def compiler(fn, **kwargs):
        compiled.append(fn)

        def call(*args):
            if fn.__self__ is first:
                raise CompilerError("test compiler failure")
            return fn(*args)

        return call

    monkeypatch.setattr(torch, "compile", compiler)
    batch = torch.zeros(256).as_subclass(_CudaBatchTensor)
    with pytest.warns(RuntimeWarning) as warnings_seen:
        result = run_compiled("collect-goal", first.goal, batch, cache=first.cache)
    assert len(warnings_seen) == 2
    torch.testing.assert_close(result, batch + 1.0)
    run_compiled("collect-goal", first.goal, batch, cache=first.cache)
    assert len(compiled) == 1
    with pytest.warns(RuntimeWarning, match="Compiling"):
        result = run_compiled("collect-goal", second.goal, batch, cache=second.cache)
    torch.testing.assert_close(result, batch + 2.0)
    assert len(compiled) == 2


def test_named_dynamics_scans_keep_separate_closure_settings(monkeypatch):
    from rlinf.algorithms.supermimic.scan import make_dynamics_window

    monkeypatch.delenv("SUPERMIMIC_SCAN_EAGER", raising=False)
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    state = default_state(256, CPU, torch.float32).as_subclass(_CudaBatchTensor)
    actions = torch.zeros(256, 1, ACTION_DIM)
    xi = torch.zeros(256, XI_DIM)
    targets = state.unsqueeze(1).clone()
    targets[:, :, 0] += 0.2

    def dynamics(state, action, xi):
        return state.new_zeros(state.shape[0], 70)

    for scaled in (False, True):
        window = make_dynamics_window(
            1,
            False,
            "pelvis",
            loss_mode="scaled" if scaled else "paper",
            compile_name="same-name",
        )
        with pytest.warns(RuntimeWarning, match="Compiling"):
            loss, _, _ = window(
                dynamics,
                None,
                state,
                actions,
                xi,
                targets,
                targets,
                torch.zeros(256, 1),
                1.0,
            )
        expected = state_prediction_loss(state, targets[:, 0], scaled=scaled)
        torch.testing.assert_close(loss, expected)


needs_warp = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("mjlab") is None,
    reason="MuJoCo Warp tests need CUDA and mjlab",
)


def _env_write_clip(directory: Path) -> None:
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


def test_solver_meaninertia_has_independent_storage_per_world(monkeypatch):
    def array(data, dtype, device):
        values = np.array(data, dtype=dtype)
        return SimpleNamespace(
            shape=values.shape,
            dtype=values.dtype,
            device=device,
            numpy=lambda: values.copy(),
            values=values,
        )

    monkeypatch.setitem(sys.modules, "warp", SimpleNamespace(array=array))
    shared = array([2.5], np.float32, "cpu")
    model = SimpleNamespace(stat=SimpleNamespace(meaninertia=shared))
    _expand_meaninertia(model, 3)
    expanded = model.stat.meaninertia
    assert expanded.shape == (3,)
    assert expanded.dtype == shared.dtype
    assert expanded.device == shared.device
    np.testing.assert_array_equal(expanded.numpy(), [2.5, 2.5, 2.5])
    expanded.values[0] = 7.0
    np.testing.assert_array_equal(expanded.numpy(), [7.0, 2.5, 2.5])
    np.testing.assert_array_equal(shared.numpy(), [2.5])
    _expand_meaninertia(model, 3)
    assert model.stat.meaninertia is expanded


@pytest.mark.parametrize(
    "model",
    [
        SimpleNamespace(),
        SimpleNamespace(stat=SimpleNamespace()),
        SimpleNamespace(stat=SimpleNamespace(meaninertia=2.5)),
        SimpleNamespace(stat=SimpleNamespace(meaninertia=np.array([2.5, 3.0]))),
    ],
)
def test_solver_meaninertia_keeps_legacy_and_per_world_storage(model, monkeypatch):
    monkeypatch.setitem(sys.modules, "warp", None)
    before = getattr(getattr(model, "stat", None), "meaninertia", None)
    _expand_meaninertia(model, 2)
    assert getattr(getattr(model, "stat", None), "meaninertia", None) is before


def test_single_world_solver_meaninertia_keeps_its_storage(monkeypatch):
    monkeypatch.setitem(sys.modules, "warp", None)
    shared = np.array([2.5])
    model = SimpleNamespace(stat=SimpleNamespace(meaninertia=shared))
    _expand_meaninertia(model, 1)
    assert model.stat.meaninertia is shared


def test_heading_features_ignore_global_yaw():
    state = default_state(1, "cpu", torch.float32)
    yawed = state.clone()
    yawed[0, 3:7] = torch.tensor([0.7071, 0.0, 0.0, 0.7071])
    f0 = heading_invariant_features(state)
    f1 = heading_invariant_features(yawed)
    assert f0.shape[-1] == 71
    # height and joint features stay aligned after removing yaw
    assert torch.allclose(f0[..., 0], f1[..., 0], atol=1e-5)


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
def test_warp_state_is_one_physical_time(tmp_path):
    """Nonzero root and joint velocity, several 5 ms substeps, one timestamp."""
    _env_write_clip(tmp_path)
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
@pytest.mark.parametrize("com_sampling", ["per_env", "per_reset"])
def test_warp_reset_recomputes_constants_only_when_the_com_moves(
    tmp_path, com_sampling
):
    frames = 12
    state = asset_default_state().unsqueeze(0).repeat(frames, 1)
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
    env = MjlabG1Env(3, motion, torch.device("cuda"), com_sampling=com_sampling)
    calls = []
    inner = env.sim.recompute_constants
    env.sim.recompute_constants = lambda level: (calls.append(level), inner(level))
    com = env.xi[:, 30:33].clone()
    friction = env.xi[:, 0].clone()
    ids = torch.tensor([0, 2], device="cuda")
    env.reset(ids)
    if com_sampling == "per_env":
        assert torch.equal(env.xi[:, 30:33], com)
        assert calls == []
    else:
        assert not torch.equal(env.xi[ids, 30:33], com[ids])
        assert len(calls) == 1
    assert not torch.equal(env.xi[ids, 0], friction[ids])
    torso = env.sim.model.body_ipos[:, env._torso_id]
    default = env._default_ipos[:, env._torso_id]
    assert torch.allclose(torso - default, env.xi[:, 30:33])


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


def _motion_write_clip(path: Path, frames: int = 30, name: str | None = None) -> None:
    payload = {
        "fps": 50.0,
        "joint_names": list(G1_JOINT_NAMES),
        "q0": torch.zeros(29),
        "state": default_state(frames, "cpu", torch.float32),
    }
    if name is not None:
        payload["name"] = name
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _reference_payload(state: torch.Tensor, fps: float = 50.0, names=None, q0=None):
    return {
        "fps": fps,
        "joint_names": list(G1_JOINT_NAMES if names is None else names),
        "q0": torch.zeros(ACTION_DIM) if q0 is None else q0,
        "state": state,
    }


def _constant_motion(
    frames: int, fps: float, speed: float, joint_rate: float, yaw_rate: float
):
    state = default_state(frames, "cpu", torch.float32)
    for i in range(frames):
        t = i / fps
        state[i, 0] = speed * t
        state[i, 7] = joint_rate * t
        state[i, ROOT_QUAT] = rotvec_to_quat(torch.tensor([0.0, 0.0, yaw_rate * t]))
    return state


def _clip(directory, index, frames=20):
    state = torch.zeros(frames, STATE_DIM)
    state[:, 2] = 0.75
    state[:, 3] = 1.0
    state[:, 0] = torch.linspace(0, 1, frames)
    path = directory / f"clip_{index:03d}.pt"
    torch.save(
        {
            "fps": 50.0,
            "joint_names": list(G1_JOINT_NAMES),
            "q0": torch.zeros(ACTION_DIM),
            "state": state,
        },
        path,
    )
    return path


def _manifest(directory, files):
    clips = [
        {
            "file": f.name,
            "source": f"src_{i}.csv",
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
        }
        for i, f in enumerate(files)
    ]
    (directory / "manifest.json").write_text(json.dumps({"clips": clips}))


def _csv(path, frames=31):
    t = np.linspace(0, 1, frames)
    rows = np.zeros((frames, 7 + ACTION_DIM))
    rows[:, 0] = t
    rows[:, 2] = 0.78
    # xyzw: 90 degrees about z.
    rows[:, 5] = np.sqrt(0.5)
    rows[:, 6] = np.sqrt(0.5)
    rows[:, 7:] = 0.1 * np.sin(t)[:, None]
    np.savetxt(path, rows, delimiter=",")


def _run_prepare(monkeypatch, argv):
    from toolkits.supermimic import prepare_lafan1

    monkeypatch.setattr(sys, "argv", ["prepare_lafan1.py", *argv])
    return prepare_lafan1.main()


def test_resampled_velocities_are_central_differences_in_the_root_frame():
    omega, tilt, dt = 3.0, 0.4, 0.02
    t = torch.arange(60, dtype=torch.float64) * dt
    state = default_state(60, "cpu", torch.float64)
    state[:, 0] = torch.sin(omega * t)
    state[:, JOINT_POS] = 0.3 * torch.sin(omega * t).unsqueeze(-1)
    yaw = torch.sin(omega * t)
    zero = torch.zeros_like(t)
    q_yaw = torch.stack((torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)), dim=-1)
    q_tilt = torch.tensor(
        [math.cos(tilt / 2), math.sin(tilt / 2), 0.0, 0.0], dtype=torch.float64
    )
    state[:, ROOT_QUAT] = quat_mul(q_yaw, q_tilt.expand(60, 4))
    out = resample_states(state, 50.0, 50.0)
    rate = omega * torch.cos(omega * t)
    inner = slice(1, -1)
    # Central differences are second order; a backward difference lags half
    # a step and is off by about omega^2 dt / 2 = 0.09 here.
    assert float((out[inner, ROOT_LIN_VEL][:, 0] - rate[inner]).abs().max()) < 5e-3
    assert float((out[inner, JOINT_VEL] - 0.3 * rate[inner, None]).abs().max()) < 2e-3
    body_axis = torch.tensor([0.0, math.sin(tilt), math.cos(tilt)], dtype=torch.float64)
    expected = rate[:, None] * body_axis
    assert float((out[inner, ROOT_ANG_VEL] - expected[inner]).abs().max()) < 5e-3
    # The two end frames use one-sided differences.
    assert float(out[0, 36]) == pytest.approx(float((state[1, 0] - state[0, 0]) / dt))
    assert float(out[-1, 36]) == pytest.approx(
        float((state[-1, 0] - state[-2, 0]) / dt)
    )


def test_null_clip_count_loads_every_bones_clip_past_the_filters(tmp_path):
    for i in range(45):
        _motion_write_clip(tmp_path / f"clip_{i:03d}.pt", name=f"walk_{i}")
    _motion_write_clip(tmp_path / "clip_045.pt", name="sit_on_chair")
    library = load_bones_library(str(tmp_path), None, "cpu")
    assert library.num_clips == 45
    assert library.filter_counts == {
        "files_read": 46,
        "files_in_dir": 46,
        "rejected_name": 1,
        "rejected_feasibility": 0,
        "kept": 45,
    }
    capped = load_bones_library(str(tmp_path), 3, "cpu")
    assert capped.num_clips == 3 and capped.clip_names == ["walk_0", "walk_1", "walk_2"]
    with pytest.raises(
        FileNotFoundError, match="kept 45 clips after filtering, need 50"
    ):
        load_bones_library(str(tmp_path), 50, "cpu")


def test_motion_payload_checks_fps_joints_and_default_pose(tmp_path):
    state = default_state(31, "cpu", torch.float32)
    state[:, 0] = torch.linspace(0.0, 1.0, 31)
    with pytest.raises(ValueError, match="bare state"):
        decode_motion_payload(state, "bare.pt")
    swapped = list(G1_JOINT_NAMES)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ValueError, match="joint order"):
        decode_motion_payload(_reference_payload(state, names=swapped), "order.pt")
    with pytest.raises(ValueError, match="fps"):
        decode_motion_payload(_reference_payload(state, fps=60.0), "fps.pt")

    decoded, q0 = decode_motion_payload(_reference_payload(state, fps=30.0), "30.pt")
    assert decoded.shape == (51, STATE_DIM)
    assert q0.shape == (ACTION_DIM,)
    assert float(decoded[-1, 0]) == pytest.approx(1.0, abs=1e-4)
    assert float(decoded[25, ROOT_LIN_VEL.start]) == pytest.approx(1.0, abs=1e-3)

    kept, _ = decode_motion_payload(_reference_payload(state[:8], fps=50.0), "50.pt")
    assert kept.shape[0] == 8

    q0_a = torch.arange(ACTION_DIM, dtype=torch.float32) * 0.01
    q0_b = q0_a + 0.2
    torch.save(_reference_payload(state[:8], q0=q0_a), tmp_path / "clip_000.pt")
    torch.save(_reference_payload(state[:8], q0=q0_b), tmp_path / "clip_001.pt")
    with pytest.raises(ValueError, match="default pose"):
        load_motion_library(str(tmp_path), 2, "cpu")
    (tmp_path / "clip_001.pt").unlink()
    torch.save(_reference_payload(state[:8], q0=q0_a), tmp_path / "clip_001.pt")
    library = load_motion_library(str(tmp_path), 2, "cpu")
    assert torch.allclose(library.q0, q0_a)


def test_goal730_uses_future_frames_and_holds_the_last():
    clip = default_state(4, "cpu", torch.float32)
    for i in range(4):
        clip[i, 0] = float(i)
        clip[i, JOINT_VEL] = 10.0
    goal = MotionLibrary([clip]).goal730(torch.tensor([0]), torch.tensor([0]))
    assert goal.shape == (1, 730)
    frames = goal.view(1, 10, 73)
    assert torch.allclose(
        frames[0, :, 0], torch.tensor([1, 2, 3, 3, 3, 3, 3, 3, 3, 3]).float()
    )
    assert torch.allclose(frames[0, 0, 44:73], torch.ones(29))


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
    assert (
        feasibility_reject(deep, feet, pelvis, upright, speed) == "ground penetration"
    )
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
    assert library.frames.device.type == "cpu"
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


def test_resample_uses_the_control_grid_for_positions_and_velocity():
    # 30 Hz, two frames, 1 m/s. The old linspace kept the endpoint and
    # reported 0.833 m/s. The 50 Hz grid keeps 1 m/s and drops the short tail.
    state = _constant_motion(2, 30.0, 1.0, 0.5, 0.4)
    out = resample_states(state, 30.0, 50.0)
    _src, dst, dt = control_time_grid(2, 30.0, 50.0)
    assert dt == pytest.approx(0.02)
    assert out.shape[0] == 2
    assert float(dst[-1]) == pytest.approx(0.02)
    assert float(out[-1, 0]) == pytest.approx(0.02, abs=1e-5)
    assert float(out[:, ROOT_LIN_VEL][:, 0].mean()) == pytest.approx(1.0, abs=1e-4)
    assert float(out[:, ROOT_LIN_VEL][:, 0].mean()) != pytest.approx(0.8333, abs=1e-3)
    assert torch.allclose(
        out[:, ROOT_QUAT].norm(dim=-1), torch.ones(out.shape[0]), atol=1e-5
    )

    for frames in (2, 4, 31, 45):
        speed, joint_rate, yaw_rate = 1.3, -0.25, 0.4
        src = _constant_motion(frames, 30.0, speed, joint_rate, yaw_rate)
        sampled = resample_states(src, 30.0, 50.0)
        steps = int(math.floor((frames - 1) * 50.0 / 30.0 + 1e-8))
        assert sampled.shape[0] == steps + 1
        assert float(sampled[1, 0] - sampled[0, 0]) == pytest.approx(
            speed * 0.02, abs=1e-4
        )
        assert torch.allclose(
            sampled[:, ROOT_LIN_VEL][:, 0],
            torch.full((sampled.shape[0],), speed),
            atol=1e-4,
        )
        assert torch.allclose(
            sampled[:, JOINT_VEL][:, 0],
            torch.full((sampled.shape[0],), joint_rate),
            atol=1e-4,
        )
        assert torch.allclose(
            sampled[:, ROOT_ANG_VEL],
            torch.tensor([0.0, 0.0, yaw_rate]).expand(sampled.shape[0], 3),
            atol=1e-3,
        )
        dots = (sampled[:-1, ROOT_QUAT] * sampled[1:, ROOT_QUAT]).sum(-1)
        assert torch.all(dots > 0)

    native = _constant_motion(7, 50.0, 1.0, 0.2, 0.0)
    same = resample_states(native, 50.0, 50.0)
    assert same.shape[0] == 7
    assert torch.allclose(same[:, :3], native[:, :3], atol=1e-5)
    assert torch.allclose(same[:, 7:36], native[:, 7:36], atol=1e-5)


def test_manifest_names_clips_and_rejects_changed_files(tmp_path):
    files = [_clip(tmp_path, 0), _clip(tmp_path, 1)]
    _manifest(tmp_path, files)
    library = load_motion_library(str(tmp_path), 2, "cpu")
    assert library.clip_names == ["src_0.csv", "src_1.csv"]

    torch.save({"tampered": True}, tmp_path / "clip_001.pt")
    with pytest.raises(ValueError, match="SHA-256"):
        load_motion_library(str(tmp_path), 2, "cpu")


def test_manifest_rejects_an_unlisted_clip(tmp_path):
    files = [_clip(tmp_path, 0)]
    _manifest(tmp_path, files)
    _clip(tmp_path, 1)
    with pytest.raises(ValueError, match="not listed"):
        load_motion_library(str(tmp_path), 2, "cpu")


def test_prepare_uses_explicit_q0_without_mjlab(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mjlab", None)
    monkeypatch.setitem(
        sys.modules, "mjlab.asset_zoo.robots.unitree_g1.g1_constants", None
    )
    source = tmp_path / "csv"
    source.mkdir()
    _csv(source / "walk.csv")
    q0 = torch.linspace(-0.2, 0.2, ACTION_DIM, dtype=torch.float64)
    q0_file = tmp_path / "g1_q0.pt"
    torch.save(q0, q0_file)
    out = tmp_path / "clips"
    assert (
        _run_prepare(
            monkeypatch,
            [
                "--out",
                str(out),
                "--source-dir",
                str(source),
                "--q0-file",
                str(q0_file),
                "--skip-remote-check",
                "--allow-partial",
                "--accept-license",
            ],
        )
        == 0
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["q0"] == "file: g1_q0.pt"
    assert (
        manifest["q0_file_sha256"] == hashlib.sha256(q0_file.read_bytes()).hexdigest()
    )
    payload = torch.load(out / "clip_000.pt", weights_only=True)
    assert torch.equal(payload["q0"], q0.float())
    library = load_motion_library(str(out), 1, "cpu")
    assert torch.equal(library.q0, q0.float())
    assert library.lengths.tolist() == [51]


@pytest.mark.parametrize(
    "q0",
    [
        torch.zeros(ACTION_DIM - 1),
        torch.zeros(1, ACTION_DIM),
        torch.zeros(ACTION_DIM, dtype=torch.int64),
        torch.full((ACTION_DIM,), float("nan")),
        torch.full((ACTION_DIM,), float("inf")),
        torch.full((ACTION_DIM,), 1e300, dtype=torch.float64),
        {"q0": torch.zeros(ACTION_DIM)},
    ],
)
def test_prepare_rejects_invalid_q0_before_writing_clips(tmp_path, q0):
    from toolkits.supermimic.prepare_lafan1 import convert

    q0_file = tmp_path / "q0.pt"
    torch.save(q0, q0_file)
    out = tmp_path / "clips"
    with pytest.raises(ValueError, match="q0"):
        convert(tmp_path, [], out, q0_file=q0_file)
    assert not out.exists()


@needs_mjlab
def test_prepare_converts_csv_and_writes_a_checked_manifest(tmp_path, monkeypatch):
    source = tmp_path / "csv"
    source.mkdir()
    _csv(source / "b_walk.csv")
    _csv(source / "a_run.csv", frames=61)
    out = tmp_path / "clips"
    assert (
        _run_prepare(
            monkeypatch,
            [
                "--out",
                str(out),
                "--source-dir",
                str(source),
                "--skip-remote-check",
                "--allow-partial",
                "--accept-license",
            ],
        )
        == 0
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["checked_against_revision"] is False
    assert [c["source"] for c in manifest["clips"]] == ["a_run.csv", "b_walk.csv"]
    assert [c["frames_50hz"] for c in manifest["clips"]] == [101, 51]
    assert "CC BY-NC-ND 4.0" in manifest["license_note"]

    payload = torch.load(out / "clip_001.pt", weights_only=True)
    assert payload["fps"] == 30.0
    assert torch.allclose(
        payload["state"][0, 3:7],
        torch.tensor([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]).float(),
    )
    library = load_motion_library(str(out), 2, "cpu")
    assert library.clip_names == ["a_run.csv", "b_walk.csv"]
    assert library.lengths.tolist() == [101, 51]


def test_prepare_requires_the_license_acknowledgement(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        _run_prepare(
            monkeypatch,
            [
                "--out",
                str(tmp_path),
                "--source-dir",
                str(tmp_path),
                "--skip-remote-check",
            ],
        )


def test_prepare_rejects_a_partial_set_by_default(tmp_path, monkeypatch):
    source = tmp_path / "csv"
    source.mkdir()
    _csv(source / "only.csv")
    with pytest.raises(SystemExit, match="expected 40"):
        _run_prepare(
            monkeypatch,
            [
                "--out",
                str(tmp_path / "out"),
                "--source-dir",
                str(source),
                "--skip-remote-check",
                "--accept-license",
            ],
        )


ROOT_HEADER = [
    "Frame",
    "root_translateX",
    "root_translateY",
    "root_translateZ",
    "root_rotateX",
    "root_rotateY",
    "root_rotateZ",
]


Q0 = torch.zeros(29)


def _header(joint_order: list[str] | None = None) -> list[str]:
    names = list(G1_JOINT_NAMES if joint_order is None else joint_order)
    return ROOT_HEADER + [f"{name}_dof" for name in names]


def _write_csv(path: Path, rows: list[list[float]], joint_order=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = _header(joint_order)
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(value) for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row(
    frame: int,
    position_cm: tuple[float, float, float],
    euler_deg: tuple[float, float, float],
    joints_deg: list[float] | None = None,
) -> list[float]:
    joints = [0.0] * len(G1_JOINT_NAMES) if joints_deg is None else joints_deg
    return [frame, *position_cm, *euler_deg, *joints]


def _constant_rows(frames: int, **kwargs) -> list[list[float]]:
    position = kwargs.get("position_cm", (0.0, 0.0, 90.0))
    euler = kwargs.get("euler_deg", (0.0, 0.0, 0.0))
    joints = kwargs.get("joints_deg")
    return [_row(index, position, euler, joints) for index in range(frames)]


def test_centimeters_become_meters_and_joints_follow_names(tmp_path):
    order = list(reversed(G1_JOINT_NAMES))
    degrees = [float(index + 1) for index in range(len(order))]
    # Column order is reversed, so the degrees travel with the joint name.
    rows = _constant_rows(8, position_cm=(100.0, -20.0, 80.0), joints_deg=degrees)
    path = tmp_path / "motion.csv"
    _write_csv(path, rows, joint_order=order)
    position_cm, _euler, joints_deg = parse_bones_csv(path)
    assert torch.allclose(
        position_cm[0],
        torch.tensor([100.0, -20.0, 80.0], dtype=position_cm.dtype),
    )
    by_name = {name: degrees[index] for index, name in enumerate(order)}
    expected = torch.tensor(
        [by_name[name] for name in G1_JOINT_NAMES], dtype=joints_deg.dtype
    )
    assert torch.allclose(joints_deg[0], expected)

    payload = build_payload(path, Path("motion.csv"), Q0, 120.0, 50.0)
    state = payload["state"]
    assert state.shape[1] == STATE_DIM
    assert state.dtype == torch.float32
    assert torch.isfinite(state).all()
    assert torch.allclose(state[0, ROOT_POS], torch.tensor([1.0, -0.2, 0.8]), atol=1e-5)
    assert torch.allclose(
        state[0, JOINT_POS],
        torch.deg2rad(expected).to(dtype=torch.float32),
        atol=1e-5,
    )


def test_bad_csv_rows_name_the_problem(tmp_path):
    good = _constant_rows(8)
    missing = tmp_path / "missing.csv"
    _write_csv(missing, good)
    text = missing.read_text(encoding="utf-8").replace(
        "left_knee_joint_dof", "not_a_joint_dof"
    )
    missing.write_text(text, encoding="utf-8")
    with pytest.raises(CsvError, match="unexpected columns"):
        parse_bones_csv(missing)

    dropped = tmp_path / "dropped.csv"
    header = _header()
    column = header.index("root_translateZ")
    header.pop(column)
    lines = [",".join(header)]
    for row in good:
        cells = [str(value) for value in row]
        cells.pop(column)
        lines.append(",".join(cells))
    dropped.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(CsvError, match="missing columns"):
        parse_bones_csv(dropped)

    illegal_text = tmp_path / "text.csv"
    _write_csv(illegal_text, good)
    illegal_text.write_text(
        illegal_text.read_text(encoding="utf-8").replace("90.0", "bad", 1),
        encoding="utf-8",
    )
    with pytest.raises(CsvError, match="non-numeric"):
        parse_bones_csv(illegal_text)

    illegal = tmp_path / "nan.csv"
    _write_csv(illegal, good)
    illegal.write_text(
        illegal.read_text(encoding="utf-8").replace("90.0", "nan", 1), encoding="utf-8"
    )
    with pytest.raises(CsvError, match="non-finite"):
        parse_bones_csv(illegal)

    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(CsvError, match="empty"):
        parse_bones_csv(empty)

    short = tmp_path / "short.csv"
    _write_csv(short, _constant_rows(2))
    with pytest.raises(CsvError, match="shorter than one"):
        build_payload(short, Path("short.csv"), Q0, 120.0, 50.0)


def test_converted_clip_loads_and_passes_the_bones_filters(tmp_path):
    root = tmp_path / "g1" / "csv"
    _write_csv(
        root / "2024-01-01" / "walk_forward.csv",
        _constant_rows(16, position_cm=(0.0, 0.0, 90.0)),
    )
    _write_csv(
        root / "2024-01-01" / "sit_on_chair.csv",
        _constant_rows(16, position_cm=(0.0, 0.0, 50.0)),
    )
    _write_csv(
        root / "2024-01-02" / "jump_off_50cm.csv",
        _constant_rows(16, position_cm=(0.0, 0.0, 90.0)),
    )
    out = tmp_path / "clips"
    summary = convert_paths([root], out, Q0)
    assert summary.failed == 0
    assert summary.converted == 3
    library = load_motion_library(str(out), None, "cpu")
    assert library.num_clips == 3
    assert library.clip(0).shape[-1] == STATE_DIM

    bones = load_bones_library(str(out), None, "cpu")
    assert bones.clip_names == ["2024-01-01/walk_forward"]
    assert name_rejected("2024-01-01/sit_on_chair")
    assert name_rejected("2024-01-02/jump_off_50cm")
    assert bones.filter_counts["rejected_name"] == 2
    assert bones.filter_counts["rejected_feasibility"] == 0

    walk = torch.load(
        out / output_filename(Path("2024-01-01/walk_forward.csv")), weights_only=True
    )
    frames = walk["state"].shape[0]
    walk["zmin"] = torch.full((frames,), -0.2)
    walk["foot_clearance"] = torch.zeros(frames, 2)
    filtered = tmp_path / "filtered"
    filtered.mkdir()
    torch.save(walk, filtered / "clip_walk.pt")
    with pytest.raises(RuntimeError, match="removed every clip"):
        load_bones_library(str(filtered), None, "cpu")

    walk["zmin"] = torch.zeros(frames)
    walk["foot_clearance"] = torch.zeros(frames, 2)
    torch.save(walk, filtered / "clip_walk.pt")
    kept = load_bones_library(str(filtered), None, "cpu")
    assert kept.num_clips == 1
    assert kept.filter_counts["rejected_feasibility"] == 0
    assert (
        clip_name_for(Path("2024-01-01/walk_forward.csv")) == "2024-01-01/walk_forward"
    )


def _fake_checkpoint(directory, world_size=1, step=0):
    directory.mkdir(parents=True)
    torch.save({"step": step}, directory / ckpt.MODEL_FILE)
    for rank in range(world_size):
        torch.save({"rank": rank}, directory / ckpt.rank_file(rank))
    return ckpt.write_meta(directory, {"world_size": world_size, "global_update": step})


def test_missing_policy_fields_mean_the_paper_recipe():
    meta = {}
    assert ckpt.policy_value(meta, "critic_features") == "root4"
    assert ckpt.policy_value(meta, "wm_state_loss") == "paper"
    assert ckpt.policy_value(meta, "reward_anchor") == "torso"
    assert ckpt.policy_value(meta, "input_norm") == "none"
    assert ckpt.policy_value(meta, "residual_frame") == "heading"


def test_latest_checkpoint_skips_unfinished_and_corrupt(tmp_path):
    _fake_checkpoint(tmp_path / "global_step_5", step=5)
    _fake_checkpoint(tmp_path / "global_step_9", step=9)
    with open(tmp_path / "global_step_9" / ckpt.MODEL_FILE, "r+b") as handle:
        handle.seek(-1, 2)
        last = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([last[0] ^ 0xFF]))
    _fake_checkpoint(tmp_path / "global_step_12.tmp", step=12)
    (tmp_path / "global_step_20").mkdir()
    torch.save({}, tmp_path / "global_step_20" / ckpt.MODEL_FILE)

    assert not ckpt.is_complete_checkpoint(tmp_path / "global_step_9")
    assert not ckpt.is_complete_checkpoint(tmp_path / "global_step_12.tmp")
    assert not ckpt.is_complete_checkpoint(tmp_path / "global_step_20")
    assert ckpt.find_latest_checkpoint(tmp_path) == str(tmp_path / "global_step_5")
    assert ckpt.find_latest_checkpoint(tmp_path / "missing") is None


def test_meta_requires_every_rank_file(tmp_path):
    directory = tmp_path / "global_step_1.tmp"
    directory.mkdir()
    torch.save({}, directory / ckpt.MODEL_FILE)
    torch.save({}, directory / ckpt.rank_file(0))
    with pytest.raises(RuntimeError, match="shared by all nodes"):
        ckpt.write_meta(directory, {"world_size": 2, "global_update": 1})


def test_rng_capture_round_trips_through_weights_only_load(tmp_path):
    torch.manual_seed(1)
    random.seed(2)
    np.random.seed(3)
    torch.save(ckpt.capture_rng(torch.device("cpu")), tmp_path / "rng.pt")
    expected = (torch.rand(4), random.random(), np.random.rand(3))
    torch.rand(9)
    random.random()
    np.random.rand(5)
    ckpt.restore_rng(
        torch.load(tmp_path / "rng.pt", weights_only=True), torch.device("cpu")
    )
    assert torch.equal(torch.rand(4), expected[0])
    assert random.random() == expected[1]
    assert np.array_equal(np.random.rand(3), expected[2])


def _checkpoint_cfg(velocity: bool = False):
    return OmegaConf.create(
        {
            "algorithm": {
                "goal_anchor": False,
                "k": 8,
                "H": 4,
                "gamma": 0.99,
                "lam": 0.95,
                "goal_dim": 680,
                "obs_dim": 93,
                "goal_encoding": "paper",
                "velocity_tracking": velocity,
                "warmup_sim_steps": 12,
                "updates_per_env_step": 1.0,
                "wm_batches": 1,
                "wm_window_batch": 4,
                "imagine_batch": 4,
                "critic_epochs": 1,
                "critic_minibatches": 1,
                "bce_weight": 1.0,
                "action_noise_std": 0.1,
                "G_max": 10.0,
                "lr": 2e-4,
                "lr_decay_updates": 10,
                "action_dim": 29,
                "max_episode_seconds": 10.0,
                "termination_body": "pelvis",
                "seed": 0,
            },
            "env": {
                "motion_format": "lafan1",
                "motion_path": synthetic_clip_dir(),
                "num_clips": 2,
                "backend": "analytic",
                "num_envs": 4,
                "replay_length": 32,
            },
        }
    )


@needs_mjlab
def test_trainer_checkpoint_restores_state_and_rng(tmp_path):
    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    trainer = SuperMimicTrainer(_checkpoint_cfg(), device=torch.device("cpu"))
    trainer.run_until_updates(2)
    trainer.reset_sampler.bin_return[0] = 3.25
    target = tmp_path / "global_step_2"
    trainer.save_checkpoint(str(target))
    next_draw = torch.rand(3)

    meta = ckpt.verify_checkpoint(target)
    assert meta["world_size"] == 1 and meta["global_update"] == 2
    assert meta["goal_dim"] == 680 and meta["obs_dim"] == 93
    assert not ckpt.temp_dir(target).exists()

    resumed = SuperMimicTrainer(_checkpoint_cfg(), device=torch.device("cpu"))
    resumed.load_checkpoint(str(target))
    assert torch.equal(torch.rand(3), next_draw)
    for key, value in trainer.models.state_dict().items():
        assert torch.equal(resumed.models.state_dict()[key], value), key
    assert (
        resumed.opt_actor.state_dict()["state"].keys()
        == trainer.opt_actor.state_dict()["state"].keys()
    )
    assert resumed.sched_wm.last_epoch == trainer.sched_wm.last_epoch
    for name in (
        "global_update",
        "sim_vector_steps",
        "env_transitions",
        "wm_optimizer_steps",
        "actor_optimizer_steps",
        "critic_optimizer_steps",
    ):
        assert getattr(resumed, name) == getattr(trainer, name), name
    assert float(resumed.reset_sampler.bin_return[0]) == 3.25
    assert torch.equal(
        resumed.reset_sampler.bin_memory, trainer.reset_sampler.bin_memory
    )
    # Replay is not saved, so the resumed trainer warms a new one first.
    assert resumed.replay_warm is False


@needs_mjlab
def test_interrupted_save_leaves_no_complete_checkpoint(tmp_path, monkeypatch):
    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    trainer = SuperMimicTrainer(_checkpoint_cfg(), device=torch.device("cpu"))
    trainer.save_checkpoint(str(tmp_path / "global_step_1"))

    def interrupted(directory, meta):
        raise OSError("disk full")

    monkeypatch.setattr(ckpt, "write_meta", interrupted)
    trainer.global_update = 2
    with pytest.raises(OSError, match="disk full"):
        trainer.save_checkpoint(str(tmp_path / "global_step_2"))
    assert not (tmp_path / "global_step_2").exists()
    assert (tmp_path / "global_step_2.tmp").exists()
    assert ckpt.find_latest_checkpoint(tmp_path) == str(tmp_path / "global_step_1")

    monkeypatch.undo()
    trainer.save_checkpoint(str(tmp_path / "global_step_2"))
    assert not (tmp_path / "global_step_2.tmp").exists()
    assert ckpt.find_latest_checkpoint(tmp_path) == str(tmp_path / "global_step_2")


@needs_mjlab
def test_checkpoint_with_other_dimensions_is_rejected(tmp_path):
    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    SuperMimicTrainer(_checkpoint_cfg(), device=torch.device("cpu")).save_checkpoint(
        str(tmp_path / "global_step_0")
    )
    other = SuperMimicTrainer(
        _checkpoint_cfg(velocity=True), device=torch.device("cpu")
    )
    with pytest.raises(ValueError, match="velocity_tracking"):
        other.load_checkpoint(str(tmp_path / "global_step_0"))


@needs_mjlab
def test_checkpoints_of_another_format_or_goal_window_are_refused(tmp_path):
    import json

    from rlinf.envs.sim.g1_mimic.motion import GOAL_WINDOW
    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    trainer = SuperMimicTrainer(_checkpoint_cfg(), device=torch.device("cpu"))
    target = tmp_path / "global_step_0"
    trainer.save_checkpoint(str(target))
    meta_path = target / ckpt.META_FILE
    meta = json.loads(meta_path.read_text())
    assert meta["format"] == ckpt.FORMAT_VERSION
    assert meta["goal_window"] == list(GOAL_WINDOW["paper"]) == [0, 9]
    # This fixture omits the four switches, so the trainer follows zita-ch.
    assert meta["reward_anchor"] == "pelvis"
    assert meta["input_norm"] == "running"
    assert meta["residual_frame"] == "body"
    assert meta["wm_state_loss"] == "running"

    def load_with(**changes):
        meta_path.write_text(json.dumps({**meta, **changes}))
        SuperMimicTrainer(
            _checkpoint_cfg(), device=torch.device("cpu")
        ).load_checkpoint(str(target))

    with pytest.raises(ValueError, match="goal_window"):
        load_with(goal_window=[1, 10])
    with pytest.raises(ValueError, match="critic_features"):
        load_with(critic_features="zita")
    without_features = {
        key: value for key, value in meta.items() if key != "critic_features"
    }
    meta_path.write_text(json.dumps(without_features))
    SuperMimicTrainer(_checkpoint_cfg(), device=torch.device("cpu")).load_checkpoint(
        str(target)
    )
    with pytest.raises(ValueError, match="format 2"):
        load_with(format=2)
    meta_path.unlink()
    with pytest.raises(ValueError, match="has no"):
        SuperMimicTrainer(
            _checkpoint_cfg(), device=torch.device("cpu")
        ).load_checkpoint(str(target))


def _eval_cfg(velocity: bool = False):
    return OmegaConf.create(
        {
            "runner": {"only_eval": True},
            "algorithm": {
                "goal_anchor": False,
                "k": 8,
                "H": 4,
                "gamma": 0.99,
                "lam": 0.95,
                "goal_dim": 680,
                "obs_dim": 93,
                "goal_encoding": "paper",
                "velocity_tracking": velocity,
                "warmup_sim_steps": 4,
                "updates_per_env_step": 0.4,
                "wm_batches": 1,
                "wm_window_batch": 4,
                "imagine_batch": 4,
                "critic_epochs": 1,
                "critic_minibatches": 1,
                "bce_weight": 1.0,
                "action_noise_std": 0.1,
                "G_max": 10.0,
                "lr": 2e-4,
                "lr_decay_updates": 10,
                "action_dim": 29,
                "max_episode_seconds": 10.0,
                "termination_body": "pelvis",
                "seed": 0,
            },
            "env": {
                "motion_format": "lafan1",
                "motion_path": synthetic_clip_dir(),
                "num_clips": 3,
                "backend": "analytic",
                "num_envs": 2,
                "replay_length": 32,
            },
        }
    )


REPO = Path(__file__).resolve().parents[2]


class _ScriptedVelocityEnv:
    """Holds a set velocity per env; the last ``steady`` steps use ``late``."""

    def __init__(self, early, late, steady, steps, fall_at=None):
        from rlinf.envs.sim.g1_mimic.features import default_state

        self.num_envs = early.shape[0]
        self.state = default_state(self.num_envs, "cpu", torch.float32)
        self.xi = torch.zeros(self.num_envs, 33)
        self.early, self.late = early, late
        self.switch = steps - steady
        self.fall_at = fall_at or {}
        self.t = 0

    def ref_state(self):
        return self.state

    def step(self, action):
        velocity = self.early if self.t < self.switch else self.late
        state = self.state.clone()
        state[:, 36:38] = velocity[:, :2]
        state[:, 41] = velocity[:, 2]
        for env, step in self.fall_at.items():
            if self.t >= step:
                state[env, 2] = 0.2
        self.state = state
        self.t += 1
        return {"next_state": state, "ref_state": state}


class _NanEnv:
    def __init__(self):
        self.num_envs = 2
        self.state = default_state(2, "cpu", torch.float32)
        self.motion_time = torch.zeros(2, dtype=torch.long)
        self.motion_id = torch.zeros(2, dtype=torch.long)
        self.xi = torch.zeros(2, 33)
        self.ticks = 0
        self.motion = self

    def clip_length(self, motion_id):
        return torch.full(motion_id.shape, 4)

    def ref_state(self):
        return default_state(2, "cpu", torch.float32)

    def step(self, action):
        self.ticks += 1
        nxt = self.state.clone()
        if self.ticks == 1:
            end = torch.tensor([EPISODE_FAILURE, EPISODE_CONT])
        elif self.ticks < 4:
            end = torch.tensor([EPISODE_FAILURE, EPISODE_CONT])
            nxt[0] = torch.nan
        else:
            end = torch.tensor([EPISODE_FAILURE, EPISODE_CLIP_END])
            nxt[0] = torch.nan
        self.state = nxt
        self.motion_time = self.motion_time + 1
        ref = self.ref_state()
        return {"next_state": nxt, "ref_state": ref, "end_kind": end}


@needs_mjlab
def test_evaluator_refuses_a_checkpoint_of_another_config(tmp_path):
    from rlinf.workers.supermimic.evaluator import SuperMimicEvaluator
    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    SuperMimicTrainer(_eval_cfg(), device=torch.device("cpu")).save_checkpoint(
        str(tmp_path / "global_step_0")
    )
    evaluator = SuperMimicEvaluator(_eval_cfg(velocity=True), torch.device("cpu"))
    with pytest.raises(ValueError, match="velocity_tracking"):
        evaluator.load(str(tmp_path / "global_step_0"))


@needs_mjlab
@needs_warp
def test_eval_entry_writes_summary_and_tables(tmp_path):
    from rlinf.envs.sim.g1_mimic.g1_fk import asset_default_state
    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    standing = asset_default_state()
    clips = tmp_path / "clips"
    clips.mkdir()
    for i in range(3):
        torch.save(
            {
                "fps": 50.0,
                "joint_names": list(G1_JOINT_NAMES),
                "q0": standing[7:36].clone(),
                "state": standing.unsqueeze(0).repeat(20, 1),
            },
            clips / f"clip_{i:03d}.pt",
        )
    cfg = _eval_cfg(velocity=True)
    # Match the explicit paper-interface overrides passed to the evaluator.
    cfg.algorithm.wm_state_loss = "paper"
    cfg.algorithm.reward_anchor = "torso"
    cfg.algorithm.input_norm = "none"
    cfg.algorithm.residual_frame = "heading"
    cfg.env.num_envs = 4
    SuperMimicTrainer(cfg, device=torch.device("cpu")).save_checkpoint(
        str(tmp_path / "run" / "global_step_5")
    )
    env = dict(os.environ)
    env["EMBODIED_PATH"] = str(REPO / "examples" / "embodiment")
    env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH', '')}"
    out = tmp_path / "eval_out"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "evaluations" / "eval_supermimic.py"),
            "--config-name",
            "supermimic_lafan1_zita_eval",
            "algorithm.velocity_tracking=true",
            "algorithm.goal_encoding=paper",
            "algorithm.goal_frame_dim=68",
            "algorithm.obs_dim=93",
            "algorithm.reward_anchor=torso",
            "algorithm.input_norm=none",
            "algorithm.residual_frame=heading",
            "algorithm.wm_state_loss=paper",
            "algorithm.critic_features=root4",
            "eval.velocity=g2",
            f"env.motion_path={clips}",
            "env.num_clips=3",
            "env.num_envs=4",
            f"eval.checkpoint={tmp_path / 'run'}",
            "eval.g2_trials=1",
            "eval.g2_seconds=0.06",
            "eval.g2_steady_seconds=0.04",
            "eval.g2_batch=336",
            f"eval.output_dir={out}",
            f"runner.logger.log_path={tmp_path / 'logs'}",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["meta"]["global_step"] == 5
    assert summary["meta"]["checkpoint"].endswith("global_step_5")
    assert "eval/normalized_return" in summary["metrics"]
    assert summary["metrics"]["eval_velocity_g2/num_commands"] == 672.0
    assert summary["metrics"]["eval_velocity_g2/num_trials"] == 1.0
    assert (out / "per_clip.csv").is_file()
    with open(out / "velocity_g2.csv") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 672 and rows[0]["command_name"].startswith("vx-vy:")
    assert (out / "config.yaml").is_file()


def test_eval_entry_rejects_an_unknown_velocity_mode_before_any_work(tmp_path):
    env = dict(os.environ)
    env["EMBODIED_PATH"] = str(REPO / "examples" / "embodiment")
    env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH', '')}"
    out = tmp_path / "eval_out"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "evaluations" / "eval_supermimic.py"),
            "env.motion_path=/unused",
            "eval.velocity=bogus",
            f"eval.checkpoint={tmp_path / 'missing'}",
            f"eval.output_dir={out}",
            f"runner.logger.log_path={tmp_path / 'logs'}",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO,
        timeout=600,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "eval.velocity must be none, smoke or g2, got 'bogus'" in output
    assert "no complete global_step" not in output
    assert not out.exists()


def test_step_test_commands_follow_appendix_g2():
    from rlinf.envs.sim.g1_mimic.velocity import (
        COMMAND_HIGH,
        COMMAND_LOW,
        step_test_commands,
    )

    commands, names, planes = step_test_commands()
    assert commands.shape == (672, 3) and len(set(names)) == 672
    assert {plane: planes.count(plane) for plane in set(planes)} == {
        "vx-vy": 224,
        "vx-wz": 224,
        "vy-wz": 224,
    }
    assert bool((commands.abs().max(dim=-1).values > 1e-6).all())
    for plane, idle in (("vx-vy", 2), ("vx-wz", 1), ("vy-wz", 0)):
        rows = commands[torch.tensor([p == plane for p in planes])]
        assert float(rows[:, idle].abs().max()) < 1e-6
    for axis in range(3):
        grid = torch.linspace(COMMAND_LOW[axis], COMMAND_HIGH[axis], 15)
        values = torch.unique(commands[:, axis])
        assert torch.allclose(values, grid, atol=1e-6)


def test_step_test_error_uses_the_last_two_seconds_and_marks_falls():
    from rlinf.workers.supermimic.eval_rollout import run_velocity_step_test

    commands = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.5, 1.0], [1.0, 0.0, 0.0]])
    late = torch.tensor([[0.9, 0.0, 0.0], [0.0, 0.5, 0.5], [1.0, 0.0, 0.0]])
    early = torch.full((3, 3), 5.0)
    env = _ScriptedVelocityEnv(early, late, steady=100, steps=250, fall_at={2: 40})

    def zeros(history, command):
        return torch.zeros(history.shape[0], 29)

    out = run_velocity_step_test(env, zeros, torch.zeros(29), commands, 250, 100)
    assert float(out["relative_error"][0]) == pytest.approx(0.1, abs=1e-6)
    assert float(out["relative_error"][1]) == pytest.approx(
        0.5 / math.sqrt(1.25), abs=1e-6
    )
    assert math.isnan(float(out["relative_error"][2]))
    assert out["fell"].tolist() == [0, 0, 1]


def test_step_test_aggregation_takes_the_median_of_trials_that_stood():
    from rlinf.workers.supermimic.eval_rollout import aggregate_step_test

    nan = float("nan")
    errors = torch.tensor(
        [[0.1, nan, nan], [0.3, 0.2, nan], [0.2, nan, nan], [nan, 0.6, nan]]
    )
    fell = torch.isnan(errors).long()
    summary, rows = aggregate_step_test(errors, fell, ["vx-vy", "vx-wz", "vx-wz"])
    assert rows["relative_error"][:2].tolist() == pytest.approx([0.2, 0.2])
    assert math.isnan(float(rows["relative_error"][2]))
    assert rows["fell_trials"].tolist() == [1, 2, 4]
    assert summary["eval_velocity_g2/commands_all_trials_fell"] == 1.0
    assert summary["eval_velocity_g2/median_relative_error"] == pytest.approx(0.2)
    assert summary["eval_velocity_g2/trial_fall_rate"] == pytest.approx(7 / 12)
    assert summary["eval_velocity_g2/median_relative_error_vx-wz"] == pytest.approx(0.2)
    assert summary["eval_velocity_g2/num_commands"] == 3.0


@needs_mjlab
def test_eval_walks_every_clip_and_keeps_the_full_denominator():
    clips = []
    for length in (2, 3, 4, 2, 3, 4, 2, 3, 4, 5, 2):
        clips.append(default_state(length, "cpu", torch.float32))
    library = MotionLibrary(clips)
    env = AnalyticG1Env(
        3, library, torch.device("cpu"), max_episode_seconds=10.0, eval_mode=True
    )

    def policy(history, goal):
        return torch.zeros(history.shape[0], ACTION_DIM)

    def goal_fn(motion_id, motion_time, state):
        return torch.zeros(state.shape[0], 1)

    parts = []
    for start in range(0, 11, 3):
        ids = torch.arange(start, min(start + 3, 11))
        env.reset(
            torch.arange(ids.numel()), ids, torch.zeros(ids.numel(), dtype=torch.long)
        )
        parts.append(
            run_tracking_eval(env, policy, library.q0, goal_fn, active=int(ids.numel()))
        )
    rows = {key: torch.cat([part[key] for part in parts], 0) for key in parts[0]}
    assert sorted(int(x) for x in rows["clip_id"]) == list(range(11))
    assert torch.equal(rows["steps"], rows["scheduled_steps"])
    summary = summarize_tracking_eval(rows)
    assert summary["eval/num_clips"] == pytest.approx(11.0)
    # Early failure must not shrink the denominator to the played steps.
    standing = default_state(8, "cpu", torch.float32)
    fail_env = AnalyticG1Env(
        1,
        MotionLibrary([standing]),
        torch.device("cpu"),
        max_episode_seconds=10.0,
        eval_mode=True,
    )
    fail_env.reset(torch.tensor([0]), torch.tensor([0]), torch.tensor([0]))
    fail_env.state[:, 2] -= 0.4
    failed = run_tracking_eval(fail_env, policy, fail_env.motion.q0, goal_fn)
    assert float(failed["steps"]) < float(failed["scheduled_steps"])
    assert torch.allclose(
        failed["normalized_return"],
        failed["return_sum"] / (0.1 * failed["scheduled_steps"]),
    )


@needs_mjlab
def test_long_eval_keeps_the_history_window_and_the_full_return(tmp_path):
    from rlinf.envs.sim.g1_mimic.motion import MotionLibrary

    standing = default_state(40, "cpu", torch.float32)
    env = AnalyticG1Env(
        1,
        MotionLibrary([standing]),
        torch.device("cpu"),
        max_episode_seconds=10.0,
        eval_mode=True,
    )
    env.reset(torch.tensor([0]), torch.tensor([0]), torch.tensor([0]))
    rows = run_tracking_eval(
        env,
        lambda history, goal: torch.zeros(history.shape[0], 29),
        torch.zeros(29),
        lambda motion_id, motion_time, state: torch.zeros(motion_id.shape[0], 4),
    )
    assert float(rows["steps"]) == 40.0
    assert float(rows["clip_end"]) == 1.0
    assert float(rows["fail"]) == 0.0
    assert float(rows["normalized_return"]) == pytest.approx(1.0, abs=1e-4)
    assert torch.isfinite(rows["return_sum"]).all()

    # Longer than the 10 s training limit. Eval must play the whole clip.
    long = default_state(520, "cpu", torch.float32)
    long_env = AnalyticG1Env(
        1,
        MotionLibrary([long]),
        torch.device("cpu"),
        max_episode_seconds=10.0,
        eval_mode=True,
    )
    long_env.reset(torch.tensor([0]), torch.tensor([0]), torch.tensor([0]))
    long_rows = run_tracking_eval(
        long_env,
        lambda history, goal: torch.zeros(history.shape[0], 29),
        torch.zeros(29),
        lambda motion_id, motion_time, state: torch.zeros(motion_id.shape[0], 4),
    )
    assert float(long_rows["steps"]) == 520.0
    assert float(long_rows["scheduled_steps"]) == 520.0
    assert float(long_rows["clip_end"]) == 1.0
    assert float(long_rows["episode_length_s"]) == pytest.approx(10.4, abs=1e-4)
    assert float(long_rows["normalized_return"]) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("active", [None, torch.tensor([True, False])])
def test_policy_action_rejects_nonfinite_active_rows(value, active):
    action = torch.zeros(2, ACTION_DIM)
    action[0, 3] = value
    with pytest.raises(RuntimeError, match=r"active evaluation rows \[0\]"):
        checked_policy_action(action, active)


def test_policy_action_masks_only_inactive_rows():
    action = torch.ones(3, ACTION_DIM)
    action[1] = float("nan")
    action[2] = float("inf")
    actual = checked_policy_action(action, torch.tensor([True, False, False]))
    assert torch.equal(actual[0], action[0])
    assert torch.equal(actual[1:], torch.zeros(2, ACTION_DIM))
    assert torch.isnan(action[1]).all()


@pytest.mark.parametrize("kind", ["tracking", "velocity", "step_test"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_rollout_rejects_nonfinite_policy_before_simulation(kind, value):
    from rlinf.workers.supermimic.eval_rollout import (
        run_velocity_eval,
        run_velocity_step_test,
    )

    env = _NanEnv()

    def policy(history, goal):
        return torch.full((2, ACTION_DIM), value)

    with pytest.raises(RuntimeError, match="non-finite policy actions"):
        if kind == "tracking":
            run_tracking_eval(
                env,
                policy,
                torch.zeros(ACTION_DIM),
                lambda ids, times, state: torch.zeros(2, 680),
            )
        elif kind == "velocity":
            run_velocity_eval(env, policy, torch.zeros(ACTION_DIM), torch.ones(2, 3), 2)
        else:
            run_velocity_step_test(
                env,
                policy,
                torch.zeros(ACTION_DIM),
                torch.ones(2, 3),
                2,
                1,
            )
    assert env.ticks == 0


@pytest.mark.parametrize("kind", ["velocity", "step_test"])
def test_velocity_rollout_ignores_nonfinite_policy_on_finished_rows(kind):
    from rlinf.workers.supermimic.eval_rollout import (
        run_velocity_eval,
        run_velocity_step_test,
    )

    command = torch.ones(2, 3)
    env = _ScriptedVelocityEnv(command, command, 1, 3, fall_at={0: 0})

    def policy(history, goal):
        action = torch.zeros(2, ACTION_DIM)
        if env.t > 0:
            action[0] = float("nan")
        return action

    if kind == "velocity":
        rows = run_velocity_eval(env, policy, torch.zeros(ACTION_DIM), command, 3)
        assert rows["fall"].tolist() == [1, 0]
    else:
        rows = run_velocity_step_test(
            env, policy, torch.zeros(ACTION_DIM), command, 3, 1
        )
        assert rows["fell"].tolist() == [1, 0]
    assert env.t == 3


@pytest.mark.parametrize("kind", ["tracking", "velocity", "step_test"])
@pytest.mark.parametrize("field", ["next_state", "ref_state"])
@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_rollout_rejects_nonfinite_active_states(kind, field, initial, value):
    from rlinf.workers.supermimic.eval_rollout import (
        run_velocity_eval,
        run_velocity_step_test,
    )

    class InvalidStateEnv(_NanEnv):
        def __init__(self):
            super().__init__()
            if initial and field == "next_state":
                self.state[0, 2] = value

        def ref_state(self):
            ref = super().ref_state()
            if initial and field == "ref_state":
                ref[0, 2] = value
            return ref

        def step(self, action):
            result = super().step(action)
            result[field][0, 2] = value
            result["end_kind"][0] = EPISODE_CLIP_END
            return result

    env = InvalidStateEnv()

    def policy(history, goal):
        return torch.zeros(2, ACTION_DIM)

    with pytest.raises(RuntimeError, match=r"non-finite evaluation states.*\[0\]"):
        if kind == "tracking":
            run_tracking_eval(
                env,
                policy,
                torch.zeros(ACTION_DIM),
                lambda ids, times, state: torch.zeros(2, 680),
            )
        elif kind == "velocity":
            run_velocity_eval(env, policy, torch.zeros(ACTION_DIM), torch.ones(2, 3), 2)
        else:
            run_velocity_step_test(
                env, policy, torch.zeros(ACTION_DIM), torch.ones(2, 3), 2, 1
            )
    assert env.ticks == (0 if initial else 1)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_eval_state_check_masks_only_inactive_rows(value):
    state = default_state(2, "cpu", torch.float32)
    ref = state.clone()
    state[1] = value
    ref[1] = value
    next_state, next_ref = checked_eval_states(state, ref, torch.tensor([True, False]))
    assert torch.equal(next_state[0], state[0])
    assert torch.equal(next_ref[0], ref[0])
    assert torch.isfinite(next_state).all() and torch.isfinite(next_ref).all()
    assert not torch.isfinite(state[1]).all()


@pytest.mark.parametrize("kind", ["velocity", "step_test"])
def test_velocity_rollout_ignores_nonfinite_states_on_finished_rows(kind):
    from rlinf.workers.supermimic.eval_rollout import (
        run_velocity_eval,
        run_velocity_step_test,
    )

    class FinishedStateEnv(_ScriptedVelocityEnv):
        def step(self, action):
            result = super().step(action)
            if self.t > 1:
                result["next_state"][0] = float("nan")
                result["ref_state"][0] = float("nan")
            return result

    command = torch.ones(2, 3)
    env = FinishedStateEnv(command, command, 1, 3, fall_at={0: 0})

    def policy(history, goal):
        return torch.zeros(2, ACTION_DIM)

    if kind == "velocity":
        rows = run_velocity_eval(env, policy, torch.zeros(ACTION_DIM), command, 3)
        assert rows["fall"].tolist() == [1, 0]
        assert torch.isfinite(rows["lin_vel_err"]).all()
    else:
        rows = run_velocity_step_test(
            env, policy, torch.zeros(ACTION_DIM), command, 3, 1
        )
        assert rows["fell"].tolist() == [1, 0]
    assert env.t == 3


def _pose(z: float = 0.76, speed_xy=(0.0, 0.0), tilt_z: float = -1.0) -> torch.Tensor:
    """Upright identity root unless ``tilt_z`` asks for a pitched gravity."""
    state = torch.zeros(STATE_DIM)
    state[2] = z
    state[3] = 1.0
    state[ROOT_LIN_VEL.start] = speed_xy[0]
    state[ROOT_LIN_VEL.start + 1] = speed_xy[1]
    if tilt_z != -1.0:
        # Pitch so body-frame gravity z equals tilt_z. Identity gives -1.
        # g_body_z = -R_zz. For a pitch p, R_zz = cos(p), so cos(p) = -tilt_z.
        cosine = max(-1.0, min(1.0, -tilt_z))
        pitch = math.acos(cosine)
        half = 0.5 * pitch
        state[3] = math.cos(half)
        state[5] = math.sin(half)
    return state


def _criteria(**kwargs) -> RecoveryCriteria:
    values = {"stable_seconds": 0.04, "dt": DT}
    values.update(kwargs)
    return RecoveryCriteria(**values)


def test_heading_push_rotates_into_the_world_frame():
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
    forward = world_push_xy(torch.tensor([1.0, 0.0]), 2.0, "world", identity)
    assert torch.allclose(forward, torch.tensor([2.0, 0.0]), atol=1e-6)
    yaw = math.pi / 2
    half = 0.5 * yaw
    turned = torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)])
    # Robot x, after a +90 deg yaw, points along world y.
    pushed = world_push_xy(torch.tensor([1.0, 0.0]), 2.0, "heading", turned)
    assert torch.allclose(pushed, torch.tensor([0.0, 2.0]), atol=1e-5)
    # A world-frame request ignores that yaw.
    unchanged = world_push_xy(torch.tensor([1.0, 0.0]), 2.0, "world", turned)
    assert torch.allclose(unchanged, torch.tensor([2.0, 0.0]), atol=1e-5)


def test_recovery_window_fall_and_timeout():
    ref = _pose()
    origin = torch.zeros(2)
    criteria = _criteria()
    required = criteria.stable_frames()
    assert required == 2
    still = _pose()
    moving = _pose(speed_xy=(1.0, 0.0))
    assert bool(is_stable(still.unsqueeze(0), ref.unsqueeze(0), origin, criteria)[0])
    assert not bool(
        is_stable(moving.unsqueeze(0), ref.unsqueeze(0), origin, criteria)[0]
    )
    tipped = _pose(tilt_z=-0.2)
    assert float(root_tilt(tipped.unsqueeze(0), ref.unsqueeze(0))[0]) > 0.35
    assert not bool(
        is_stable(tipped.unsqueeze(0), ref.unsqueeze(0), origin, criteria)[0]
    )

    # Two stable steps complete a 0.04 s window. Time is the completing step.
    recovered = judge_post_push(
        torch.tensor([False, True, True]),
        torch.tensor([False, False, False]),
        criteria,
    )
    assert recovered["outcome"] == "recovered"
    assert recovered["recovery_time_s"] == pytest.approx(3 * DT)
    assert recovered["stable_onset_time_s"] == pytest.approx(2 * DT)

    # A fall with zero speed is still a fall. The time is not zero.
    fallen = judge_post_push(
        torch.tensor([True, True]),
        torch.tensor([False, True]),
        criteria,
    )
    assert fallen["outcome"] == "fallen"
    assert fallen["recovery_time_s"] is None

    # One stable step does not finish a two-step window.
    waiting = judge_post_push(
        torch.tensor([False, True]),
        torch.tensor([False, False]),
        criteria,
    )
    assert waiting["outcome"] == "timeout"
    assert waiting["recovery_time_s"] is None

    # The window resets when a fast step breaks it, then completes later.
    restarted = judge_post_push(
        torch.tensor([True, False, True, True]),
        torch.zeros(4, dtype=torch.bool),
        criteria,
    )
    assert restarted["outcome"] == "recovered"
    assert restarted["stable_onset_time_s"] == pytest.approx(3 * DT)
    assert restarted["recovery_time_s"] == pytest.approx(4 * DT)


def test_trial_plan_repeats_for_a_fixed_seed_and_rejects_bad_speeds():
    first = trial_plan([0.5, 4.0], 2, "random", 7)
    second = trial_plan([0.5, 4.0], 2, "random", 7)
    other = trial_plan([0.5, 4.0], 2, "random", 8)
    assert [(row["direction_x"], row["direction_y"]) for row in first] == [
        (row["direction_x"], row["direction_y"]) for row in second
    ]
    assert [(row["direction_x"], row["direction_y"]) for row in first] != [
        (row["direction_x"], row["direction_y"]) for row in other
    ]
    assert [row["speed_m_s"] for row in first] == [0.5, 0.5, 4.0, 4.0]
    fixed = trial_plan([1.0], 2, [0.0, 2.0], 1)
    assert fixed[0]["direction_x"] == pytest.approx(0.0)
    assert fixed[0]["direction_y"] == pytest.approx(1.0)
    criteria = _criteria(stable_seconds=0.5)
    with pytest.raises(ValueError, match="finite and > 0"):
        check_push_protocol(
            [0.5, float("nan")], 1, 0.0, 1.0, criteria, "world", "add", "random"
        )
    with pytest.raises(ValueError, match="longer than max_seconds"):
        check_push_protocol([0.5], 1, 0.0, 0.2, criteria, "heading", "set", "random")
    with pytest.raises(ValueError, match="non-zero"):
        check_push_protocol([0.5], 1, 0.0, 1.0, criteria, "world", "add", [0.0, 0.0])


def test_summary_files_leave_failed_recovery_times_blank(tmp_path):
    rows = [
        {
            "trial": 0,
            "speed_m_s": 0.5,
            "direction_x": 1.0,
            "direction_y": 0.0,
            "frame": "world",
            "push_mode": "add",
            "seed": 3,
            "outcome": "timeout",
            "reason": "timeout",
            "recovered": 0,
            "fallen": 0,
            "timeout": 1,
            "initialization_failed": 0,
            "recovery_time_s": None,
            "stable_onset_time_s": None,
            "world_push_x": 0.5,
            "world_push_y": 0.0,
            "measured_dvx": 0.5,
            "measured_dvy": 0.0,
        }
    ]
    summary = {
        "meta": {"seed": 3},
        **summarize_trials(rows, [0.5]),
    }
    write_push_outputs(
        tmp_path,
        summary,
        rows,
        trajectory=[
            {
                "trial": 0,
                "phase": "push",
                "root_x": 0.0,
                "root_y": 0.0,
                "root_z": 0.76,
                "root_vx": 0.5,
                "root_vy": 0.0,
                "root_vz": 0.0,
                "tilt": 0.0,
            }
        ],
    )
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert loaded["by_speed"][0]["recovery_time_mean_s"] is None
    assert loaded["by_speed"][0]["timeout_rate"] == 1.0
    with (tmp_path / "per_trial.csv").open(encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    assert table[0]["outcome"] == "timeout"
    assert table[0]["recovery_time_s"] == ""
    assert table[0]["measured_dvx"] == "0.5"
    with (tmp_path / "trajectory.csv").open(encoding="utf-8") as handle:
        trace = list(csv.DictReader(handle))
    assert trace[0]["phase"] == "push"
    assert trace[0]["root_vx"] == "0.5"


def test_missing_and_mismatched_checkpoints_are_refused(tmp_path):
    empty = tmp_path / "global_step_1"
    empty.mkdir()
    with pytest.raises(ValueError, match="no checkpoint_meta.json"):
        verify_checkpoint(empty)
    with pytest.raises(FileNotFoundError, match="no complete"):
        resolve_checkpoint(str(tmp_path))
    with pytest.raises(ValueError, match="goal_dim"):
        require_matching_checkpoint(
            {
                "goal_dim": 730,
                "obs_dim": 93,
                "goal_encoding": "paper",
                "goal_window": [0, 9],
                "velocity_tracking": False,
            },
            {
                "goal_dim": 680,
                "obs_dim": 93,
                "goal_encoding": "paper",
                "goal_window": [0, 9],
                "velocity_tracking": False,
            },
        )


@needs_mjlab
def test_standing_pose_is_inside_joint_limits_and_matches_fk():
    from rlinf.envs.sim.g1_mimic.g1_fk import asset_joint_limits, g1_kinematics
    from rlinf.envs.sim.g1_mimic.reward import failure_signals

    state = default_standing_state()
    low, high = asset_joint_limits()
    joints = state[JOINT_POS]
    assert torch.all(joints >= low - 1e-4)
    assert torch.all(joints <= high + 1e-4)
    assert torch.allclose(state[ROOT_QUAT].norm(), torch.tensor(1.0), atol=1e-5)
    assert torch.count_nonzero(state[ROOT_LIN_VEL]) == 0
    assert float(state[2]) == pytest.approx(0.76, abs=1e-3)
    body, _rot, _lin, _ang = g1_kinematics().forward(state.unsqueeze(0))
    assert torch.isfinite(body).all()
    free = g1_kinematics().free_tracked
    assert torch.allclose(body[0, free], state[ROOT_POS], atol=1e-4)
    ref = state.unsqueeze(0)
    _height, _tilt, failed = failure_signals(ref, ref)
    assert not bool(failed[0])
    dropped = state.clone()
    dropped[2] = float(state[2]) - 0.4
    _height, _tilt, failed = failure_signals(dropped.unsqueeze(0), ref)
    assert bool(failed[0])
    # A fallen pose can have zero speed. The fall still wins.
    verdict = judge_post_push(
        torch.tensor([True]),
        failed[:1].cpu(),
        _criteria(),
    )
    assert verdict["outcome"] == "fallen"
    assert verdict["recovery_time_s"] is None


@pytest.mark.parametrize("mode", ["add", "set"])
def test_push_operation_updates_selected_root_velocity(mode):
    # The robot and simulation are vendor boundaries; exercise the real env method.
    data = SimpleNamespace(
        root_link_pos_w=torch.zeros(2, 3),
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        root_link_lin_vel_w=torch.tensor([[0.5, 0.2, 0.1], [0.3, -0.1, 0.4]]),
        root_link_ang_vel_w=torch.tensor([[0.2, 0.1, 0.3], [-0.2, 0.4, 0.1]]),
        joint_pos=torch.full((2, ACTION_DIM), 0.2),
        joint_vel=torch.full((2, ACTION_DIM), 0.4),
    )

    class Robot:
        def __init__(self):
            self.data = data

        def write_root_link_velocity_to_sim(self, velocity, env_ids):
            data.root_link_lin_vel_w[env_ids] = velocity[:, :3]
            data.root_link_ang_vel_w[env_ids] = velocity[:, 3:]

    env = MjlabG1Env.__new__(MjlabG1Env)
    env.device, env.dtype = torch.device("cpu"), torch.float32
    env.robot = Robot()
    env.sim = SimpleNamespace(forward=lambda: None)
    env._origins = torch.zeros(2, 3)
    before = env._read_state().clone()
    env.state = before.clone()
    push = torch.tensor([[1.25, -0.5]])
    after = env.apply_horizontal_velocity_push(torch.tensor([1]), push, mode=mode)
    expected = before.clone()
    horizontal = before[1, ROOT_LIN_VEL][:2] + push[0] if mode == "add" else push[0]
    expected[1, ROOT_LIN_VEL.start : ROOT_LIN_VEL.start + 2] = horizontal
    torch.testing.assert_close(after, expected)
    torch.testing.assert_close(env.state, expected)


@needs_warp
@pytest.mark.parametrize("mode", ["add", "set"])
def test_push_changes_warp_root_velocity_before_the_next_integration(mode):
    from rlinf.envs.sim.g1_mimic.env import make_env

    state = default_standing_state()
    motion = hold_pose_library(state, 30)
    device = torch.device("cuda:0")
    env = make_env(
        "mjlab",
        num_envs=1,
        motion=motion,
        device=device,
        max_episode_seconds=1.0e6,
        eval_mode=True,
    )
    env_ids = torch.zeros(1, dtype=torch.long, device=device)
    env.reset(env_ids, env_ids, env_ids)
    initial_velocity = torch.tensor([[0.3, -0.1, 0.4, 0.2, 0.1, -0.15]], device=device)
    env.robot.write_root_link_velocity_to_sim(initial_velocity, env_ids)
    env.sim.forward()
    before = env._read_state().clone()
    push = torch.tensor([[1.25, -0.5]], device=device)
    env.apply_horizontal_velocity_push(env_ids, push, mode=mode)
    after = env.state
    delta = after[0, ROOT_LIN_VEL] - before[0, ROOT_LIN_VEL]
    expected = initial_velocity[0, :2] + push[0] if mode == "add" else push[0]
    torch.testing.assert_close(
        after[0, ROOT_LIN_VEL][:2], expected, atol=1e-4, rtol=1e-4
    )
    assert torch.allclose(delta[2], torch.zeros((), device=device), atol=1e-4)
    assert torch.allclose(after[0, ROOT_POS], before[0, ROOT_POS], atol=1e-5)
    assert torch.allclose(after[0, ROOT_QUAT], before[0, ROOT_QUAT], atol=1e-5)
    assert torch.allclose(after[0, ROOT_ANG_VEL], before[0, ROOT_ANG_VEL], atol=1e-4)
    assert torch.allclose(after[0, JOINT_POS], before[0, JOINT_POS], atol=1e-5)
    assert torch.allclose(after[0, JOINT_VEL], before[0, JOINT_VEL], atol=1e-4)
    stepped = env.step(torch.zeros(1, 29, device=device))
    assert torch.isfinite(stepped["next_state"]).all()
    moved = stepped["next_state"][0, 0] - before[0, 0]
    assert float(moved) > 1e-4


@needs_mjlab
@pytest.mark.parametrize("settle_steps", [0, 1])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_push_trial_rejects_nonfinite_policy_before_simulation(settle_steps, value):
    from rlinf.envs.sim.g1_mimic.features import default_state
    from rlinf.envs.sim.g1_mimic.push_recovery import run_one_trial

    class PushEnv:
        num_envs = 1
        termination_body = "pelvis"

        def __init__(self):
            self.state = default_state(1, "cpu", torch.float32)
            self.reference = self.state.clone()
            self.xi = torch.zeros(1, 33)
            self.motion_id = torch.zeros(1, dtype=torch.long)
            self.motion_time = torch.zeros(1, dtype=torch.long)

        def reset(self, *args):
            self.state = self.reference.clone()

        def ref_state(self):
            return self.reference

        def apply_horizontal_velocity_push(self, env_ids, delta, mode):
            self.state[env_ids, 36:38] += delta

        def step(self, action):
            raise AssertionError("invalid actions reached the simulator")

    env = PushEnv()
    plan = trial_plan([0.5], 1, [1.0, 0.0], 0)[0]
    with pytest.raises(RuntimeError, match="non-finite policy actions"):
        run_one_trial(
            env,
            lambda history, goal: torch.full((1, 29), value),
            lambda ids, times, state: torch.zeros(1, 680),
            q0=torch.zeros(29),
            plan=plan,
            criteria=RecoveryCriteria(),
            frame="world",
            push_mode="add",
            settle_steps=settle_steps,
            max_steps=1,
            include_ref_ori=False,
            joint_low=torch.full((29,), -10.0),
            joint_high=torch.full((29,), 10.0),
        )


SCRIPT = REPO / "examples" / "embodiment" / "slurm_supermimic.sh"


def _stub(path: Path, body: str) -> None:
    path.write_text("#! /bin/bash\n" + body)
    path.chmod(0o755)


def _setup(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    _stub(bin_dir / "scontrol", "printf 'node-a\\nnode-b\\nnode-c\\n'\n")
    _stub(
        bin_dir / "srun",
        f'echo "$*" >> {log}\n'
        'case "$*" in *hostname*) echo "192.0.2.10 172.17.0.1" ;; esac\n',
    )
    _stub(bin_dir / "ray", "exit 0\n")
    repo = tmp_path / "repo"
    embodied = repo / "examples" / "embodiment"
    embodied.mkdir(parents=True)
    (embodied / "train_supermimic.py").write_text("")
    args = tmp_path / "driver_args.txt"
    _stub(
        embodied / "run_supermimic.sh", f'echo "RAY_ADDRESS=$RAY_ADDRESS $*" > {args}\n'
    )
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SLURM_JOB_NODELIST": "node-[a-c]",
        "SLURM_SUBMIT_DIR": str(repo),
        "RAY_HEAD_WAIT": "0",
    }
    return env, log, args


def test_slurm_script_starts_one_ray_node_per_host_and_spans_all_gpus(tmp_path):
    env, log, args = _setup(tmp_path)
    env["SLURM_GPUS_ON_NODE"] = "4"
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "supermimic_lafan1_zita",
            "env.motion_path=/shared/lafan1",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert args.read_text().split() == [
        "RAY_ADDRESS=192.0.2.10:6379",
        "supermimic_lafan1_zita",
        "cluster.num_nodes=3",
        "cluster.component_placement.supermimic=0-11",
        "env.motion_path=/shared/lafan1",
    ]
    calls = log.read_text()
    assert (
        "-w node-a env RLINF_NODE_RANK=0 ray start --head "
        "--node-ip-address=192.0.2.10 --port=6379 --num-gpus=4 --block"
    ) in calls
    for rank, node in ((1, "node-b"), (2, "node-c")):
        assert (
            f"-w {node} env RLINF_NODE_RANK={rank} ray start "
            "--address=192.0.2.10:6379 --num-gpus=4 --block"
        ) in calls
    # The stop step must overlap the blocking start steps that hold every slot.
    assert "--overlap --nodes=3 --ntasks=3 ray stop" in calls


def test_slurm_script_needs_a_gpu_count(tmp_path):
    env, _log, args = _setup(tmp_path)
    result = subprocess.run(
        ["bash", str(SCRIPT), "supermimic_lafan1_zita"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "GPUS_PER_NODE" in result.stderr
    assert not args.exists()


CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples" / "embodiment" / "config"


def _analytic_cfg(**algorithm):
    from rlinf.algorithms.supermimic.validate import validate_supermimic_cfg

    cfg = small_config()
    cfg.env.backend = "analytic"
    cfg.env.device = "cpu"
    cfg.env.motion_path = synthetic_clip_dir()
    cfg.env.num_clips = 4
    cfg.env.num_envs = 8
    cfg.env.replay_length = 64
    cfg.algorithm.imagine_batch = 8
    cfg.algorithm.wm_window_batch = 16
    cfg.algorithm.seed = 0
    for key, value in algorithm.items():
        cfg.algorithm[key] = value
    return validate_supermimic_cfg(cfg, world_size=1)


SELF_ANCHOR = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])


SELF_ROT6D = SELF_ANCHOR[3:]


STEP_X = 0.02


def _frame_of(state: torch.Tensor) -> torch.Tensor:
    return torch.round(state[..., 0] / STEP_X).long()


def _perfect_tracker_cls():
    from rlinf.envs.sim.g1_mimic.reward import failure_signals

    class PerfectTracker(AnalyticG1Env):
        """Lands exactly on the next reference frame, whatever the action."""

        def step(self, action_executed):
            del action_executed
            next_time = self.motion_time + 1
            ref = self.motion.state_at(self.motion_id, next_time)
            height, tilt, failed = failure_signals(
                ref, ref, termination_body=self.termination_body
            )
            timeout = (self.time_in_episode + 1 >= self.max_steps) & (
                not self.eval_mode
            )
            clip_end = self.motion.past_clip_end(self.motion_id, next_time)
            self.state = ref.clone()
            self.motion_time = next_time
            self.time_in_episode = self.time_in_episode + 1
            return {
                "next_state": self.state,
                "end_kind": compose_end_kind(failed, timeout, clip_end),
                "motion_failure": failed,
                "timeout": timeout,
                "clip_end": clip_end,
                "height_err": height,
                "tilt_err": tilt,
                "ref_state": ref,
            }

    return PerfectTracker


def _timeline_state(frames: int, clip: int) -> torch.Tensor:
    """A reference whose every channel moves, so a one-frame shift shows up."""
    f = torch.arange(frames, dtype=torch.float64)
    joint = torch.arange(29, dtype=torch.float64)
    state = torch.zeros(frames, 71, dtype=torch.float64)
    state[:, 0] = STEP_X * f
    state[:, 1] = 10.0 * clip
    state[:, 2] = 0.78 + 0.05 * torch.sin(0.3 * f)
    yaw, pitch = 0.05 * f, 0.08 * torch.sin(0.2 * f)
    zero = torch.zeros_like(f)
    q_yaw = torch.stack((torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)), dim=-1)
    q_pitch = torch.stack(
        (torch.cos(pitch / 2), zero, torch.sin(pitch / 2), zero), dim=-1
    )
    state[:, 3:7] = quat_mul(q_yaw, q_pitch)
    state[:, JOINT_POS] = 0.03 + 0.02 * torch.sin(0.2 * f[:, None] + 0.3 * joint)
    state[:, 36] = 1.0 + 0.1 * torch.sin(0.2 * f)
    state[:, 38] = 0.2 * torch.cos(0.3 * f)
    state[:, 39] = 0.1 * torch.sin(0.2 * f)
    state[:, 40] = 0.2 * torch.cos(0.2 * f)
    state[:, 41] = 2.5
    state[:, 42:71] = 0.2 * torch.cos(0.2 * f[:, None] + 0.3 * joint)
    return state.float()


def _write_timeline_clips(directory: Path, lengths: tuple[int, ...]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for clip, frames in enumerate(lengths):
        payload = {
            "fps": 50.0,
            "joint_names": list(G1_JOINT_NAMES),
            "q0": torch.zeros(29),
            "state": _timeline_state(frames, clip),
        }
        torch.save(payload, directory / f"clip_{clip:03d}.pt")
    return directory


def _zero_policy(history, *args):
    return torch.zeros(history.shape[0], 29)


def _perfect_trainer(tmp_path, lengths=(40, 70), **algorithm):
    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    cfg = _analytic_cfg(**algorithm)
    cfg.env.motion_path = str(_write_timeline_clips(tmp_path / "clips", lengths))
    cfg.env.num_clips = len(lengths)
    trainer = SuperMimicTrainer(cfg, device=torch.device("cpu"))
    trainer.env = _perfect_tracker_cls()(
        trainer.num_envs, trainer.motion, torch.device("cpu"), eval_mode=True
    )
    ids = torch.arange(trainer.num_envs)
    trainer.env.reset(ids, *trainer._reset_points(ids))
    trainer._refresh_history()
    trainer.models.policy = _zero_policy
    trainer.models.velocity_policy = _zero_policy
    return trainer


class _PerfectDynamics(nn.Module):
    """Residual that moves a timeline state exactly onto its next frame."""

    def __init__(self, motion):
        super().__init__()
        self.motion = motion

    def forward(self, state, action, xi):
        del action, xi
        clip = torch.round(state[:, 1] / 10.0).long()
        target = self.motion.state_at(clip, _frame_of(state) + 1)
        quat = quat_normalize(state[:, 3:7])
        heading_t = yaw_rotation(heading_yaw(quat)).transpose(-1, -2)

        def to_heading(delta):
            return torch.einsum("bij,bj->bi", heading_t, delta)

        return torch.cat(
            (
                to_heading(target[:, 0:3] - state[:, 0:3]),
                quat_to_rotvec(quat_mul(quat_inv(quat), target[:, 3:7])),
                target[:, JOINT_POS] - state[:, JOINT_POS],
                to_heading(target[:, 36:39] - state[:, 36:39]),
                target[:, 39:42] - state[:, 39:42],
                target[:, 42:71] - state[:, 42:71],
            ),
            dim=-1,
        )


class _Recorder(nn.Module):
    """Passes inputs through to a real module and keeps a copy of them."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.inputs: list[tuple[torch.Tensor, ...]] = []

    def forward(self, *args):
        self.inputs.append(tuple(arg.detach().clone() for arg in args))
        return self.inner(*args)


def _changed(before: dict[str, torch.Tensor], module: torch.nn.Module) -> bool:
    return any(
        not torch.equal(before[name], value)
        for name, value in module.state_dict().items()
    )


def _snap(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _write_asset_pose_clip(directory: Path, frames: int = 200) -> None:
    """Repeat the G1 asset keyframe. This is not a LAFAN1 retarget."""
    from rlinf.envs.sim.g1_mimic.constants import G1_JOINT_NAMES
    from rlinf.envs.sim.g1_mimic.g1_fk import asset_default_state

    state = asset_default_state()
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "fps": 50.0,
            "joint_names": list(G1_JOINT_NAMES),
            "q0": state[7:36].clone(),
            "state": state.unsqueeze(0).repeat(frames, 1),
        },
        directory / "clip_000.pt",
    )


@needs_mjlab
@pytest.mark.parametrize(
    ("encoding", "anchor", "obs_dim"),
    [
        ("paper", False, 93),
        ("paper", True, 93),
        ("paper", False, 99),
        ("goal730", False, 93),
        ("goal730", False, 99),
    ],
)
def test_collection_scores_each_state_against_the_reference_of_its_time(
    tmp_path, encoding, anchor, obs_dim
):
    trainer = _perfect_trainer(
        tmp_path,
        goal_encoding=encoding,
        goal_anchor=anchor,
        obs_dim=obs_dim,
        goal_frame_dim=73 if encoding == "goal730" else 68,
    )
    motion, buffer = trainer.motion, trainer.buffer
    zeros = torch.zeros(1, 29)
    ref = motion.state_at(torch.tensor([0]), torch.tensor([3]))
    assert float(analytic_reward(ref, ref, zeros, zeros)["joint_limit"]) == 0.0
    rows = torch.arange(trainer.num_envs)
    ends = 0
    for _ in range(60):
        mid = trainer.env.motion_id.clone()
        t = trainer.env.motion_time.clone()
        length = motion.clip_length(mid)
        goal = trainer._goal(mid, t, trainer.env.state)
        if encoding == "paper":
            frames = goal[:, :680].view(-1, 10, 68)
            for k in range(10):
                expected = motion.state_at(mid, t + k)[:, JOINT_POS] - motion.q0
                assert torch.allclose(frames[:, k, :29], expected, atol=1e-6)
            if anchor:
                assert torch.allclose(
                    goal[:, 680:], SELF_ANCHOR.expand(len(rows), 9), atol=1e-5
                )
        else:
            frames = goal.view(-1, 10, 73)
            for k in range(10):
                expected = motion.state_at(mid, t + 1 + k)[:, JOINT_POS]
                assert torch.allclose(frames[:, k, 15:44], expected, atol=1e-6)
        slot = buffer.cursor.clone()
        trainer.collect_steps(1, accumulate=False)
        reward = buffer.reward[rows, slot]
        assert torch.allclose(
            reward, torch.full_like(reward, MAX_TRACKING_REWARD), atol=1e-6
        )
        assert torch.equal(buffer.motion_time[rows, slot], t)
        assert torch.equal(_frame_of(buffer.state[rows, slot]), t)
        assert torch.equal(
            _frame_of(buffer.next_state[rows, slot]), torch.minimum(t + 1, length - 1)
        )
        if obs_dim == 99:
            relative = buffer.obs_noisy[rows, slot, 93:]
            assert torch.allclose(relative, SELF_ROT6D.expand(len(rows), 6), atol=1e-5)
        end = buffer.end_kind[rows, slot]
        assert torch.equal(end == EPISODE_CLIP_END, t + 1 >= length)
        assert not torch.any(end == EPISODE_FAILURE)
        ends += int((end == EPISODE_CLIP_END).sum())
    assert ends > 0


@needs_mjlab
def test_wm_only_updates_fit_the_world_model_without_the_actor(tmp_path):
    trainer = _perfect_trainer(tmp_path, wm_only_updates=2)
    calls = {"wm": 0, "actor": 0}

    def fit():
        calls["wm"] += 1
        return {"wm/state_loss": 0.0}

    def actor():
        calls["actor"] += 1
        return {"actor/objective": 1.0}

    trainer.fit_world_model = fit
    trainer.imagine_and_update_actor = actor
    trainer.buffer.num_valid_windows = lambda _k: 8
    trainer.step_acc = 3.0
    wm_epoch = trainer.sched_wm.last_epoch
    actor_epoch = trainer.sched_actor.last_epoch
    metrics = trainer.maybe_update()
    assert calls == {"wm": 3, "actor": 1}
    assert trainer.global_update == 3
    assert trainer.sched_wm.last_epoch == wm_epoch + 3
    assert trainer.sched_actor.last_epoch == actor_epoch + 1
    assert metrics["actor/objective"] == 1.0
    assert metrics["train/wm_only"] == 0.0


@needs_mjlab
def test_world_model_and_imagination_score_each_state_against_its_own_reference(
    tmp_path,
):
    trainer = _perfect_trainer(tmp_path, goal_anchor=True)
    for _ in range(40):
        trainer.collect_steps(1, accumulate=False)
    motion = trainer.motion
    trainer.models.dynamics = _PerfectDynamics(motion)
    head = _Recorder(trainer.models.termination)
    trainer.models.termination = head
    metrics = trainer.fit_world_model()
    assert metrics["wm/one_step_loss"] < 1e-8
    assert metrics["wm/k_step_loss"] < 1e-8
    assert max(float(inputs[0].abs().max()) for inputs in head.inputs) < 1e-5
    head.inputs.clear()
    starts = trainer.buffer.sample_starts(8)
    with torch.no_grad():
        rollout = trainer._imagine_motion(starts)
    rewards = rollout["rewards"]
    assert torch.allclose(
        rewards, torch.full_like(rewards, MAX_TRACKING_REWARD), atol=1e-5
    )
    assert float(rollout["features"][..., -4:].abs().max()) < 1e-4
    assert max(float(inputs[0].abs().max()) for inputs in head.inputs) < 1e-4
    mid, tau0 = starts["motion_id"], starts["motion_time"]
    length = motion.clip_length(mid)
    assert bool((tau0 + trainer.H > length - 1).any())
    for t in range(trainer.H + 1):
        state = rollout["states"][:, t]
        assert torch.equal(_frame_of(state), torch.minimum(tau0 + t, length - 1))
        goal = rollout["goals"][:, t]
        assert torch.allclose(goal, trainer._goal(mid, tau0 + t, state), atol=1e-5)
        first = goal[:, :680].view(-1, 10, 68)[:, 0, :29]
        expected = motion.state_at(mid, tau0 + t)[:, JOINT_POS] - motion.q0
        assert torch.allclose(first, expected, atol=1e-5)
        assert torch.allclose(goal[:, 680:], SELF_ANCHOR.expand(len(mid), 9), atol=1e-4)


@needs_mjlab
def test_tracking_eval_gives_a_perfect_tracker_full_return(tmp_path):
    from rlinf.models.embodiment.supermimic import SuperMimicModels
    from rlinf.workers.supermimic.evaluator import goal_function, motion_eval_rows

    lengths = (40, 70)
    motion = load_motion_library(
        str(_write_timeline_clips(tmp_path / "clips", lengths)), 2, "cpu"
    )
    env = _perfect_tracker_cls()(2, motion, torch.device("cpu"), eval_mode=True)
    models = SuperMimicModels(680)
    models.policy = _zero_policy
    cfg = OmegaConf.create({"algorithm": {"obs_dim": 93}})
    rows = motion_eval_rows(
        cfg,
        motion,
        models,
        goal_function(motion, "paper", False),
        torch.device("cpu"),
        2,
        env=env,
    )
    assert torch.allclose(rows["normalized_return"], torch.ones(2), atol=1e-5)
    assert float(rows["torso_pos_err"].abs().max()) < 1e-5
    assert float(rows["free_root_pos_err"].abs().max()) < 1e-5
    assert rows["fail"].tolist() == [0, 0]
    assert rows["clip_end"].tolist() == [1, 1]
    assert rows["steps"].tolist() == list(lengths)


@needs_mjlab
def test_terminal_state_written_before_reset():
    motion = synthetic_clips(num_clips=2, clip_len=30)
    env = AnalyticG1Env(1, motion, torch.device("cpu"), max_episode_seconds=10.0)
    env.state[0, 2] = 0.0
    env.motion_time[0] = 0
    ref = env.ref_state()
    env.state[0, 2] = ref[0, 2] - 0.40
    executed = torch.zeros(1, ACTION_DIM)
    before = env.state.clone()
    result = env.step(executed)
    assert int(result["end_kind"][0]) == EPISODE_FAILURE
    assert torch.allclose(result["next_state"][0, :3], env.state[0, :3])
    fallen = result["next_state"][0, 2].item()
    env.reset(torch.tensor([0]))
    standing = env.state[0, 2].item()
    assert fallen != pytest.approx(standing, abs=1e-4)
    assert before[0, 2].item() != pytest.approx(standing, abs=1e-4)


def test_only_eval_runner_does_not_train():
    from rlinf.runners.supermimic_runner import SuperMimicRunner

    cfg = small_config()
    cfg.runner.only_eval = True
    seen = {"eval": 0, "train": 0}

    class _Result:
        def wait(self):
            return [{"eval/normalized_return": 0.5, "eval/fail_rate": 0.0}]

    class _Worker:
        def evaluate(self):
            seen["eval"] += 1
            return _Result()

        def run_iteration(self):
            seen["train"] += 1
            raise AssertionError("only_eval must not train")

    metrics = SuperMimicRunner(cfg, _Worker()).run()
    assert seen == {"eval": 1, "train": 0}
    assert metrics["eval/normalized_return"] == pytest.approx(0.5)


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
    assert any(
        param.grad is not None and float(param.grad.abs().sum()) > 0
        for param in models.velocity_encoder.parameters()
    )
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
    models.critic(torch.randn(2, models.critic.net[0].in_features)).sum().backward()
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
    models.velocity_critic(
        torch.randn(2, 71), command, torch.randn(2, 33)
    ).sum().backward()
    opt_velocity_critic.step()
    assert _changed(before["velocity_critic"], models.velocity_critic)
    assert not _changed(before["critic"], models.critic)


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
    assert any(
        param.grad is not None and float(param.grad.abs().sum()) > 0
        for param in models.velocity_encoder.parameters()
    )
    assert all(param.grad is None for param in models.decoder.parameters())
    assert all(param.grad is None for param in models.encoder.parameters())
    models.zero_grad()
    z_motion = models.encoder(goal).detach()
    z_vel = models.velocity_encoder(command, history)
    (0.1 * (z_vel - z_motion).pow(2).mean()).backward()
    assert all(param.grad is None for param in models.encoder.parameters())
    assert any(param.grad is not None for param in models.velocity_encoder.parameters())


def test_runner_saves_and_evaluates_once_per_new_update(tmp_path):
    from rlinf.runners.supermimic_runner import SuperMimicRunner

    class _Result:
        def __init__(self, payload):
            self.payload = payload

        def wait(self):
            return [self.payload]

    class _Worker:
        def __init__(self, updates):
            self.updates = list(updates)
            self.saves = []
            self.evals = []

        def run_iteration(self):
            value = self.updates.pop(0)
            return _Result({"train/global_update": value})

        def save_checkpoint(self, directory):
            self.saves.append(directory)
            return _Result(None)

        def evaluate(self):
            self.evals.append(int(self.saves[-1].rsplit("_", 1)[-1]))
            return _Result({"eval/normalized_return": 0.0})

    cfg = OmegaConf.create(
        {
            "runner": {
                "logger": {
                    "log_path": str(tmp_path),
                    "project_name": "rlinf",
                    "experiment_name": "supermimic_smoke",
                    "logger_backends": [],
                },
                "save_interval": 1,
                "val_check_interval": 1,
                "only_eval": False,
            },
            "algorithm": {"total_updates": 3},
        }
    )
    worker = _Worker([0, 0, 1, 1, 2, 2, 2, 3])
    SuperMimicRunner(cfg, worker).run()
    assert [int(path.rsplit("_", 1)[-1]) for path in worker.saves] == [1, 2, 3]
    assert worker.evals == [1, 2, 3]
    assert all(path.count("supermimic_smoke") == 1 for path in worker.saves)
    assert all(
        path.endswith(f"global_step_{step}")
        for path, step in zip(worker.saves, [1, 2, 3], strict=True)
    )

    # Evaluation also runs on the last update, as in the embodied runner.
    cfg.runner.save_interval = 2
    cfg.runner.val_check_interval = 2
    worker = _Worker([0, 1, 1, 2, 3])
    runner = SuperMimicRunner(cfg, worker)
    logged, finished = [], []
    log = runner.metric_logger.log
    runner.metric_logger.log = lambda data, step: (
        logged.append(step) or log(data, step=step)
    )
    runner.metric_logger.finish = lambda: finished.append(True)
    runner.run()
    assert worker.evals == [2, 3]
    assert finished == [True]
    assert logged == [0, 1, 2, 2, 3, 3]

    # A resumed run does not log the restored step a second time.
    worker = _Worker([3, 3, 4, 4])
    cfg.algorithm.total_updates = 4
    runner = SuperMimicRunner(cfg, worker)
    runner.global_step = 3
    logged = []
    runner.metric_logger.log = lambda data, step: logged.append(step)
    runner.run()
    assert logged == [4, 4]


@needs_mjlab
def test_trainer_history_noise_is_drawn_once_per_frame():
    from omegaconf import OmegaConf

    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    cfg = OmegaConf.create(
        {
            "algorithm": {
                "goal_anchor": False,
                "k": 8,
                "H": 32,
                "gamma": 0.99,
                "lam": 0.95,
                "goal_dim": 680,
                "obs_dim": 93,
                "goal_encoding": "paper",
                "warmup_sim_steps": 1,
                "updates_per_env_step": 0.4,
                "wm_batches": 1,
                "wm_window_batch": 2,
                "imagine_batch": 2,
                "action_noise_std": 0.1,
                "G_max": 10.0,
                "lr": 2e-4,
                "lr_decay_updates": 10,
                "action_dim": 29,
                "max_episode_seconds": 10.0,
                "termination_body": "pelvis",
            },
            "env": {
                "motion_format": "lafan1",
                "motion_path": synthetic_clip_dir(),
                "num_clips": 2,
                "backend": "analytic",
                "num_envs": 4,
                "replay_length": 32,
            },
        }
    )
    torch.manual_seed(0)
    trainer = SuperMimicTrainer(cfg, device=torch.device("cpu"))
    frames = trainer.obs_hist.frames
    # Reset fill: one noised frame repeated over the window.
    assert torch.equal(frames, frames[:, :1].expand_as(frames))
    for _ in range(5):
        before = trainer.obs_hist.frames.clone()
        trainer.collect_steps(1, accumulate=False)
        after = trainer.obs_hist.frames
        # Stored frames shift by one and are not redrawn.
        assert torch.equal(after[:, :-1], before[:, 1:])
        assert not torch.equal(after[:, -1], before[:, -1])


@needs_warp
def test_smoke_three_updates_and_checkpoint(tmp_path):
    cfg = small_config()
    _write_asset_pose_clip(tmp_path / "pose")
    cfg.env.motion_path = str(tmp_path / "pose")
    cfg.env.num_clips = 1
    cfg = validate_supermimic_cfg(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = SuperMimicTrainer(cfg, device=device)
    metrics = trainer.run_until_updates(3)

    assert int(metrics["train/global_update"]) == 3
    assert metrics["buffer/valid_windows"] > 0
    assert math.isfinite(metrics["wm/state_loss"])
    assert math.isfinite(metrics["actor/objective"])
    assert math.isfinite(metrics["critic/loss"])
    assert math.isfinite(metrics["wm/grad_norm"]) and metrics["wm/grad_norm"] > 0
    assert math.isfinite(metrics["actor/grad_norm"]) and metrics["actor/grad_norm"] > 0
    assert 0.0 <= metrics["imagination/survival"] <= 1.0
    assert math.isfinite(metrics["sim/terminal_rate"])
    # Torch, Warp and whole-device counters are reported separately.
    assert (
        0
        < metrics["memory/torch_peak_allocated"]
        <= metrics["memory/torch_peak_reserved"]
    )
    assert metrics["memory/warp_mempool_peak"] > 0
    assert metrics["memory/torch_peak_reserved"] < metrics["memory/device_used"]
    assert metrics["memory/device_used"] <= metrics["memory/device_total"]
    assert metrics["time/replay_sample_s"] > 0.0
    assert metrics["time/world_model_s"] > 0.0
    assert metrics["time/bptt_s"] > 0.0
    assert metrics["time/critic_s"] > 0.0
    assert metrics["time/sim_s"] > 0.0
    assert metrics["wm_optimizer_steps"] >= 10
    assert metrics["actor_optimizer_steps"] >= 3

    ckpt = tmp_path / "global_step_3"
    trainer.save_checkpoint(str(ckpt))
    assert (ckpt / "supermimic.pt").is_file()
    saved_bins = trainer.reset_sampler.bin_return.clone()
    trainer.reset_sampler.reset_statistics()
    trainer.load_checkpoint(str(ckpt))
    assert trainer.global_update == 3
    assert trainer.replay_warm is False
    # Reset-bin statistics are saved with the weights; replay is not.
    assert torch.equal(trainer.reset_sampler.bin_return, saved_bins)
    steps_before = trainer.sim_vector_steps
    acc_before = trainer.step_acc
    trainer.run_iteration()
    # Warmup refills the new buffer and must not add to the UTD accumulator.
    # Only the one post-warmup collection step does.
    accounted = trainer.step_acc + (trainer.global_update - 3)
    assert trainer.replay_warm is True
    assert trainer.sim_vector_steps == steps_before + trainer.warmup + 1
    assert math.isclose(
        accounted, acc_before + trainer.updates_per_env_step, abs_tol=1e-6
    )

    eval_metrics = trainer.evaluate()
    for key in (
        "eval/normalized_return",
        "eval/torso_pos_err",
        "eval/free_root_pos_err",
        "eval/free_root_xy_drift",
        "eval/free_root_yaw_drift",
        "eval/fail_rate",
        "eval/fail_time_s",
        "eval/episode_length_s",
        "eval/clip_end_rate",
    ):
        assert math.isfinite(eval_metrics[key])
    assert eval_metrics["eval/num_clips"] == pytest.approx(1.0)

    # A new trainer must warm the new buffer. The restored step count is not
    # evidence that the empty replay is ready.
    del trainer
    torch.cuda.empty_cache()
    resumed = SuperMimicTrainer(cfg, device=device)
    resumed.load_checkpoint(str(ckpt))
    assert resumed.replay_warm is False
    restored_steps = resumed.sim_vector_steps
    resumed.run_iteration()
    assert resumed.replay_warm is True
    assert resumed.sim_vector_steps == restored_steps + resumed.warmup + 1


@pytest.fixture
def config_factory(monkeypatch):
    hydra = pytest.importorskip("hydra")
    monkeypatch.setenv("EMBODIED_PATH", str(CONFIG_DIR.parent))

    def compose(name="supermimic_lafan1_zita", overrides=()):
        directory = (
            CONFIG_DIR
            if name == "supermimic_lafan1_zita"
            else REPO / "evaluations/supermimic"
        )
        with hydra.initialize_config_dir(config_dir=str(directory), version_base="1.1"):
            return hydra.compose(config_name=name, overrides=list(overrides))

    return compose


def test_main_recipe_and_evaluators_keep_their_checkpoint_interfaces(config_factory):
    train = validate_supermimic_cfg(
        config_factory(overrides=["env.motion_path=/tmp/lafan1"])
    )
    evaluate = validate_supermimic_cfg(
        config_factory("supermimic_lafan1_zita_eval", ["env.motion_path=/tmp/lafan1"])
    )
    push = validate_supermimic_cfg(config_factory("supermimic_push_recovery"))
    assert train.env.num_envs == 4096
    assert train.algorithm.total_updates == 16000
    assert train.runner.save_interval == 2000
    assert train.algorithm.warmup_sim_steps == 64
    assert train.algorithm.wm_only_updates == 500
    assert train.env.com_sampling == "per_env"
    expected = {
        "goal_encoding": "goal730",
        "goal_dim": 730,
        "obs_dim": 99,
        "reward_anchor": "pelvis",
        "input_norm": "running",
        "residual_frame": "body",
        "wm_state_loss": "running",
        "critic_features": "zita",
        "matmul": "bf16",
        "termination_head": "analytic",
    }
    for key, value in expected.items():
        assert train.algorithm[key] == evaluate.algorithm[key] == value
    assert train.algorithm.wm_window_batch == 2048
    assert train.algorithm.imagine_batch == 4096
    assert evaluate.runner.only_eval and evaluate.env.num_envs == 40
    assert evaluate.eval.checkpoint is None
    assert (push.algorithm.goal_dim, push.algorithm.obs_dim) == (680, 93)
    assert push.algorithm.input_norm == "none"
    assert push.env.num_envs == 1 and not push.eval.motion


@pytest.mark.parametrize(
    "override,match",
    [
        ("algorithm.k=7", "algorithm.k"),
        ("algorithm.goal_anchor=true", "does not combine with the anchor"),
        ("algorithm.input_norm=invalid", "input_norm"),
        ("algorithm.matmul=invalid", "matmul"),
        ("env.motion_path=", "motion_path"),
    ],
)
def test_config_rejects_invalid_overrides(config_factory, override, match):
    cfg = config_factory(overrides=["env.motion_path=/tmp/lafan1", override])
    with pytest.raises((AssertionError, ValueError), match=match):
        validate_supermimic_cfg(cfg)


def test_global_batches_split_over_ranks_without_changing_the_recipe(config_factory):
    cfg = config_factory(
        overrides=[
            "env.motion_path=/tmp/lafan1",
            "cluster.component_placement.supermimic=0-1",
        ]
    )
    before = OmegaConf.to_container(cfg.algorithm, resolve=True)
    validated = validate_supermimic_cfg(cfg)
    assert validated.algorithm.world_size == 2
    assert validated.algorithm.k == before["k"]
    assert validated.algorithm.H == before["H"]
    cfg = config_factory(overrides=["env.motion_path=/tmp/lafan1", "env.num_envs=3"])
    with pytest.raises(AssertionError, match="divisible"):
        validate_supermimic_cfg(cfg, world_size=2)


@pytest.mark.parametrize(
    "name,value,match",
    [
        ("velocity", "bogus", "none, smoke or g2"),
        ("g2_trials", 0, "g2_trials"),
        ("g2_batch", 0, "g2_batch"),
        ("g2_steady_seconds", 6.0, "g2_steady_seconds"),
        ("velocity_smoke_seconds", 0.0, "velocity_smoke_seconds"),
        ("seed", "abc", "seed"),
    ],
)
def test_eval_options_fail_before_loading_a_checkpoint(
    config_factory, name, value, match
):
    from rlinf.algorithms.supermimic.validate import validate_supermimic_eval_cfg

    cfg = config_factory("supermimic_lafan1_zita_eval")
    cfg.eval[name] = value
    with pytest.raises((AssertionError, ValueError), match=match):
        validate_supermimic_eval_cfg(cfg.eval)


@pytest.mark.parametrize("explicit", [False, True])
def test_launcher_defaults_to_small_batches_without_changing_explicit_runs(
    tmp_path, monkeypatch, explicit
):
    binary = tmp_path / "bin"
    binary.mkdir()
    capture = tmp_path / "args.json"
    stub = binary / "python"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['SUPERMIMIC_TEST_ARGS']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binary}:{os.environ['PATH']}")
    monkeypatch.setenv("SUPERMIMIC_TEST_ARGS", str(capture))
    command = ["bash", str(REPO / "examples/embodiment/run_supermimic.sh")]
    if explicit:
        command += ["supermimic_lafan1_zita", f"runner.logger.log_path={tmp_path}"]
    subprocess.run(command, check=True, capture_output=True, text=True)
    arguments = json.loads(capture.read_text())
    assert arguments[1:3] == ["--config-name", "supermimic_lafan1_zita"]
    if explicit:
        assert not any(arg.startswith("env.num_envs=") for arg in arguments)
    else:
        for value in [
            "env.num_envs=8",
            "env.num_clips=1",
            "env.replay_length=64",
            "algorithm.total_updates=3",
            "algorithm.imagine_batch=8",
            "algorithm.wm_window_batch=64",
        ]:
            assert value in arguments


def test_reset_sampling_prefers_the_low_return_bin():
    sampler = BinResetSampler(
        torch.tensor([100]), 1, 10.0, 0.99, 500, torch.device("cpu")
    )
    sampler.bin_return[0] = 0.0
    expected = _expected_mixture(sampler, update_memory=True)
    torch.manual_seed(0)
    _motion_id, motion_time = sampler.sample(8000)
    fraction = float((motion_time < 50).float().mean())
    assert fraction > 0.55
    assert fraction == pytest.approx(float(expected[0]), abs=0.03)
    # The other bin is the rest of the 100-frame clip.
    assert float((motion_time >= 50).float().mean()) == pytest.approx(
        float(expected[1]), abs=0.03
    )


def _expected_mixture(sampler: BinResetSampler, update_memory: bool) -> torch.Tensor:
    hard = torch.softmax(-TEMPERATURE * sampler.bin_return / sampler.g_max, dim=0)
    memory = sampler.bin_memory
    if update_memory:
        memory = (1.0 - MEMORY_EMA) * memory + MEMORY_EMA * hard
    probs = (
        UNIFORM_PROB / sampler.num_bins
        + ADAPTIVE_PROB * hard
        + MEMORY_PROB * memory / memory.sum()
    )
    return probs / probs.sum()
