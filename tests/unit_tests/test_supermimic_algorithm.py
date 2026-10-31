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

"""SuperMimic objectives, networks, replay and return-based reset sampling."""

from __future__ import annotations

import importlib.util
import math
import time

import pytest
import torch
from torch import nn

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
from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    EPISODE_CLIP_END,
    EPISODE_CONT,
    EPISODE_FAILURE,
    EPISODE_TIMEOUT,
    ROOT_QUAT,
    STATE_DIM,
    XI_DIM,
    critic_input_dim,
)
from rlinf.envs.sim.g1_mimic.features import (
    default_state,
    heading_invariant_features,
    integrate_residual,
    root_error_4d,
    rotvec_to_quat,
    so3_relative_rotvec,
)
from rlinf.envs.sim.g1_mimic.obs import add_sensor_noise, analytic_obs
from rlinf.envs.sim.g1_mimic.reward import analytic_reward, motion_critic_input
from rlinf.models.embodiment.supermimic import (
    DynamicsNet,
    MotionEncoder,
    SuperMimicModels,
)

needs_mjlab = pytest.mark.skipif(
    importlib.util.find_spec("mjlab") is None,
    reason="mjlab is an optional SuperMimic extra",
)


CPU = torch.device("cpu")


def _ema(ret: float, g_max: float = 10.0) -> float:
    return (1.0 - RETURN_EMA) * g_max + RETURN_EMA * ret


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


def _state_with_quat(quat: torch.Tensor) -> torch.Tensor:
    state = default_state(quat.shape[0], "cpu", torch.float32)
    state[:, ROOT_QUAT] = quat
    return state


def _write_episode(buf, steps, action_of):
    env_ids = torch.zeros(1, dtype=torch.long)
    for t in range(steps):
        state = default_state(1, "cpu", torch.float32)
        state[0, 7] = float(t)
        action = torch.full((1, ACTION_DIM), action_of(t))
        prev = (
            torch.zeros(1, ACTION_DIM)
            if t == 0
            else torch.full((1, ACTION_DIM), action_of(t - 1))
        )
        obs = analytic_obs(state, prev, torch.zeros(1, 33))
        buf.add(
            env_ids,
            state,
            state,
            action,
            action + 0.25,
            torch.zeros(1),
            torch.tensor([EPISODE_CONT]),
            torch.zeros(1, 33),
            torch.zeros(1, dtype=torch.long),
            torch.tensor([t]),
            torch.tensor([t]),
            prev_action=prev,
            obs_noisy=obs,
        )


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


@needs_mjlab
def test_replay_window_and_prev_action():
    buf = SuperMimicReplay(1, 16, torch.device("cpu"))
    env_ids = torch.zeros(1, dtype=torch.long)
    q0 = torch.zeros(ACTION_DIM)
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
    obs = analytic_obs(
        batch["state"][0, 0], batch["prev_action"][0], batch["xi"][0], q0
    )
    recon = analytic_reward(
        batch["state"][0, 1],
        batch["state"][0, 1],
        batch["action_policy"][0, 0],
        batch["prev_action"][0],
    )
    assert obs.shape[-1] == 93
    assert recon["action_smooth"].ndim == 0
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


def test_dynamics_width_matches_table_a4():
    linears = [module for module in DynamicsNet().net if isinstance(module, nn.Linear)]
    assert [layer.out_features for layer in linears[:-1]] == [2048, 2048, 2048]


def test_replay_history_ignores_overwritten_slots():
    buf = SuperMimicReplay(1, 16, torch.device("cpu"))
    env_ids = torch.zeros(1, dtype=torch.long)
    q0 = torch.zeros(ACTION_DIM)
    for t in range(20):
        state = default_state(1, "cpu", torch.float32)
        state[0, 7] = float(t)
        action = torch.full((1, ACTION_DIM), float(t + 1))
        buf.add(
            env_ids,
            state,
            state,
            action,
            action,
            torch.zeros(1),
            torch.tensor([EPISODE_CONT]),
            torch.zeros(1, 33),
            torch.zeros(1, dtype=torch.long),
            torch.tensor([t]),
            torch.tensor([t]),
        )
    history = buf.history_at(0, 4, q0).view(10, 93)
    assert torch.allclose(history[-1, 0], torch.tensor(4.0))
    assert not torch.any(torch.isclose(history[:, 0], torch.tensor(19.0)))
    assert torch.equal(buf.prev_action_at(0, 4), torch.zeros(ACTION_DIM))


def test_reset_bins_are_uniform_when_returns_match():
    sampler = BinResetSampler(
        torch.tensor([100, 100]), 2, 10.0, 0.99, 500, torch.device("cpu")
    )
    assert sampler.num_bins == 4
    probs = sampler.mixture(update_memory=False)
    assert torch.allclose(probs, torch.full_like(probs, 0.25))
    before = sampler.bin_memory.clone()
    sampler.mixture(update_memory=False)
    assert torch.equal(sampler.bin_memory, before)


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


def test_reset_return_ema_and_timeout_bootstrap():
    def fresh() -> BinResetSampler:
        return BinResetSampler(
            torch.tensor([50]), 1, 10.0, 0.99, 20, torch.device("cpu")
        )

    failed = fresh()
    failed.log_step(torch.tensor([0.0]), torch.tensor([0]), torch.tensor([0]))
    failed.finish(
        torch.tensor([0]),
        torch.tensor([EPISODE_FAILURE]),
        torch.tensor([100.0]),
    )
    assert float(failed.bin_return) == pytest.approx(0.95 * 10.0)
    assert int(failed.ep_len) == 0

    timed_out = fresh()
    timed_out.log_step(torch.tensor([0.0]), torch.tensor([0]), torch.tensor([0]))
    value = 4.0
    timed_out.finish(
        torch.tensor([0]),
        torch.tensor([EPISODE_TIMEOUT]),
        torch.tensor([value]),
    )
    # One step remains, so the critic is discounted by one gamma.
    tail = 0.99 * value
    assert float(timed_out.bin_return) == pytest.approx(
        (1.0 - RETURN_EMA) * 10.0 + RETURN_EMA * tail
    )

    ended = fresh()
    ended.log_step(torch.tensor([0.0]), torch.tensor([0]), torch.tensor([0]))
    ended.finish(
        torch.tensor([0]),
        torch.tensor([EPISODE_CLIP_END]),
        torch.tensor([100.0]),
    )
    assert float(ended.bin_return) == pytest.approx(0.95 * 10.0)


def test_window_sampling_does_not_scan_candidates_in_python():
    num_envs, capacity, k = 32, 512, 8
    buf = SuperMimicReplay(num_envs, capacity, torch.device("cpu"))
    env_ids = torch.arange(num_envs)
    state = torch.zeros(num_envs, STATE_DIM)
    action = torch.zeros(num_envs, ACTION_DIM)
    for t in range(capacity):
        buf.add(
            env_ids,
            state,
            state,
            action,
            action,
            torch.zeros(num_envs),
            torch.zeros(num_envs, dtype=torch.long),
            torch.zeros(num_envs, 33),
            torch.zeros(num_envs, dtype=torch.long),
            torch.zeros(num_envs, dtype=torch.long),
            torch.full((num_envs,), t),
        )
    assert buf.num_valid_windows(k) == num_envs * (capacity - k + 1)

    def _forbid(*_args, **_kwargs):
        raise AssertionError("sample_windows scanned candidate windows in Python")

    buf.valid_window_starts = _forbid
    started = time.perf_counter()
    batch = buf.sample_windows(k, 64)
    elapsed = time.perf_counter() - started
    assert batch is not None
    assert batch["state"].shape[0] == 64
    assert elapsed < 1.0

    def _forbid_rescan(*_args, **_kwargs):
        raise AssertionError("a later batch scanned the buffer again")

    buf._window_env_phys = _forbid_rescan
    again = buf.sample_windows(k, 64)
    assert again is not None and again["state"].shape[0] == 64


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
    q0 = torch.zeros(ACTION_DIM)

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
            assert buf.sample_starts(4, q0) is not None
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
    assert buf.sample_starts(2, q0) is not None


def test_imagination_noise_does_not_stop_the_state_gradient():
    state = default_state(2, "cpu", torch.float32).requires_grad_(True)
    obs = analytic_obs(state, torch.zeros(2, ACTION_DIM), torch.zeros(2, 33))
    noisy = add_sensor_noise(obs, True)
    noisy.sum().backward()
    assert state.grad is not None
    assert torch.isfinite(state.grad).all()
    assert float(state.grad.abs().sum()) > 0.0


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


def test_paper_state_loss_is_unweighted_and_scaled_divides_blocks():
    pred = default_state(1, "cpu", torch.float32)
    target = pred.clone()
    target[..., 0] = pred[..., 0] + 0.05
    paper = state_prediction_loss(pred, target)
    scaled = state_prediction_loss(pred, target, scaled=True)
    assert float(paper) == pytest.approx(0.05**2, rel=1e-5)
    assert float(scaled) == pytest.approx(1.0, rel=1e-5)


def test_root4_critic_input_keeps_the_historical_order():
    state = default_state(2, "cpu", torch.float32)
    ref = state.clone()
    ref[..., 0] = 1.0
    goal = torch.randn(2, 680)
    xi = torch.randn(2, XI_DIM)
    got = motion_critic_input(state, goal, xi, ref, "root4")
    want = torch.cat(
        (heading_invariant_features(state), goal, xi, root_error_4d(state, ref)),
        dim=-1,
    )
    assert torch.allclose(got, want)
    assert got.shape[-1] == critic_input_dim(680, "root4")
    models = SuperMimicModels(680, critic_features="root4")
    assert models.critic.net[0].in_features == got.shape[-1]
    zita = SuperMimicModels(730, critic_features="zita")
    assert zita.critic.net[0].in_features == critic_input_dim(730, "zita")


def test_world_model_rollout_loss_is_finite_for_k8():
    torch.manual_seed(0)
    net = DynamicsNet()
    state = default_state(2, "cpu", torch.float32)
    state.requires_grad_(True)
    action = torch.zeros(2, ACTION_DIM)
    xi = torch.zeros(2, 33)
    target = default_state(2, "cpu", torch.float32)
    loss = state.new_zeros(())
    pred = state
    for _ in range(8):
        pred = integrate_residual(pred, net(pred, action, xi))
        loss = loss + state_prediction_loss(pred, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(state.grad).all()


def test_history_keeps_a3_after_the_ring_overwrites_it():
    buf = SuperMimicReplay(1, 16, torch.device("cpu"))
    _write_episode(buf, 20, lambda t: float(100 + t))
    history = buf.history_at(0, 13, torch.zeros(ACTION_DIM)).view(10, 93)
    oldest_action = history[0, -ACTION_DIM:]
    assert torch.allclose(oldest_action, torch.full((ACTION_DIM,), 103.0))
    assert not torch.allclose(oldest_action, torch.zeros(ACTION_DIM))


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
        history = buf.history_at(env, slot, torch.zeros(ACTION_DIM)).view(10, 93)
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


def test_dynamics_window_matches_the_open_loop():
    """The fused k-step loss is the same sum the trainer used to accumulate."""
    from rlinf.algorithms.supermimic.scan import make_dynamics_window
    from rlinf.envs.sim.g1_mimic.features import integrate_residual

    torch.manual_seed(0)
    dynamics = DynamicsNet()
    twin = DynamicsNet()
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
        step = state_prediction_loss(pred, targets[:, t])
        if t == 0:
            first = step
        last = step
        manual = manual + step
    window = make_dynamics_window(k, False, False, "pelvis")
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
        integrate_residual,
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


def test_running_norm_merges_split_batches_like_one_batch():
    from rlinf.models.embodiment.supermimic import RunningNorm, merge_welford_batches
    from rlinf.workers.supermimic.distributed import DataParallel

    torch.manual_seed(0)
    prior = torch.randn(5, 3)
    values = torch.randn(7, 3)
    parts = (values[:2], values[2:2], values[2:])
    whole = RunningNorm(3)
    whole.update(prior)
    whole.update(values)
    probe = RunningNorm(3)
    counts, means, variances = zip(*(probe.batch_stats(part) for part in parts))
    count, mean, var = merge_welford_batches(list(counts), list(means), list(variances))
    split = RunningNorm(3)
    split.update(prior)
    split.update_from_stats(count, mean, var)
    assert torch.allclose(split.mean, whole.mean, atol=1e-5, rtol=1e-5)
    assert torch.allclose(split.var, whole.var, atol=1e-5, rtol=1e-5)
    assert torch.allclose(split.count, whole.count)

    local = RunningNorm(3)
    synced = RunningNorm(3)
    local.update(values)
    DataParallel(1, 0, torch.device("cpu")).update_norm(synced, values)
    assert torch.equal(synced.mean, local.mean)
    assert torch.equal(synced.var, local.var)
    assert torch.equal(synced.count, local.count)
    DataParallel(1, 0, torch.device("cpu")).update_norm(synced, torch.zeros(0, 3))
    assert torch.equal(synced.count, local.count)


def test_dynamics_running_norm_accepts_an_empty_transition():
    from rlinf.envs.sim.g1_mimic.constants import ACTION_DIM, STATE_DIM, XI_DIM

    net = DynamicsNet(input_norm=True, output_norm=True)
    state = torch.zeros(0, STATE_DIM)
    action = torch.zeros(0, ACTION_DIM)
    xi = torch.zeros(0, XI_DIM)
    net.update_norm(state, action, xi, state, "body")
    assert float(net.in_norm.count) == 0
    assert float(net.out_norm.count) == 0


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


def test_running_norm_modules_are_absent_from_the_paper_state_dict():
    plain = SuperMimicModels(8, velocity=True)
    running = SuperMimicModels(8, velocity=True, input_norm=True, output_norm=True)
    plain_names = set(plain.state_dict())
    running_names = set(running.state_dict())
    assert "dynamics.in_norm.mean" not in plain_names
    assert "dynamics.out_norm.mean" not in plain_names
    assert "encoder.goal_norm.mean" not in plain_names
    assert "dynamics.in_norm.mean" in running_names
    assert "dynamics.out_norm.mean" in running_names
    assert "decoder.obs_norm.mean" in running_names
    assert running.velocity_encoder.obs_norm is running.decoder.obs_norm
    assert not any(
        name.startswith("velocity_encoder.obs_norm") for name in running_names
    )


def test_normalize_xi_uses_zita_fixed_scales():
    from rlinf.envs.sim.g1_mimic.features import normalize_xi

    xi = torch.zeros(33)
    xi[0] = 0.75
    xi[1] = 0.01
    xi[30] = 0.05
    scaled = normalize_xi(xi)
    assert float(scaled[0]) == pytest.approx(0.0)
    assert float(scaled[1]) == pytest.approx(1.0)
    assert float(scaled[30]) == pytest.approx(1.0)
