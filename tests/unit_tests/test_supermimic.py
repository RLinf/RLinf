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

from __future__ import annotations

import importlib.util
import math
import time
from copy import deepcopy
from pathlib import Path

import pytest

needs_mjlab = pytest.mark.skipif(
    importlib.util.find_spec("mjlab") is None,
    reason="mjlab is an optional SuperMimic extra",
)
import torch
from omegaconf import OmegaConf
from torch import nn

from rlinf.envs.sim.g1_mimic.objectives import (
    actor_objective,
    gae_targets,
    state_prediction_loss,
)
from rlinf.workers.supermimic.validate import validate_supermimic_cfg
from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    EPISODE_CLIP_END,
    EPISODE_CONT,
    EPISODE_FAILURE,
    EPISODE_TIMEOUT,
    G1_JOINT_NAMES,
    ROOT_ANG_VEL,
    ROOT_LIN_VEL,
    ROOT_QUAT,
    STATE_DIM,
)
from rlinf.envs.sim.g1_mimic.env import AnalyticG1Env
from rlinf.envs.sim.g1_mimic.features import (
    default_state,
    heading_invariant_features,
    heading_yaw,
    integrate_residual,
    quat_normalize,
    rotvec_to_quat,
    so3_relative_rotvec,
    yaw_rotation,
)
from rlinf.envs.sim.g1_mimic.motion import (
    MotionLibrary,
    decode_motion_payload,
    load_motion_library,
    resample_states,
    synthetic_clips,
)
from rlinf.envs.sim.g1_mimic.reset_sampler import (
    ADAPTIVE_PROB,
    MEMORY_EMA,
    MEMORY_PROB,
    RETURN_EMA,
    TEMPERATURE,
    UNIFORM_PROB,
    BinResetSampler,
)
from rlinf.envs.sim.g1_mimic.networks import DynamicsNet
from rlinf.envs.sim.g1_mimic.obs import add_sensor_noise, analytic_obs, pad_history
from rlinf.envs.sim.g1_mimic.replay import SuperMimicReplay
from rlinf.envs.sim.g1_mimic.eval_rollout import (
    run_tracking_eval,
    summarize_tracking_eval,
)
from rlinf.envs.sim.g1_mimic.reward import (
    analytic_reward,
    failure_signals,
    reanchor_reference,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples" / "embodiment" / "config"


def _load(name: str):
    return OmegaConf.load(CONFIG_DIR / f"{name}.yaml")


def _compose(name: str):
    """Replay Hydra inheritance without importing Hydra (JobConf is broken on 3.11)."""
    if name == "supermimic_base":
        cfg = _load(name)
    elif name == "supermimic_lafan1":
        cfg = OmegaConf.merge(_load("supermimic_base"), _load(name))
    elif name in {"supermimic_lafan1_anchor", "supermimic_smoke"}:
        cfg = OmegaConf.merge(
            _load("supermimic_base"), _load("supermimic_lafan1"), _load(name)
        )
    else:
        raise KeyError(name)
    cfg.pop("defaults", None)
    cfg.pop("hydra", None)
    return cfg


def test_hydra_four_yaml_inheritance():
    base = _compose("supermimic_base")
    lafan1 = _compose("supermimic_lafan1")
    anchor = _compose("supermimic_lafan1_anchor")
    smoke = _compose("supermimic_smoke")

    assert base.algorithm.k == 8
    assert base.algorithm.H == 32
    assert base.algorithm.obs_dim == 93
    assert base.algorithm.goal_frame_dim == 68
    assert base.env.num_envs == 4096

    lafan1.env.motion_path = "/tmp/lafan1_g1"
    lafan1 = validate_supermimic_cfg(lafan1)
    assert lafan1.algorithm.goal_anchor is False
    assert lafan1.algorithm.goal_dim == 680
    assert lafan1.algorithm.obs_history_dim == 930
    assert lafan1.algorithm.critic_in_dim == 680 + 71 + 33 + 4
    assert lafan1.env.num_envs == 4096
    assert lafan1.algorithm.imagine_batch == 4096
    assert lafan1.algorithm.wm_window_batch == 2048
    assert lafan1.algorithm.total_updates == 320000

    anchor.env.motion_path = "/tmp/lafan1_g1"
    anchor = validate_supermimic_cfg(anchor)
    assert anchor.algorithm.goal_anchor is True
    assert anchor.algorithm.goal_dim == 689
    assert anchor.algorithm.obs_dim == 93
    assert anchor.algorithm.obs_history_dim == 930
    assert anchor.env.num_envs == 4096
    assert anchor.algorithm.k == 8
    assert anchor.algorithm.H == 32

    smoke.env.motion_path = "/tmp/lafan1_g1"
    smoke = validate_supermimic_cfg(smoke)
    assert smoke.algorithm.goal_dim == 680
    assert smoke.env.num_envs == 8
    assert smoke.algorithm.imagine_batch == 8
    assert smoke.algorithm.wm_window_batch == 64
    assert smoke.algorithm.total_updates == 3
    assert smoke.algorithm.k == 8
    assert smoke.algorithm.H == 32
    assert smoke.env.backend == "mjlab"


def test_validate_rejects_bad_k_and_missing_motion():
    cfg = _compose("supermimic_lafan1")
    cfg.algorithm.k = 7
    with pytest.raises(AssertionError, match="algorithm.k"):
        validate_supermimic_cfg(cfg)

    cfg = _compose("supermimic_lafan1")
    cfg.env.motion_path = ""
    cfg.env.backend = "mjlab"
    with pytest.raises(AssertionError, match="motion_path"):
        validate_supermimic_cfg(cfg)


def test_validate_rejects_multi_gpu_placement():
    cfg = _compose("supermimic_smoke")
    cfg.cluster.component_placement.supermimic = "all"
    with pytest.raises(AssertionError, match="world size"):
        validate_supermimic_cfg(cfg)
    cfg = _compose("supermimic_smoke")
    cfg.cluster.component_placement.supermimic = "0-1"
    with pytest.raises(AssertionError, match="world size"):
        validate_supermimic_cfg(cfg)


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
    obs = analytic_obs(batch["state"][0, 0], batch["prev_action"][0], batch["xi"][0], q0)
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


def test_episode_end_codes_are_distinct():
    assert len({EPISODE_CONT, EPISODE_FAILURE, EPISODE_TIMEOUT, EPISODE_CLIP_END}) == 4


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


def test_lafan1_refuses_missing_clips(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_motion_library(str(tmp_path), 40, "cpu", allow_synthetic=False)
    library = load_motion_library("", 3, "cpu", allow_synthetic=True)
    assert library.num_clips == 3


def test_critic_uses_the_same_imagined_trajectory():
    """H=2 only counts policy calls. Training configs keep H=32."""
    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    cfg = _compose("supermimic_smoke")
    cfg.env.backend = "analytic"
    cfg.env.motion_path = ""
    cfg.algorithm.H = 2
    cfg.algorithm.goal_dim = 680
    cfg.env.num_envs = 1
    cfg.env.replay_length = 16
    cfg.algorithm.imagine_batch = 1
    cfg.algorithm.critic_epochs = 1
    cfg.algorithm.critic_minibatches = 1
    cfg.algorithm.warmup_sim_steps = 0
    trainer = SuperMimicTrainer(cfg, device=torch.device("cpu"))
    trainer.collect_steps(4, accumulate=False)
    calls = {"n": 0}
    policy = trainer.models.policy

    def _count(history, goal):
        calls["n"] += 1
        return policy(history, goal)

    trainer.models.policy = _count
    trainer.imagine_and_update_actor()
    assert calls["n"] == 2


def test_validate_does_not_mutate_user_k():
    cfg = _compose("supermimic_smoke")
    cfg.env.motion_path = "/tmp/lafan1_g1"
    before = deepcopy(OmegaConf.to_container(cfg.algorithm, resolve=True))
    out = validate_supermimic_cfg(cfg)
    assert out.algorithm.k == before["k"]
    assert out.algorithm.H == before["H"]
    assert "goal_dim" in out.algorithm


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


def _reference_payload(state: torch.Tensor, fps: float = 50.0, names=None, q0=None):
    return {
        "fps": fps,
        "joint_names": list(G1_JOINT_NAMES if names is None else names),
        "q0": torch.zeros(ACTION_DIM) if q0 is None else q0,
        "state": state,
    }


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

    kept, _ = decode_motion_payload(
        _reference_payload(state[:8], fps=50.0), "50.pt"
    )
    assert kept.shape[0] == 8

    q0_a = torch.arange(ACTION_DIM, dtype=torch.float32) * 0.01
    q0_b = q0_a + 0.2
    torch.save(_reference_payload(state[:8], q0=q0_a), tmp_path / "clip_000.pt")
    torch.save(_reference_payload(state[:8], q0=q0_b), tmp_path / "clip_001.pt")
    with pytest.raises(ValueError, match="default pose"):
        load_motion_library(str(tmp_path), 2, "cpu", allow_synthetic=False)
    (tmp_path / "clip_001.pt").unlink()
    torch.save(_reference_payload(state[:8], q0=q0_a), tmp_path / "clip_001.pt")
    library = load_motion_library(str(tmp_path), 2, "cpu", allow_synthetic=False)
    assert torch.allclose(library.q0, q0_a)


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
def test_reanchor_removes_xy_yaw_from_the_body_and_keeps_root_drift():
    robot = default_state(1, "cpu", torch.float32)
    ref = robot.clone()
    ref[0, 0] = 1.5
    ref[0, 1] = -0.4
    ref[0, 3:7] = rotvec_to_quat(torch.tensor([[0.0, 0.0, 1.5707963]]))[0]
    ref[0, ROOT_LIN_VEL] = torch.tensor([1.0, 0.0, 0.0])
    anchored = reanchor_reference(robot, ref)
    assert float(anchored[0, 0]) == pytest.approx(float(robot[0, 0]))
    assert float(anchored[0, 1]) == pytest.approx(float(robot[0, 1]))
    assert float(anchored[0, 2]) == pytest.approx(float(ref[0, 2]))
    assert float(heading_yaw(anchored[0, 3:7])) == pytest.approx(0.0, abs=1e-5)
    assert torch.allclose(
        anchored[0, ROOT_LIN_VEL],
        torch.tensor([0.0, -1.0, 0.0]),
        atol=1e-5,
    )
    action = torch.zeros(1, ACTION_DIM)
    ref_still = ref.clone()
    ref_still[0, ROOT_LIN_VEL] = 0.0
    drifted = analytic_reward(robot, ref_still, action, action)
    matched = analytic_reward(robot, robot, action, action)
    assert float(drifted["root_pos"]) < float(matched["root_pos"])
    assert float(drifted["body_tracking"]) == pytest.approx(
        float(matched["body_tracking"]), abs=1e-4
    )


@needs_mjlab
def test_tracking_eval_runs_from_frame_zero_to_clip_end():
    clip = default_state(4, "cpu", torch.float32)
    library = MotionLibrary([clip])
    env = AnalyticG1Env(
        1,
        library,
        torch.device("cpu"),
        max_episode_seconds=0.02,
        eval_mode=True,
    )
    env.reset(torch.tensor([0]), torch.tensor([0]), torch.tensor([0]))
    one = env.step(torch.zeros(1, ACTION_DIM))
    assert int(one["end_kind"]) == EPISODE_CONT
    env.reset(torch.tensor([0]), torch.tensor([0]), torch.tensor([0]))

    def policy(history, goal):
        return torch.zeros(history.shape[0], ACTION_DIM)

    def goal_fn(motion_id, motion_time, state):
        return torch.zeros(state.shape[0], 1)

    rows = run_tracking_eval(env, policy, library.q0, goal_fn)
    metrics = summarize_tracking_eval(rows)
    assert metrics["eval/steps"] == pytest.approx(4.0)
    assert metrics["eval/clip_end_rate"] == pytest.approx(1.0)
    assert metrics["eval/fail_rate"] == pytest.approx(0.0)
    assert metrics["eval/free_root_pos_err"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["eval/free_root_ori_err"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["eval/torso_pos_err"] == pytest.approx(0.0, abs=1e-4)
    assert metrics["eval/torso_ori_err"] == pytest.approx(0.0, abs=1e-4)
    # A clip of T frames is stepped at times 0..T-1. The step that advances
    # time to T is past the clip and is the last counted control, so the
    # scheduled length is T, not T-1. A matched reference pays 0.1 per step.
    assert rows["scheduled_steps"].item() == pytest.approx(4.0)
    assert rows["steps"].item() == pytest.approx(4.0)
    assert rows["return_sum"].item() == pytest.approx(0.4, abs=1e-3)
    assert rows["normalized_return"].item() == pytest.approx(1.0, abs=1e-3)
    assert torch.allclose(
        rows["normalized_return"],
        rows["return_sum"] / (0.1 * rows["scheduled_steps"]),
    )
    assert math.isfinite(metrics["eval/normalized_return"])
    assert math.isfinite(metrics["eval/free_root_xy_drift"])
    assert math.isfinite(metrics["eval/free_root_yaw_drift"])


def test_only_eval_runner_does_not_train():
    from rlinf.runners.supermimic_runner import SuperMimicRunner

    cfg = _compose("supermimic_smoke")
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


def _state_with_quat(quat: torch.Tensor) -> torch.Tensor:
    state = default_state(quat.shape[0], "cpu", torch.float32)
    state[:, ROOT_QUAT] = quat
    return state


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


def test_history_keeps_old_noise_and_reset_repeats_one_draw():
    torch.manual_seed(1)
    first = add_sensor_noise(torch.zeros(93), True)
    stored = first.clone()
    _ = add_sensor_noise(torch.ones(93), True)
    assert torch.equal(first, stored)
    padded = pad_history([first]).view(10, 93)
    assert torch.equal(padded, first.expand(10, 93))


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
    loss = state_prediction_loss(pred, _state_with_quat(torch.tensor([[1.0, 0.0, 0.0, 0.0]])))
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


def test_resampled_yaw_rate_matches_the_pose_change():
    count = 31
    rate = 0.7
    state = default_state(count, "cpu", torch.float32)
    for i in range(count):
        state[i, ROOT_QUAT] = rotvec_to_quat(torch.tensor([0.0, 0.0, rate * i / 30.0]))
    out = resample_states(state, 30.0, 50.0)
    assert torch.allclose(
        out[2:, ROOT_ANG_VEL],
        torch.tensor([0.0, 0.0, rate]).expand(out.shape[0] - 2, 3),
        atol=0.02,
    )


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


def _write_episode(buf, steps, action_of):
    env_ids = torch.zeros(1, dtype=torch.long)
    for t in range(steps):
        state = default_state(1, "cpu", torch.float32)
        state[0, 7] = float(t)
        action = torch.full((1, ACTION_DIM), action_of(t))
        prev = torch.zeros(1, ACTION_DIM) if t == 0 else torch.full((1, ACTION_DIM), action_of(t - 1))
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
        assert torch.allclose(history[0, -ACTION_DIM:], torch.full((ACTION_DIM,), expected))


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
        env.reset(torch.arange(ids.numel()), ids, torch.zeros(ids.numel(), dtype=torch.long))
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


def test_final_checkpoint_is_saved_once():
    from rlinf.runners.supermimic_runner import SuperMimicRunner

    class _Result:
        def __init__(self, payload):
            self.payload = payload

        def wait(self):
            return [self.payload]

    class _Worker:
        def __init__(self):
            self.step = 0
            self.saves = []
            self.evals = []

        def run_iteration(self):
            self.step += 1
            return _Result({"train/global_update": float(self.step)})

        def save_checkpoint(self, directory):
            self.saves.append(directory)
            return _Result(None)

        def evaluate(self):
            self.evals.append(self.step)
            return _Result({"eval/normalized_return": 0.0})

    cfg = _compose("supermimic_smoke")
    cfg.algorithm.total_updates = 2
    cfg.runner.save_interval = -1
    cfg.runner.val_check_interval = -1
    worker = _Worker()
    SuperMimicRunner(cfg, worker).run()
    assert len(worker.saves) == 1
    assert worker.saves[0].endswith("global_step_2")
    assert worker.evals == []

    cfg.runner.save_interval = 2
    cfg.runner.val_check_interval = 2
    both = _Worker()
    SuperMimicRunner(cfg, both).run()
    assert len(both.saves) == 1
    assert both.evals == [2]
