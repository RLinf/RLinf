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

"""SuperMimic trainer and runner: collection, updates, time alignment, velocity task."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch
from _supermimic_support import AnalyticG1Env, synthetic_clip_dir, synthetic_clips
from omegaconf import OmegaConf
from torch import nn

from rlinf.algorithms.supermimic.networks import SuperMimicModels
from rlinf.algorithms.supermimic.reset_sampler import RETURN_EMA, BinResetSampler
from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    EPISODE_CLIP_END,
    EPISODE_CONT,
    EPISODE_FAILURE,
    EPISODE_TIMEOUT,
    G1_JOINT_NAMES,
    HISTORY_LEN,
    JOINT_POS,
    MAX_TRACKING_REWARD,
)
from rlinf.envs.sim.g1_mimic.env import compose_end_kind
from rlinf.envs.sim.g1_mimic.features import (
    default_state,
    heading_yaw,
    quat_inv,
    quat_mul,
    quat_normalize,
    quat_to_rotvec,
    yaw_rotation,
)
from rlinf.envs.sim.g1_mimic.motion import load_motion_library
from rlinf.envs.sim.g1_mimic.obs import add_sensor_noise, pad_history
from rlinf.envs.sim.g1_mimic.reward import analytic_reward, failure_signals
from rlinf.workers.supermimic.eval_rollout import RollingHistory
from rlinf.workers.supermimic.trainer import SuperMimicTrainer
from rlinf.workers.supermimic.validate import validate_supermimic_cfg

CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples" / "embodiment" / "config"

needs_mjlab = pytest.mark.skipif(
    importlib.util.find_spec("mjlab") is None,
    reason="mjlab is an optional SuperMimic extra",
)

needs_warp = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("mjlab") is None,
    reason="MuJoCo Warp tests need CUDA and mjlab",
)


def _analytic_cfg(**algorithm):
    from rlinf.workers.supermimic.validate import validate_supermimic_cfg

    cfg = OmegaConf.merge(
        *(
            OmegaConf.load(CONFIG_DIR / f"{name}.yaml")
            for name in ("supermimic_base", "supermimic_lafan1", "supermimic_smoke")
        )
    )
    cfg.pop("defaults", None)
    cfg.pop("hydra", None)
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


def _changed(before: dict[str, torch.Tensor], module: torch.nn.Module) -> bool:
    return any(
        not torch.equal(before[name], value)
        for name, value in module.state_dict().items()
    )


def _snap(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


class _Probe(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(value)))
        self.seen = []

    def forward(self, *args):
        self.seen.append(
            tuple(arg.detach().clone() for arg in args if torch.is_tensor(arg))
        )
        return self.scale * torch.ones(args[0].shape[0])


def _clip(path: Path, frames: int, z: float) -> None:
    state = default_state(frames, "cpu", torch.float32)
    state[:, 2] = z
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "fps": 50.0,
            "joint_names": list(G1_JOINT_NAMES),
            "q0": torch.zeros(29),
            "state": state,
        },
        path,
    )


def _max_change(before: dict[str, torch.Tensor], module: torch.nn.Module) -> float:
    return max(
        float((before[k] - v).abs().max()) for k, v in module.state_dict().items()
    )


def _snap_velocity_grad(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _velocity_cfg():
    return OmegaConf.create(
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
                "velocity_tracking": True,
                "grad_diagnostics": True,
                "warmup_sim_steps": 12,
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
def test_velocity_episodes_never_touch_the_motion_reset_bins():
    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    # A 10-step limit makes both tasks finish episodes within the run.
    cfg = _analytic_cfg(velocity_tracking=True, max_episode_seconds=0.2)
    trainer = SuperMimicTrainer(cfg, device=torch.device("cpu"))
    sampler = trainer.reset_sampler
    velocity = trainer.velocity_env.clone()
    assert velocity.any() and (~velocity).any()
    logged_bins: set[int] = set()
    finished_rows: list[int] = []
    draws = {"mixture": 0, "uniform": 0}
    log_step, finish = sampler.log_step, sampler.finish
    sample, sample_uniform = sampler.sample, sampler.sample_uniform

    def spy_log(reward, motion_id, motion_time, rows=None):
        active = torch.ones_like(velocity) if rows is None else rows
        logged_bins.update(
            sampler.bin_of(motion_id[active], motion_time[active]).tolist()
        )
        return log_step(reward, motion_id, motion_time, rows)

    def spy_finish(env_ids, end_kind, bootstrap):
        finished_rows.extend(env_ids.tolist())
        return finish(env_ids, end_kind, bootstrap)

    def spy_sample(n):
        draws["mixture"] += n
        return sample(n)

    def spy_uniform(n):
        draws["uniform"] += n
        return sample_uniform(n)

    sampler.log_step, sampler.finish = spy_log, spy_finish
    sampler.sample, sampler.sample_uniform = spy_sample, spy_uniform
    before = sampler.bin_return.clone()
    velocity_episodes = 0
    for _ in range(30):
        trainer.collect_steps(1, accumulate=False)
        ended = trainer.buffer.end_kind.gather(
            1, ((trainer.buffer.cursor - 1) % trainer.buffer.capacity).unsqueeze(1)
        ).squeeze(1)
        velocity_episodes += int(((ended != 0) & velocity).sum())
        assert torch.all(sampler.ep_len[velocity] == 0)
    assert velocity_episodes > 0
    assert not any(bool(velocity[row]) for row in finished_rows)
    assert draws["uniform"] == velocity_episodes
    assert draws["mixture"] == len(finished_rows)
    changed = (sampler.bin_return != before).nonzero().flatten().tolist()
    assert changed and set(changed) <= logged_bins


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
    starts = trainer.buffer.sample_starts(8, trainer.q0)
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
def test_velocity_commands_and_alignment_follow_the_reference_clock(tmp_path):
    from rlinf.envs.sim.g1_mimic.velocity import COMMAND_SCALE

    trainer = _perfect_trainer(tmp_path, velocity_tracking=True)
    motion, buffer = trainer.motion, trainer.buffer
    rows = torch.arange(trainer.num_envs)
    checked = 0
    for _ in range(30):
        mid = trainer.env.motion_id.clone()
        t = trainer.env.motion_time.clone()
        slot = buffer.cursor.clone()
        trainer.collect_steps(1, accumulate=False)
        from_ref = buffer.command_from_ref[rows, slot]
        if bool(from_ref.any()):
            expected = motion.reference_velocity_command(mid[from_ref], t[from_ref])
            assert torch.allclose(
                buffer.command[rows, slot][from_ref], expected, atol=1e-6
            )
            checked += int(from_ref.sum())
        motion_rows = ~trainer.velocity_env
        reward = buffer.reward[rows, slot][motion_rows]
        assert torch.allclose(
            reward, torch.full_like(reward, MAX_TRACKING_REWARD), atol=1e-6
        )
    assert checked > 0
    trainer.models.dynamics = _PerfectDynamics(motion)
    encoder = _Recorder(trainer.models.encoder)
    velocity_encoder = _Recorder(trainer.models.velocity_encoder)
    trainer.models.encoder = encoder
    trainer.models.velocity_encoder = velocity_encoder
    drawn = {}
    sample_starts = buffer.sample_starts

    def spy_starts(batch, q0, task=None):
        drawn[task] = sample_starts(batch, q0, task)
        return drawn[task]

    buffer.sample_starts = spy_starts
    trainer._velocity_imagine()
    starts = drawn[1]
    from_ref = starts["command_from_ref"]
    assert bool(from_ref.any())
    mid, tau0 = starts["motion_id"][from_ref], starts["motion_time"][from_ref]
    scale = torch.tensor(COMMAND_SCALE)
    assert len(velocity_encoder.inputs) == trainer.H
    assert len(encoder.inputs) == trainer.H
    for t in range(trainer.H):
        command = velocity_encoder.inputs[t][0][from_ref] * scale
        expected = motion.reference_velocity_command(mid, tau0 + t)
        assert torch.allclose(command, expected, atol=1e-5)
        goal = encoder.inputs[t][0][from_ref]
        assert torch.allclose(goal, motion.goal_ref(mid, tau0 + t), atol=1e-5)


@needs_mjlab
def test_analytic_termination_leaves_the_learned_head_untouched(tmp_path):
    from rlinf.envs.sim.g1_mimic.reward import analytic_failure_prob

    trainer = _perfect_trainer(tmp_path, termination_head="analytic")
    for _ in range(30):
        trainer.collect_steps(1, accumulate=False)
    head = {k: v.clone() for k, v in trainer.models.termination.state_dict().items()}
    trainer.fit_world_model()
    for key, value in trainer.models.termination.state_dict().items():
        assert torch.equal(value, head[key]), key
    starts = trainer.buffer.sample_starts(8, trainer.q0)
    with torch.no_grad():
        rollout = trainer._imagine_motion(starts)
    # A perfect tracker of a moving dynamics-free model: recompute d_hat at t=1.
    state, ref = (
        rollout["states"][:, 1],
        trainer.motion.state_at(starts["motion_id"], starts["motion_time"] + 1),
    )
    height, tilt, _ = failure_signals(state, ref)
    assert torch.allclose(rollout["d"][:, 0], analytic_failure_prob(height, tilt))
    assert trainer._checkpoint_meta()["termination_head"] == "analytic"


@needs_mjlab
@pytest.mark.parametrize("body", ["pelvis", "torso"])
def test_imagined_termination_uses_the_configured_body(tmp_path, body):
    from rlinf.envs.sim.g1_mimic.reward import termination_head_input

    trainer = _perfect_trainer(tmp_path, termination_body=body)
    assert trainer.termination_body == body
    for _ in range(30):
        trainer.collect_steps(1, accumulate=False)
    starts = trainer.buffer.sample_starts(8, trainer.q0)
    # Pitching the root forward keeps the pelvis height but lowers the torso,
    # so the two bodies give different height errors against one reference.
    half = torch.tensor(0.6)
    zero = torch.tensor(0.0)
    pitch = torch.stack((half.cos(), zero, half.sin(), zero)).expand(8, 4)
    starts["state"][:, 3:7] = quat_mul(starts["state"][:, 3:7], pitch)
    seen = []
    hook = trainer.models.termination.register_forward_hook(
        lambda module, inputs, output: seen.append(inputs[0].detach().clone())
    )
    with torch.no_grad():
        rollout = trainer._imagine_motion(starts)
    hook.remove()
    state = rollout["states"][:, 1]
    ref = trainer.motion.state_at(starts["motion_id"], starts["motion_time"] + 1)
    other = "torso" if body == "pelvis" else "pelvis"
    expected = termination_head_input(*failure_signals(state, ref, body)[:2])
    wrong = termination_head_input(*failure_signals(state, ref, other)[:2])
    assert not torch.allclose(expected, wrong, atol=1e-3)
    assert torch.allclose(seen[0], expected, atol=1e-6)


@needs_mjlab
def test_termination_bias_init_sets_the_failure_prior_of_the_learned_head(tmp_path):
    from rlinf.envs.sim.g1_mimic.reward import termination_head_input

    gen = torch.Generator().manual_seed(0)
    features = termination_head_input(
        0.2 * torch.randn(4096, generator=gen), 0.3 * torch.randn(4096, generator=gen)
    )
    biased = _perfect_trainer(tmp_path / "biased", termination_bias_init=-4.6)
    assert float(biased.models.termination.net[-1].bias) == pytest.approx(-4.6)
    with torch.no_grad():
        prior = torch.sigmoid(biased.models.termination(features)).mean()
    assert float(prior) < 0.03
    assert biased._checkpoint_meta()["termination_bias_init"] == -4.6
    default = _perfect_trainer(tmp_path / "default")
    with torch.no_grad():
        prior = torch.sigmoid(default.models.termination(features)).mean()
    assert 0.2 < float(prior) < 0.8
    assert default._checkpoint_meta()["termination_bias_init"] is None


@needs_mjlab
def test_tracking_eval_gives_a_perfect_tracker_full_return(tmp_path):
    from rlinf.algorithms.supermimic.networks import SuperMimicModels
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


@needs_mjlab
def test_critic_uses_the_same_imagined_trajectory():
    """H=2 only counts policy calls. Training configs keep H=32."""
    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    cfg = _compose("supermimic_smoke")
    cfg.env.backend = "analytic"
    cfg.env.motion_path = synthetic_clip_dir()
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


def test_history_keeps_old_noise_and_reset_repeats_one_draw():
    torch.manual_seed(1)
    first = add_sensor_noise(torch.zeros(93), True)
    stored = first.clone()
    _ = add_sensor_noise(torch.ones(93), True)
    assert torch.equal(first, stored)
    padded = pad_history([first]).view(10, 93)
    assert torch.equal(padded, first.expand(10, 93))


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


def test_reset_bin_timeouts_bootstrap_with_the_motion_critic_only():
    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    trainer = SuperMimicTrainer.__new__(SuperMimicTrainer)
    trainer.velocity = True
    trainer.device = torch.device("cpu")
    n = 4
    trainer.velocity_env = torch.tensor([False, True, True, False])
    trainer.env = type("Env", (), {})()
    trainer.env.state = default_state(n, "cpu", torch.float32)
    trainer.env.motion_id = torch.arange(n)
    trainer.env.motion_time = torch.tensor([17, 17, 4, 8])
    trainer.env.xi = torch.zeros(n, 33)
    motion = type("Motion", (), {})()

    def state_at(motion_id, time):
        return default_state(time.shape[0], "cpu", torch.float32)

    motion.state_at = state_at
    trainer.motion = motion
    goal_times = []

    def _goal(motion_id, motion_time, state):
        goal_times.append(motion_time.detach().clone())
        return torch.zeros(motion_id.shape[0], 680)

    trainer._goal = _goal
    motion_probe = _Probe(10.0)
    vel_probe = _Probe(1.0)
    trainer.models = type(
        "Models", (), {"critic": motion_probe, "velocity_critic": vel_probe}
    )()
    motion_ids = torch.tensor([0, 3])
    values = trainer._critic_bootstrap(motion_ids)
    assert torch.allclose(values.detach(), torch.tensor([10.0, 10.0]))
    # The step already advanced motion_time to the terminal reference frame.
    assert torch.allclose(goal_times[0], torch.tensor([17, 8]))
    assert vel_probe.seen == []
    values[0].backward()
    assert float(motion_probe.scale.grad) == pytest.approx(1.0)
    motion_probe.zero_grad(set_to_none=True)
    detached = trainer._bootstrap_value(motion_ids)
    assert not detached.requires_grad
    assert motion_probe.scale.grad is None

    lengths = torch.tensor([50, 50, 50, 50])
    sampler = BinResetSampler(lengths, n, 10.0, 0.99, 10, torch.device("cpu"))
    trainer.reset_sampler = sampler
    sampler.log_step(
        torch.zeros(n),
        trainer.env.motion_id,
        torch.zeros(n, dtype=torch.long),
        rows=~trainer.velocity_env,
    )
    assert sampler.ep_len.tolist() == [1, 0, 0, 1]
    end = torch.tensor([EPISODE_TIMEOUT, EPISODE_CLIP_END])
    trainer._update_reset_bins(motion_ids, end)
    gamma = 0.99

    def expected(ret):
        return (1.0 - RETURN_EMA) * 10.0 + RETURN_EMA * ret

    assert float(sampler.bin_return[0]) == pytest.approx(expected(gamma * 10.0))
    assert float(sampler.bin_return[3]) == pytest.approx(expected(0.0))
    # Velocity rows never logged a step, so their bins keep the initial G_max.
    assert sampler.bin_return[1:3].tolist() == [10.0, 10.0]
    assert vel_probe.seen == []
    assert int(sampler.ep_len.sum()) == 0


@needs_mjlab
def test_velocity_episode_keeps_timeout_clip_end_and_resets(tmp_path):
    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    _clip(tmp_path / "clip_000.pt", 80, 1.2)
    _clip(tmp_path / "clip_001.pt", 1, 1.2)
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
                "velocity_tracking": True,
                "warmup_sim_steps": 1,
                "updates_per_env_step": 0.4,
                "wm_batches": 1,
                "wm_window_batch": 2,
                "imagine_batch": 6,
                "bce_weight": 1.0,
                "action_noise_std": 0.0,
                "G_max": 10.0,
                "lr": 1e-4,
                "lr_decay_updates": 10,
                "action_dim": 29,
                "max_episode_seconds": 10.0,
                "termination_body": "pelvis",
            },
            "env": {
                "motion_format": "lafan1",
                "motion_path": str(tmp_path),
                "num_clips": 2,
                "backend": "analytic",
                "num_envs": 6,
                "replay_length": 16,
            },
        }
    )
    trainer = SuperMimicTrainer(cfg, device=torch.device("cpu"))

    def zeros(history, *args):
        return torch.zeros(history.shape[0], 29)

    trainer.models.policy = zeros
    trainer.models.velocity_policy = zeros
    ids = torch.arange(6)
    motion_id = torch.tensor([0, 1, 0, 0, 1, 0])
    motion_time = torch.zeros(6, dtype=torch.long)
    trainer.env.reset(ids, motion_id, motion_time)
    state = trainer.motion.state_at(motion_id, motion_time).clone()
    state[0, 2] = 0.8
    state[3, 2] = 0.8
    state[4, 2] = 0.8
    state[5, 2] = 0.2
    trainer.env.state = state
    trainer.env.motion_id = motion_id.clone()
    trainer.env.motion_time = motion_time.clone()
    trainer.env.time_in_episode.zero_()
    trainer.env.time_in_episode[0] = trainer.env.max_steps - 1
    trainer.env.time_in_episode[3] = trainer.env.max_steps - 1
    trainer._refresh_history()
    trainer.collect_steps(1, accumulate=False)
    assert trainer.buffer.end_kind[:, 0].tolist() == [
        EPISODE_FAILURE,
        EPISODE_CLIP_END,
        EPISODE_CONT,
        EPISODE_TIMEOUT,
        EPISODE_CLIP_END,
        EPISODE_FAILURE,
    ]
    finished = [0, 1, 3, 4, 5]
    assert torch.all(trainer.env.time_in_episode[finished] == 0)
    assert int(trainer.env.time_in_episode[2]) == 1
    assert trainer.buffer.next_episode_id.tolist() == [1, 1, 0, 1, 1, 1]
    assert float(trainer.buffer.next_state[3, 0, 2]) == pytest.approx(0.8, abs=1e-4)
    assert float(trainer.env.state[3, 2]) == pytest.approx(1.2, abs=1e-4)
    # A reset env's window is one fresh frame repeated; a live env shifted
    # one new frame in after the refresh.
    frames = trainer.obs_hist.frames
    assert frames.shape[1] == HISTORY_LEN
    assert torch.equal(frames[3], frames[3, -1:].expand_as(frames[3]))
    assert torch.equal(frames[2, :-1], frames[2, :1].expand(HISTORY_LEN - 1, -1))
    assert not torch.equal(frames[2, -1], frames[2, 0])
    assert int(trainer.reset_sampler.ep_len[3]) == 0
    assert int(trainer.reset_sampler.ep_len[2]) == 1

    # Motion error alone does not end a velocity episode, and the limit still does.
    trainer.env.motion_id[3] = 0
    trainer.env.motion_time[3] = 0
    trainer.env.state[3] = state[3]
    trainer.env.time_in_episode[3] = trainer.env.max_steps - 2
    trainer.collect_steps(1, accumulate=False)
    assert int(trainer.buffer.end_kind[3, 1]) == EPISODE_CONT
    assert int(trainer.env.time_in_episode[3]) == trainer.env.max_steps - 1
    assert int(trainer.buffer.next_episode_id[3]) == 1
    trainer.collect_steps(1, accumulate=False)
    assert int(trainer.buffer.end_kind[3, 2]) == EPISODE_TIMEOUT
    assert int(trainer.env.time_in_episode[3]) == 0
    # The first timeout closed episode 0. The next two steps are episode 1.
    assert int(trainer.buffer.episode_id[3, 0]) == 0
    assert int(trainer.buffer.episode_id[3, 1]) == 1
    assert int(trainer.buffer.episode_id[3, 2]) == 1
    assert int(trainer.buffer.next_episode_id[3]) == 2

    # A fall outranks the time limit. A plain tracking error does not.
    trainer.env.motion_id[3] = 0
    trainer.env.motion_time[3] = 0
    trainer.env.state[3, 2] = 0.2
    trainer.env.time_in_episode[3] = trainer.env.max_steps - 1
    before = int(trainer.buffer.next_episode_id[3])
    trainer.collect_steps(1, accumulate=False)
    assert int(trainer.buffer.end_kind[3, 3]) == EPISODE_FAILURE
    assert int(trainer.buffer.next_episode_id[3]) == before + 1
    assert int(trainer.env.time_in_episode[3]) == 0


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
    runner.metric_logger.log = lambda data, step: logged.append(step) or log(
        data, step=step
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


def test_experiments_do_not_share_a_tensorboard_directory(tmp_path):
    """The shared MetricLogger keeps ``log_path/tensorboard``.

    ``convert_pt_to_hf.py`` and the logger guide rely on that path. SuperMimic
    isolates runs through ``log_path`` itself, as ``run_embodiment.sh`` does.
    """
    from rlinf.utils.metric_logger import MetricLogger

    def logger_for(name: str) -> tuple[MetricLogger, Path]:
        log_path = tmp_path / "results" / name
        cfg = OmegaConf.create(
            {
                "runner": {
                    "per_worker_log": False,
                    "logger": {
                        "log_path": str(log_path),
                        "project_name": "rlinf",
                        "experiment_name": name,
                        "logger_backends": ["tensorboard"],
                    },
                }
            }
        )
        return MetricLogger(cfg), log_path

    first, smoke = logger_for("supermimic_smoke")
    second, velocity = logger_for("supermimic_velocity")
    try:
        smoke_cfg = smoke / "tensorboard" / "config.yaml"
        velocity_cfg = velocity / "tensorboard" / "config.yaml"
        assert smoke_cfg.is_file() and velocity_cfg.is_file()
        assert not (smoke / "tensorboard" / "supermimic_smoke").exists()
        assert "supermimic_velocity" not in smoke_cfg.read_text(encoding="utf-8")
        assert "supermimic_velocity" in velocity_cfg.read_text(encoding="utf-8")
        assert list((smoke / "tensorboard").glob("events.out.tfevents.*"))
        assert list((velocity / "tensorboard").glob("events.out.tfevents.*"))
    finally:
        first.logger["tensorboard"].finish()
        second.logger["tensorboard"].finish()


def test_vectorized_history_matches_the_list_history():
    num_envs, dim, steps = 6, 5, 40
    gen = torch.Generator().manual_seed(3)
    first = torch.randn(num_envs, dim, generator=gen)
    legacy = [[first[i]] * HISTORY_LEN for i in range(num_envs)]
    history = RollingHistory(first)
    for _ in range(steps):
        done = torch.rand(num_envs, generator=gen) < 0.15
        live = (~done).nonzero(as_tuple=True)[0]
        dead = done.nonzero(as_tuple=True)[0]
        live_frames = torch.randn(int(live.numel()), dim, generator=gen)
        reset_frames = torch.randn(int(dead.numel()), dim, generator=gen)
        for n, i in enumerate(live.tolist()):
            legacy[i] = (legacy[i] + [live_frames[n]])[-HISTORY_LEN:]
        history.push(live_frames, live)
        for n, i in enumerate(dead.tolist()):
            legacy[i] = [reset_frames[n]]
        history.fill(dead, reset_frames)
        expected = torch.stack([pad_history(frames) for frames in legacy])
        assert torch.equal(history.flat(), expected)


@needs_mjlab
def test_trainer_history_matches_the_phase2_list_code():
    from omegaconf import OmegaConf

    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    class ListHistoryTrainer(SuperMimicTrainer):
        """Phase 2 history: Python lists, one env at a time."""

        def _history_tensor(self):
            return torch.stack([pad_history(f) for f in self.obs_hist], dim=0)

        def _refresh_history(self):
            ref = self._ref_quat(self.env.motion_id, self.env.motion_time)
            self.obs_hist = []
            for i in range(self.num_envs):
                ref_i = None if ref is None else ref[i]
                frame = self._noisy_obs(
                    self.env.state[i], self.prev_action[i], self.env.xi[i], ref_i, True
                ).detach()
                self.obs_hist.append([frame] * HISTORY_LEN)

        def _append_obs(self, env_ids):
            ids = env_ids.tolist()
            if not ids:
                return
            ref = self._ref_quat(
                self.env.motion_id[env_ids], self.env.motion_time[env_ids]
            )
            for n, i in enumerate(ids):
                ref_i = None if ref is None else ref[n]
                frame = self._noisy_obs(
                    self.env.state[i], self.prev_action[i], self.env.xi[i], ref_i, True
                ).detach()
                self.obs_hist[i] = (self.obs_hist[i] + [frame])[-HISTORY_LEN:]

        def _reset_history(self, env_ids):
            for i in env_ids.tolist():
                self.obs_hist[i] = []
            ids = env_ids.tolist()
            if not ids:
                return
            ref = self._ref_quat(
                self.env.motion_id[env_ids], self.env.motion_time[env_ids]
            )
            for n, i in enumerate(ids):
                ref_i = None if ref is None else ref[n]
                frame = self._noisy_obs(
                    self.env.state[i], self.prev_action[i], self.env.xi[i], ref_i, True
                ).detach()
                self.obs_hist[i] = [frame]

    cfg = OmegaConf.create(
        {
            "algorithm": {
                "goal_anchor": False,
                "k": 8,
                "H": 32,
                "gamma": 0.99,
                "lam": 0.95,
                "goal_dim": 680,
                "obs_dim": 99,
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
                # Short episodes, so resets happen inside the comparison.
                "max_episode_seconds": 0.2,
                "termination_body": "pelvis",
            },
            "env": {
                "motion_format": "lafan1",
                "motion_path": synthetic_clip_dir(),
                "num_clips": 3,
                "backend": "analytic",
                "num_envs": 5,
                "replay_length": 64,
            },
        }
    )
    # Noise is drawn per env in the list code and per batch here. The two
    # draw orders differ, so the frame noise is switched off for the
    # bit-for-bit comparison. Noise-once semantics are checked above.
    import rlinf.workers.supermimic.trainer as trainer_module

    original_noise = trainer_module.add_sensor_noise
    trainer_module.add_sensor_noise = lambda obs, enabled: obs
    try:
        torch.manual_seed(7)
        new = SuperMimicTrainer(cfg, device=torch.device("cpu"))
        torch.manual_seed(7)
        old = ListHistoryTrainer(cfg, device=torch.device("cpu"))
        resets = 0
        for step in range(30):
            assert torch.equal(new._history_tensor(), old._history_tensor())
            before = int(new.buffer.next_episode_id.sum())
            # Both trainers share the global generator; give each step the
            # same action-noise and reset draws.
            torch.manual_seed(100 + step)
            new.collect_steps(1, accumulate=False)
            torch.manual_seed(100 + step)
            old.collect_steps(1, accumulate=False)
            resets += int(new.buffer.next_episode_id.sum()) - before
        assert resets > 0
        assert torch.equal(new._history_tensor(), old._history_tensor())
        assert torch.equal(new.buffer.obs_noisy, old.buffer.obs_noisy)
        assert torch.equal(new.buffer.state, old.buffer.state)
    finally:
        trainer_module.add_sensor_noise = original_noise


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


@needs_mjlab
def test_velocity_step_moves_only_the_velocity_encoder():
    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    torch.manual_seed(0)
    trainer = SuperMimicTrainer(_velocity_cfg(), device=torch.device("cpu"))
    trainer.collect_steps(trainer.warmup, accumulate=False)
    # Commands are redrawn every 250 steps, so a short warmup may hold no
    # reference-following start. Force them so the alignment path always runs.
    trainer.buffer.command_from_ref[:] = True
    trainer.step_acc = 1.0
    models = trainer.models
    seen: dict[str, float] = {}
    original = trainer._velocity_imagine

    def wrapped():
        frozen = {
            name: _snap_velocity_grad(getattr(models, name))
            for name in ("encoder", "decoder", "dynamics", "termination", "critic")
        }
        vel_before = _snap_velocity_grad(models.velocity_encoder)
        metrics = original()
        for name, before in frozen.items():
            seen[name] = _max_change(before, getattr(models, name))
        seen["velocity_encoder"] = _max_change(vel_before, models.velocity_encoder)
        return metrics

    trainer._velocity_imagine = wrapped
    metrics = trainer.maybe_update()

    assert metrics["train/global_update"] == 1.0
    for name in ("encoder", "decoder", "dynamics", "termination", "critic"):
        assert seen[name] == 0.0, name
    assert seen["velocity_encoder"] > 0.0

    grad = metrics["velocity/grad_norm"]
    assert math.isfinite(grad) and grad > 0.0
    parts = [
        metrics["velocity/grad_norm_reward"],
        metrics["velocity/grad_norm_bootstrap"],
        metrics["velocity/grad_norm_align"],
    ]
    assert all(math.isfinite(p) and p >= 0.0 for p in parts)
    # The total is the norm of a sum, so it cannot exceed the parts' sum.
    assert grad <= sum(parts) * (1.0 + 1e-4) + 1e-12
    assert parts[0] + parts[1] > 0.0
    assert metrics["velocity/align"] > 0.0
    assert parts[2] > 0.0
    assert 0.0 <= metrics["velocity/survival"] <= 1.0
    assert 0.0 <= metrics["velocity/fall_prob"] <= 1.0
    assert math.isfinite(metrics["velocity/imagined_lin_err"])

    # The decoder's last layer is initialized with gain 0.01. The motion
    # encoder sits behind it, so its gradient is far smaller than the
    # decoder's. actor/grad_norm mixes both; velocity/grad_norm is encoder-only.
    assert metrics["actor/encoder_grad_norm"] < metrics["actor/decoder_grad_norm"]
    assert math.isfinite(metrics["actor/encoder_grad_norm"])


@needs_warp
def test_smoke_three_updates_and_checkpoint(tmp_path):
    cfg = OmegaConf.merge(
        OmegaConf.load(CONFIG_DIR / "supermimic_base.yaml"),
        OmegaConf.load(CONFIG_DIR / "supermimic_lafan1.yaml"),
        OmegaConf.load(CONFIG_DIR / "supermimic_smoke.yaml"),
    )
    cfg.pop("defaults", None)
    cfg.pop("hydra", None)
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
