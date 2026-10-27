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

"""SuperMimic evaluation rollouts, the standalone evaluator, and the G.2 step test."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from _supermimic_support import AnalyticG1Env, synthetic_clip_dir
from omegaconf import OmegaConf

from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    EPISODE_CLIP_END,
    EPISODE_CONT,
    EPISODE_FAILURE,
    G1_JOINT_NAMES,
    HISTORY_LEN,
)
from rlinf.envs.sim.g1_mimic.features import default_state
from rlinf.envs.sim.g1_mimic.g1_fk import asset_default_state
from rlinf.envs.sim.g1_mimic.mjlab_env import MjlabG1Env
from rlinf.envs.sim.g1_mimic.motion import MotionLibrary, load_motion_library
from rlinf.envs.sim.g1_mimic.reward import analytic_reward
from rlinf.workers.supermimic.eval_rollout import (
    RollingHistory,
    run_tracking_eval,
    summarize_tracking_eval,
)

needs_mjlab = pytest.mark.skipif(
    importlib.util.find_spec("mjlab") is None,
    reason="mjlab is an optional SuperMimic extra",
)

needs_warp = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("mjlab") is None,
    reason="MuJoCo Warp tests need CUDA and mjlab",
)


def _cfg(velocity: bool = False):
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


@needs_mjlab
def test_standalone_evaluation_matches_the_trainer(tmp_path):
    from rlinf.workers.supermimic.evaluator import SuperMimicEvaluator
    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    cfg = _cfg()
    trainer = SuperMimicTrainer(cfg, device=torch.device("cpu"))
    trainer.save_checkpoint(str(tmp_path / "global_step_0"))
    expected = trainer.evaluate()

    evaluator = SuperMimicEvaluator(cfg, torch.device("cpu"))
    meta = evaluator.load(str(tmp_path / "global_step_0"))
    assert meta["goal_dim"] == 680
    summary, rows = evaluator.evaluate_motion(seed=0)
    assert rows["clip_id"].tolist() == [0, 1, 2]
    for key, value in expected.items():
        assert summary[key] == pytest.approx(value, rel=1e-6, abs=1e-7), key


@needs_mjlab
def test_velocity_evaluation_exports_rows_per_command(tmp_path):
    from rlinf.workers.supermimic.evaluator import (
        DEFAULT_VELOCITY_COMMANDS,
        SuperMimicEvaluator,
        write_eval_outputs,
    )
    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    cfg = _cfg(velocity=True)
    SuperMimicTrainer(cfg, device=torch.device("cpu")).save_checkpoint(
        str(tmp_path / "global_step_0")
    )
    evaluator = SuperMimicEvaluator(cfg, torch.device("cpu"))
    evaluator.load(str(tmp_path / "global_step_0"))
    motion, motion_rows = evaluator.evaluate_motion(seed=0)
    velocity, rows, names = evaluator.evaluate_velocity(seconds=0.2, seed=0)
    assert names == list(DEFAULT_VELOCITY_COMMANDS)
    assert rows["command"].shape == (len(names), 3)
    for key in ("lin_vel_err", "yaw_rate_err", "survival_s"):
        assert torch.isfinite(rows[key]).all(), key
    assert torch.all(rows["survival_s"] <= 0.2 + 1e-6)
    assert velocity["eval_velocity_smoke/num_commands"] == float(len(names))
    again, rows_again, _ = evaluator.evaluate_velocity(seconds=0.2, seed=0)
    assert torch.equal(rows["lin_vel_err"], rows_again["lin_vel_err"])

    # A G.2 median over commands where every trial fell is undefined.
    undefined = {"eval_velocity_g2/median_relative_error": float("nan")}
    out = write_eval_outputs(
        tmp_path / "eval",
        {**motion, **velocity, **undefined},
        {"checkpoint": "global_step_0"},
        motion_rows=motion_rows,
        clip_names=evaluator.motion.clip_names,
        velocity_rows=rows,
        command_names=names,
    )

    def reject(constant):
        raise ValueError(f"summary.json is not strict JSON: {constant}")

    summary = json.loads((out / "summary.json").read_text(), parse_constant=reject)
    assert summary["meta"]["checkpoint"] == "global_step_0"
    assert math.isfinite(summary["metrics"]["eval/normalized_return"])
    assert summary["metrics"]["eval_velocity_g2/median_relative_error"] is None
    with open(out / "per_clip.csv") as handle:
        clips = list(csv.DictReader(handle))
    assert [row["clip_name"] for row in clips] == [
        "clip_000.pt",
        "clip_001.pt",
        "clip_002.pt",
    ]
    assert [row["clip_id"] for row in clips] == ["0", "1", "2"]
    for row in clips:
        for key in ("fail", "clip_end", "fail_step", "steps", "scheduled_steps"):
            assert row[key].lstrip("-").isdigit(), (key, row[key])
    with open(out / "velocity_smoke.csv") as handle:
        commands = list(csv.DictReader(handle))
    assert [row["command_name"] for row in commands] == names
    assert float(commands[1]["command_0"]) == pytest.approx(0.5)


@needs_mjlab
def test_evaluator_refuses_a_checkpoint_of_another_config(tmp_path):
    from rlinf.workers.supermimic.evaluator import SuperMimicEvaluator
    from rlinf.workers.supermimic.trainer import SuperMimicTrainer

    SuperMimicTrainer(_cfg(), device=torch.device("cpu")).save_checkpoint(
        str(tmp_path / "global_step_0")
    )
    evaluator = SuperMimicEvaluator(_cfg(velocity=True), torch.device("cpu"))
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
    cfg = _cfg(velocity=True)
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
            "supermimic_velocity_eval",
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


def test_eval_history_stays_ten_frames():
    first = torch.zeros(2, 93)
    history = RollingHistory(first)
    for step in range(50):
        history.push(torch.full((2, 93), float(step)))
    assert history.length == HISTORY_LEN
    assert history.flat().shape[-1] == HISTORY_LEN * 93
    assert float(history.frames[0, -1, 0]) == 49.0


@needs_mjlab
def test_finished_eval_row_ignores_later_nan_state():
    env = _NanEnv()
    rows = run_tracking_eval(
        env,
        lambda history, goal: torch.zeros(history.shape[0], 29),
        torch.zeros(29),
        lambda motion_id, motion_time, state: torch.zeros(motion_id.shape[0], 680),
    )
    first = default_state(1, "cpu", torch.float32)
    expected = analytic_reward(first, first, torch.zeros(1, 29), torch.zeros(1, 29))
    assert rows["steps"].tolist() == [1.0, 4.0]
    assert float(rows["fail"][0]) == 1.0
    assert float(rows["clip_end"][1]) == 1.0
    assert float(rows["return_sum"][0]) == pytest.approx(
        float(expected["reward"]), abs=1e-5
    )
    assert torch.isfinite(rows["return_sum"]).all()
    assert torch.isfinite(rows["normalized_return"]).all()
    assert torch.isfinite(rows["body_tracking"]).all()
    summary_return = float(rows["normalized_return"].mean())
    assert math.isfinite(summary_return)


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


@needs_warp
def test_warp_step_test_starts_from_standing(tmp_path):
    from omegaconf import OmegaConf

    from rlinf.envs.sim.g1_mimic.velocity import step_test_commands
    from rlinf.workers.supermimic.evaluator import SuperMimicEvaluator

    _write_clip(tmp_path)
    standing = asset_default_state()
    motion = load_motion_library(str(tmp_path), 1, "cuda")
    env = MjlabG1Env(2, motion, torch.device("cuda"), eval_mode=True)
    env.reset_standing(torch.arange(2, device="cuda"))
    assert torch.allclose(env.state[:, 2].cpu(), standing[2].expand(2), atol=1e-4)
    assert torch.allclose(
        env.state[:, 7:36].cpu(), standing[7:36].expand(2, 29), atol=1e-4
    )
    assert float(env.state[:, 36:].abs().max()) < 1e-4

    cfg = OmegaConf.create(
        {
            "algorithm": {
                "goal_dim": 680,
                "obs_dim": 93,
                "goal_encoding": "paper",
                "goal_anchor": False,
                "velocity_tracking": True,
                "termination_body": "pelvis",
            },
            "env": {
                "backend": "mjlab",
                "motion_format": "lafan1",
                "motion_path": str(tmp_path),
                "num_clips": 1,
                "num_envs": 4,
            },
        }
    )
    evaluator = SuperMimicEvaluator(cfg, torch.device("cuda"))
    commands, names, planes = step_test_commands()
    subset = (commands[:3], names[:3], planes[:3])
    summary, rows, out_names = evaluator.evaluate_velocity_g2(
        seed=0, trials=2, seconds=0.2, steady_seconds=0.1, batch=4, commands=subset
    )
    assert out_names == names[:3]
    assert rows["trial_0_relative_error"].shape == (3,)
    errors = torch.stack(
        (rows["trial_0_relative_error"], rows["trial_1_relative_error"])
    )
    assert torch.equal(torch.isnan(errors).sum(dim=0), rows["fell_trials"])
    assert summary["eval_velocity_g2/num_trials"] == 2.0
    assert summary["eval_velocity_g2/num_commands"] == 3.0
