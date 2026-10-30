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

"""SuperMimic policy evaluation without a replay buffer or optimizers.

Used by ``evaluations/eval_supermimic.py``, by the training entry with
``runner.only_eval=true``, and by the trainer's periodic evaluation.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from rlinf.algorithms.supermimic.networks import SuperMimicModels
from rlinf.envs.sim.g1_mimic.bones import load_bones_library
from rlinf.envs.sim.g1_mimic.constants import DT
from rlinf.envs.sim.g1_mimic.env import make_env
from rlinf.envs.sim.g1_mimic.g1_fk import asset_joint_limits
from rlinf.envs.sim.g1_mimic.motion import (
    GOAL_WINDOW,
    MotionLibrary,
    load_motion_library,
)
from rlinf.envs.sim.g1_mimic.reward import compose_goal
from rlinf.envs.sim.g1_mimic.velocity import step_test_commands
from rlinf.workers.supermimic import checkpoint as ckpt
from rlinf.workers.supermimic.eval_rollout import (
    aggregate_step_test,
    run_tracking_eval,
    run_velocity_eval,
    run_velocity_step_test,
    summarize_tracking_eval,
    summarize_velocity_eval,
)

# Seven fixed commands (vx, vy, yaw rate) for a quick functional check from a
# reference start. A local choice, not the Appendix G.2 step test.
DEFAULT_VELOCITY_COMMANDS = {
    "stand": (0.0, 0.0, 0.0),
    "forward_0.5": (0.5, 0.0, 0.0),
    "forward_1.0": (1.0, 0.0, 0.0),
    "backward_0.5": (-0.5, 0.0, 0.0),
    "left_0.5": (0.0, 0.5, 0.0),
    "turn_left_1.0": (0.0, 0.0, 1.0),
    "forward_turn": (0.8, 0.0, 0.5),
}


def build_motion_library(cfg: DictConfig, device: torch.device | str) -> MotionLibrary:
    """Load the LAFAN1 or BONES-SEED library the config names."""
    env_cfg = cfg.env
    motion_format = str(env_cfg.get("motion_format", "lafan1"))
    # null loads every clip in the directory (after filtering, for BONES).
    num_clips = env_cfg.get("num_clips", None)
    num_clips = None if num_clips is None else int(num_clips)
    if motion_format == "bones_seed":
        return load_bones_library(
            str(env_cfg.get("motion_path", "")), num_clips, device
        )
    if motion_format == "lafan1":
        # motion_on_cpu keeps the frames in host memory; queries gather rows
        # there and copy only those rows to the simulator device.
        store = "cpu" if bool(env_cfg.get("motion_on_cpu", False)) else device
        return load_motion_library(env_cfg.get("motion_path", None), num_clips, store)
    raise ValueError(f"unknown env.motion_format {motion_format!r}")


def goal_function(motion: MotionLibrary, encoding: str, anchor: bool) -> Callable:
    """Goal builder ``(motion_id, motion_time, state) -> goal`` for one encoding."""

    def goal(motion_id, motion_time, state):
        if encoding == "goal730":
            return motion.goal730(motion_id, motion_time)
        goal_ref = motion.goal_ref(motion_id, motion_time)
        ref = motion.state_at(motion_id, motion_time)
        return compose_goal(goal_ref, state, ref, anchor)

    return goal


def motion_eval_rows(
    cfg: DictConfig,
    motion: MotionLibrary,
    models: SuperMimicModels,
    goal_fn: Callable,
    device: torch.device,
    batch: int,
    env: Any = None,
) -> dict[str, torch.Tensor]:
    """Every clip once from frame 0, ``batch`` clips per simulator vector."""
    n_clips = int(motion.num_clips)
    batch = max(1, min(int(batch), n_clips))
    if env is None:
        env = make_env(
            str(cfg.env.backend),
            num_envs=batch,
            motion=motion,
            device=device,
            max_episode_seconds=1.0e6,
            termination_body=str(cfg.algorithm.termination_body),
            eval_mode=True,
        )
    joint_low, joint_high = (t.to(device) for t in asset_joint_limits())
    q0 = motion.q0.to(device)
    include_ref_ori = int(cfg.algorithm.obs_dim) == 99

    def policy(history, goal):
        with torch.no_grad():
            return models.policy(history, goal)

    parts = []
    for start in range(0, n_clips, env.num_envs):
        ids = torch.arange(start, min(start + env.num_envs, n_clips), device=device)
        count = int(ids.shape[0])
        env.reset(
            torch.arange(count, device=device),
            ids,
            torch.zeros(count, dtype=torch.long, device=device),
        )
        parts.append(
            run_tracking_eval(
                env,
                policy,
                q0,
                goal_fn,
                joint_low=joint_low,
                joint_high=joint_high,
                active=count,
                include_ref_ori=include_ref_ori,
                reward_anchor=str(cfg.algorithm.get("reward_anchor", "pelvis")),
            )
        )
    merged = {key: torch.cat([part[key] for part in parts], dim=0) for key in parts[0]}
    if int(merged["clip_id"].unique().numel()) != n_clips:
        raise RuntimeError(
            f"evaluation must cover all {n_clips} clips once, covered "
            f"{int(merged['clip_id'].unique().numel())} distinct clips"
        )
    return merged


def velocity_eval_rows(
    cfg: DictConfig,
    motion: MotionLibrary,
    models: SuperMimicModels,
    device: torch.device,
    commands: torch.Tensor,
    seconds: float,
    start_clip: int = 0,
) -> dict[str, torch.Tensor]:
    """Each command in its own env, starting from ``start_clip`` frame 0."""
    if models.velocity_encoder is None:
        raise RuntimeError(
            "velocity evaluation needs a checkpoint trained with "
            "algorithm.velocity_tracking=true; these models have no velocity encoder"
        )
    n = int(commands.shape[0])
    env = make_env(
        str(cfg.env.backend),
        num_envs=n,
        motion=motion,
        device=device,
        max_episode_seconds=1.0e6,
        termination_body=str(cfg.algorithm.termination_body),
        eval_mode=True,
    )
    clip = torch.full((n,), int(start_clip), dtype=torch.long, device=device)
    env.reset(torch.arange(n, device=device), clip, torch.zeros_like(clip))

    def policy(history, command):
        with torch.no_grad():
            return models.velocity_policy(history, command)

    steps = int(round(float(seconds) / 0.02))
    return run_velocity_eval(env, policy, motion.q0.to(device), commands, steps)


class SuperMimicEvaluator:
    """Motion library, networks and eval simulator for one config."""

    def __init__(self, cfg: DictConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        algo = cfg.algorithm
        self.goal_dim = int(algo.goal_dim)
        self.obs_dim = int(algo.obs_dim)
        self.goal_encoding = str(algo.get("goal_encoding", "paper"))
        self.velocity = bool(algo.get("velocity_tracking", False))
        self.motion = build_motion_library(cfg, device)
        self.critic_features = str(algo.get("critic_features", "root4"))
        self.wm_state_loss = str(algo.get("wm_state_loss", "running"))
        self.reward_anchor = str(algo.get("reward_anchor", "pelvis"))
        self.input_norm = str(algo.get("input_norm", "running"))
        self.residual_frame = str(algo.get("residual_frame", "body"))
        self.models = SuperMimicModels(
            self.goal_dim,
            self.obs_dim,
            velocity=self.velocity,
            critic_features=self.critic_features,
            input_norm=self.input_norm == "running",
            output_norm=self.wm_state_loss == "running",
        ).to(device)
        self.models.set_bf16_matmul(str(algo.get("matmul", "tf32")) == "bf16")
        self.models.eval()
        self.goal_fn = goal_function(
            self.motion, self.goal_encoding, bool(algo.goal_anchor)
        )
        self.checkpoint: str | None = None
        self._env = None

    def load(self, directory: str) -> dict | None:
        """Load network weights only. Checks dimensions against the config."""
        path = Path(directory)
        meta = ckpt.verify_checkpoint(path)
        expected = {
            "goal_dim": self.goal_dim,
            "obs_dim": self.obs_dim,
            "goal_encoding": self.goal_encoding,
            "goal_window": list(GOAL_WINDOW[self.goal_encoding]),
            "velocity_tracking": self.velocity,
            "critic_features": self.critic_features,
            "wm_state_loss": self.wm_state_loss,
            "reward_anchor": self.reward_anchor,
            "input_norm": self.input_norm,
            "residual_frame": self.residual_frame,
        }
        for key in ckpt.POLICY_KEYS:
            found = ckpt.policy_value(meta, key)
            want = ckpt.policy_value(expected, key)
            if found != want:
                raise ValueError(
                    f"checkpoint {path} has {key}={found!r}, "
                    f"the eval config has {want!r}"
                )
        blob = torch.load(
            path / ckpt.MODEL_FILE, map_location=self.device, weights_only=True
        )
        self.models.load_state_dict(blob["models"])
        self.checkpoint = str(path)
        return meta

    def _eval_env(self):
        batch = min(int(self.cfg.env.num_envs), int(self.motion.num_clips))
        if self._env is None or self._env.num_envs != batch:
            self._env = make_env(
                str(self.cfg.env.backend),
                num_envs=batch,
                motion=self.motion,
                device=self.device,
                max_episode_seconds=1.0e6,
                termination_body=str(self.cfg.algorithm.termination_body),
                eval_mode=True,
            )
        return self._env

    def evaluate_motion(self, seed: int | None = 0) -> tuple[dict[str, float], dict]:
        """Play every clip once from frame 0.

        Returns:
            Summary metrics and per-clip rows.
        """
        if seed is not None:
            torch.manual_seed(int(seed))
        rows = motion_eval_rows(
            self.cfg,
            self.motion,
            self.models,
            self.goal_fn,
            self.device,
            int(self.cfg.env.num_envs),
            env=self._eval_env(),
        )
        return summarize_tracking_eval(rows), rows

    def evaluate_velocity(
        self,
        commands: dict[str, tuple[float, float, float]] | None = None,
        seconds: float = 10.0,
        start_clip: int = 0,
        seed: int | None = 0,
    ) -> tuple[dict[str, float], dict, list[str]]:
        """Hold each smoke-test command for ``seconds``.

        Returns:
            Summary metrics, per-command rows, and the command names.
        """
        if seed is not None:
            torch.manual_seed(int(seed))
        commands = dict(commands or DEFAULT_VELOCITY_COMMANDS)
        names = list(commands)
        tensor = torch.tensor([commands[name] for name in names], dtype=torch.float32)
        rows = velocity_eval_rows(
            self.cfg, self.motion, self.models, self.device, tensor, seconds, start_clip
        )
        return summarize_velocity_eval(rows), rows, names

    def evaluate_velocity_g2(
        self,
        seed: int = 0,
        trials: int = 4,
        seconds: float = 5.0,
        steady_seconds: float = 2.0,
        batch: int = 448,
        commands: tuple[torch.Tensor, list[str], list[str]] | None = None,
    ) -> tuple[dict[str, float], dict[str, torch.Tensor], list[str]]:
        """Appendix G.2 step test: every command in ``trials`` trials from standing.

        Trials differ in the sim context and the initial joint noise, both
        drawn from the training distribution; the paper does not say what
        varies between its four trials. ``batch`` rollouts share one vector.
        """
        if self.models.velocity_encoder is None:
            raise RuntimeError(
                "velocity evaluation needs a checkpoint trained with "
                "algorithm.velocity_tracking=true; these models have no velocity "
                "encoder"
            )
        command_table, names, planes = commands or step_test_commands()
        n_commands = int(command_table.shape[0])
        jobs = [
            (trial, index) for trial in range(trials) for index in range(n_commands)
        ]
        batch = max(1, min(int(batch), len(jobs)))
        env = make_env(
            str(self.cfg.env.backend),
            num_envs=batch,
            motion=self.motion,
            device=self.device,
            max_episode_seconds=1.0e6,
            termination_body=str(self.cfg.algorithm.termination_body),
            eval_mode=False,
        )
        ids = torch.arange(batch, device=self.device)
        errors = torch.empty(trials, n_commands, device=self.device)
        fell = torch.empty(trials, n_commands, dtype=torch.long, device=self.device)

        def policy(history, command):
            with torch.no_grad():
                return self.models.velocity_policy(history, command)

        for chunk_index, start in enumerate(range(0, len(jobs), batch)):
            chunk = jobs[start : start + batch]
            rows = [index for _, index in chunk]
            rows += [rows[-1]] * (batch - len(chunk))
            torch.manual_seed(int(seed) + chunk_index)
            env.reset_standing(ids)
            out = run_velocity_step_test(
                env,
                policy,
                self.motion.q0.to(self.device),
                command_table[rows],
                int(round(seconds / DT)),
                int(round(steady_seconds / DT)),
            )
            for i, (trial, index) in enumerate(chunk):
                errors[trial, index] = out["relative_error"][i]
                fell[trial, index] = out["fell"][i]
        summary, table = aggregate_step_test(errors, fell, planes)
        table["command"] = command_table.to(self.device)
        return summary, table, names


def _row_dicts(
    rows: dict[str, torch.Tensor], labels: list[str], label_key: str
) -> list[dict]:
    out = []
    for index, label in enumerate(labels):
        row = {label_key: label}
        for key, value in rows.items():
            item = value[index]
            if item.numel() == 1:
                row[key] = item.item()
            else:
                for j, part in enumerate(item.tolist()):
                    row[f"{key}_{j}"] = part
        out.append(row)
    return out


def json_metrics(summary: dict[str, float]) -> dict[str, float | None]:
    """Metrics for strict JSON: an undefined value (NaN) becomes ``null``."""
    return {
        key: None if isinstance(value, float) and not math.isfinite(value) else value
        for key, value in summary.items()
    }


def write_eval_outputs(
    out_dir: str | Path,
    summary: dict[str, float],
    meta: dict,
    motion_rows: dict | None = None,
    clip_names: list[str] | None = None,
    velocity_rows: dict | None = None,
    command_names: list[str] | None = None,
    g2_rows: dict | None = None,
    g2_names: list[str] | None = None,
) -> Path:
    """``summary.json`` plus ``per_clip.csv``, ``velocity_smoke.csv`` and
    ``velocity_g2.csv`` for the evaluations that ran."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(
            {"meta": meta, "metrics": json_metrics(summary)},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    tables = []
    if motion_rows is not None:
        order = motion_rows["clip_id"].tolist()
        names = [
            clip_names[int(i)] if clip_names else f"clip_{int(i):03d}" for i in order
        ]
        tables.append(("per_clip.csv", _row_dicts(motion_rows, names, "clip_name")))
    if velocity_rows is not None:
        tables.append(
            (
                "velocity_smoke.csv",
                _row_dicts(velocity_rows, command_names, "command_name"),
            )
        )
    if g2_rows is not None:
        tables.append(
            (
                "velocity_g2.csv",
                _row_dicts(g2_rows, g2_names, "command_name"),
            )
        )
    for name, rows in tables:
        with open(out / name, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return out
