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

"""Evaluate a SuperMimic checkpoint on one GPU, without Ray.

Motion tracking plays every clip once from frame 0. ``eval.velocity`` picks
the velocity evaluation: ``smoke`` holds seven fixed commands from a
reference start (a functional check), ``g2`` runs the Appendix G.2 step test.
Writes ``summary.json``, ``per_clip.csv``, and ``velocity_smoke.csv`` or
``velocity_g2.csv``. Run through ``evaluations/run_eval.sh supermimic_*``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.supermimic.validate import (
    validate_supermimic_cfg,
    validate_supermimic_eval_cfg,
)
from rlinf.utils.checkpoint import parse_global_step_from_checkpoint_path
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.workers.supermimic.checkpoint import find_latest_checkpoint
from rlinf.workers.supermimic.evaluator import (
    SuperMimicEvaluator,
    json_metrics,
    write_eval_outputs,
)


def resolve_checkpoint(path: str) -> str:
    """A ``global_step_<N>`` directory, or a folder holding several of them."""
    if not path:
        raise ValueError("set eval.checkpoint=/path/to/global_step_N (or its parent)")
    candidate = Path(path)
    if candidate.name.startswith("global_step_"):
        return str(candidate)
    latest = find_latest_checkpoint(candidate)
    if latest is None:
        raise FileNotFoundError(
            f"no complete global_step_<N> checkpoint under {candidate}"
        )
    return latest


@hydra.main(
    version_base="1.1", config_path="supermimic", config_name="supermimic_lafan1_eval"
)
def main(cfg) -> None:
    cfg = validate_supermimic_cfg(cfg, world_size=1)
    eval_cfg = cfg.eval
    validate_supermimic_eval_cfg(eval_cfg)
    velocity = str(eval_cfg.velocity)
    checkpoint = resolve_checkpoint(str(eval_cfg.get("checkpoint") or ""))
    step = parse_global_step_from_checkpoint_path(checkpoint)
    device = torch.device(
        str(cfg.env.get("device", "cuda")) if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    seed = int(eval_cfg.get("seed", 0))
    torch.manual_seed(seed)

    evaluator = SuperMimicEvaluator(cfg, device)
    meta = evaluator.load(checkpoint) or {}
    summary: dict[str, float] = {}
    motion_rows = velocity_rows = command_names = g2_rows = g2_names = None
    if bool(eval_cfg.get("motion", True)):
        metrics, motion_rows = evaluator.evaluate_motion(seed=seed)
        summary.update(metrics)
    if velocity == "smoke":
        commands = eval_cfg.get("velocity_smoke_commands", None)
        commands = (
            {k: tuple(float(x) for x in v) for k, v in commands.items()}
            if commands
            else None
        )
        metrics, velocity_rows, command_names = evaluator.evaluate_velocity(
            commands=commands,
            seconds=float(eval_cfg.velocity_smoke_seconds),
            start_clip=int(eval_cfg.velocity_smoke_start_clip),
            seed=seed,
        )
        summary.update(metrics)
    if velocity == "g2":
        metrics, g2_rows, g2_names = evaluator.evaluate_velocity_g2(
            seed=seed,
            trials=int(eval_cfg.g2_trials),
            seconds=float(eval_cfg.g2_seconds),
            steady_seconds=float(eval_cfg.g2_steady_seconds),
            batch=int(eval_cfg.g2_batch),
        )
        summary.update(metrics)

    out_dir = eval_cfg.get("output_dir") or os.path.join(
        cfg.runner.logger.log_path, f"eval_global_step_{step}"
    )
    info = {
        "checkpoint": checkpoint,
        "global_step": step,
        "config": cfg.runner.logger.experiment_name,
        "seed": seed,
        "device": str(device),
        "num_clips": int(evaluator.motion.num_clips),
        "motion_path": str(cfg.env.get("motion_path", "")),
        "checkpoint_world_size": meta.get("world_size"),
    }
    write_eval_outputs(
        out_dir,
        summary,
        info,
        motion_rows=motion_rows,
        clip_names=list(evaluator.motion.clip_names),
        velocity_rows=velocity_rows,
        command_names=command_names,
        g2_rows=g2_rows,
        g2_names=g2_names,
    )
    logger = MetricLogger(cfg)
    logger.log(summary, step=step)
    logger.finish()
    get_logger().info(
        "supermimic eval "
        + json.dumps({"output_dir": str(out_dir), **json_metrics(summary)}, indent=2)
    )
    OmegaConf.save(cfg, os.path.join(out_dir, "config.yaml"))


if __name__ == "__main__":
    main()
