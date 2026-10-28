# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stationary standing-pose push recovery, without Ray.

Paper §4.3 holds a static T-pose. This entry does not. It loads an existing
tracking checkpoint, stands in mjlab's knees-bent keyframe, applies velocity
pushes, and writes ``summary.json`` and ``per_trial.csv``. It does not train.
Run it through ``evaluations/run_eval.sh supermimic_push_recovery``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from evaluations.eval_supermimic import resolve_checkpoint
from rlinf.algorithms.supermimic.networks import SuperMimicModels
from rlinf.envs.sim.g1_mimic.env import make_env
from rlinf.envs.sim.g1_mimic.g1_fk import asset_joint_limits
from rlinf.envs.sim.g1_mimic.motion import GOAL_WINDOW
from rlinf.envs.sim.g1_mimic.push_recovery import (
    PAPER_PROTOCOL,
    PROTOCOL_NAME,
    REFERENCE_POSE,
    RecoveryCriteria,
    check_push_protocol,
    control_steps,
    criteria_dict,
    default_standing_state,
    hold_pose_library,
    reference_steps,
    run_one_trial,
    summarize_trials,
    trial_plan,
    write_push_outputs,
)
from rlinf.utils.checkpoint import parse_global_step_from_checkpoint_path
from rlinf.utils.logging import get_logger
from rlinf.workers.supermimic import checkpoint as ckpt
from rlinf.workers.supermimic.evaluator import goal_function
from rlinf.workers.supermimic.validate import validate_supermimic_cfg


def _criteria(eval_cfg) -> RecoveryCriteria:
    drift = eval_cfg.get("max_xy_drift", None)
    return RecoveryCriteria(
        speed_threshold=float(eval_cfg.speed_threshold),
        tilt_threshold=float(eval_cfg.tilt_threshold),
        torso_ori_threshold=float(eval_cfg.torso_ori_threshold),
        stable_seconds=float(eval_cfg.stable_seconds),
        max_xy_drift=None if drift is None else float(drift),
    )


def require_matching_checkpoint(meta: dict, expected: dict) -> None:
    """Refuse a checkpoint whose policy interface is not this config's."""
    for key in ckpt.POLICY_KEYS:
        if meta.get(key) != expected[key]:
            raise ValueError(
                f"checkpoint has {key}={meta.get(key)!r}, "
                f"the eval config has {expected[key]!r}"
            )


def _load_models(
    cfg, checkpoint: str, device: torch.device
) -> tuple[SuperMimicModels, dict]:
    algo = cfg.algorithm
    models = SuperMimicModels(
        int(algo.goal_dim),
        int(algo.obs_dim),
        velocity=bool(algo.get("velocity_tracking", False)),
    ).to(device)
    models.eval()
    path = Path(checkpoint)
    meta = ckpt.verify_checkpoint(path)
    require_matching_checkpoint(
        meta,
        {
            "goal_dim": int(algo.goal_dim),
            "obs_dim": int(algo.obs_dim),
            "goal_encoding": str(algo.get("goal_encoding", "paper")),
            "goal_window": list(GOAL_WINDOW[str(algo.get("goal_encoding", "paper"))]),
            "velocity_tracking": bool(algo.get("velocity_tracking", False)),
        },
    )
    blob = torch.load(path / ckpt.MODEL_FILE, map_location=device, weights_only=True)
    models.load_state_dict(blob["models"])
    return models, meta


@hydra.main(
    version_base="1.1",
    config_path="supermimic",
    config_name="supermimic_push_recovery",
)
def main(cfg) -> None:
    cfg = validate_supermimic_cfg(cfg, world_size=1)
    if str(cfg.env.backend) != "mjlab":
        raise ValueError(
            "push recovery integrates MuJoCo Warp and requires env.backend=mjlab, "
            f"got {cfg.env.backend!r}"
        )
    eval_cfg = cfg.eval
    criteria = _criteria(eval_cfg)
    speeds = check_push_protocol(
        list(eval_cfg.speeds),
        int(eval_cfg.trials),
        float(eval_cfg.settle_seconds),
        float(eval_cfg.max_seconds),
        criteria,
        str(eval_cfg.frame),
        str(eval_cfg.push_mode),
        eval_cfg.direction,
    )
    checkpoint = resolve_checkpoint(str(eval_cfg.get("checkpoint") or ""))
    step = parse_global_step_from_checkpoint_path(checkpoint)
    device = torch.device(
        str(cfg.env.get("device", "cuda")) if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    seed = int(eval_cfg.seed)
    torch.manual_seed(seed)

    settle_steps = control_steps(float(eval_cfg.settle_seconds))
    max_steps = control_steps(float(eval_cfg.max_seconds))
    standing = default_standing_state()
    motion = hold_pose_library(standing, reference_steps(settle_steps, max_steps))
    env = make_env(
        "mjlab",
        num_envs=1,
        motion=motion,
        device=device,
        max_episode_seconds=1.0e6,
        termination_body=str(cfg.algorithm.termination_body),
        eval_mode=True,
    )
    models, meta = _load_models(cfg, checkpoint, device)
    goal_fn = goal_function(
        motion,
        str(cfg.algorithm.get("goal_encoding", "paper")),
        bool(cfg.algorithm.goal_anchor),
    )
    include_ref_ori = int(cfg.algorithm.obs_dim) == 99
    joint_low, joint_high = asset_joint_limits()
    q0 = motion.q0

    def policy(history, goal):
        with torch.no_grad():
            return models.policy(history, goal)

    plan = trial_plan(speeds, int(eval_cfg.trials), eval_cfg.direction, seed)
    rows = []
    trajectory = []
    record = bool(eval_cfg.get("save_trajectory", False))
    for item in plan:
        row, trace = run_one_trial(
            env,
            policy,
            goal_fn,
            q0,
            item,
            criteria,
            str(eval_cfg.frame),
            str(eval_cfg.push_mode),
            settle_steps,
            max_steps,
            include_ref_ori,
            joint_low,
            joint_high,
            record_trajectory=record,
        )
        rows.append(row)
        trajectory.extend(trace)
        get_logger().info(
            "push trial "
            + json.dumps(
                {
                    "trial": row["trial"],
                    "speed_m_s": row["speed_m_s"],
                    "outcome": row["outcome"],
                    "reason": row["reason"],
                    "recovery_time_s": row["recovery_time_s"],
                }
            )
        )

    stats = summarize_trials(rows, speeds)
    summary = {
        "meta": {
            "checkpoint": checkpoint,
            "global_step": step,
            "config": cfg.runner.logger.experiment_name,
            "seed": seed,
            "device": str(device),
            "checkpoint_world_size": meta.get("world_size"),
            "protocol": PROTOCOL_NAME,
            "reference_pose": REFERENCE_POSE,
            "paper": PAPER_PROTOCOL,
            "implementation": {
                **criteria_dict(criteria),
                "speeds_m_s": speeds,
                "trials_per_speed": int(eval_cfg.trials),
                "frame": str(eval_cfg.frame),
                "push_mode": str(eval_cfg.push_mode),
                "direction": (
                    "random"
                    if str(eval_cfg.direction).lower() == "random"
                    else [float(item) for item in eval_cfg.direction]
                ),
                "settle_seconds": float(eval_cfg.settle_seconds),
                "max_seconds": float(eval_cfg.max_seconds),
                "note": (
                    "A low recovery rate on a short-trained checkpoint is a "
                    "property of that policy, not a failed evaluator."
                ),
            },
        },
        **stats,
    }
    out_dir = Path(
        eval_cfg.get("output_dir")
        or os.path.join(cfg.runner.logger.log_path, f"push_recovery_global_step_{step}")
    )
    write_push_outputs(out_dir, summary, rows, trajectory if record else None)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    get_logger().info(
        "supermimic push recovery "
        + json.dumps({"output_dir": str(out_dir), **stats["overall"]}, indent=2)
    )


if __name__ == "__main__":
    main()
