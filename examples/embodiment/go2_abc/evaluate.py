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

"""Evaluate a native Go2 locomotion checkpoint on its simulated training task."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
RUNTIME = PROJECT.parent / "public-evaluator"
MIN_EPISODES = 1024
EVAL_SEEDS = (23456, 34567)


@dataclass(frozen=True)
class Task:
    """Identifiers used by the benchmark CLI and its episode records."""

    gym_id: str
    name: str
    task_id: int
    family_id: int


TASKS = {
    "A": Task("MTL-Velocity-Flat-Unitree-Go2-A2-Omni-v0", "A2_omni", 1, 0),
    "B": Task("MTL-Velocity-Rough-Unitree-Go2-B1-RoughWalk-v0", "B1_rough", 2, 1),
    "C": Task("MTL-Velocity-Rough-Unitree-Go2-B2-StairClimb-v0", "B2_stairs", 3, 1),
}


def export_policy(checkpoint: Path, initial: Path | None, destination: Path) -> None:
    """Export native weights for RSL-RL inference, optionally verifying PPO updates."""
    import torch

    from rlinf.models.embodiment.go2_policy import Go2Policy

    model = Go2Policy()

    def validate(weights: dict, source: Path) -> None:
        model.load_state_dict(weights, strict=True)
        if not all(torch.isfinite(value).all() for value in weights.values()):
            raise ValueError(f"Nonfinite policy weights in {source}")
        if not (model.std > 0).all():
            raise ValueError(f"Policy standard deviations must be positive in {source}")

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    validate(state, checkpoint)
    if initial is not None:
        baseline = torch.load(initial, map_location="cpu", weights_only=True)
        baseline = baseline.get("model_state_dict", baseline)
        validate(baseline, initial)
        for prefix in ("actor.", "critic."):
            if not any(
                not torch.equal(value, baseline[name])
                for name, value in state.items()
                if name.startswith(prefix)
            ):
                raise ValueError(f"No update to {prefix} relative to {initial}")
    torch.save(
        {
            "model_state_dict": state,
            "iter": 1000,
            "optimizer_state_dict": torch.optim.Adam(model.parameters()).state_dict(),
            "infos": {
                "inference_export_only": True,
                "native_checkpoint": str(checkpoint),
            },
        },
        destination,
    )


def evaluation_args(
    task: Task, train_name: str, policy: Path, output: Path, asset_root: Path | None
) -> list[str]:
    """Build the benchmark's single-task evaluation command for one seed."""
    kit = "--/app/extensions/excluded/0=omni.kit.telemetry"
    if asset_root is not None:
        kit += f" --/persistent/isaac/asset_root/cloud={asset_root.resolve()}"
    return [
        "--task",
        task.gym_id,
        "--checkpoint",
        str(policy),
        "--headless",
        "--num_envs",
        "256",
        "--num_episodes",
        str(MIN_EPISODES),
        "--report_train_task",
        train_name,
        "--eval_task",
        task.name,
        "--output_dir",
        str(output),
        f"--kit_args={kit}",
        "env.rewards.base_height.func=rlinf.envs.sim.isaaclab.tasks.go2:terrain_base_height_l2",
    ]


def read_episodes(directory: Path, task: Task, train_name: str) -> list[dict]:
    """Cross-check the task report and summary before pooling their episodes."""
    folder = directory / train_name
    report = json.loads((folder / f"{task.name}.json").read_text())
    rows, summary = report["episodes"], report["summary"]
    if json.loads((folder / "summary.json").read_text())[task.name] != summary:
        raise ValueError(f"Evaluation summaries disagree in {folder}")
    if (
        not isinstance(rows, list)
        or len(rows) < MIN_EPISODES
        or summary["num_episodes"] != len(rows)
        or (summary["train_task"], summary["eval_task"]) != (train_name, task.gym_id)
    ):
        raise ValueError(f"Incomplete or unexpected {task.name} evaluation in {folder}")
    if any(
        (row["task_id"], row["family_id"]) != (task.task_id, task.family_id)
        or type(row["episode_success"]) is not bool
        for row in rows
    ):
        raise ValueError(f"Invalid {task.name} episode in {folder}")
    tracking = sum(row["episode_success"] for row in rows)
    if not math.isclose(
        summary["success_rate"], tracking / len(rows), rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError(f"Tracking counts and rate disagree in {folder}")
    if task == TASKS["C"]:
        if any(type(row["physical_obstacle_passage"]) is not bool for row in rows):
            raise ValueError(f"Invalid stair passage measurement in {folder}")
        passage = sum(row["physical_obstacle_passage"] for row in rows)
        if (
            summary["required_obstacle_metrics_complete"] is not True
            or summary["physical_measured_episodes"] != len(rows)
            or summary["physical_passage_successes"] != passage
            or not math.isclose(
                summary["physical_obstacle_passage_rate"],
                passage / len(rows),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"Stair passage counts or rate disagree in {folder}")
    return rows


def main(*, legacy_stairs: bool = False) -> None:
    """Export the native weights and pool the two verified benchmark reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    if legacy_stairs:
        data = Path(os.environ.get("GO2_ABC_DATA", PROJECT / "data/go2_abc"))
        parser.set_defaults(
            task="C", train_seed=0, initial_checkpoint=data / "models/C/baseline.pt"
        )
    else:
        parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument(
        "checkpoint", type=Path, help="RLinf step 1,000 full_weights.pt"
    )
    parser.add_argument("output", type=Path, help="New directory for policy and scores")
    parser.add_argument("--asset-root", type=Path, help="Local Isaac Sim 5.1 assets")
    parser.add_argument(
        "--train-seed", type=int, help="Training seed to include in the report label"
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="Initial native or RSL-RL weights for checking actor and critic updates",
    )
    args = parser.parse_args()
    if args.train_seed is not None and args.train_seed < 0:
        parser.error("--train-seed must be nonnegative")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("run inside a Slurm GPU allocation")

    from rlinf.utils.logging import get_logger

    task = TASKS[args.task]
    train_name = task.name
    if args.train_seed is not None:
        train_name += f"_s{args.train_seed}"

    output = args.output.resolve()
    output.mkdir(parents=True)
    policy = output / "policy.pt"
    export_policy(args.checkpoint.resolve(), args.initial_checkpoint, policy)

    imports = [str(PROJECT), str(RUNTIME / "benchmark"), str(RUNTIME / "a40")]
    if os.environ.get("PYTHONPATH"):
        imports.append(os.environ["PYTHONPATH"])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(imports))
    if args.task == "B":
        env["GO2_B1_NUMERIC_GUARD"] = "1"
    else:
        env.pop("GO2_B1_NUMERIC_GUARD", None)

    entry = RUNTIME / "a40/evaluate_continuation.py"
    episodes = tracking = passage = 0
    for seed in EVAL_SEEDS:
        directory = output / str(seed)
        argv = evaluation_args(task, train_name, policy, directory, args.asset_root)
        subprocess.run(
            [sys.executable, str(entry), *argv],
            env={**env, "GO2_EVAL_SEED": str(seed)},
            check=True,
        )
        rows = read_episodes(directory, task, train_name)
        episodes += len(rows)
        tracking += sum(row["episode_success"] for row in rows)
        if args.task == "C":
            passage += sum(row["physical_obstacle_passage"] for row in rows)

    summary = {
        "num_episodes": episodes,
        "tracking_successes": tracking,
        "tracking_rate": tracking / episodes,
    }
    message = f"Tracking {tracking}/{episodes} ({tracking / episodes:.2%})"
    if args.task == "C":
        summary.update(passage_successes=passage, passage_rate=passage / episodes)
        message += f", stair passage {passage}/{episodes} ({passage / episodes:.2%})"
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    get_logger().info(message)


if __name__ == "__main__":
    main()
