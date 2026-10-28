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

"""Run timestamp-first LIBERO ART diagnostic, probe, and training matrices."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

SAMPLERS = ("legacy", "uniform", "success_mixture", "early_failure")


def experiment_matrix(
    phase: str, sampler: str | None, seed: int, seeds: list[int]
) -> list[dict]:
    """Return explicit ablation arms; legacy remains the state-sampling control."""
    if phase == "diagnose":
        return [
            {
                "sampler": "legacy",
                "master": master,
                "reference": reference,
                "seed": seed,
            }
            for reference in ("rollout", "actor")
            for master in (False, True)
        ]
    if phase == "probe":
        return [
            {"sampler": name, "master": False, "reference": "rollout", "seed": seed}
            for name in SAMPLERS
        ]
    if phase in {"train", "confirm"}:
        if sampler not in SAMPLERS[1:]:
            raise ValueError(
                "Choose a non-legacy --sampler from probe results before training."
            )
        return [
            {"sampler": name, "master": True, "reference": "actor", "seed": trial_seed}
            for trial_seed in (seeds if phase == "confirm" else [seed])
            for name in ("legacy", sampler)
        ]
    return [
        {
            "sampler": "legacy",
            "master": phase != "eval",
            "reference": "actor",
            "seed": seed,
        }
    ]


def build_command(
    args: argparse.Namespace, arm: dict, run_dir: Path, name: str
) -> list[str]:
    """Build the full Hydra command without relying on shell word splitting."""
    repo = Path(__file__).resolve().parents[2]
    suite = "object" if args.phase == "object" else args.suite
    horizon = 240 if suite == "object" else 520
    config = (
        f"libero_{'object' if suite == 'object' else '10'}_0_art_diagnostics_openpi"
    )
    defaults = {
        "diagnose": 5,
        "probe": 10,
        "train": 30,
        "confirm": 50,
        "object": 25,
        "eval": 0,
    }
    steps = defaults[args.phase] if args.steps is None else args.steps
    evaluate = args.phase in {"train", "confirm", "object", "eval"}
    repeats = args.eval_repeats or (
        5 if args.phase == "confirm" else 4 if args.phase == "eval" else 3
    )
    overrides = {
        "actor.model.model_path": str(args.checkpoint),
        "rollout.model.model_path": str(args.checkpoint),
        "actor.seed": arm["seed"],
        "++rollout.seed": arm["seed"],
        "env.train.seed": arm["seed"],
        "env.eval.seed": 0,
        "env.train.total_num_envs": 128,
        "env.eval.total_num_envs": 48,
        "env.eval.max_steps_per_rollout_epoch": 2 * horizon,
        "actor.micro_batch_size": 2,
        "actor.global_batch_size": 64,
        "actor.optim.lr": args.lr,
        "actor.optim.use_fp32_master_params": arm["master"],
        "actor.model.num_steps": 4,
        "actor.model.num_action_chunks": 5,
        "algorithm.art_state_sampler": arm["sampler"],
        "algorithm.art_states_per_trajectory": 4,
        "algorithm.art_success_state_fraction": 0.7,
        "algorithm.art_failure_max_step_fraction": 0.5,
        "algorithm.art_reference_source": arm["reference"],
        "algorithm.art_replay_diagnostics": args.phase == "diagnose",
        "algorithm.art_parameter_diagnostics": args.phase == "diagnose",
        "algorithm.art_replay_batch_sizes": "[1,2,8]"
        if args.phase == "diagnose"
        else "[]",
        "algorithm.art_probe_only": args.phase == "probe",
        "runner.max_steps": steps,
        "runner.max_epochs": steps,
        "runner.eval_at_start": evaluate,
        "runner.eval_repeats": repeats,
        "runner.eval_seed": 12345,
        "runner.val_check_interval": (5 if args.phase == "object" else 10)
        if evaluate
        else -1,
        "runner.save_interval": -1,
        "runner.resume_dir": "null",
        "runner.ckpt_path": "null",
        "runner.logger.log_path": str(run_dir),
        "runner.logger.experiment_name": name,
    }
    command = [
        sys.executable,
        str(repo / "examples/embodiment/train_embodied_agent.py"),
        "--config-path",
        str(repo / "examples/embodiment/config"),
        "--config-name",
        config,
    ]
    for key, value in overrides.items():
        if isinstance(value, bool):
            value = str(value).lower()
        command.append(f"{key}={value}")
    return command + args.override


def summarize_probes(log_root: Path) -> None:
    """Report signal yield, including holdout reset states, without choosing a winner."""
    grouped = defaultdict(list)
    for path in sorted(log_root.rglob("*_art_probe_rank*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            grouped[(row["sampler"], "all")].append(row)
            # Keep every snapshot/continuation of one initial reset in one split.
            split = (
                "holdout"
                if int(row["art_source_reset_state_id"]) % 5 == 0
                else "screen"
            )
            grouped[(row["sampler"], split)].append(row)
    print(
        "sampler           split     groups  active  all_fail all_success source_success mean_step"
    )
    for (sampler, split), rows in sorted(grouped.items()):
        count = len(rows)
        active = sum(row["active"] for row in rows) / count
        failed = sum(max(row["returns"]) == 0 for row in rows) / count
        successful = sum(min(row["returns"]) > 0 for row in rows) / count
        source = sum(row["art_source_success"] for row in rows) / count
        step = sum(row["art_state_steps"] for row in rows) / count
        print(
            f"{sampler:17} {split:8} {count:6} {active:7.1%} {failed:8.1%} {successful:11.1%} {source:14.1%} {step:9.1f}"
        )
    if not grouped:
        raise ValueError(f"No ART probe JSONL files found under {log_root}.")


def main() -> None:
    """Run one phase sequentially on the selected GPUs and stop on any failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("diagnose", "probe", "train", "object", "confirm", "eval", "summary"),
    )
    parser.add_argument("--suite", choices=("long", "object"), default="long")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--log-root", type=Path, default=Path("logs/art_ablation"))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--sampler", choices=SAMPLERS[1:])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--eval-repeats", type=int)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Append one Hydra override; may be repeated.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.phase == "summary":
        summarize_probes(args.log_root)
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required.")
    if not args.dry_run and not args.checkpoint.is_dir():
        parser.error(f"Checkpoint directory does not exist: {args.checkpoint}")
    try:
        arms = experiment_matrix(args.phase, args.sampler, args.seed, args.seeds)
    except ValueError as error:
        parser.error(str(error))
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=args.gpus,
        EMBODIED_PATH=str(repo / "examples/embodiment"),
        REPO_PATH=str(repo),
        ROBOT_PLATFORM="LIBERO",
        LIBERO_TYPE="standard",
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        HYDRA_FULL_ERROR="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="1",
    )
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    for arm in arms:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        suite = "object" if args.phase == "object" else args.suite
        lr = f"{args.lr:g}".replace("e-0", "e").replace("e-", "e")
        name = f"{stamp}_{suite}0_states_T4_lr{lr}_{args.phase}_{arm['sampler']}_master_{str(arm['master']).lower()}_ref_{arm['reference']}_seed{arm['seed']}"
        run_dir = args.log_root.resolve() / name
        command = build_command(args, arm, run_dir, name)
        print(shlex.join(command), flush=True)
        if args.dry_run:
            continue
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / f"{name}_command.json").write_text(
            json.dumps({"command": command, "gpus": args.gpus}, indent=2)
        )
        log_file = args.log_root.resolve() / f"{name}.log"
        with log_file.open("w") as stream:
            stream.write(shlex.join(command) + "\n")
            stream.flush()
            process = subprocess.Popen(
                command,
                cwd=repo,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    stream.write(line)
                    stream.flush()
                code = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
        if code:
            raise subprocess.CalledProcessError(code, command)


if __name__ == "__main__":
    main()
