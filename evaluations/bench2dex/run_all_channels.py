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

"""Run the native four-channel protocol through RLinf's evaluation launcher."""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from rlinf.utils.logging import get_logger

CHANNELS = ("none", "cov", "inv", "inv_cov")
logger = get_logger()


def channel_command(
    repo: Path,
    config: str,
    channel: str,
    episodes: int,
    start: int,
    output: Path,
    anchor_dir: Path,
    seed: int,
    inv_cov_seed: int,
    split: str,
    overrides: list[str],
) -> list[str]:
    """Build one single-channel run with explicit official episode semantics."""
    profile = {"cov": "cov_only", "inv": "inv_only"}.get(channel, channel)
    return [
        "bash",
        str(repo / "evaluations/run_eval.sh"),
        "bench2dex",
        config,
        *overrides,
        f"runner.logger.log_path={output}",
        f"env.eval.rollout_epoch={episodes}",
        "env.eval.total_num_envs=1",
        "env.eval.auto_reset=false",
        f"env.eval.seed={seed}",
        f"env.eval.init_params.generalization_profile={profile}",
        f"env.eval.init_params.generalization_split={split}",
        f"env.eval.init_params.inv_cov_seed={inv_cov_seed}",
        f"env.eval.init_params.start_episode={start}",
        f"env.eval.init_params.output_dir={output}",
        "env.eval.init_params.episode_manifest=null",
        "env.eval.init_params.anchor_hdf5=null",
        f"env.eval.init_params.anchor_dir={anchor_dir}",
    ]


def summarize_channel(source: Path, directories: list[Path], output: Path) -> dict:
    """Merge optional process chunks with Bench2Dex's native result writer."""
    if str(source) not in sys.path:
        sys.path.append(str(source))
    from benchmark.results import load_episode_results, write_results

    results = [result for path in directories for result in load_episode_results(path)]
    policy_name = json.loads((directories[0] / "summary.json").read_text())["metadata"][
        "policy_name"
    ]
    if directories != [output]:
        write_results(output, policy_name, results)
    return json.loads((output / "summary.json").read_text())


def evaluate_channel(
    repo: Path,
    args: argparse.Namespace,
    channel: str,
    output: Path,
    env: dict[str, str],
    overrides: list[str],
    source: Path,
) -> tuple[str, dict | None]:
    """Execute one channel's process batches and collect native episode results."""
    chunk_size = args.episodes_per_process or args.num_episodes
    channel_dir = output / channel
    directories = []
    status = "ok"
    for start in range(1, args.num_episodes + 1, chunk_size):
        count = min(chunk_size, args.num_episodes - start + 1)
        directory = (
            channel_dir
            if chunk_size >= args.num_episodes
            else channel_dir / "chunks" / f"{start:06d}"
        )
        command = channel_command(
            repo,
            args.config_name,
            channel,
            count,
            start,
            directory,
            args.anchor_dir.resolve(),
            args.seed,
            args.inv_cov_seed,
            args.generalization_split,
            overrides,
        )
        result = subprocess.run(command, env=env, check=False)
        if (directory / "summary.json").is_file():
            directories.append(directory)
        if result.returncode:
            status = "failed"
            break
    summary = (
        summarize_channel(Path(source).resolve(), directories, channel_dir)
        if directories
        else None
    )
    if summary is None or summary["metadata"]["n_episodes"] != args.num_episodes:
        status = "failed"
    return status, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_name")
    parser.add_argument(
        "--anchor-dir", default=os.environ.get("BENCH2DEX_ANCHOR_DIR"), type=Path
    )
    parser.add_argument("--num-episodes", default=50, type=int)
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument("--seed", default=100_000_000, type=int)
    parser.add_argument("--inv-cov-seed", default=200_000_000, type=int)
    parser.add_argument(
        "--generalization-split", choices=("seen", "unseen", "all"), default="unseen"
    )
    parser.add_argument("--channels", default=",".join(CHANNELS))
    parser.add_argument("--episodes-per-process", type=int, default=0)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--keep-going", action="store_true")
    args, overrides = parser.parse_known_args()
    channels = args.channels.split(",")
    if any(channel not in CHANNELS for channel in channels):
        parser.error(f"Expected channels from {CHANNELS}, got {channels}")
    if args.num_episodes <= 0 or (
        args.episodes_per_process is not None and args.episodes_per_process < 0
    ):
        parser.error(
            "num-episodes must be positive and episodes-per-process nonnegative"
        )
    if args.anchor_dir is None or not args.anchor_dir.is_dir():
        parser.error(
            "Set --anchor-dir or BENCH2DEX_ANCHOR_DIR to the official replay anchor directory"
        )
    source = os.environ.get("BENCH2DEX_PATH")
    if not source:
        parser.error("Set BENCH2DEX_PATH to the Bench2Dex checkout")
    repo = Path(__file__).resolve().parents[2]
    output = (
        args.output_root
        or repo
        / "logs/bench2dex_four_channels"
        / args.config_name
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu_id))
    failed = False
    with (output / "success_rates.tsv").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(
            [
                "channel",
                "status",
                "episodes",
                "successes",
                "success_rate",
                "success_percent",
                "output_dir",
            ]
        )
        for channel in channels:
            channel_dir = output / channel
            status, summary = evaluate_channel(
                repo, args, channel, output, env, overrides, Path(source).resolve()
            )
            metadata = summary["metadata"] if summary else {}
            rate = (
                summary["core_metrics"]["completion"]["success_rate"]
                if summary
                else None
            )
            episodes = metadata.get("n_episodes")
            writer.writerow(
                [
                    channel,
                    status,
                    episodes,
                    metadata.get("successes"),
                    f"{rate:.6f}" if rate is not None else "-",
                    f"{rate * 100:.2f}%" if rate is not None else "-",
                    str(channel_dir),
                ]
            )
            stream.flush()
            if status == "failed":
                failed = True
                if not args.keep_going:
                    break
    logger.info(f"Bench2Dex channel summary: {output / 'success_rates.tsv'}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
