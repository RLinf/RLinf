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

"""Build unbalanced terminal-success windows for VLM Trend Success SFT.

Example:
    python examples/reward/vlm_trend/preprocess_terminal_success.py \\
        --raw-data-path logs/xxx/step0 \\
        --output-dir logs/xxx/success_data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from examples.reward.vlm_trend.data import (
    Candidate,
    first_success_transition,
    load_episode,
    transition_observations,
    write_labeled_sample,
    write_manifest,
)
from rlinf.utils.logging import get_logger

logger = get_logger()


def build_terminal_success_rows(
    raw_data_paths: list[str],
    output_dir: str | Path,
    window_size: int,
    interval: int,
    val_split: float,
    workers: int,
    seed: int,
    task_description: str | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Materialize unbalanced terminal-success windows for Qwen SFT."""
    entries = sorted(
        (
            (Path(root).resolve(), path.resolve())
            for root in raw_data_paths
            for path in Path(root).glob("*.pkl")
        ),
        key=lambda entry: str(entry[1]),
    )

    def inspect(entry: tuple[Path, Path]) -> dict[str, Any] | None:
        root, path = entry
        episode = load_episode(path)
        if episode is None:
            return None
        observations, observation_offset = transition_observations(episode)
        if len(observations) < window_size:
            return None
        end_step = len(observations) - 1
        first_success = first_success_transition(episode, len(observations))
        success = bool(episode.get("success", False) or first_success is not None)
        if first_success is not None:
            end_step = min(end_step, first_success)
        task = str(
            episode.get("task")
            or episode.get("task_description")
            or episode.get("task_name")
            or task_description
            or ""
        ).strip()
        if not task:
            return None
        return {
            "path": path,
            "task": task,
            "end_step": end_step,
            "observation_offset": observation_offset,
            "success_steps": [end_step] if success else [],
            "success": success,
            "is_complete": (
                success
                or bool(episode.get("terminated", []) and episode["terminated"][-1])
                or bool(episode.get("truncated", []) and episode["truncated"][-1])
            ),
            "source_run": root.parent.name,
            "split_key": str(path),
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        items = [item for item in executor.map(inspect, entries) if item is not None]
    rows_by_split = {"train": [], "eval": []}
    stats: dict[str, Any] = {"input_episodes": len(entries), "splits": {}}
    output_dir = Path(output_dir)
    for item in items:
        fraction = (
            int(hashlib.sha256(item["split_key"].encode()).hexdigest()[:8], 16) / 2**32
        )
        split = "eval" if fraction < val_split else "train"
        first = window_size - 1
        end_steps = list(range(first, item["end_step"] + 1, interval))
        success_steps = {step for step in item["success_steps"] if step >= first}
        end_steps.extend(success_steps - set(end_steps))
        source_cache: dict[str, dict[str, Any]] = {}
        for end_step in sorted(end_steps):
            answer = "1" if end_step in success_steps else "0"
            source_end = end_step + item["observation_offset"]
            candidate = Candidate(
                source_path=str(item["path"]),
                source_run=item["source_run"],
                split=split,
                sample_type="potential",
                task=item["task"],
                episode_success=answer == "1",
                start_idx=source_end - window_size + 1,
                end_idx=source_end,
                teacher_value=float(answer),
                teacher_delta=0.0,
                answer=answer,
                terminal_success=True,
                is_complete=item["is_complete"],
            )
            row = write_labeled_sample(
                candidate,
                output_dir,
                source_cache,
                num_bins=10,
                window_size=window_size,
            )
            if row is not None:
                rows_by_split[split].append(row)
    for split, rows in rows_by_split.items():
        random.Random(seed + (split == "eval")).shuffle(rows)
        positives = sum(row["answer"] == "1" for row in rows)
        stats["splits"][split] = {
            "positive": positives,
            "negative": len(rows) - positives,
            "interval": interval,
        }
    stats["complete_episodes"] = sum(item["is_complete"] for item in items)
    stats["partial_episodes"] = len(items) - stats["complete_episodes"]
    return rows_by_split, stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess VLM Trend terminal-success SFT windows."
    )
    parser.add_argument("--raw-data-path", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-description", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows_by_split, stats = build_terminal_success_rows(
        args.raw_data_path,
        args.output_dir,
        args.window_size,
        args.interval,
        args.val_split,
        args.workers,
        args.seed,
        args.task_description,
    )
    output_dir = Path(args.output_dir)
    for split, rows in rows_by_split.items():
        write_manifest(rows, output_dir, split)
    (output_dir / "dataset_info.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    logger.info("%s", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
