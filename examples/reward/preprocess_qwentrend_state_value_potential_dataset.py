#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build online-matched QwenTrend potential and progress SFT data.

Absolute samples contain one five-frame dual-view window and predict a discrete
success potential. Relative samples contain two adjacent five-frame clips and
predict whether the state-value teacher moves up, down, or stays effectively
unchanged. Successful and failed episodes are sampled independently so neither
outcome can dominate a bucket.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from examples.reward.preprocess_qwentrend_state_value_dataset import (
    _extract_dual_view_frames,
    _to_numpy,
    _to_uint8_rgb,
    load_value_model,
    score_states,
)


@dataclass(frozen=True)
class Candidate:
    source_path: str
    split: str
    sample_type: str
    task: str
    episode_success: bool
    start_idx: int
    end_idx: int
    teacher_value: float
    teacher_delta: float
    answer: str
    progress_gap_steps: int | None = None


class Reservoir:
    """Bounded deterministic reservoir for one stratification bucket."""

    def __init__(self, capacity: int, rng: random.Random) -> None:
        self.capacity = max(0, capacity)
        self.rng = rng
        self.seen = 0
        self.items: list[Candidate] = []

    def add(self, item: Candidate) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        replacement = self.rng.randrange(self.seen)
        if replacement < self.capacity:
            self.items[replacement] = item


def potential_prompt(task: str, window_size: int, num_bins: int) -> str:
    return (
        "You are estimating task-conditioned success potential for a robot "
        f"manipulation state. Task: {task}. The two synchronized videos show "
        f"the same {window_size}-frame history from two camera views. Predict "
        f"the final state's potential as exactly one digit from 0 to {num_bins - 1}, "
        f"where 0 is furthest from eventual success and {num_bins - 1} is closest."
    )


def progress_prompt(task: str, window_size: int, gap_steps: int | None = None) -> str:
    gap_steps = window_size if gap_steps is None else gap_steps
    relation = (
        "immediately adjacent"
        if gap_steps == window_size
        else f"separated by {gap_steps} environment steps"
    )
    return (
        "You are judging local task progress in a robot manipulation trajectory. "
        f"Task: {task}. In each synchronized camera video, the first {window_size} "
        f"frames are the earlier clip and the next {window_size} frames are the "
        f"later clip; their final states are {relation}. Compare their final states. "
        "Answer with exactly one word: up, same, or down."
    )


def potential_bin(value: float, num_bins: int) -> int:
    return int(round(np.clip(value, 0.0, 1.0) * (num_bins - 1)))


def progress_label(delta: float, deadband: float) -> str:
    if delta > deadband:
        return "up"
    if delta < -deadband:
        return "down"
    return "same"


def smooth_values(values: np.ndarray, window_size: int) -> np.ndarray:
    """Denoise a teacher trajectory with an edge-padded moving average."""
    if window_size <= 1:
        return values
    if window_size % 2 == 0:
        raise ValueError("temporal_smoothing_window must be odd")
    radius = window_size // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    kernel = np.full(window_size, 1.0 / window_size, dtype=np.float32)
    return np.clip(np.convolve(padded, kernel, mode="valid"), 0.0, 1.0)


def _messages(prompt: str, answer: str) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]


def _bucket_capacity(args: argparse.Namespace, split: str, sample_type: str) -> int:
    suffix = "train" if split == "train" else "eval"
    if sample_type == "potential":
        return int(getattr(args, f"potential_per_bucket_{suffix}"))
    return int(getattr(args, f"progress_per_bucket_{suffix}"))


def _write_sample(
    candidate: Candidate,
    output_dir: Path,
    source_cache: dict[str, dict[str, Any]],
    num_bins: int,
    window_size: int,
) -> dict[str, Any] | None:
    episode = source_cache.get(candidate.source_path)
    if episode is None:
        with open(candidate.source_path, "rb") as f:
            episode = pickle.load(f)
        source_cache.clear()
        source_cache[candidate.source_path] = episode
    observations = episode.get("observations", [])
    if candidate.sample_type == "progress":
        earlier_frames = _extract_dual_view_frames(
            observations,
            candidate.start_idx,
            candidate.start_idx + window_size - 1,
        )
        later_frames = _extract_dual_view_frames(
            observations,
            candidate.end_idx - window_size + 1,
            candidate.end_idx,
        )
        frames = (
            None
            if earlier_frames is None or later_frames is None
            else (
                earlier_frames[0] + later_frames[0],
                earlier_frames[1] + later_frames[1],
            )
        )
    else:
        frames = _extract_dual_view_frames(
            observations, candidate.start_idx, candidate.end_idx
        )
    if frames is None:
        return None

    main_frames, extra_frames = frames
    stem = os.path.splitext(os.path.basename(candidate.source_path))[0]
    source_run = Path(candidate.source_path).parent.parent.name
    gap_suffix = (
        f"_gap_{candidate.progress_gap_steps}"
        if candidate.progress_gap_steps is not None
        else ""
    )
    sample_id = (
        f"{candidate.sample_type}_{source_run}_{stem}_{candidate.start_idx:04d}_"
        f"{candidate.end_idx:04d}{gap_suffix}"
    )
    pkl_dir = output_dir / candidate.split / "pkl"
    pkl_dir.mkdir(parents=True, exist_ok=True)
    sample_pkl = pkl_dir / f"{sample_id}.pkl"
    payload = {
        "main_frames": [_to_uint8_rgb(frame) for frame in main_frames],
        "extra_view_frames": [_to_uint8_rgb(frame) for frame in extra_frames],
        "label": candidate.answer,
        "sample_type": candidate.sample_type,
        "teacher_value": candidate.teacher_value,
        "teacher_delta": candidate.teacher_delta,
        "source_episode_path": candidate.source_path,
        "start_idx": candidate.start_idx,
        "end_idx": candidate.end_idx,
        "progress_gap_steps": candidate.progress_gap_steps,
    }
    with sample_pkl.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    if candidate.sample_type == "potential":
        prompt = potential_prompt(candidate.task, window_size, num_bins)
    else:
        prompt = progress_prompt(
            candidate.task, window_size, candidate.progress_gap_steps
        )
    return {
        "task": candidate.task,
        "prompt": prompt,
        "question": prompt,
        "answer": candidate.answer,
        "pkl_path": str(sample_pkl.resolve()),
        "messages": _messages(prompt, candidate.answer),
        "source_episode_path": candidate.source_path,
        "source_run": source_run,
        "segment_metadata": {
            "start_step": candidate.start_idx,
            "end_step": candidate.end_idx,
            "window_size": window_size,
            "progress_gap_steps": candidate.progress_gap_steps,
            "source_run": source_run,
            "success": candidate.episode_success,
            "sample_type": candidate.sample_type,
            "views": ["main_images", "extra_view_images[0]"],
        },
        "supervision": {
            "score_name": "state_success_value_potential",
            "teacher_value": candidate.teacher_value,
            "teacher_delta": candidate.teacher_delta,
            "potential_bin": potential_bin(candidate.teacher_value, num_bins),
            "progress_label": (
                candidate.answer if candidate.sample_type == "progress" else None
            ),
        },
    }


def preprocess(args: argparse.Namespace) -> dict[str, Any]:
    if not 2 <= args.num_bins <= 10:
        raise ValueError("num_bins must be between 2 and 10 for single digit labels")
    if args.temporal_smoothing_window < 1 or args.temporal_smoothing_window % 2 == 0:
        raise ValueError("temporal_smoothing_window must be a positive odd integer")
    if any(gap < 1 for gap in args.progress_gap_steps):
        raise ValueError("progress_gap_steps must contain only positive values")
    args.progress_gap_steps = sorted(set(args.progress_gap_steps))
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg, mean, std = load_value_model(args.value_checkpoint, device)
    files_by_root = {
        str(Path(root).resolve()): sorted(glob(os.path.join(root, "*.pkl")))
        for root in args.raw_data_path
    }
    pkl_files = list(
        dict.fromkeys(path for files in files_by_root.values() for path in files)
    )
    if args.max_episodes is not None:
        pkl_files = pkl_files[: args.max_episodes]
    if not pkl_files:
        raise ValueError(f"No episode pkl files found in {args.raw_data_path}")
    split_by_path: dict[str, str] = {}
    for root_files in files_by_root.values():
        root_files = [path for path in root_files if path in pkl_files]
        rng.shuffle(root_files)
        eval_count = min(
            len(root_files), max(1, int(round(len(root_files) * args.val_split)))
        )
        split_by_path.update(
            {
                path: ("eval" if index < eval_count else "train")
                for index, path in enumerate(root_files)
            }
        )
    if args.only_split is not None:
        pkl_files = [
            path for path in pkl_files if split_by_path[path] == args.only_split
        ]

    reservoirs: dict[tuple[Any, ...], Reservoir] = {}
    episode_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()

    def add_candidate(key: tuple[Any, ...], candidate: Candidate) -> None:
        if key not in reservoirs:
            reservoirs[key] = Reservoir(
                _bucket_capacity(args, candidate.split, candidate.sample_type), rng
            )
        reservoirs[key].add(candidate)

    for pkl_path in tqdm(pkl_files, desc="Scoring episodes", unit="episode"):
        with open(pkl_path, "rb") as f:
            episode = pickle.load(f)
        observations = episode.get("observations", [])
        if len(observations) < args.window_size * 2:
            skipped["short_episode"] += 1
            continue
        states = []
        for observation in observations:
            if "states" not in observation:
                states = []
                break
            states.append(_to_numpy(observation["states"]).reshape(-1))
        if len(states) != len(observations):
            skipped["missing_states"] += 1
            continue
        values = score_states(
            model, cfg, mean, std, states, device, args.score_batch_size
        ).reshape(-1)
        values = smooth_values(values, args.temporal_smoothing_window)
        split = split_by_path[pkl_path]
        success = bool(episode.get("success", False))
        outcome = "success" if success else "failure"
        episode_counts[f"{split}_{outcome}"] += 1
        task = str(
            episode.get("task")
            or episode.get("task_description")
            or args.task_description
            or "robot manipulation progress judgment"
        )

        first_end = args.window_size - 1
        for end_idx in range(first_end, len(states), args.stride):
            start_idx = end_idx - args.window_size + 1
            value = float(values[end_idx])
            bin_id = potential_bin(value, args.num_bins)
            potential = Candidate(
                pkl_path,
                split,
                "potential",
                task,
                success,
                start_idx,
                end_idx,
                value,
                0.0,
                str(bin_id),
            )
            add_candidate((split, "potential", bin_id, outcome), potential)

            for gap_steps in args.progress_gap_steps:
                earlier_end = end_idx - gap_steps
                if earlier_end < first_end:
                    continue
                pair_start = earlier_end - args.window_size + 1
                delta = value - float(values[earlier_end])
                label = progress_label(delta, args.progress_deadband)
                progress = Candidate(
                    pkl_path,
                    split,
                    "progress",
                    task,
                    success,
                    pair_start,
                    end_idx,
                    value,
                    delta,
                    label,
                    gap_steps,
                )
                add_candidate((split, "progress", gap_steps, label, outcome), progress)

    selected = [item for reservoir in reservoirs.values() for item in reservoir.items]
    selected.sort(key=lambda item: (item.source_path, item.sample_type, item.start_idx))
    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_cache: dict[str, dict[str, Any]] = {}
    for candidate in tqdm(selected, desc="Writing samples", unit="sample"):
        row = _write_sample(
            candidate,
            output_dir,
            source_cache,
            args.num_bins,
            args.window_size,
        )
        if row is None:
            skipped["missing_frames"] += 1
            continue
        rows_by_split[candidate.split].append(row)

    metadata: dict[str, Any] = {
        "raw_data_paths": args.raw_data_path,
        "output_dir": args.output_dir,
        "value_checkpoint": args.value_checkpoint,
        "num_bins": args.num_bins,
        "window_size": args.window_size,
        "stride": args.stride,
        "progress_deadband": args.progress_deadband,
        "progress_gap_steps": args.progress_gap_steps,
        "temporal_smoothing_window": args.temporal_smoothing_window,
        "num_episodes": len(pkl_files),
        "episode_counts": dict(episode_counts),
        "skipped": dict(skipped),
        "reservoirs": {
            "/".join(map(str, key)): {
                "seen": reservoir.seen,
                "selected": len(reservoir.items),
            }
            for key, reservoir in reservoirs.items()
        },
        "splits": {},
    }
    for split in ("train", "eval"):
        rows = rows_by_split[split]
        rng.shuffle(rows)
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        manifest = split_dir / "segments.jsonl"
        with manifest.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        metadata["splits"][split] = {
            "manifest": str(manifest),
            "num_samples": len(rows),
            "sample_type_counts": dict(
                Counter(row["segment_metadata"]["sample_type"] for row in rows)
            ),
            "answer_counts": dict(Counter(row["answer"] for row in rows)),
            "outcome_counts": dict(
                Counter(
                    "success" if row["segment_metadata"]["success"] else "failure"
                    for row in rows
                )
            ),
            "source_run_counts": dict(Counter(row["source_run"] for row in rows)),
            "progress_gap_counts": dict(
                Counter(
                    str(row["segment_metadata"]["progress_gap_steps"])
                    for row in rows
                    if row["segment_metadata"]["sample_type"] == "progress"
                )
            ),
        }
    with (output_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-data-path",
        required=True,
        action="append",
        help="Collected-data directory; repeat to merge independent collection runs.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--value-checkpoint", required=True)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--progress-deadband", type=float, default=0.03)
    parser.add_argument(
        "--progress-gap-steps",
        type=int,
        nargs="+",
        default=None,
        help="One or more temporal gaps, for example: 10 20 40.",
    )
    parser.add_argument("--temporal-smoothing-window", type=int, default=1)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--only-split", choices=("train", "eval"), default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--score-batch-size", type=int, default=4096)
    parser.add_argument("--potential-per-bucket-train", type=int, default=800)
    parser.add_argument("--potential-per-bucket-eval", type=int, default=120)
    parser.add_argument("--progress-per-bucket-train", type=int, default=1200)
    parser.add_argument("--progress-per-bucket-eval", type=int, default=180)
    parser.add_argument("--task-description", type=str, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.progress_gap_steps is None:
        args.progress_gap_steps = [args.window_size]
    return args


if __name__ == "__main__":
    preprocess(parse_args())
