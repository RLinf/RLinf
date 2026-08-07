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

"""Label VLM Trend SFT windows with a state success value model."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from collections import Counter
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from rlinf.data.datasets.vlm_trend_io import (
    extract_dual_view_frames,
    to_numpy_float32,
    to_uint8_rgb,
)
from rlinf.utils.logging import get_logger
from rlinf.utils.state_success_value import (
    StateSuccessValue,
    load_value_model,
    score_states,
)

logger = get_logger()


def _build_prompt(task: str, window_size: int) -> str:
    return (
        f"You are currently performing the task: {task}. "
        f"Please judge whether the operation shown in these two {window_size}-frame "
        "videos, which capture the same time window from two different views, "
        "increases, decreases, or does not clearly change the likelihood of "
        "eventual task success. Answer with exactly one word: positive, negative, "
        "or unclear."
    )


def _build_messages(prompt: str, label: str) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": label}]},
    ]


def label_from_score(
    delta: float,
    start_value: float,
    end_value: float,
    episode_success: bool,
    window_pos: float,
    pos_delta: float,
    neg_delta: float,
    pos_value: float,
    neg_value: float,
    late_negative_pos: float,
    use_outcome_rules: bool,
) -> str:
    """Map a value-window delta (and optional outcome rules) to a text label.

    Returns:
        One of ``positive``, ``negative``, or ``unclear``.
    """
    if delta >= pos_delta or (
        use_outcome_rules and episode_success and end_value >= pos_value
    ):
        return "positive"
    if delta <= neg_delta or (
        use_outcome_rules
        and (not episode_success)
        and window_pos >= late_negative_pos
        and end_value <= neg_value
    ):
        return "negative"
    return "unclear"


def infer_delta_thresholds(
    args: argparse.Namespace,
    pkl_files: list[str],
    model: StateSuccessValue,
    cfg: dict[str, Any],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> tuple[float, float, dict[str, Any]]:
    """Resolve positive/negative delta thresholds from args or data quantiles.

    Returns:
        ``(pos_delta, neg_delta, threshold_metadata)``.
    """
    if args.label_mode == "threshold":
        return args.pos_delta, args.neg_delta, {}

    deltas: list[float] = []
    for pkl_path in tqdm(pkl_files, desc="Scanning value deltas", unit="episode"):
        try:
            with open(pkl_path, "rb") as f:
                episode = pickle.load(f)
        except (EOFError, pickle.UnpicklingError, OSError):
            continue
        observations = episode.get("observations", [])
        if len(observations) < args.window_size:
            continue
        states = []
        for obs in observations:
            if "states" not in obs:
                states = []
                break
            states.append(to_numpy_float32(obs["states"]).reshape(-1))
        if len(states) < args.window_size:
            continue
        values = score_states(
            model=model,
            cfg=cfg,
            mean=mean,
            std=std,
            states=states,
            device=device,
            batch_size=args.score_batch_size,
        )
        for start_idx in range(0, len(states) - args.window_size + 1, args.stride):
            end_idx = start_idx + args.window_size - 1
            deltas.append(float(values[end_idx] - values[start_idx]))

    if not deltas:
        raise ValueError("Failed to infer quantile thresholds from empty delta set")
    delta_arr = np.asarray(deltas, dtype=np.float32)
    pos_delta = float(np.quantile(delta_arr, 1.0 - args.positive_fraction))
    neg_delta = float(np.quantile(delta_arr, args.negative_fraction))
    if pos_delta <= neg_delta:
        raise ValueError(
            "Invalid quantile thresholds: "
            f"pos_delta={pos_delta:.6f}, neg_delta={neg_delta:.6f}"
        )
    return (
        pos_delta,
        neg_delta,
        {
            "delta_count": int(delta_arr.shape[0]),
            "delta_mean": float(delta_arr.mean()),
            "delta_std": float(delta_arr.std()),
            "delta_min": float(delta_arr.min()),
            "delta_max": float(delta_arr.max()),
            "positive_fraction": args.positive_fraction,
            "negative_fraction": args.negative_fraction,
        },
    )


def preprocess(args: argparse.Namespace) -> dict[str, Any]:
    """Build dual-view progress SFT manifests labeled by the state-value teacher.

    Returns:
        Dataset metadata written to ``dataset_info.json``.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg, mean, std = load_value_model(args.value_checkpoint, device)

    pkl_files = sorted(glob(os.path.join(args.raw_data_path, "*.pkl")))
    if args.max_episodes is not None:
        pkl_files = pkl_files[: args.max_episodes]
    if not pkl_files:
        raise ValueError(f"No episode pkl files found in {args.raw_data_path}")
    rng = random.Random(args.seed)
    rng.shuffle(pkl_files)
    val_count = max(1, int(round(len(pkl_files) * args.val_split)))
    split_by_path = {
        pkl_path: ("eval" if idx < val_count else "train")
        for idx, pkl_path in enumerate(pkl_files)
    }
    pos_delta, neg_delta, threshold_metadata = infer_delta_thresholds(
        args=args,
        pkl_files=pkl_files,
        model=model,
        cfg=cfg,
        mean=mean,
        std=std,
        device=device,
    )

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    written_by_split_label: dict[str, Counter[str]] = {
        "train": Counter(),
        "eval": Counter(),
    }
    pkl_dirs = {split: output_dir / split / "pkl" for split in ("train", "eval")}
    for pkl_dir in pkl_dirs.values():
        pkl_dir.mkdir(parents=True, exist_ok=True)

    for pkl_path in tqdm(pkl_files, desc="Scoring episodes", unit="episode"):
        if args.max_samples_per_label_per_split and all(
            written_by_split_label[split][label] >= args.max_samples_per_label_per_split
            for split in ("train", "eval")
            for label in ("positive", "negative", "unclear")
        ):
            break
        try:
            with open(pkl_path, "rb") as f:
                episode = pickle.load(f)
        except (EOFError, pickle.UnpicklingError, OSError):
            continue
        observations = episode.get("observations", [])
        if len(observations) < args.window_size:
            continue
        states = []
        for obs in observations:
            if "states" not in obs:
                states = []
                break
            states.append(to_numpy_float32(obs["states"]).reshape(-1))
        if len(states) < args.window_size:
            continue
        values = score_states(
            model=model,
            cfg=cfg,
            mean=mean,
            std=std,
            states=states,
            device=device,
            batch_size=args.score_batch_size,
        )
        split = split_by_path[pkl_path]
        episode_success = bool(episode.get("success", False))
        task = str(
            episode.get("task")
            or episode.get("task_description")
            or args.task_description
            or "robot manipulation progress judgment"
        )
        for start_idx in range(0, len(states) - args.window_size + 1, args.stride):
            end_idx = start_idx + args.window_size - 1
            start_value = float(values[start_idx])
            end_value = float(values[end_idx])
            delta = end_value - start_value
            denom = max(1, len(states) - 1)
            window_pos = end_idx / denom
            label = label_from_score(
                delta=delta,
                start_value=start_value,
                end_value=end_value,
                episode_success=episode_success,
                window_pos=window_pos,
                pos_delta=pos_delta,
                neg_delta=neg_delta,
                pos_value=args.pos_value,
                neg_value=args.neg_value,
                late_negative_pos=args.late_negative_pos,
                use_outcome_rules=args.use_outcome_rules,
            )
            if (
                label == "unclear"
                and args.unclear_mode == "zero_delta"
                and abs(delta) > args.unclear_abs_delta
            ):
                continue
            if args.drop_unclear and label == "unclear":
                continue
            if (
                args.max_samples_per_label_per_split
                and written_by_split_label[split][label]
                >= args.max_samples_per_label_per_split
            ):
                continue
            frames = extract_dual_view_frames(observations, start_idx, end_idx)
            if frames is None:
                continue
            sample_id = (
                f"{os.path.splitext(os.path.basename(pkl_path))[0]}"
                f"_frames_{start_idx:04d}_{end_idx:04d}"
            )
            main_frames, extra_view_frames = frames
            sample_pkl = pkl_dirs[split] / f"{label}_{sample_id}.pkl"
            with sample_pkl.open("wb") as f:
                pickle.dump(
                    {
                        "main_frames": [to_uint8_rgb(frame) for frame in main_frames],
                        "extra_view_frames": [
                            to_uint8_rgb(frame) for frame in extra_view_frames
                        ],
                        "label": label,
                        "score": delta,
                        "state_value_start": start_value,
                        "state_value_end": end_value,
                        "source_episode_path": pkl_path,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            prompt = _build_prompt(task, args.window_size)
            rows_by_split[split].append(
                {
                    "task": task,
                    "prompt": prompt,
                    "question": prompt,
                    "answer": label,
                    "pkl_path": str(sample_pkl.resolve()),
                    "messages": _build_messages(prompt, label),
                    "source_episode_path": pkl_path,
                    "segment_metadata": {
                        "start_step": start_idx,
                        "end_step": end_idx,
                        "window_size": args.window_size,
                        "episode_id": episode.get("episode_id"),
                        "env_idx": episode.get("env_idx"),
                        "success": episode_success,
                        "views": ["main_images", "extra_view_images[0]"],
                    },
                    "supervision": {
                        "label": label,
                        "score": delta,
                        "score_name": "state_success_value_delta",
                        "state_value_start": start_value,
                        "state_value_end": end_value,
                        "pos_delta": pos_delta,
                        "neg_delta": neg_delta,
                        "pos_value": args.pos_value,
                        "neg_value": args.neg_value,
                    },
                }
            )
            written_by_split_label[split][label] += 1

    metadata: dict[str, Any] = {
        "raw_data_path": args.raw_data_path,
        "output_dir": args.output_dir,
        "value_checkpoint": args.value_checkpoint,
        "window_size": args.window_size,
        "stride": args.stride,
        "val_split": args.val_split,
        "label_mode": args.label_mode,
        "pos_delta": pos_delta,
        "neg_delta": neg_delta,
        "pos_value": args.pos_value,
        "neg_value": args.neg_value,
        "late_negative_pos": args.late_negative_pos,
        "use_outcome_rules": args.use_outcome_rules,
        "threshold_metadata": threshold_metadata,
        "drop_unclear": args.drop_unclear,
        "unclear_mode": args.unclear_mode,
        "unclear_abs_delta": args.unclear_abs_delta,
        "num_episodes": len(pkl_files),
        "splits": {},
    }
    for split, rows in rows_by_split.items():
        rng.shuffle(rows)
        if args.max_samples_per_split and len(rows) > args.max_samples_per_split:
            rows = rows[: args.max_samples_per_split]
            rows_by_split[split] = rows
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        manifest = split_dir / "segments.jsonl"
        with manifest.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        metadata["splits"][split] = {
            "manifest": str(manifest),
            "num_samples": len(rows),
            "label_counts": dict(Counter(row["answer"] for row in rows)),
        }

    with (output_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("%s", json.dumps(metadata, indent=2, ensure_ascii=False))
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for state-value progress dataset preprocessing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--value-checkpoint", required=True)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--max-samples-per-label-per-split", type=int, default=None)
    parser.add_argument("--score-batch-size", type=int, default=4096)
    parser.add_argument(
        "--label-mode",
        choices=("threshold", "quantile"),
        default="threshold",
        help="Use fixed delta thresholds or infer them from value-delta quantiles.",
    )
    parser.add_argument("--positive-fraction", type=float, default=0.25)
    parser.add_argument("--negative-fraction", type=float, default=0.25)
    parser.add_argument("--pos-delta", type=float, default=0.03)
    parser.add_argument("--neg-delta", type=float, default=-0.03)
    parser.add_argument("--pos-value", type=float, default=0.75)
    parser.add_argument("--neg-value", type=float, default=0.25)
    parser.add_argument("--late-negative-pos", type=float, default=0.6)
    parser.add_argument(
        "--use-outcome-rules",
        action="store_true",
        default=True,
        help="Also use episode success/failure rules in addition to value deltas.",
    )
    parser.add_argument(
        "--no-outcome-rules",
        action="store_false",
        dest="use_outcome_rules",
        help="Label only by state-value delta thresholds.",
    )
    parser.add_argument("--drop-unclear", action="store_true", default=False)
    parser.add_argument(
        "--unclear-mode",
        choices=("all", "zero_delta"),
        default="all",
        help="Select unclear samples from all middle windows or only near-zero deltas.",
    )
    parser.add_argument("--unclear-abs-delta", type=float, default=0.005)
    parser.add_argument("--task-description", type=str, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


if __name__ == "__main__":
    preprocess(parse_args())
