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

"""Build potential and progress labels for VLM Trend Potential SFT.

Example:
    python examples/reward/vlm_trend/preprocess_potential.py \\
        --raw-data-path logs/xxx/step0 \\
        --value-checkpoint logs/xxx/teacher/best.pt \\
        --output-dir logs/xxx/potential_data
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from examples.reward.vlm_trend.data import (
    Candidate,
    first_success_transition,
    load_episode,
    potential_bin,
    progress_label,
    transition_observations,
    write_labeled_sample,
    write_manifest,
)
from rlinf.models.embodiment.modules.utils import make_mlp
from rlinf.utils.logging import get_logger

logger = get_logger()


def load_value_model(
    checkpoint_path: str, device: torch.device
) -> tuple[nn.Module, dict[str, Any], np.ndarray, np.ndarray]:
    """Load the lightweight state-success teacher checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    channels = [int(config["hidden_dim"])] * int(config["num_layers"]) + [1]
    model_layers = make_mlp(
        int(config["state_dim"]) * int(config["history_size"]),
        channels,
        act_builder=nn.SiLU,
        last_act=False,
        use_layer_norm=True,
    )
    dropout = float(config.get("dropout", 0.0))
    if dropout > 0:
        layers_with_dropout = []
        for layer in model_layers:
            layers_with_dropout.append(layer)
            if isinstance(layer, nn.SiLU):
                layers_with_dropout.append(nn.Dropout(dropout))
        model_layers = layers_with_dropout
    model = nn.Sequential(*model_layers).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return (
        model,
        config,
        np.asarray(config["mean"], dtype=np.float32),
        np.asarray(config["std"], dtype=np.float32),
    )


def score_states(
    model: nn.Module,
    config: dict[str, Any],
    mean: np.ndarray,
    std: np.ndarray,
    states: list[np.ndarray],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Score all episode states with the state-success teacher."""
    history_size = int(config["history_size"])
    first = states[0]
    inputs = np.stack(
        [
            np.concatenate(
                [
                    states[index - offset] if index >= offset else first
                    for offset in range(history_size - 1, -1, -1)
                ]
            ).astype(np.float32)
            for index in range(len(states))
        ]
    )
    inputs = (inputs - mean[None]) / std[None]
    outputs = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(inputs[start : start + batch_size]).to(device)
            outputs.append(torch.sigmoid(model(batch).squeeze(-1)).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


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


def run_potential(args: argparse.Namespace) -> dict[str, Any]:
    """Build potential/progress labels through the existing Qwen SFT pipeline."""
    if args.progress_gap_steps is None:
        args.progress_gap_steps = [args.window_size]
    if not 2 <= args.num_bins <= 10:
        raise ValueError("num_bins must be between 2 and 10 for single digit labels")
    if args.temporal_smoothing_window < 1 or args.temporal_smoothing_window % 2 == 0:
        raise ValueError("temporal_smoothing_window must be a positive odd integer")
    if any(gap < 1 for gap in args.progress_gap_steps):
        raise ValueError("progress_gap_steps must contain only positive values")
    if not 0.0 <= args.val_split <= 1.0:
        raise ValueError(
            f"val_split must be in [0, 1], got {args.val_split}; "
            "use 0 for train-only or a fraction for the eval hold-out."
        )
    args.progress_gap_steps = sorted(set(args.progress_gap_steps))

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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

    source_run_by_path = {
        path: Path(root).parent.name
        for root, root_files in files_by_root.items()
        for path in root_files
    }
    split_by_path: dict[str, str] = {}
    for root_files in files_by_root.values():
        root_files = [path for path in root_files if path in pkl_files]
        rng.shuffle(root_files)
        eval_count = (
            0
            if args.val_split <= 0
            else min(
                len(root_files),
                max(1, int(round(len(root_files) * args.val_split))),
            )
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
        if not pkl_files:
            raise ValueError(
                f"--only-split {args.only_split!r} selected no episodes after "
                f"applying --val-split {args.val_split}; for example "
                "--only-split eval requires val_split > 0."
            )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg, mean, std = load_value_model(args.value_checkpoint, device)
    candidates: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    episode_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    first_end = args.window_size - 1
    for pkl_path in tqdm(pkl_files, desc="Scoring episodes", unit="episode"):
        episode = load_episode(pkl_path, log_errors=True)
        if episode is None:
            skipped["unreadable_episode"] += 1
            continue
        observations, observation_offset = transition_observations(episode)
        first_success = first_success_transition(episode, len(observations))
        if first_success is not None:
            observations = observations[: first_success + 1]
        if len(observations) < args.window_size * 2:
            skipped["short_episode"] += 1
            continue
        states = [
            np.asarray(
                observation["states"].detach().cpu().numpy()
                if torch.is_tensor(observation["states"])
                else observation["states"],
                dtype=np.float32,
            ).reshape(-1)
            for observation in observations
            if "states" in observation
        ]
        if len(states) != len(observations):
            skipped["missing_states"] += 1
            continue
        values = score_states(
            model, cfg, mean, std, states, device, args.score_batch_size
        ).reshape(-1)
        values = smooth_values(values, args.temporal_smoothing_window)
        split = split_by_path[pkl_path]
        success = bool(episode.get("success", False))
        episode_counts[f"{split}_{'success' if success else 'failure'}"] += 1
        task = str(
            episode.get("task")
            or episode.get("task_description")
            or args.task_description
            or ""
        ).strip()
        if not task:
            skipped["missing_task_description"] += 1
            continue
        for end_idx in range(first_end, len(values), args.stride):
            start_idx = end_idx - args.window_size + 1
            value = float(values[end_idx])
            candidates[(split, "potential")].append(
                Candidate(
                    pkl_path,
                    source_run_by_path[pkl_path],
                    split,
                    "potential",
                    task,
                    success,
                    start_idx + observation_offset,
                    end_idx + observation_offset,
                    value,
                    0.0,
                    str(potential_bin(value, args.num_bins)),
                )
            )
            for gap_steps in args.progress_gap_steps:
                earlier_end = end_idx - gap_steps
                if earlier_end < first_end:
                    continue
                delta = value - float(values[earlier_end])
                candidates[(split, "progress")].append(
                    Candidate(
                        pkl_path,
                        source_run_by_path[pkl_path],
                        split,
                        "progress",
                        task,
                        success,
                        earlier_end - args.window_size + 1 + observation_offset,
                        end_idx + observation_offset,
                        value,
                        delta,
                        progress_label(delta, args.progress_deadband),
                        gap_steps,
                    )
                )

    selected = []
    selection = {}
    for (split, sample_type), items in sorted(candidates.items()):
        suffix = "train" if split == "train" else "eval"
        limit = int(getattr(args, f"{sample_type}_samples_{suffix}"))
        chosen = (
            rng.sample(items, limit)
            if limit > 0 and len(items) > limit
            else list(items)
        )
        selected.extend(chosen)
        selection[f"{split}/{sample_type}"] = {
            "seen": len(items),
            "selected": len(chosen),
            "method": "uniform_without_replacement",
        }
    selected.sort(key=lambda item: (item.source_path, item.sample_type, item.start_idx))
    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_cache: dict[str, dict[str, Any]] = {}
    for candidate in tqdm(selected, desc="Writing samples", unit="sample"):
        row = write_labeled_sample(
            candidate,
            output_dir,
            source_cache,
            args.num_bins,
            args.window_size,
        )
        if row is None:
            skipped["missing_frames"] += 1
        else:
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
        "selection": selection,
        "splits": {},
    }
    for split in ("train", "eval"):
        rows = rows_by_split[split]
        rng.shuffle(rows)
        manifest = write_manifest(rows, output_dir, split)
        metadata["splits"][split] = {
            "manifest": manifest,
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
    with (output_dir / "dataset_info.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    logger.info("%s", json.dumps(metadata, indent=2, ensure_ascii=False))
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess VLM Trend potential and progress SFT windows."
    )
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
    parser.add_argument("--potential-samples-train", type=int, default=16000)
    parser.add_argument("--potential-samples-eval", type=int, default=2400)
    parser.add_argument("--progress-samples-train", type=int, default=7200)
    parser.add_argument("--progress-samples-eval", type=int, default=1080)
    parser.add_argument("--task-description", type=str, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run_potential(parse_args(argv))


if __name__ == "__main__":
    main()
