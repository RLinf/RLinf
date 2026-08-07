#!/usr/bin/env python3
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

"""Plot value-model curves for ALOHA sandwich RECAP episodes.

The script uses the value predictions written by
``compute_advantages.py`` (``value_current`` column), and labels each raw HDF5
episode by final success and whether human-in-the-loop segments are present.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RAW_DIR = Path(
    "/inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/aloha-data/sandwich_rl"
)
DEFAULT_DATASET_DIR = Path(
    "/inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/aloha-data/sandwich_lerobot_v21"
)
DEFAULT_ADVANTAGE_TAG = "sandwich_fail300_N10_q30_teleop"


@dataclass(frozen=True)
class EpisodeMeta:
    """Raw episode metadata needed for plotting."""

    episode_index: int
    episode_name: str
    num_frames: int
    raw_reward: float
    is_success: bool
    has_teleop: bool
    teleop_segments: list[list[int]]


CATEGORY_ORDER = (
    "hitl_success",
    "hitl_failure",
    "inference_success",
    "inference_failure",
)

CATEGORY_LABELS = {
    "hitl_success": "inference failed, HITL success",
    "hitl_failure": "inference failed, HITL failure",
    "inference_success": "inference success",
    "inference_failure": "inference failure",
}

CATEGORY_COLORS = {
    "hitl_success": "#1b9e77",
    "hitl_failure": "#d95f02",
    "inference_success": "#377eb8",
    "inference_failure": "#6a3d9a",
}


def _scalar_reward(value: Any) -> float:
    reward = np.asarray(value)
    if reward.shape == ():
        return float(reward)
    if reward.shape == (1,):
        return float(reward[0])
    raise ValueError(f"reward must be scalar or shape (1,), got {reward.shape}")


def _classify(is_success: bool, has_teleop: bool) -> str:
    if has_teleop and is_success:
        return "hitl_success"
    if has_teleop and not is_success:
        return "hitl_failure"
    if not has_teleop and is_success:
        return "inference_success"
    return "inference_failure"


def _read_raw_hdf5_meta(raw_dir: Path) -> dict[int, EpisodeMeta]:
    files = sorted([*raw_dir.glob("*.hdf5"), *raw_dir.glob("*.h5")])
    if not files:
        raise FileNotFoundError(f"No .hdf5 or .h5 files found under {raw_dir}")

    episodes: dict[int, EpisodeMeta] = {}
    for episode_index, episode_path in enumerate(files):
        with h5py.File(episode_path, "r") as episode:
            raw_reward = _scalar_reward(episode["reward"][()])
            num_frames = int(episode["action"].shape[0])
            if "teleop_segments" in episode:
                teleop_segments = np.asarray(episode["teleop_segments"], dtype=np.int64)
            else:
                teleop_segments = np.empty((0, 2), dtype=np.int64)
        if teleop_segments.size == 0:
            teleop_segments = teleop_segments.reshape(0, 2)
        episodes[episode_index] = EpisodeMeta(
            episode_index=episode_index,
            episode_name=episode_path.name,
            num_frames=num_frames,
            raw_reward=raw_reward,
            is_success=raw_reward > 0.0,
            has_teleop=bool(teleop_segments.size > 0),
            teleop_segments=teleop_segments.astype(int).tolist(),
        )
    return episodes


def _read_hil_segments(dataset_dir: Path) -> dict[int, EpisodeMeta] | None:
    hil_path = dataset_dir / "meta" / "hil_segments.json"
    if not hil_path.exists():
        return None

    data = json.loads(hil_path.read_text(encoding="utf-8"))
    episodes: dict[int, EpisodeMeta] = {}
    for entry in data.get("episodes", []):
        episode_index = int(entry["episode_index"])
        raw_reward = float(entry["raw_reward"])
        teleop_segments = entry.get("teleop_segments", [])
        episodes[episode_index] = EpisodeMeta(
            episode_index=episode_index,
            episode_name=str(entry["episode_name"]),
            num_frames=int(entry["num_frames"]),
            raw_reward=raw_reward,
            is_success=bool(entry["is_success"]),
            has_teleop=bool(teleop_segments),
            teleop_segments=teleop_segments,
        )
    return episodes


def load_episode_meta(raw_dir: Path, dataset_dir: Path) -> dict[int, EpisodeMeta]:
    """Load episode metadata using converted metadata when available."""
    from_hil = _read_hil_segments(dataset_dir)
    if from_hil is not None:
        return from_hil
    return _read_raw_hdf5_meta(raw_dir)


def enrich_values(values_path: Path, episodes: dict[int, EpisodeMeta]) -> pd.DataFrame:
    if not values_path.exists():
        raise FileNotFoundError(
            f"Value sidecar not found: {values_path}. "
            "Run compute_advantages.py first; it writes the value_current column."
        )

    df = pd.read_parquet(values_path)
    required = {"episode_index", "frame_index", "value_current"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{values_path} is missing required columns: {missing}")

    rows = []
    for episode_index, group in df.groupby("episode_index", sort=True):
        ep_idx = int(episode_index)
        if ep_idx not in episodes:
            raise KeyError(f"episode_index={ep_idx} not found in raw metadata")
        meta = episodes[ep_idx]
        category = _classify(meta.is_success, meta.has_teleop)
        g = group.sort_values("frame_index").copy()
        g["episode_name"] = meta.episode_name
        g["category"] = category
        g["category_label"] = CATEGORY_LABELS[category]
        g["raw_reward"] = meta.raw_reward
        g["is_success"] = meta.is_success
        g["has_teleop"] = meta.has_teleop
        g["episode_num_frames"] = meta.num_frames
        rows.append(g)

    return pd.concat(rows, ignore_index=True)


def _resample_curve(values: np.ndarray, num_points: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.full(num_points, np.nan, dtype=np.float64)
    if values.size == 1:
        return np.full(num_points, values[0], dtype=np.float64)
    x = np.linspace(0.0, 1.0, values.size)
    target = np.linspace(0.0, 1.0, num_points)
    return np.interp(target, x, values)


def _category_resampled(df: pd.DataFrame, category: str, num_points: int) -> np.ndarray:
    curves = []
    cat_df = df[df["category"] == category]
    for _, group in cat_df.groupby("episode_index", sort=True):
        values = group.sort_values("frame_index")["value_current"].to_numpy()
        curves.append(_resample_curve(values, num_points))
    if not curves:
        return np.empty((0, num_points), dtype=np.float64)
    return np.vstack(curves)


def write_summaries(df: pd.DataFrame, output_dir: Path) -> None:
    episode_rows = []
    for episode_index, group in df.groupby("episode_index", sort=True):
        values = group.sort_values("frame_index")["value_current"]
        first = group.iloc[0]
        episode_rows.append(
            {
                "episode_index": int(episode_index),
                "episode_name": first["episode_name"],
                "category": first["category"],
                "category_label": first["category_label"],
                "raw_reward": float(first["raw_reward"]),
                "is_success": bool(first["is_success"]),
                "has_teleop": bool(first["has_teleop"]),
                "num_frames": int(first["episode_num_frames"]),
                "value_mean": float(values.mean()),
                "value_min": float(values.min()),
                "value_max": float(values.max()),
                "value_first": float(values.iloc[0]),
                "value_last": float(values.iloc[-1]),
            }
        )

    episode_df = pd.DataFrame(episode_rows)
    episode_df.to_csv(output_dir / "value_curves_episode_summary.csv", index=False)
    df.to_parquet(output_dir / "value_curves_frame_values.parquet", index=False)

    counts = (
        episode_df.groupby(["category", "category_label"], sort=False)
        .size()
        .reset_index(name="num_episodes")
    )
    summary = {
        "num_frames": int(len(df)),
        "num_episodes": int(len(episode_df)),
        "category_counts": counts.to_dict(orient="records"),
    }
    (output_dir / "value_curves_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def plot_by_category(df: pd.DataFrame, output_dir: Path, num_points: int) -> None:
    x = np.linspace(0.0, 1.0, num_points)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)

    for ax, category in zip(axes.ravel(), CATEGORY_ORDER, strict=True):
        curves = _category_resampled(df, category, num_points)
        color = CATEGORY_COLORS[category]
        for curve in curves:
            ax.plot(x, curve, color=color, alpha=0.24, linewidth=1.0)
        if curves.size:
            mean = np.nanmean(curves, axis=0)
            ax.plot(x, mean, color="black", linewidth=2.3, label="mean")
        ax.set_title(f"{CATEGORY_LABELS[category]} (n={curves.shape[0]})")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("normalized episode progress")
        ax.set_ylabel("value_current")
        if curves.size:
            ax.legend(loc="best")

    fig.suptitle("ALOHA sandwich value curves by episode category")
    fig.tight_layout()
    fig.savefig(output_dir / "value_curves_by_category.png", dpi=180)
    plt.close(fig)


def plot_category_means(df: pd.DataFrame, output_dir: Path, num_points: int) -> None:
    x = np.linspace(0.0, 1.0, num_points)
    fig, ax = plt.subplots(figsize=(12, 7))

    for category in CATEGORY_ORDER:
        curves = _category_resampled(df, category, num_points)
        if not curves.size:
            continue
        color = CATEGORY_COLORS[category]
        mean = np.nanmean(curves, axis=0)
        if curves.shape[0] > 1:
            sem = np.nanstd(curves, axis=0, ddof=1) / np.sqrt(curves.shape[0])
            ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.16)
        ax.plot(
            x,
            mean,
            color=color,
            linewidth=2.4,
            label=f"{CATEGORY_LABELS[category]} (n={curves.shape[0]})",
        )

    ax.set_title("ALOHA sandwich mean value curves")
    ax.set_xlabel("normalized episode progress")
    ax.set_ylabel("value_current")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "value_curves_category_means.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--advantage-tag", default=DEFAULT_ADVANTAGE_TAG)
    parser.add_argument("--values-path", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/aloha_sandwich_recap/value_curves"),
    )
    parser.add_argument("--num-points", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values_path = args.values_path
    if values_path is None:
        values_path = (
            args.dataset_dir
            / "meta"
            / f"advantages_{args.advantage_tag}.parquet"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_episode_meta(args.raw_dir, args.dataset_dir)
    df = enrich_values(values_path, episodes)
    write_summaries(df, args.output_dir)
    plot_by_category(df, args.output_dir, args.num_points)
    plot_category_means(df, args.output_dir, args.num_points)

    counts = df.drop_duplicates("episode_index")["category"].value_counts()
    print(f"Wrote value plots to: {args.output_dir.resolve()}")
    for category in CATEGORY_ORDER:
        print(f"  {CATEGORY_LABELS[category]}: {int(counts.get(category, 0))}")


if __name__ == "__main__":
    main()
