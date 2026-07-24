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

"""Plot ALOHA sandwich RECAP advantage curves and distributions.

This script reads the sidecar written by ``compute_advantages.py`` directly.
It does not load LeRobot or decode episode videos, so it works with the v2.1
ALOHA dataset without depending on a particular LeRobot Python API version.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rlinf.data.process.mixture_config import read_mixture_config

try:
    from .plot_aloha_sandwich_value_curves import (
        CATEGORY_COLORS,
        CATEGORY_LABELS,
        CATEGORY_ORDER,
        DEFAULT_ADVANTAGE_TAG,
        DEFAULT_DATASET_DIR,
        DEFAULT_RAW_DIR,
        _resample_curve,
        enrich_values,
        load_episode_meta,
    )
except ImportError:
    from plot_aloha_sandwich_value_curves import (
        CATEGORY_COLORS,
        CATEGORY_LABELS,
        CATEGORY_ORDER,
        DEFAULT_ADVANTAGE_TAG,
        DEFAULT_DATASET_DIR,
        DEFAULT_RAW_DIR,
        _resample_curve,
        enrich_values,
        load_episode_meta,
    )


DEFAULT_OUTPUT_DIR = Path("logs/aloha_sandwich_recap/advantage_curves")


def resolve_advantages_path(
    dataset_dir: Path,
    advantage_tag: str,
    advantages_path: Path | None,
) -> Path:
    """Resolve an explicit or tag-derived advantage sidecar path."""
    if advantages_path is not None:
        return advantages_path
    return dataset_dir / "meta" / f"advantages_{advantage_tag}.parquet"


def resolve_threshold(
    dataset_dir: Path,
    advantage_tag: str,
    threshold: float | None,
) -> float | None:
    """Resolve the positive-label threshold from CLI or mixture metadata."""
    if threshold is not None:
        return threshold

    mixture_cfg = read_mixture_config(dataset_dir)
    tags = mixture_cfg.get("tags", {})
    if isinstance(tags, dict):
        tag_cfg = tags.get(advantage_tag, {})
        if isinstance(tag_cfg, dict) and "unified_threshold" in tag_cfg:
            return float(tag_cfg["unified_threshold"])
    if "unified_threshold" in mixture_cfg:
        return float(mixture_cfg["unified_threshold"])
    return None


def category_resampled(
    df: pd.DataFrame,
    category: str,
    num_points: int,
) -> np.ndarray:
    """Return normalized-progress advantage curves for one episode category."""
    curves = []
    category_df = df[df["category"] == category]
    for _, group in category_df.groupby("episode_index", sort=True):
        values = group.sort_values("frame_index")["advantage_continuous"].to_numpy()
        curves.append(_resample_curve(values, num_points))
    if not curves:
        return np.empty((0, num_points), dtype=np.float64)
    return np.vstack(curves)


def positive_mask(df: pd.DataFrame, threshold: float | None) -> pd.Series | None:
    """Return stored advantage labels, or threshold-derived labels as fallback."""
    if "advantage" in df.columns:
        return df["advantage"].astype(bool)
    if threshold is not None:
        return df["advantage_continuous"] >= threshold
    return None


def write_summaries(
    df: pd.DataFrame,
    output_dir: Path,
    threshold: float | None,
) -> None:
    """Write episode-level advantage statistics and a compact JSON summary."""
    labels = positive_mask(df, threshold)
    if labels is not None:
        df = df.copy()
        df["_positive"] = labels

    episode_rows = []
    for episode_index, group in df.groupby("episode_index", sort=True):
        values = group.sort_values("frame_index")["advantage_continuous"]
        first = group.iloc[0]
        row = {
            "episode_index": int(episode_index),
            "episode_name": first["episode_name"],
            "category": first["category"],
            "category_label": first["category_label"],
            "raw_reward": float(first["raw_reward"]),
            "is_success": bool(first["is_success"]),
            "has_teleop": bool(first["has_teleop"]),
            "num_frames": int(len(group)),
            "advantage_mean": float(values.mean()),
            "advantage_std": float(values.std()),
            "advantage_min": float(values.min()),
            "advantage_max": float(values.max()),
            "advantage_first": float(values.iloc[0]),
            "advantage_last": float(values.iloc[-1]),
        }
        if "_positive" in group:
            row["positive_frames"] = int(group["_positive"].sum())
            row["positive_rate"] = float(group["_positive"].mean())
        episode_rows.append(row)

    episode_df = pd.DataFrame(episode_rows)
    episode_df.to_csv(output_dir / "advantage_curves_episode_summary.csv", index=False)

    counts = (
        episode_df.groupby(["category", "category_label"], sort=False)
        .size()
        .reset_index(name="num_episodes")
    )
    summary = {
        "num_frames": int(len(df)),
        "num_episodes": int(len(episode_df)),
        "threshold": threshold,
        "advantage_mean": float(df["advantage_continuous"].mean()),
        "advantage_std": float(df["advantage_continuous"].std()),
        "category_counts": counts.to_dict(orient="records"),
    }
    if labels is not None:
        summary["positive_frames"] = int(labels.sum())
        summary["positive_rate"] = float(labels.mean())
    (output_dir / "advantage_curves_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def plot_by_category(
    df: pd.DataFrame,
    output_dir: Path,
    num_points: int,
    threshold: float | None,
) -> None:
    """Plot individual and mean advantage curves in four episode categories."""
    x = np.linspace(0.0, 1.0, num_points)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)

    for ax, category in zip(axes.ravel(), CATEGORY_ORDER, strict=True):
        curves = category_resampled(df, category, num_points)
        color = CATEGORY_COLORS[category]
        for curve in curves:
            ax.plot(x, curve, color=color, alpha=0.24, linewidth=1.0)
        if curves.size:
            mean = np.nanmean(curves, axis=0)
            ax.plot(x, mean, color="black", linewidth=2.3, label="mean")
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
        if threshold is not None:
            ax.axhline(
                threshold,
                color="orange",
                linestyle=":",
                linewidth=1.2,
                label=f"positive threshold={threshold:.4g}",
            )
        ax.set_title(f"{CATEGORY_LABELS[category]} (n={curves.shape[0]})")
        ax.set_xlabel("normalized episode progress")
        ax.set_ylabel("advantage_continuous")
        ax.grid(True, alpha=0.25)
        if curves.size or threshold is not None:
            ax.legend(fontsize=8, loc="best")

    fig.suptitle("ALOHA sandwich advantage curves by episode category")
    fig.tight_layout()
    fig.savefig(output_dir / "advantage_curves_by_category.png", dpi=180)
    plt.close(fig)


def plot_category_means(
    df: pd.DataFrame,
    output_dir: Path,
    num_points: int,
    threshold: float | None,
) -> None:
    """Plot category mean advantage curves with standard-error bands."""
    x = np.linspace(0.0, 1.0, num_points)
    fig, ax = plt.subplots(figsize=(12, 7))

    for category in CATEGORY_ORDER:
        curves = category_resampled(df, category, num_points)
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

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    if threshold is not None:
        ax.axhline(
            threshold,
            color="orange",
            linestyle=":",
            linewidth=1.4,
            label=f"positive threshold={threshold:.4g}",
        )
    ax.set_title("ALOHA sandwich mean advantage curves")
    ax.set_xlabel("normalized episode progress")
    ax.set_ylabel("advantage_continuous")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "advantage_curves_category_means.png", dpi=180)
    plt.close(fig)


def plot_distribution(
    df: pd.DataFrame,
    output_dir: Path,
    threshold: float | None,
) -> None:
    """Plot continuous advantages, labels, value correlation, and episode stats."""
    advantage = df["advantage_continuous"].to_numpy(dtype=np.float64)
    labels = positive_mask(df, threshold)
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))

    ax = axes[0, 0]
    p1, p99 = np.nanpercentile(advantage, [1, 99])
    margin = max((p99 - p1) * 0.05, 1e-6)
    xlim_lo = min(p1 - margin, 0.0)
    xlim_hi = max(p99 + margin, 0.0)
    if threshold is not None:
        xlim_lo = min(xlim_lo, threshold - margin)
        xlim_hi = max(xlim_hi, threshold + margin)
    ax.hist(
        advantage,
        bins=100,
        range=(xlim_lo, xlim_hi),
        density=True,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1.2, label="zero")
    ax.axvline(
        float(np.nanmean(advantage)),
        color="green",
        linewidth=1.2,
        label=f"mean={np.nanmean(advantage):.4g}",
    )
    if threshold is not None:
        ax.axvline(
            threshold,
            color="orange",
            linewidth=1.5,
            label=f"threshold={threshold:.4g}",
        )
    ax.set_title(f"Advantage distribution (n={len(df):,})")
    ax.set_xlabel("advantage_continuous")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    values = df["value_current"].to_numpy(dtype=np.float64)
    ax.hist(values, bins=100, density=True, alpha=0.7, color="forestgreen")
    ax.set_title(f"Value predictions (mean={np.nanmean(values):.4g})")
    ax.set_xlabel("value_current")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 2]
    stride = max(1, math.ceil(len(df) / 20_000))
    ax.scatter(values[::stride], advantage[::stride], alpha=0.25, s=4, color="purple")
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.0)
    if threshold is not None:
        ax.axhline(threshold, color="orange", linewidth=1.2)
    ax.set_title(f"Value vs advantage (every {stride} frame(s))")
    ax.set_xlabel("value_current")
    ax.set_ylabel("advantage_continuous")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    if labels is not None:
        positive_df = pd.DataFrame(
            {"episode_index": df["episode_index"], "positive": labels}
        )
        episode_positive = positive_df.groupby("episode_index")["positive"].mean()
        ax.bar(
            episode_positive.index,
            episode_positive.to_numpy() * 100.0,
            color="coral",
            alpha=0.75,
        )
        ax.axhline(
            float(labels.mean()) * 100.0,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label=f"overall={labels.mean() * 100.0:.1f}%",
        )
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No labels or threshold", ha="center", va="center")
    ax.set_title("Positive-label rate by episode")
    ax.set_xlabel("episode_index")
    ax.set_ylabel("positive rate (%)")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    episode_stats = (
        df.groupby("episode_index")["advantage_continuous"]
        .agg(["mean", "std"])
        .reset_index()
    )
    ax.errorbar(
        episode_stats["episode_index"],
        episode_stats["mean"],
        yerr=episode_stats["std"],
        fmt="o",
        markersize=3,
        alpha=0.75,
        capsize=2,
        color="teal",
    )
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.0)
    ax.set_title(f"Advantage by episode (n={len(episode_stats)})")
    ax.set_xlabel("episode_index")
    ax.set_ylabel("mean +/- std")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 2]
    ax.axis("off")
    stats = [
        f"Frames: {len(df):,}",
        f"Episodes: {df['episode_index'].nunique()}",
        "",
        f"Advantage mean: {np.nanmean(advantage):.6g}",
        f"Advantage std: {np.nanstd(advantage):.6g}",
        f"Advantage min: {np.nanmin(advantage):.6g}",
        f"Advantage max: {np.nanmax(advantage):.6g}",
        f"Advantage median: {np.nanmedian(advantage):.6g}",
        "",
        f"Value mean: {np.nanmean(values):.6g}",
        f"Value std: {np.nanstd(values):.6g}",
    ]
    if threshold is not None:
        stats.extend(["", f"Threshold: {threshold:.6g}"])
    if labels is not None:
        stats.append(f"Positive: {int(labels.sum()):,} ({labels.mean() * 100:.1f}%)")
    ax.text(
        0.05,
        0.95,
        "\n".join(stats),
        transform=ax.transAxes,
        va="top",
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "lightgray", "alpha": 0.3},
    )
    ax.set_title("Statistics summary")

    fig.suptitle("ALOHA sandwich advantage analysis", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "advantage_distribution.png", dpi=170)
    plt.close(fig)


def generate_advantage_plots(
    raw_dir: Path,
    dataset_dir: Path,
    advantages_path: Path,
    output_dir: Path,
    num_points: int = 200,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Generate all advantage plots and summaries, returning enriched frames."""
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = load_episode_meta(raw_dir, dataset_dir)
    df = enrich_values(advantages_path, episodes)
    if "advantage_continuous" not in df.columns:
        raise ValueError(
            f"{advantages_path} is missing required column: advantage_continuous"
        )

    write_summaries(df, output_dir, threshold)
    plot_by_category(df, output_dir, num_points, threshold)
    plot_category_means(df, output_dir, num_points, threshold)
    plot_distribution(df, output_dir, threshold)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--advantage-tag", default=DEFAULT_ADVANTAGE_TAG)
    parser.add_argument("--advantages-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-points", type=int, default=200)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Positive threshold. Defaults to mixture_config.yaml for the tag.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    advantages_path = resolve_advantages_path(
        args.dataset_dir,
        args.advantage_tag,
        args.advantages_path,
    )
    threshold = resolve_threshold(
        args.dataset_dir,
        args.advantage_tag,
        args.threshold,
    )
    df = generate_advantage_plots(
        raw_dir=args.raw_dir,
        dataset_dir=args.dataset_dir,
        advantages_path=advantages_path,
        output_dir=args.output_dir,
        num_points=args.num_points,
        threshold=threshold,
    )

    print(f"Wrote advantage plots to: {args.output_dir.resolve()}")
    print(f"  frames: {len(df):,}")
    print(f"  episodes: {df['episode_index'].nunique()}")
    print(f"  threshold: {threshold}")


if __name__ == "__main__":
    main()
