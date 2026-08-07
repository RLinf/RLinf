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

"""Plot per-episode value and advantage curves with HITL interval shading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .plot_aloha_sandwich_advantage_curves import (
        positive_mask,
        resolve_advantages_path,
        resolve_threshold,
    )
    from .plot_aloha_sandwich_value_curves import (
        DEFAULT_ADVANTAGE_TAG,
        DEFAULT_DATASET_DIR,
        DEFAULT_RAW_DIR,
        EpisodeMeta,
        enrich_values,
        load_episode_meta,
    )
except ImportError:
    from plot_aloha_sandwich_advantage_curves import (
        positive_mask,
        resolve_advantages_path,
        resolve_threshold,
    )
    from plot_aloha_sandwich_value_curves import (
        DEFAULT_ADVANTAGE_TAG,
        DEFAULT_DATASET_DIR,
        DEFAULT_RAW_DIR,
        EpisodeMeta,
        enrich_values,
        load_episode_meta,
    )


DEFAULT_OUTPUT_DIR = Path("logs/aloha_sandwich_recap/per_episode_value_advantage")


def intervention_mask(
    frame_indices: np.ndarray,
    teleop_segments: list[list[int]],
) -> np.ndarray:
    """Return a mask for half-open human-intervention intervals."""
    mask = np.zeros(len(frame_indices), dtype=bool)
    for start, end in teleop_segments:
        mask |= (frame_indices >= int(start)) & (frame_indices < int(end))
    return mask


def add_intervention_spans(
    axes: np.ndarray,
    teleop_segments: list[list[int]],
) -> None:
    """Shade human-intervention intervals on every axis."""
    for axis_index, ax in enumerate(axes):
        for segment_index, (start, end) in enumerate(teleop_segments):
            label = None
            if axis_index == 0 and segment_index == 0:
                label = "human intervention"
            ax.axvspan(
                int(start),
                int(end),
                color="tab:orange",
                alpha=0.15,
                linewidth=0.0,
                label=label,
            )


def plot_episode(
    group: pd.DataFrame,
    meta: EpisodeMeta,
    output_path: Path,
    threshold: float | None,
    dpi: int,
) -> dict[str, object]:
    """Plot one episode and return its summary row."""
    group = group.sort_values("frame_index")
    frames = group["frame_index"].to_numpy(dtype=np.int64)
    values = group["value_current"].to_numpy(dtype=np.float64)
    advantages = group["advantage_continuous"].to_numpy(dtype=np.float64)
    labels = positive_mask(group, threshold)
    teleop = intervention_mask(frames, meta.teleop_segments)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 5),
        sharex=True,
        constrained_layout=True,
    )
    add_intervention_spans(axes, meta.teleop_segments)

    axes[0].plot(
        frames,
        values,
        color="tab:blue",
        linewidth=1.1,
        label="value_current",
    )
    axes[0].set_ylabel("value_current")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8, loc="best")

    axes[1].plot(
        frames,
        advantages,
        color="tab:orange",
        linewidth=1.1,
        label="advantage_continuous",
    )
    axes[1].axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=0.9,
        alpha=0.55,
        label="zero",
    )
    if threshold is not None:
        axes[1].axhline(
            threshold,
            color="tab:green",
            linestyle=":",
            linewidth=1.2,
            label=f"positive threshold={threshold:.4g}",
        )
    axes[1].set_xlabel("frame_index")
    axes[1].set_ylabel("advantage_continuous")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8, loc="best")

    intervention_frames = int(teleop.sum())
    intervention_rate = intervention_frames / len(group) if len(group) else 0.0
    fig.suptitle(
        f"Episode {meta.episode_index:03d}: {meta.episode_name} | "
        f"{group.iloc[0]['category_label']} | "
        f"frames={len(group):,} | "
        f"human intervention={intervention_frames:,} ({intervention_rate:.1%})",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    row: dict[str, object] = {
        "episode_index": meta.episode_index,
        "episode_name": meta.episode_name,
        "category": group.iloc[0]["category"],
        "category_label": group.iloc[0]["category_label"],
        "plot_path": str(output_path),
        "num_frames": int(len(group)),
        "teleop_segments": meta.teleop_segments,
        "teleop_frames": intervention_frames,
        "teleop_rate": intervention_rate,
        "value_first": float(values[0]),
        "value_last": float(values[-1]),
        "advantage_first": float(advantages[0]),
        "advantage_last": float(advantages[-1]),
    }
    if labels is not None:
        row["advantage_positive_frames"] = int(labels.sum())
        row["advantage_positive_rate"] = float(labels.mean())
    return row


def generate_per_episode_plots(
    raw_dir: Path,
    dataset_dir: Path,
    advantages_path: Path,
    output_dir: Path,
    threshold: float | None = None,
    dpi: int = 160,
    hitl_only: bool = False,
) -> pd.DataFrame:
    """Generate per-episode plots and CSV/JSON summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = load_episode_meta(raw_dir, dataset_dir)
    df = enrich_values(advantages_path, episodes)
    if "advantage_continuous" not in df.columns:
        raise ValueError(
            f"{advantages_path} is missing required column: advantage_continuous"
        )

    rows = []
    for episode_index, group in df.groupby("episode_index", sort=True):
        meta = episodes[int(episode_index)]
        if hitl_only and not meta.has_teleop:
            continue
        category = str(group.iloc[0]["category"])
        output_path = output_dir / (f"episode_{int(episode_index):03d}_{category}.png")
        rows.append(plot_episode(group, meta, output_path, threshold, dpi))

    summary_df = pd.DataFrame(rows)
    csv_path = output_dir / "per_episode_value_advantage_summary.csv"
    json_path = output_dir / "per_episode_value_advantage_summary.json"
    summary_df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    return summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--advantage-tag", default=DEFAULT_ADVANTAGE_TAG)
    parser.add_argument("--advantages-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument(
        "--hitl-only",
        action="store_true",
        help="Plot only episodes that contain human-intervention intervals.",
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
    summary = generate_per_episode_plots(
        raw_dir=args.raw_dir,
        dataset_dir=args.dataset_dir,
        advantages_path=advantages_path,
        output_dir=args.output_dir,
        threshold=threshold,
        dpi=args.dpi,
        hitl_only=args.hitl_only,
    )

    print(f"Wrote per-episode plots to: {args.output_dir.resolve()}")
    print(f"  episodes: {len(summary)}")
    print(f"  HITL episodes: {int((summary['teleop_frames'] > 0).sum())}")
    print(f"  threshold: {threshold}")


if __name__ == "__main__":
    main()
