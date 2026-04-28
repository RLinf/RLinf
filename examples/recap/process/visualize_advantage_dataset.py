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

"""Visualize advantage distribution and episode videos for datasets with pre-computed advantages.

This script is designed for datasets that already have advantage_continuous computed,
supporting various observation key naming conventions (e.g., 'image' vs 'observation.images.*').

Usage:
    # Generate distribution plot and episode videos
    python visualiz/visualize_advantage_dataset.py \
        --dataset /path/to/your/dataset \
        --output /path/to/output \
        --num-episodes 10
        --tag <tag>

    # Distribution plot only (no videos)
    python visualize/visualize_advantage_dataset.py \
        --dataset /path/to/your/dataset \
        --output /path/to/output \
        --no-video \
        --tag <tag>

    # All episodes
    python visualize/visualize_advantage_dataset.py \
        --dataset /path/to/your/dataset \
        --output /path/to/output \
        --num-episodes 10 \
        --tag <tag>
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from matplotlib.animation import FFMpegWriter, FuncAnimation
from tqdm import tqdm

from rlinf.data.datasets.recap.utils import decode_image_struct_batch


def to_numpy(x):
    import torch

    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


def to_scalar(x):
    if hasattr(x, "item"):
        return x.item()
    return x


def load_dataset(
    dataset_path: Path,
) -> tuple[LeRobotDataset, LeRobotDatasetMetadata, dict]:
    """Load LeRobot dataset with metadata."""
    meta = LeRobotDatasetMetadata(dataset_path.name, root=dataset_path)

    dataset = LeRobotDataset(
        dataset_path.name,
        root=dataset_path,
        delta_timestamps=None,
        download_videos=False,
    )
    dataset.hf_dataset.set_transform(decode_image_struct_batch)

    tasks = {}
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        with open(tasks_path, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                if "task_index" in entry and "task" in entry:
                    tasks[entry["task_index"]] = entry["task"]

    return dataset, meta, tasks


def detect_image_keys(sample: dict) -> list[str]:
    """Auto-detect image keys from sample."""
    image_keys = []
    for key in sample.keys():
        # Support both naming conventions
        if key.startswith("observation.images.") or key in [
            "image",
            "wrist_image",
            "front_image",
            "left_image",
            "right_image",
        ]:
            val = sample[key]
            arr = to_numpy(val)
            # Check if it looks like an image (3D or 4D with channel dim)
            if arr.ndim >= 3:
                image_keys.append(key)
    return image_keys


def get_episode_indices(dataset: LeRobotDataset, episode_index: int) -> list[int]:
    """Get all sample indices for a given episode."""
    if (
        hasattr(dataset, "episode_data_index")
        and dataset.episode_data_index is not None
    ):
        ep_data = dataset.episode_data_index
        if episode_index < len(ep_data["from"]):
            start = int(ep_data["from"][episode_index].item())
            end = int(ep_data["to"][episode_index].item())
            return list(range(start, end))

    # Fallback: scan
    indices = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if int(to_scalar(sample["episode_index"])) == episode_index:
            indices.append(idx)
    return sorted(indices)


def create_advantage_distribution_plot(
    dataset_path: Path,
    output_path: Path,
    threshold: float | None = None,
    tag: str | None = None,
    advantage_key: str = "advantage_continuous",
    figsize: tuple = (14, 10),
):
    """Create comprehensive advantage distribution plots from meta/advantages.parquet."""
    adv_filename = f"advantages_{tag}.parquet" if tag else "advantages.parquet"
    adv_parquet = dataset_path / "meta" / adv_filename
    if not adv_parquet.exists():
        if tag:
            raise FileNotFoundError(
                f"Advantages file for tag '{tag}' not found: {adv_parquet}. "
                f"Available files: {sorted(p.name for p in (dataset_path / 'meta').glob('advantages*.parquet'))}"
            )
        raise FileNotFoundError(f"Advantages file not found: {adv_parquet}")
    print(f"Loading advantage data from {adv_parquet}...")

    df = pd.read_parquet(adv_parquet)

    # Rename advantage_continuous to advantage for consistent downstream use
    if "advantage_continuous" in df.columns and "advantage" in df.columns:
        # advantage is bool, advantage_continuous is float — use continuous
        df["advantage_value"] = df["advantage_continuous"]
    elif "advantage_continuous" in df.columns:
        df["advantage_value"] = df["advantage_continuous"]
    elif "advantage" in df.columns:
        df["advantage_value"] = df["advantage"].astype(float)
    else:
        print(f"Warning: no advantage column found in {adv_parquet}")
        return None

    print(f"Loaded {len(df)} samples from {df['episode_index'].nunique()} episodes")

    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(
        f"Advantage Analysis: {dataset_path.name}", fontsize=14, fontweight="bold"
    )

    adv = df["advantage_value"].values

    # 1. Advantage histogram
    ax = axes[0, 0]
    # Clip x-axis to 1st-99th percentile to avoid outliers compressing the view
    p1, p99 = np.percentile(adv, [1, 99])
    margin = (p99 - p1) * 0.05
    xlim_lo, xlim_hi = p1 - margin, p99 + margin
    # Ensure threshold and zero are visible if they fall within a reasonable range
    if threshold is not None:
        xlim_lo = min(xlim_lo, threshold - margin)
        xlim_hi = max(xlim_hi, threshold + margin)
    xlim_lo = min(xlim_lo, -margin)  # always show zero line
    ax.hist(
        adv,
        bins=100,
        range=(xlim_lo, xlim_hi),
        density=True,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="Zero")
    ax.axvline(
        x=np.mean(adv),
        color="green",
        linestyle="-",
        linewidth=1.5,
        label=f"Mean={np.mean(adv):.4f}",
    )
    if threshold is not None:
        ax.axvline(
            x=threshold,
            color="orange",
            linestyle="-",
            linewidth=2.0,
            label=f"Threshold={threshold:.4f}",
        )
    ax.set_xlim(xlim_lo, xlim_hi)
    ax.set_xlabel("Advantage")
    ax.set_ylabel("Density")
    ax.set_title(f"Advantage Distribution (n={len(adv)})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Value prediction histogram
    ax = axes[0, 1]
    if len(df["value_current"]) > 0:
        v_curr = df["value_current"].values
        ax.hist(
            v_curr,
            bins=100,
            density=True,
            alpha=0.7,
            color="forestgreen",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xlabel("V(o_t)")
        ax.set_ylabel("Density")
        ax.set_title(f"Value Predictions (mean={np.mean(v_curr):.4f})")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "No value_current data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    # 3. Value vs Advantage scatter
    ax = axes[0, 2]
    if "value_current" in df.columns and len(df["value_current"]) > 0:
        v_curr = df["value_current"].values
        ax.scatter(v_curr, adv, alpha=0.3, s=5, c="purple")
        ax.axhline(y=0, color="red", linestyle="--", linewidth=1, label="Adv=0")
        if threshold is not None:
            ax.axhline(
                y=threshold,
                color="orange",
                linestyle="-",
                linewidth=1.5,
                label=f"Thresh={threshold:.4f}",
            )
        ax.set_xlabel("V(o_t)")
        ax.set_ylabel("Advantage")
        ax.set_title("Value vs Advantage")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "No value_current data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    # 4. Positive rate per episode
    ax = axes[1, 0]
    if threshold is not None:
        df["_positive"] = df["advantage_value"] >= threshold
        ep_pos = df.groupby("episode_index")["_positive"].mean().reset_index()
        ep_pos.columns = ["episode_index", "positive_rate"]
        ax.bar(
            ep_pos["episode_index"],
            ep_pos["positive_rate"] * 100,
            color="coral",
            alpha=0.7,
            edgecolor="black",
            linewidth=0.3,
        )
        ax.axhline(
            y=ep_pos["positive_rate"].mean() * 100,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean={ep_pos['positive_rate'].mean() * 100:.1f}%",
        )
        ax.set_xlabel("Episode Index")
        ax.set_ylabel("Positive Rate (%)")
        ax.set_title(f"Positive Rate by Episode (thresh={threshold:.4f})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        df.drop(columns=["_positive"], inplace=True)
    else:
        ax.text(
            0.5,
            0.5,
            "No threshold set",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    # 5. Advantage by episode
    ax = axes[1, 1]
    ep_stats = (
        df.groupby("episode_index")["advantage_value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    ax.errorbar(
        ep_stats["episode_index"],
        ep_stats["mean"],
        yerr=ep_stats["std"],
        fmt="o",
        markersize=3,
        alpha=0.7,
        capsize=2,
        color="teal",
    )
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Episode Index")
    ax.set_ylabel("Advantage (mean ± std)")
    ax.set_title(f"Advantage by Episode ({len(ep_stats)} episodes)")
    ax.grid(True, alpha=0.3)

    # 6. Statistics summary
    ax = axes[1, 2]
    ax.axis("off")
    stats_text = [
        f"Samples: {len(df):,}",
        f"Episodes: {df['episode_index'].nunique()}",
        "",
        "Advantage:",
        f"  Mean: {adv.mean():.5f}",
        f"  Std: {adv.std():.5f}",
        f"  Min: {adv.min():.5f}",
        f"  Max: {adv.max():.5f}",
        f"  Median: {np.median(adv):.5f}",
    ]
    if threshold is not None:
        positive_count = int((adv >= threshold).sum())
        positive_pct = positive_count / len(adv) * 100
        stats_text.extend(
            [
                "",
                f"Threshold: {threshold:.5f}",
                f"  Positive: {positive_count:,} ({positive_pct:.1f}%)",
            ]
        )
    if len(df["value_current"]) > 0:
        v_curr = df["value_current"].values
        stats_text.extend(
            [
                "",
                "Value V(o_t):",
                f"  Mean: {v_curr.mean():.5f}",
                f"  Std: {v_curr.std():.5f}",
                f"  Range: [{v_curr.min():.4f}, {v_curr.max():.4f}]",
            ]
        )
    ax.text(
        0.1,
        0.95,
        "\n".join(stats_text),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "lightgray", "alpha": 0.3},
    )
    ax.set_title("Statistics Summary")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved distribution plot: {output_path}")

    return df


def get_episode_data(
    dataset: LeRobotDataset,
    episode_index: int,
    tasks: dict,
    image_keys: list[str],
    adv_df: pd.DataFrame,
) -> dict[str, Any]:
    """Extract all data for an episode.

    Reads advantage/value/reward data from the pre-loaded advantages DataFrame
    (from meta/advantages.parquet), and images from the dataset.
    """
    indices = get_episode_indices(dataset, episode_index)

    if not indices:
        return None

    # Get advantage data for this episode from the DataFrame
    ep_adv = adv_df[adv_df["episode_index"] == episode_index].sort_values("frame_index")

    data = {
        "frames": [],
        "images": {key: [] for key in image_keys},
        "values": [],
        "advantages": [],
        "task": "",
        "episode_index": episode_index,
    }

    # Build lookup from frame_index -> row for fast access
    adv_lookup = {}
    for _, row in ep_adv.iterrows():
        adv_lookup[int(row["frame_index"])] = row

    for idx in tqdm(indices, desc=f"Loading episode {episode_index}", leave=False):
        sample = dataset[idx]
        frame_idx = int(to_scalar(sample["frame_index"]))
        data["frames"].append(frame_idx)

        # Load images
        for key in image_keys:
            if key in sample:
                img = to_numpy(sample[key])
                if img.ndim == 4:
                    img = img[0]
                if img.dtype == np.float32 or img.dtype == np.float64:
                    img = (img * 255).astype(np.uint8)
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                data["images"][key].append(img)

        # Load advantage/value/reward from adv_lookup
        if frame_idx in adv_lookup:
            row = adv_lookup[frame_idx]
            data["advantages"].append(float(row.get("advantage_continuous", 0.0)))
            data["values"].append(float(row.get("value_current", 0.0)))
        else:
            data["advantages"].append(0.0)
            data["values"].append(0.0)

        # Get task
        if not data["task"] and "task_index" in sample and tasks:
            task_idx = int(to_scalar(sample["task_index"]))
            data["task"] = tasks.get(task_idx, f"Task {task_idx}")

    return data


def create_episode_video(
    episode_data: dict[str, Any],
    output_path: Path,
    threshold: float | None = None,
    fps: int = 10,
    figsize: tuple[int, int] = (14, 8),
    dpi: int = 100,
):
    """Create a video for an episode with images and plots."""
    frames = episode_data["frames"]
    n_frames = len(frames)

    if n_frames == 0:
        return

    image_keys = [k for k, v in episode_data["images"].items() if len(v) > 0]
    n_cameras = len(image_keys)

    if n_cameras == 0:
        print(f"  No images found for episode {episode_data['episode_index']}")
        return

    has_values = any(v != 0 for v in episode_data["values"])
    has_advantages = any(a != 0 for a in episode_data["advantages"])

    n_plots = 0
    if has_values:
        n_plots += 1
    if has_advantages:
        n_plots += 1
    if n_plots == 0:
        n_plots = 1  # At least show something

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(2, max(n_cameras, n_plots), figure=fig, height_ratios=[2, 1])

    # Camera views
    camera_axes = []
    for i, key in enumerate(image_keys):
        ax = fig.add_subplot(gs[0, i])
        display_name = key.replace("observation.images.", "").replace("_", " ").title()
        ax.set_title(display_name, fontsize=10)
        ax.axis("off")
        camera_axes.append(ax)

    # Plots
    plot_axes = []
    plot_data = []

    plot_idx = 0
    if has_values:
        ax_value = fig.add_subplot(gs[1, plot_idx])
        ax_value.set_title("Value V(o_t)", fontsize=10)
        ax_value.set_xlabel("Frame")
        ax_value.set_ylabel("Value")
        ax_value.grid(True, alpha=0.3)
        plot_axes.append(ax_value)
        plot_data.append(("values", episode_data["values"], "tab:green"))
        plot_idx += 1

    if has_advantages:
        ax_adv = fig.add_subplot(gs[1, plot_idx])
        ax_adv.set_title("Advantage", fontsize=10)
        ax_adv.set_xlabel("Frame")
        ax_adv.set_ylabel("Value")
        ax_adv.grid(True, alpha=0.3)
        ax_adv.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        if threshold is not None:
            ax_adv.axhline(
                y=threshold,
                color="orange",
                linestyle="-",
                linewidth=1.5,
                alpha=0.8,
                label=f"Threshold={threshold:.4f}",
            )
            ax_adv.legend(fontsize=7, loc="upper right")
        plot_axes.append(ax_adv)
        plot_data.append(("advantages", episode_data["advantages"], "tab:red"))

    # Set up static plot lines
    for ax, (name, values, color) in zip(plot_axes, plot_data):
        ax.plot(frames, values, color=color, alpha=0.7, linewidth=1)
        ax.set_xlim(frames[0], frames[-1])
        if values:
            margin = (max(values) - min(values)) * 0.1 + 0.01
            ax.set_ylim(min(values) - margin, max(values) + margin)

    # Initialize images
    camera_ims = []
    for ax, key in zip(camera_axes, image_keys):
        im = ax.imshow(episode_data["images"][key][0])
        camera_ims.append(im)

    # Add thick border + semi-transparent overlay for threshold indication
    from matplotlib.patches import Rectangle

    border_patches = []
    overlay_patches = []
    for ax in camera_axes:
        # Thick green border
        rect = Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            linewidth=8,
            edgecolor="lime",
            facecolor="none",
            visible=False,
            zorder=10,
            clip_on=False,
        )
        ax.add_patch(rect)
        border_patches.append(rect)
        # Semi-transparent green overlay
        overlay = Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            linewidth=0,
            edgecolor="none",
            facecolor="lime",
            alpha=0.15,
            visible=False,
            zorder=9,
        )
        ax.add_patch(overlay)
        overlay_patches.append(overlay)

    # Initialize markers
    plot_markers = []
    for ax, (name, values, color) in zip(plot_axes, plot_data):
        (marker,) = ax.plot([frames[0]], [values[0]], "o", color=color, markersize=8)
        plot_markers.append(marker)

    advantages = episode_data["advantages"]
    task_text = episode_data.get("task", "")[:60]
    title = fig.suptitle(
        f"Episode {episode_data['episode_index']} - {task_text}\nFrame: {frames[0]}",
        fontsize=11,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    def update(frame_num):
        for im, key in zip(camera_ims, image_keys):
            im.set_array(episode_data["images"][key][frame_num])

        for marker, (name, values, _) in zip(plot_markers, plot_data):
            marker.set_data([frames[frame_num]], [values[frame_num]])

        # Show green border + overlay when advantage >= threshold
        above = threshold is not None and advantages[frame_num] >= threshold
        for rect in border_patches:
            rect.set_visible(above)
        for overlay in overlay_patches:
            overlay.set_visible(above)

        adv_val = advantages[frame_num]
        indicator = " [ABOVE THRESHOLD]" if above else ""
        title.set_text(
            f"Episode {episode_data['episode_index']} - {task_text}\n"
            f"Frame: {frames[frame_num]}  Adv: {adv_val:.4f}{indicator}"
        )

        return camera_ims + plot_markers + border_patches + overlay_patches + [title]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=True)

    writer = FFMpegWriter(fps=fps, bitrate=2000)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)


def create_episode_summary_plot(
    episode_data: dict[str, Any],
    output_path: Path,
    threshold: float | None = None,
    figsize: tuple[int, int] = (14, 8),
):
    """Create a static summary plot for an episode."""
    frames = episode_data["frames"]
    n_frames = len(frames)

    if n_frames == 0:
        return

    image_keys = [k for k, v in episode_data["images"].items() if len(v) > 0]
    n_cameras = len(image_keys)

    if n_cameras == 0:
        return

    # Sample frames
    sample_indices = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]
    sample_indices = sorted({i for i in sample_indices if i < n_frames})

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        n_cameras + 2,
        len(sample_indices),
        figure=fig,
        height_ratios=[1] * n_cameras + [1, 1],
    )

    # Image frames
    advantages = episode_data["advantages"]
    for cam_idx, key in enumerate(image_keys):
        cam_name = key.replace("observation.images.", "").replace("_", " ").title()
        for col_idx, frame_idx in enumerate(sample_indices):
            ax = fig.add_subplot(gs[cam_idx, col_idx])
            ax.imshow(episode_data["images"][key][frame_idx])
            ax.axis("off")
            # Green border if advantage >= threshold at this frame
            above = threshold is not None and advantages[frame_idx] >= threshold
            if above:
                import matplotlib.patches as mpatches

                rect = mpatches.FancyBboxPatch(
                    (0, 0),
                    1,
                    1,
                    transform=ax.transAxes,
                    boxstyle="round,pad=0",
                    linewidth=4,
                    edgecolor="lime",
                    facecolor="none",
                    zorder=10,
                )
                ax.add_patch(rect)
            if cam_idx == 0:
                title_str = f"t={frames[frame_idx]}"
                if above:
                    title_str += " *"
                ax.set_title(
                    title_str,
                    fontsize=9,
                    color="green" if above else "black",
                    fontweight="bold" if above else "normal",
                )
            if col_idx == 0:
                ax.text(
                    -0.1,
                    0.5,
                    cam_name,
                    transform=ax.transAxes,
                    fontsize=9,
                    va="center",
                    ha="right",
                    rotation=90,
                )

    # Value plot
    ax_value = fig.add_subplot(gs[n_cameras, :])
    values = episode_data["values"]
    if any(v != 0 for v in values):
        ax_value.plot(frames, values, "g-", label="V(o_t)", linewidth=1.5)
    ax_value.set_ylabel("Value")
    ax_value.legend(loc="upper right", fontsize=8)
    ax_value.grid(True, alpha=0.3)
    ax_value.set_xlim(frames[0], frames[-1])

    # Advantage plot
    ax_adv = fig.add_subplot(gs[n_cameras + 1, :])
    adv_values = episode_data["advantages"]
    ax_adv.plot(frames, adv_values, "r-", label="Advantage", linewidth=1.5)
    ax_adv.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    if threshold is not None:
        ax_adv.axhline(
            y=threshold,
            color="orange",
            linestyle="-",
            linewidth=1.5,
            alpha=0.8,
            label=f"Threshold={threshold:.4f}",
        )
        # Shade above-threshold regions green
        adv_arr_plot = np.array(adv_values)
        above_mask = adv_arr_plot >= threshold
        ax_adv.fill_between(
            frames,
            adv_arr_plot,
            threshold,
            where=above_mask,
            alpha=0.2,
            color="lime",
            label="Above threshold",
        )
    ax_adv.set_xlabel("Frame")
    ax_adv.set_ylabel("Advantage")
    ax_adv.legend(loc="upper right", fontsize=8)
    ax_adv.grid(True, alpha=0.3)
    ax_adv.set_xlim(frames[0], frames[-1])

    task_text = episode_data.get("task", "")[:80]
    adv_arr = np.array(episode_data["advantages"])
    fig.suptitle(
        f"Episode {episode_data['episode_index']}: {task_text}\n"
        f"Advantage: mean={adv_arr.mean():.4f}, std={adv_arr.std():.4f}",
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize advantage distribution and episodes"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to LeRobot dataset"
    )
    parser.add_argument(
        "--output", type=str, default="outputs/advantage_viz", help="Output directory"
    )
    parser.add_argument(
        "--episodes", type=int, nargs="+", help="Specific episode indices to visualize"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=10, help="Number of episodes (0=all)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS")
    parser.add_argument("--no-video", action="store_true", help="Skip video generation")
    parser.add_argument(
        "--no-distribution", action="store_true", help="Skip distribution plot"
    )
    parser.add_argument(
        "--advantage-key",
        type=str,
        default="advantage_continuous",
        help="Column name for advantage values",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Advantage threshold to draw on plots (auto-detected from mixture_config.yaml if not set)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Advantage tag: loads meta/advantages_{tag}.parquet and reads threshold from mixture_config.yaml tags.{tag}",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine parquet filename based on tag
    tag = args.tag
    if tag:
        adv_filename = f"advantages_{tag}.parquet"
    else:
        adv_filename = "advantages.parquet"

    adv_parquet = dataset_path / "meta" / adv_filename
    if not adv_parquet.exists():
        if tag:
            raise FileNotFoundError(
                f"Advantages file for tag '{tag}' not found: {adv_parquet}. "
                f"Available files: {sorted(p.name for p in (dataset_path / 'meta').glob('advantages*.parquet'))}"
            )
        raise FileNotFoundError(f"Advantages file not found: {adv_parquet}")

    print(f"Using advantages parquet: {adv_parquet}")

    # Detect threshold: CLI arg > mixture_config.yaml (tag-aware) > infer from data
    threshold = args.threshold
    if threshold is None:
        # Try to read from mixture_config.yaml in dataset dir or parent dir
        for config_dir in [dataset_path, dataset_path.parent]:
            mixture_path = config_dir / "mixture_config.yaml"
            if mixture_path.exists():
                import yaml

                with open(mixture_path, "r") as f:
                    mixture_cfg = yaml.safe_load(f) or {}
                # Tag-aware lookup: tags.<tag>.unified_threshold > unified_threshold
                if tag and "tags" in mixture_cfg and tag in mixture_cfg["tags"]:
                    tag_cfg = mixture_cfg["tags"][tag]
                    if "unified_threshold" in tag_cfg:
                        threshold = float(tag_cfg["unified_threshold"])
                        print(
                            f"Auto-detected threshold from {mixture_path} [tags.{tag}]: {threshold:.4f}"
                        )
                        break
                elif "unified_threshold" in mixture_cfg:
                    threshold = float(mixture_cfg["unified_threshold"])
                    print(
                        f"Auto-detected threshold from {mixture_path}: {threshold:.4f}"
                    )
                    break

    if threshold is None:
        # Infer from advantages parquet: min advantage_continuous where advantage==True
        if adv_parquet.exists():
            _df = pd.read_parquet(
                adv_parquet, columns=["advantage", "advantage_continuous"]
            )
            if "advantage" in _df.columns and "advantage_continuous" in _df.columns:
                positive_mask = _df["advantage"].astype(bool)
                if positive_mask.any():
                    threshold = float(
                        _df.loc[positive_mask, "advantage_continuous"].min()
                    )
                    print(f"Inferred threshold from data: {threshold:.4f}")
            del _df

    if threshold is not None:
        print(f"Using threshold: {threshold:.4f}")
    else:
        print("Warning: no threshold detected, threshold lines will not be drawn")

    # Load adv_df from parquet for episode data
    adv_df = pd.read_parquet(adv_parquet) if adv_parquet.exists() else pd.DataFrame()

    # Step 1: Create distribution plot
    if not args.no_distribution:
        dist_path = output_dir / "advantage_distribution.png"
        create_advantage_distribution_plot(
            dataset_path,
            dist_path,
            threshold=threshold,
            tag=tag,
            advantage_key=args.advantage_key,
        )

    # Step 2: Create episode visualizations
    print(f"\nLoading dataset from {dataset_path}...")
    dataset, meta, tasks = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples, {meta.total_episodes} episodes")

    # Auto-detect image keys
    sample = dataset[0]
    image_keys = detect_image_keys(sample)
    print(f"Detected image keys: {image_keys}")

    # Determine episodes to process
    if args.episodes:
        episode_indices = args.episodes
    else:
        all_episodes = list(range(meta.total_episodes))
        if args.num_episodes <= 0:
            episode_indices = all_episodes
        else:
            np.random.seed(args.seed)
            episode_indices = np.random.choice(
                all_episodes, min(args.num_episodes, len(all_episodes)), replace=False
            )
            episode_indices = sorted(episode_indices)

    print(
        f"\nProcessing {len(episode_indices)} episodes: {episode_indices[:10]}{'...' if len(episode_indices) > 10 else ''}"
    )

    for ep_idx in tqdm(episode_indices, desc="Processing episodes"):
        ep_data = get_episode_data(
            dataset=dataset,
            episode_index=ep_idx,
            tasks=tasks,
            image_keys=image_keys,
            adv_df=adv_df,
        )

        if ep_data is None:
            print(f"No data found for episode {ep_idx}")
            continue

        # Create summary plot
        plot_path = output_dir / f"episode_{ep_idx:04d}_summary.png"
        create_episode_summary_plot(ep_data, plot_path, threshold=threshold)

        # Create video
        if not args.no_video:
            video_path = output_dir / f"episode_{ep_idx:04d}.mp4"
            create_episode_video(ep_data, video_path, threshold=threshold, fps=args.fps)

    print(f"\nVisualization complete! Output saved to {output_dir}")


if __name__ == "__main__":
    main()
