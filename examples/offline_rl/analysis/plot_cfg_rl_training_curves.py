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

"""Plot CFG finetuning curves from TensorBoard event files.

The script treats each event file as a separate launch. This is important for
RECAP CFG runs because reusing the same ``cfg_rl/tensorboard`` directory after
restarting training creates multiple event files whose steps all begin at zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DEFAULT_LOG_DIR = Path("logs/aloha_sandwich_recap/20260707-065405/cfg_rl")
DEFAULT_OUTPUT_DIR = DEFAULT_LOG_DIR
LOSS_TAGS = (
    "train/loss",
    "train/conditional_loss",
    "train/unconditional_loss",
)
RATIO_TAGS = (
    "train/conditional_ratio",
    "train/unconditional_ratio",
    "train/positive_label_ratio",
    "train/negative_label_ratio",
)


def load_event_file(path: Path) -> dict[str, pd.DataFrame]:
    """Load all scalar tags from one TensorBoard event file."""
    event_acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
    event_acc.Reload()

    scalars = {}
    for tag in event_acc.Tags().get("scalars", []):
        rows = [
            {"step": event.step, "wall_time": event.wall_time, "value": event.value}
            for event in event_acc.Scalars(tag)
        ]
        scalars[tag] = pd.DataFrame(rows)
    return scalars


def smooth(values: pd.Series, window: int) -> pd.Series:
    """Apply a centered rolling mean without hiding short runs."""
    if window <= 1 or len(values) < 3:
        return values
    actual_window = min(window, max(3, len(values) // 10))
    return values.rolling(actual_window, min_periods=1, center=True).mean()


def event_label(index: int, event_file: Path, scalars: dict[str, pd.DataFrame]) -> str:
    """Build a compact label with the event index and step range."""
    loss = scalars.get("train/loss")
    if loss is None or loss.empty:
        return f"event_{index}"
    return (
        f"event_{index}: steps {int(loss['step'].iloc[0])}-{int(loss['step'].iloc[-1])}"
    )


def plot_scalar(
    ax: plt.Axes,
    runs: dict[str, dict[str, pd.DataFrame]],
    tag: str,
    *,
    smoothing_window: int,
    include_tag_in_label: bool = False,
) -> None:
    """Plot one scalar tag for all event files."""
    for label, scalars in runs.items():
        if tag not in scalars:
            continue
        df = scalars[tag].sort_values("step")
        y = smooth(df["value"], smoothing_window)
        curve_label = f"{label} {tag}" if include_tag_in_label else label
        ax.plot(df["step"], y, linewidth=1.4, label=curve_label)
    ax.set_xlabel("global step within event file")
    ax.set_ylabel(tag)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best")


def write_summary(
    output_path: Path,
    tensorboard_dir: Path,
    event_files: list[Path],
    runs: dict[str, dict[str, pd.DataFrame]],
) -> None:
    """Write scalar ranges so plots can be audited without TensorBoard."""
    summary = {
        "source_dir": str(tensorboard_dir),
        "note": (
            "Each TensorBoard event file is treated as a separate launch because "
            "restarted runs usually reset steps to zero."
        ),
        "event_files": [],
    }

    for event_file, (label, scalars) in zip(event_files, runs.items(), strict=True):
        file_summary = {"file": event_file.name, "label": label, "tags": {}}
        for tag, df in scalars.items():
            if df.empty:
                continue
            values = df["value"]
            file_summary["tags"][tag] = {
                "num_points": int(len(df)),
                "first_step": int(df["step"].iloc[0]),
                "last_step": int(df["step"].iloc[-1]),
                "first": float(values.iloc[0]),
                "last": float(values.iloc[-1]),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        summary["event_files"].append(file_summary)

    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def plot_training_dashboard(
    output_path: Path,
    runs: dict[str, dict[str, pd.DataFrame]],
    smoothing_window: int,
    loss_zoom_max: float,
    title: str,
) -> None:
    """Write the multi-panel CFG diagnostics figure."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 13), constrained_layout=True)
    fig.suptitle(title, fontsize=15)

    plot_scalar(axes[0, 0], runs, "train/loss", smoothing_window=smoothing_window)
    axes[0, 0].set_title("train/loss by event file")

    plot_scalar(axes[0, 1], runs, "train/loss", smoothing_window=smoothing_window)
    axes[0, 1].set_title("train/loss zoom: low-loss region")
    axes[0, 1].set_ylim(-0.05, loss_zoom_max)

    for tag in ("train/conditional_loss", "train/unconditional_loss"):
        plot_scalar(
            axes[1, 0],
            runs,
            tag,
            smoothing_window=smoothing_window,
            include_tag_in_label=True,
        )
    axes[1, 0].set_title("conditional / unconditional loss")

    plot_scalar(axes[1, 1], runs, "train/grad_norm", smoothing_window=smoothing_window)
    axes[1, 1].set_title("gradient norm")

    plot_scalar(axes[2, 0], runs, "train/learning_rate", smoothing_window=1)
    axes[2, 0].set_title("learning rate")

    for tag in RATIO_TAGS:
        plot_scalar(
            axes[2, 1],
            runs,
            tag,
            smoothing_window=smoothing_window,
            include_tag_in_label=True,
        )
    axes[2, 1].set_title("sampling / label ratios")

    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_loss_only(
    output_path: Path,
    runs: dict[str, dict[str, pd.DataFrame]],
    smoothing_window: int,
    loss_zoom_max: float,
    title: str,
) -> None:
    """Write a compact figure focused on CFG loss."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True)
    fig.suptitle(title, fontsize=15)

    plot_scalar(axes[0], runs, "train/loss", smoothing_window=smoothing_window)
    axes[0].set_title("All event files, full loss scale")

    plot_scalar(axes[1], runs, "train/loss", smoothing_window=smoothing_window)
    axes[1].set_title("Low-loss zoom")
    axes[1].set_ylim(-0.05, loss_zoom_max)

    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="CFG RL log directory containing tensorboard/event files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plots. Defaults to --log-dir.",
    )
    parser.add_argument(
        "--event-files",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Specific TensorBoard event files to plot. Relative paths are resolved "
            "against --log-dir/tensorboard. Defaults to all event files."
        ),
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=25,
        help="Centered rolling mean window for noisy scalar curves.",
    )
    parser.add_argument(
        "--loss-zoom-max",
        type=float,
        default=1.0,
        help="Upper y-limit for the low-loss zoom panels.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Figure title. Defaults to the log directory name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir
    output_dir = args.output_dir or log_dir
    tensorboard_dir = log_dir / "tensorboard"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.event_files:
        event_files = [
            event_file if event_file.is_absolute() else tensorboard_dir / event_file
            for event_file in args.event_files
        ]
    else:
        event_files = sorted(tensorboard_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(
            f"No TensorBoard event files found in {tensorboard_dir}"
        )
    missing = [event_file for event_file in event_files if not event_file.exists()]
    if missing:
        raise FileNotFoundError(f"TensorBoard event files not found: {missing}")

    runs = {}
    for index, event_file in enumerate(event_files):
        scalars = load_event_file(event_file)
        runs[event_label(index, event_file, scalars)] = scalars

    title = args.title or f"CFG RL finetune diagnostics: {log_dir.name}"
    plot_training_dashboard(
        output_dir / "cfg_rl_training_curves.png",
        runs,
        args.smoothing_window,
        args.loss_zoom_max,
        title,
    )
    plot_loss_only(
        output_dir / "cfg_rl_loss_curves.png",
        runs,
        args.smoothing_window,
        args.loss_zoom_max,
        title.replace("diagnostics", "loss diagnostics"),
    )
    write_summary(
        output_dir / "cfg_rl_training_summary.json",
        tensorboard_dir,
        event_files,
        runs,
    )

    print(f"Wrote {output_dir / 'cfg_rl_training_curves.png'}")
    print(f"Wrote {output_dir / 'cfg_rl_loss_curves.png'}")
    print(f"Wrote {output_dir / 'cfg_rl_training_summary.json'}")


if __name__ == "__main__":
    main()
