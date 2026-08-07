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

"""Generate value, advantage, and CFG curves for one ALOHA RECAP run."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .plot_aloha_sandwich_advantage_curves import (
        generate_advantage_plots,
        resolve_advantages_path,
        resolve_threshold,
    )
    from .plot_aloha_sandwich_value_curves import (
        DEFAULT_ADVANTAGE_TAG,
        DEFAULT_DATASET_DIR,
        DEFAULT_RAW_DIR,
        enrich_values,
        load_episode_meta,
    )
    from .plot_aloha_sandwich_value_curves import (
        plot_by_category as plot_values_by_category,
    )
    from .plot_aloha_sandwich_value_curves import (
        plot_category_means as plot_value_category_means,
    )
    from .plot_aloha_sandwich_value_curves import (
        write_summaries as write_value_summaries,
    )
    from .plot_cfg_rl_training_curves import (
        event_label,
        load_event_file,
        plot_loss_only,
        plot_training_dashboard,
    )
    from .plot_cfg_rl_training_curves import (
        write_summary as write_cfg_summary,
    )
except ImportError:
    from plot_aloha_sandwich_advantage_curves import (
        generate_advantage_plots,
        resolve_advantages_path,
        resolve_threshold,
    )
    from plot_aloha_sandwich_value_curves import (
        DEFAULT_ADVANTAGE_TAG,
        DEFAULT_DATASET_DIR,
        DEFAULT_RAW_DIR,
        enrich_values,
        load_episode_meta,
    )
    from plot_aloha_sandwich_value_curves import (
        plot_by_category as plot_values_by_category,
    )
    from plot_aloha_sandwich_value_curves import (
        plot_category_means as plot_value_category_means,
    )
    from plot_aloha_sandwich_value_curves import (
        write_summaries as write_value_summaries,
    )
    from plot_cfg_rl_training_curves import (
        event_label,
        load_event_file,
        plot_loss_only,
        plot_training_dashboard,
    )
    from plot_cfg_rl_training_curves import (
        write_summary as write_cfg_summary,
    )


def generate_value_plots(
    raw_dir: Path,
    dataset_dir: Path,
    values_path: Path,
    output_dir: Path,
    num_points: int,
) -> None:
    """Generate the existing value plots and summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = load_episode_meta(raw_dir, dataset_dir)
    df = enrich_values(values_path, episodes)
    write_value_summaries(df, output_dir)
    plot_values_by_category(df, output_dir, num_points)
    plot_value_category_means(df, output_dir, num_points)


def generate_cfg_plots(
    cfg_log_dir: Path,
    output_dir: Path,
    smoothing_window: int,
    loss_zoom_max: float,
    title: str,
) -> None:
    """Generate the existing CFG loss dashboard and scalar summary."""
    tensorboard_dir = cfg_log_dir / "tensorboard"
    event_files = sorted(tensorboard_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(
            f"No TensorBoard event files found in {tensorboard_dir}"
        )

    runs = {}
    for index, event_file in enumerate(event_files):
        scalars = load_event_file(event_file)
        runs[event_label(index, event_file, scalars)] = scalars

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_training_dashboard(
        output_dir / "cfg_rl_training_curves.png",
        runs,
        smoothing_window,
        loss_zoom_max,
        title,
    )
    plot_loss_only(
        output_dir / "cfg_rl_loss_curves.png",
        runs,
        smoothing_window,
        loss_zoom_max,
        title.replace("diagnostics", "loss diagnostics"),
    )
    write_cfg_summary(
        output_dir / "cfg_rl_training_summary.json",
        tensorboard_dir,
        event_files,
        runs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Run root, for example logs/aloha_sandwich_recap/20260709-152247.",
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--advantage-tag", default=DEFAULT_ADVANTAGE_TAG)
    parser.add_argument("--advantages-path", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Value/advantage output. Defaults to RUN_DIR/value_advantage_visualizations.",
    )
    parser.add_argument(
        "--cfg-log-dir",
        type=Path,
        default=None,
        help="CFG log input. Defaults to RUN_DIR/cfg_rl.",
    )
    parser.add_argument(
        "--cfg-output-dir",
        type=Path,
        default=None,
        help="CFG output. Defaults to --cfg-log-dir.",
    )
    parser.add_argument("--num-points", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--smoothing-window", type=int, default=25)
    parser.add_argument("--loss-zoom-max", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    output_dir = args.output_dir or run_dir / "value_advantage_visualizations"
    cfg_log_dir = args.cfg_log_dir or run_dir / "cfg_rl"
    cfg_output_dir = args.cfg_output_dir or cfg_log_dir
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

    generate_value_plots(
        raw_dir=args.raw_dir,
        dataset_dir=args.dataset_dir,
        values_path=advantages_path,
        output_dir=output_dir,
        num_points=args.num_points,
    )
    generate_advantage_plots(
        raw_dir=args.raw_dir,
        dataset_dir=args.dataset_dir,
        advantages_path=advantages_path,
        output_dir=output_dir,
        num_points=args.num_points,
        threshold=threshold,
    )
    generate_cfg_plots(
        cfg_log_dir=cfg_log_dir,
        output_dir=cfg_output_dir,
        smoothing_window=args.smoothing_window,
        loss_zoom_max=args.loss_zoom_max,
        title=f"CFG RL finetune diagnostics: {run_dir.name}",
    )

    print(f"Wrote value/advantage plots to: {output_dir.resolve()}")
    print(f"Wrote CFG plots to: {cfg_output_dir.resolve()}")


if __name__ == "__main__":
    main()
