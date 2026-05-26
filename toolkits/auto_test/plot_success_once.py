# Copyright 2025 The RLinf Authors.
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

# !/usr/bin/env python3
"""
Plotting module for success_once metrics.

This module provides functionality to plot success_once curves from parsed
log data. It depends on parse_success_once for data parsing.

Usage:
    python plot_success_once.py <log_path> [options]

Example:
    # Plot single experiment
    python plot_success_once.py logs/experiment1/run_embodiment.log --output curve.png

    # Plot all experiments in a directory
    python plot_success_once.py logs/ --output comparison.png

    # Save data and plot
    python plot_success_once.py logs/ --output comparison.png --plot-data data.csv
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from parse_success_once import (
    print_results,
    process_log_directory,
    process_single_log,
    save_success_once_data,
)


def plot_success_once_curves(
    results: list, output_path: str | None = None, figsize: tuple = (12, 8)
):
    """
    Plot success_once curves for each experiment and save

    Args:
        results: List of results, each element contains step_to_success data
        output_path: Output image path, if not specified use default path
        figsize: Figure size
    """
    if not results:
        print("No data to plot")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Color cycle
    colors = plt.cm.tab10.colors
    color_idx = 0

    for r in results:
        step_to_success = r.get("step_to_success", {})
        if not step_to_success:
            continue

        # Sort by step
        sorted_steps = sorted(step_to_success.keys())
        steps = sorted_steps
        success_values = [step_to_success[s] for s in sorted_steps]

        # Plot curve
        experiment_name = r["experiment_name"]
        ax.plot(
            steps,
            success_values,
            marker="o",
            markersize=4,
            linewidth=1.5,
            color=colors[color_idx % len(colors)],
            label=experiment_name,
        )
        color_idx += 1

    ax.set_xlabel("Global Step", fontsize=12)
    ax.set_ylabel("Success Once", fontsize=12)
    ax.set_title("Success Once vs Global Step", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Determine output path
    if output_path is None:
        output_path = "success_once_curve.png"

    # Save image
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Curve saved to: {output_path}")
    plt.close()


def plot_single_experiment(
    result: dict, output_path: str | None = None, figsize: tuple = (10, 6)
):
    """
    Plot success_once curve for single experiment and save

    Args:
        result: Single experiment result dictionary
        output_path: Output image path
        figsize: Figure size
    """
    step_to_success = result.get("step_to_success", {})
    if not step_to_success:
        print("No data to plot")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Sort by step
    sorted_steps = sorted(step_to_success.keys())
    steps = sorted_steps
    success_values = [step_to_success[s] for s in sorted_steps]

    # Plot curve
    ax.plot(
        steps,
        success_values,
        marker="o",
        markersize=5,
        linewidth=2,
        color="#2E86AB",
        label=result["experiment_name"],
    )

    ax.set_xlabel("Global Step", fontsize=12)
    ax.set_ylabel("Success Once", fontsize=12)
    ax.set_title(f"Success Once Curve - {result['experiment_name']}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set y-axis range
    ax.set_ylim(0, max(max(success_values) * 1.1, 0.1) if success_values else 1.0)

    # Adjust layout padding
    plt.tight_layout()

    # Determine output path if not specified
    if output_path is None:
        output_path = f"{result['experiment_name']}_success_once_curve.png"

    # Save image
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Image saved to: {output_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot success_once curves from training logs"
    )
    parser.add_argument("path", type=str, help="Log file or directory path")
    parser.add_argument(
        "--step", type=int, default=100, help="Target step (default: 100)"
    )
    parser.add_argument(
        "--log-filename",
        type=str,
        default="run_embodiment.log",
        help="Log filename (default: run_embodiment.log)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold for check_global_step, if not specified use 10%% of total steps",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (default: success_once_curve.png or <experiment_name>_success_once_curve.png)",
    )
    parser.add_argument(
        "--plot-data",
        type=str,
        default=None,
        help="Save success_once data for each step to CSV file (default: success_once_data.csv)",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="12,8",
        help="Figure size as width,height (default: 12,8)",
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default=None,
        help="Baseline directory path for comparison (default: logs_baseline)",
    )
    parser.add_argument(
        "--baseline-filename",
        type=str,
        default="run_embodiment.log",
        help="Baseline log filename (default: run_embodiment.log)",
    )
    parser.add_argument(
        "--similarity-method",
        type=str,
        default="pearson",
        choices=["pearson", "spearman", "mse", "mae", "cosine", "dtw", "all"],
        help="Similarity calculation method for baseline comparison (default: pearson)",
    )
    parser.add_argument(
        "--comparison-output",
        type=str,
        default=None,
        help="Output file path for comparison results (default: baseline_comparison.csv)",
    )
    parser.add_argument(
        "--no-comparison-plot",
        action="store_true",
        help="Do not plot comparison curves with baseline",
    )

    args = parser.parse_args()

    path = Path(args.path)

    if path.is_file():
        # Process single file
        results = [process_single_log(str(path), args.step, args.threshold)]
    elif path.is_dir():
        # Process directory
        results = process_log_directory(
            str(path), args.step, args.log_filename, args.threshold
        )
    else:
        print(f"Error: {path} does not exist")
        sys.exit(1)

    # Print results
    print_results(results)

    # Parse figsize
    try:
        figsize = tuple(map(int, args.figsize.split(",")))
    except ValueError:
        figsize = (12, 8)

    # Plot curves
    if len(results) == 1:
        # Single experiment, plot individual curve
        plot_single_experiment(results[0], args.output, figsize)
    else:
        # Multiple experiments, plot comparison curves
        plot_success_once_curves(results, args.output, figsize)

    # Save each step's data to CSV
    if args.plot_data:
        save_success_once_data(results, args.plot_data)

    # Compare with baseline (lazy import to avoid circular dependency with compare_baseline.py)
    from compare_baseline import (
        _find_baseline_log,
        compare_results_with_baseline,
        plot_comparison_with_baseline,
        print_comparison_results,
        save_comparison_results,
    )

    baseline_dir = args.baseline_dir
    if baseline_dir is None:
        # Default baseline directory
        baseline_dir = "logs_baseline"

    baseline_path = Path(baseline_dir)
    if baseline_path.exists() and baseline_path.is_dir():
        print("\n" + "=" * 60)
        print("Baseline Comparison")
        print("=" * 60)

        # Find matching baseline logs for each experiment
        comparison_results = []
        for result in results:
            experiment_name = result["experiment_name"]
            baseline_log_path = _find_baseline_log(
                baseline_path, experiment_name, args.baseline_filename
            )

            if baseline_log_path:
                try:
                    comparison = compare_results_with_baseline(
                        [result], baseline_log_path, args.similarity_method
                    )[0]
                    comparison_results.append(comparison)
                except Exception as e:
                    comparison_results.append(
                        {
                            "experiment_name": experiment_name,
                            "baseline_log_path": baseline_log_path,
                            "error": str(e),
                        }
                    )
            else:
                comparison_results.append(
                    {
                        "experiment_name": experiment_name,
                        "baseline_log_path": None,
                        "error": f"No matching baseline found for {experiment_name}",
                    }
                )

        # Print comparison results
        print_comparison_results(comparison_results)

        # Save comparison results
        save_comparison_results(comparison_results, args.comparison_output)

        # Plot comparison curves
        if not args.no_comparison_plot:
            for i, result in enumerate(results):
                experiment_name = result["experiment_name"]
                baseline_log_path = _find_baseline_log(
                    baseline_path, experiment_name, args.baseline_filename
                )

                if baseline_log_path:
                    try:
                        if args.output:
                            output_path = (
                                f"{args.output.rsplit('.', 1)[0]}_vs_baseline_{i}.png"
                            )
                        else:
                            output_path = f"{experiment_name}_vs_baseline.png"
                        plot_comparison_with_baseline(
                            result, baseline_log_path, output_path, figsize
                        )
                    except Exception as e:
                        print(f"Failed to plot comparison for {experiment_name}: {e}")
    else:
        if args.baseline_dir is not None:
            print(f"\nWarning: Baseline directory {baseline_dir} does not exist")


if __name__ == "__main__":
    main()
