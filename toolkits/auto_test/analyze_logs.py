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
Top-level script for log analysis workflow.

This script integrates all log analysis functionalities:
1. Parse logs to extract success_once metrics
2. Plot success_once curves
3. Compare with baseline experiments

Usage:
    python analyze_logs.py <log_path> [options]

Example:
    # Full analysis with default settings
    python analyze_logs.py logs/

    # Specify output paths
    python analyze_logs.py logs/ --output-dir results/

    # Custom baseline directory
    python analyze_logs.py logs/ --baseline-dir logs_baseline/
"""

import argparse
import sys
from pathlib import Path

from compare_baseline import (
    _find_baseline_log,
    compare_results_with_baseline,
    plot_comparison_with_baseline,
    print_comparison_results,
    save_comparison_results,
)
from parse_success_once import (
    print_results,
    process_log_directory,
    process_single_log,
    save_success_once_data,
)
from plot_success_once import (
    plot_single_experiment,
)


def analyze_logs(
    log_path: str,
    output_dir: str | None = None,
    baseline_dir: str = "logs_baseline",
    log_filename: str = "run_embodiment.log",
    baseline_filename: str = "run_embodiment.log",
    target_step: int = 100,
    threshold: float | None = None,
    similarity_method: str = "pearson",
    similarity_threshold: float | None = None,
    figsize: tuple = (12, 8),
    skip_plot: bool = False,
    skip_comparison: bool = False,
    skip_comparison_plot: bool = False,
):
    """
    Complete log analysis workflow.

    Args:
        log_path: Log file or directory path
        output_dir: Output directory for all results
        baseline_dir: Baseline directory path
        log_filename: Log filename to search
        baseline_filename: Baseline log filename
        target_step: Target step for analysis
        threshold: Threshold for check_global_step
        similarity_method: Similarity calculation method
        similarity_threshold: Threshold for similarity alert
        figsize: Figure size for plots
        skip_plot: Skip plotting curves
        skip_comparison: Skip baseline comparison
        skip_comparison_plot: Skip comparison plots
    """
    path = Path(log_path)

    # Setup output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(".")

    # Step 1: Parse logs
    print("\n" + "=" * 60)
    print("Parsing Logs ...")
    print("=" * 60)

    if path.is_file():
        results = [process_single_log(str(path), target_step, threshold)]
    elif path.is_dir():
        results = process_log_directory(str(path), target_step, log_filename, threshold)
    else:
        print(f"Error: {path} does not exist")
        sys.exit(1)

    # Print parsed results
    print_results(results)

    # Save parsed data
    parse_output = output_path / "parse_results.csv"
    save_success_once_data(results, str(parse_output))

    # Step 2: Plot curves (each experiment vs baseline in separate figures)
    if not skip_plot:
        print("\n" + "=" * 60)
        print("Plotting Curves ...")
        print("=" * 60)

        plot_data_output = output_path / "plot_data.csv"
        save_success_once_data(results, str(plot_data_output))

        # Check if baseline directory exists for comparison plots
        baseline_path = Path(baseline_dir)
        has_baseline = baseline_path.exists() and baseline_path.is_dir()

        # Create a directory for individual plots
        curves_dir = output_path / "curves"
        curves_dir.mkdir(parents=True, exist_ok=True)

        for result in results:
            experiment_name = result["experiment_name"]
            output_img = curves_dir / f"{experiment_name}_curve.png"

            if has_baseline:
                # Find matching baseline log
                baseline_log_path = _find_baseline_log(
                    baseline_path, experiment_name, baseline_filename
                )

                if baseline_log_path:
                    # Plot experiment vs baseline comparison
                    try:
                        plot_comparison_with_baseline(
                            result, baseline_log_path, str(output_img), figsize
                        )
                    except Exception as e:
                        print(f"Failed to plot comparison for {experiment_name}: {e}")
                        # Fall back to single experiment plot
                        plot_single_experiment(result, str(output_img), figsize)
                else:
                    # No matching baseline, plot single experiment
                    plot_single_experiment(result, str(output_img), figsize)
            else:
                # No baseline directory, plot single experiment
                plot_single_experiment(result, str(output_img), figsize)

    # Step 3: Compare with baseline
    if not skip_comparison:
        print("\n" + "=" * 60)
        print("Conducting Baseline Comparison")
        print("=" * 60)

        baseline_path = Path(baseline_dir)
        if not baseline_path.exists() or not baseline_path.is_dir():
            print(
                f"Warning: Baseline directory {baseline_dir} does not exist, skipping comparison"
            )
            return results

        # Find matching baseline logs for each experiment
        # Use full directory name (with timestamp) for precise matching
        comparison_results = []
        for result in results:
            experiment_name = result["experiment_name"]
            log_path = result.get("log_path", "")
            baseline_log_path = _find_baseline_log(
                baseline_path, experiment_name, baseline_filename
            )

            if baseline_log_path:
                try:
                    comparison = compare_results_with_baseline(
                        [result], baseline_log_path, similarity_method
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
        print_comparison_results(comparison_results, similarity_threshold)

        # Save comparison results
        comparison_output = output_path / "baseline_comparison.csv"
        save_comparison_results(comparison_results, str(comparison_output))

        # Plot comparison curves
        if not skip_comparison_plot:
            print("\n" + "=" * 60)
            print("Plotting Comparison Curves ...")
            print("=" * 60)

            comparison_plot_dir = output_path / "comparison_plots"
            comparison_plot_dir.mkdir(parents=True, exist_ok=True)

            for i, result in enumerate(results):
                experiment_name = result["experiment_name"]
                baseline_log_path = _find_baseline_log(
                    baseline_path, experiment_name, baseline_filename
                )

                if baseline_log_path:
                    try:
                        output_img = (
                            comparison_plot_dir / f"{experiment_name}_vs_baseline.png"
                        )
                        plot_comparison_with_baseline(
                            result, baseline_log_path, str(output_img), figsize
                        )
                    except Exception as e:
                        print(f"Failed to plot comparison for {experiment_name}: {e}")

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Results saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Complete log analysis workflow: parse, plot, and compare with baseline"
    )
    parser.add_argument("path", type=str, help="Log file or directory path")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for all results (default: current directory)",
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default="logs_baseline",
        help="Baseline directory path (default: logs_baseline)",
    )
    parser.add_argument(
        "--log-filename",
        type=str,
        default="run_embodiment.log",
        help="Log filename (default: run_embodiment.log)",
    )
    parser.add_argument(
        "--baseline-filename",
        type=str,
        default="run_embodiment.log",
        help="Baseline log filename (default: run_embodiment.log)",
    )
    parser.add_argument(
        "--step", type=int, default=100, help="Target step (default: 100)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold for check_global_step",
    )
    parser.add_argument(
        "--similarity-method",
        type=str,
        default="pearson",
        choices=["pearson", "spearman", "mse", "mae", "cosine", "dtw", "all"],
        help="Similarity calculation method (default: pearson)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.2,
        help="Threshold for similarity alert. For correlation methods (pearson, spearman, cosine): alert if value < threshold. For error methods (mse, mae, dtw): alert if value > threshold",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="12,8",
        help="Figure size as width,height (default: 12,8)",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip plotting curves",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip baseline comparison",
    )
    parser.add_argument(
        "--skip-comparison-plot",
        action="store_true",
        help="Skip comparison plots",
    )

    args = parser.parse_args()

    # Parse figsize
    try:
        figsize = tuple(map(int, args.figsize.split(",")))
    except ValueError:
        figsize = (12, 8)

    # Run analysis
    analyze_logs(
        log_path=args.path,
        output_dir=args.output_dir,
        baseline_dir=args.baseline_dir,
        log_filename=args.log_filename,
        baseline_filename=args.baseline_filename,
        target_step=args.step,
        threshold=args.threshold,
        similarity_method=args.similarity_method,
        similarity_threshold=args.similarity_threshold,
        figsize=figsize,
        skip_plot=args.skip_plot,
        skip_comparison=args.skip_comparison,
        skip_comparison_plot=args.skip_comparison_plot,
    )


if __name__ == "__main__":
    main()
