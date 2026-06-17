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

# !/usr/bin/env python3
"""
Baseline comparison module for success_once metrics.

This module provides functionality to compare experiment results with baseline
results using various similarity metrics.

Usage:
    python compare_baseline.py <log_path> --baseline <baseline_log_path> [options]

Example:
    # Compare single experiment with baseline
    python compare_baseline.py logs/experiment1/run_embodiment.log --baseline log-baseline/run_embodiment.log

    # Compare all experiments in a directory with baseline
    python compare_baseline.py logs/ --baseline log-baseline/run_embodiment.log

    # Use specific similarity method
    python compare_baseline.py logs/ --baseline log-baseline/run_embodiment.log --similarity-method spearman
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from parse_success_once import (
    parse_log_file,
    print_results,
    process_log_directory,
    process_single_log,
    save_success_once_data,
)
from scipy import stats


def _find_baseline_log(baseline_path, experiment_name, filename):
    """Find a baseline log file by matching experiment name via directory suffix.

    Args:
        baseline_path: Path to the baseline directory.
        experiment_name: Experiment name to match (directory must end with '-{experiment_name}').
        filename: Log filename inside the matched subdirectory.

    Returns:
        String path to the log file, or None if no match found.
    """
    for d in baseline_path.iterdir():
        if d.is_dir() and d.name.endswith(f"-{experiment_name}"):
            log = d / filename
            if log.exists():
                return str(log)
    return None


def compute_curve_similarity(
    step_to_success_1: dict[int, float],
    step_to_success_2: dict[int, float],
    method: str = "pearson",
) -> dict[str, float]:
    """
    Compute similarity between two curves based on success_once data

    Args:
        step_to_success_1: First curve step-to-success mapping
        step_to_success_2: Second curve step-to-success mapping
        method: Similarity calculation method, supported:
            - 'pearson': Pearson correlation coefficient (default)
            - 'spearman': Spearman correlation coefficient
            - 'mse': Mean squared error (MSE)
            - 'mae': Mean absolute error (MAE)
            - 'cosine': Cosine similarity
            - 'dtw': Dynamic time warping distance
            - 'all': Return all metrics

    Returns:
        Dictionary containing similarity metrics
    """
    # Find common steps
    steps_1 = set(step_to_success_1.keys())
    steps_2 = set(step_to_success_2.keys())
    common_steps = sorted(steps_1 & steps_2)

    if len(common_steps) < 2:
        return {
            "method": method,
            "similarity": None,
            "common_steps": len(common_steps),
            "error": "Not enough common steps for comparison (need at least 2)",
        }

    values_1 = np.array([step_to_success_1[s] for s in common_steps])
    values_2 = np.array([step_to_success_2[s] for s in common_steps])

    results = {"common_steps": len(common_steps)}

    def pearson_correlation(v1, v2):
        """Pearson correlation coefficient"""
        if np.std(v1) == 0 or np.std(v2) == 0:
            return None
        return stats.pearsonr(v1, v2)[0]

    def spearman_correlation(v1, v2):
        """Spearman correlation coefficient"""
        if np.std(v1) == 0 or np.std(v2) == 0:
            return None
        return stats.spearmanr(v1, v2)[0]

    def mse(v1, v2):
        """Mean squared error (MSE)"""
        return np.mean((v1 - v2) ** 2)

    def mae(v1, v2):
        """Mean absolute error (MAE)"""
        return np.mean(np.abs(v1 - v2))

    def cosine_similarity(v1, v2):
        """Cosine similarity"""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return None
        return np.dot(v1, v2) / (norm1 * norm2)

    def dtw_distance(v1, v2):
        """Dynamic time warping distance"""
        n, m = len(v1), len(v2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(v1[i - 1] - v2[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],  # insertion
                    dtw_matrix[i, j - 1],  # deletion
                    dtw_matrix[i - 1, j - 1],  # match
                )
        return dtw_matrix[n, m]

    # Calculate similarity based on method
    if method == "pearson":
        results["method"] = "pearson"
        results["similarity"] = pearson_correlation(values_1, values_2)
        results["description"] = (
            "Pearson correlation coefficient (range [-1, 1], closer to 1 means more similar)"
        )
    elif method == "spearman":
        results["method"] = "spearman"
        results["similarity"] = spearman_correlation(values_1, values_2)
        results["description"] = (
            "Spearman correlation coefficient (range [-1, 1], closer to 1 means more similar)"
        )
    elif method == "mse":
        results["method"] = "mse"
        results["similarity"] = mse(values_1, values_2)
        results["description"] = "Mean Squared Error (smaller means more similar)"
    elif method == "mae":
        results["method"] = "mae"
        results["similarity"] = mae(values_1, values_2)
        results["description"] = "Mean Absolute Error (smaller means more similar)"
    elif method == "cosine":
        results["method"] = "cosine"
        results["similarity"] = cosine_similarity(values_1, values_2)
        results["description"] = (
            "Cosine similarity (range [-1, 1], closer to 1 means more similar)"
        )
    elif method == "dtw":
        results["method"] = "dtw"
        results["similarity"] = dtw_distance(values_1, values_2)
        results["description"] = (
            "Dynamic Time Warping distance (smaller means more similar)"
        )
    elif method == "all":
        results["method"] = "all"
        results["pearson"] = pearson_correlation(values_1, values_2)
        results["spearman"] = spearman_correlation(values_1, values_2)
        results["mse"] = mse(values_1, values_2)
        results["mae"] = mae(values_1, values_2)
        results["cosine"] = cosine_similarity(values_1, values_2)
        results["dtw"] = dtw_distance(values_1, values_2)
    else:
        raise ValueError(
            f"Unknown method: {method}. Supported methods: pearson, spearman, mse, mae, cosine, dtw, all"
        )

    return results


def compare_with_baseline(
    result: dict, baseline_log_path: str, method: str = "pearson"
) -> dict:
    """
    Compare single experiment result with baseline

    Args:
        result: Experiment result dictionary (contains step_to_success)
        baseline_log_path: Baseline log file path
        method: Similarity calculation method

    Returns:
        Dictionary containing comparison results
    """
    baseline_name, baseline_step_to_success = parse_log_file(baseline_log_path)

    similarity_result = compute_curve_similarity(
        result.get("step_to_success", {}), baseline_step_to_success, method=method
    )

    # Calculate last step value difference
    exp_step_to_success = result.get("step_to_success", {})
    last_step_diff = None
    if exp_step_to_success and baseline_step_to_success:
        # Find the last step for experiment
        exp_last_step = max(exp_step_to_success.keys())
        exp_last_value = exp_step_to_success[exp_last_step]

        # Find the last step for baseline
        baseline_last_step = max(baseline_step_to_success.keys())
        baseline_last_value = baseline_step_to_success[baseline_last_step]

        # Calculate difference: experiment - baseline
        last_step_diff = exp_last_value - baseline_last_value

    return {
        "experiment_name": result["experiment_name"],
        "baseline_name": baseline_name,
        "baseline_log_path": baseline_log_path,
        "similarity_result": similarity_result,
        "last_step_diff": last_step_diff,
    }


def compare_results_with_baseline(
    results: list, baseline_log_path: str, method: str = "pearson"
) -> list:
    """
    Batch compare multiple experiment results with baseline

    Args:
        results: Experiment result list
        baseline_log_path: Baseline log file path
        method: Similarity calculation method

    Returns:
        Comparison result list
    """
    comparison_results = []
    for result in results:
        try:
            comparison = compare_with_baseline(result, baseline_log_path, method)
            comparison_results.append(comparison)
        except Exception as e:
            comparison_results.append(
                {
                    "experiment_name": result.get("experiment_name", "unknown"),
                    "baseline_log_path": baseline_log_path,
                    "error": str(e),
                }
            )
    return comparison_results


# ANSI color codes for terminal output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_comparison_results(
    comparison_results: list, similarity_threshold: float | None = None
):
    """
    Print comparison results with threshold alert

    Args:
        comparison_results: Comparison result list
        similarity_threshold: Threshold for similarity alert (default: None, no alert)
                             For correlation methods (pearson, spearman, cosine): alert if value < threshold
                             For error methods (mse, mae, dtw): alert if value > threshold
    """
    # Determine if we need to show alert column
    show_alert = similarity_threshold is not None

    if show_alert:
        print("=" * 160)
        print(
            f"{'Experiment Name':<45} {'Baseline Name':<35} {'Method':<10} {'Similarity':<12} {'Steps':<8} {'Last Step Diff':<15} {'Alert':<20}"
        )
        print("=" * 160)
    else:
        print("=" * 140)
        print(
            f"{'Experiment Name':<45} {'Baseline Name':<35} {'Similarity Method':<12} {'Similarity Value':<15} {'Common Steps':<10} {'Last Step Diff':<15}"
        )
        print("=" * 140)

    alert_count = 0
    for c in comparison_results:
        if "error" in c:
            if show_alert:
                print(
                    f"{c['experiment_name']:<45} {'N/A':<35} {'N/A':<10} {'N/A':<12} {'0':<8} {'N/A':<15} {'Error: ' + c['error']:<20}"
                )
            else:
                print(
                    f"{c['experiment_name']:<45} {'N/A':<35} {'N/A':<12} {'Error: ' + c['error']:<15} {'N/A':<15}"
                )
            continue

        sim_result = c["similarity_result"]
        method = sim_result.get("method", "N/A")
        similarity = sim_result.get("similarity")
        common_steps = sim_result.get("common_steps", 0)
        last_step_diff = c.get("last_step_diff")

        if similarity is not None:
            sim_str = f"{similarity:.6f}"
        else:
            sim_str = "N/A"

        # Format last step difference
        if last_step_diff is not None:
            diff_str = f"{last_step_diff:+.6f}"
            # Color based on positive/negative difference
            if last_step_diff > 0:
                diff_display = f"{GREEN}{diff_str}{RESET}"
            elif last_step_diff < 0:
                diff_display = f"{RED}{diff_str}{RESET}"
            else:
                diff_display = diff_str
        else:
            diff_display = "N/A"

        # Check threshold and generate alert
        alert_msg = ""
        is_alert = False
        if show_alert and similarity is not None:
            # For correlation methods (higher is better), alert if below threshold
            if method in ["pearson", "spearman", "cosine"]:
                if similarity < similarity_threshold:
                    is_alert = True
                    alert_msg = f"{RED}{BOLD}⚠ LOW ({similarity:.4f} < {similarity_threshold}){RESET}"
                    alert_count += 1
                else:
                    alert_msg = f"{GREEN}✓ OK{RESET}"
            # For error methods (lower is better), alert if above threshold
            elif method in ["mse", "mae", "dtw"]:
                if similarity > similarity_threshold:
                    is_alert = True
                    alert_msg = f"{RED}{BOLD}⚠ HIGH ({similarity:.4f} > {similarity_threshold}){RESET}"
                    alert_count += 1
                else:
                    alert_msg = f"{GREEN}✓ OK{RESET}"

        if show_alert:
            # Print with color for similarity value if alert
            if is_alert:
                sim_display = f"{RED}{BOLD}{sim_str}{RESET}"
            else:
                sim_display = sim_str
            print(
                f"{c['experiment_name']:<45} {c['baseline_name']:<35} {method:<10} {sim_display:<12} {common_steps:<8} {diff_display:<15} {alert_msg}"
            )
        else:
            print(
                f"{c['experiment_name']:<45} {c['baseline_name']:<35} {method:<12} {sim_str:<15} {common_steps:<10} {diff_display:<15}"
            )

    if show_alert:
        print("=" * 160)
        if alert_count > 0:
            print(f"{RED}{BOLD}Total Alerts: {alert_count}{RESET}")
        else:
            print(f"{GREEN}All experiments passed threshold check{RESET}")
    else:
        print("=" * 140)


def plot_comparison_with_baseline(
    result: dict,
    baseline_log_path: str,
    output_path: str | None = None,
    figsize: tuple = (12, 8),
):
    """
    Plot comparison between experiment curve and baseline curve

    Args:
        result: Experiment result dictionary
        baseline_log_path: Baseline log file path
        output_path: Output image path
        figsize: Figure size
    """
    # Parse baseline
    baseline_name, baseline_step_to_success = parse_log_file(baseline_log_path)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot experiment curve
    step_to_success = result.get("step_to_success", {})
    if step_to_success:
        sorted_steps = sorted(step_to_success.keys())
        steps = sorted_steps
        success_values = [step_to_success[s] for s in sorted_steps]
        ax.plot(
            steps,
            success_values,
            marker="o",
            markersize=4,
            linewidth=2,
            color="#2E86AB",
            label=result["experiment_name"],
        )

    # Plot baseline curve
    if baseline_step_to_success:
        sorted_steps = sorted(baseline_step_to_success.keys())
        steps = sorted_steps
        success_values = [baseline_step_to_success[s] for s in sorted_steps]
        ax.plot(
            steps,
            success_values,
            marker="s",
            markersize=4,
            linewidth=2,
            color="#E94F37",
            label=f"Baseline: {baseline_name}",
            linestyle="--",
        )

    ax.set_xlabel("Global Step", fontsize=12)
    ax.set_ylabel("Success Once", fontsize=12)
    ax.set_title(
        f"Success Once Comparison: {result['experiment_name']} vs Baseline", fontsize=14
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path is None:
        output_path = f"{result['experiment_name']}_vs_baseline.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved to: {output_path}")
    plt.close()


def save_comparison_results(comparison_results: list, output_path: str | None = None):
    """
    Save comparison results to CSV file

    Args:
        comparison_results: Comparison result list
        output_path: Output file path
    """
    if not comparison_results:
        return

    if output_path is None:
        output_path = "baseline_comparison.csv"

    with open(output_path, "w") as f:
        f.write(
            "experiment_name,baseline_name,method,similarity,common_steps,last_step_diff,baseline_log_path\n"
        )

        for c in comparison_results:
            if "error" in c:
                f.write(
                    f"{c['experiment_name']},N/A,N/A,Error: {c['error']},0,N/A,{c['baseline_log_path']}\n"
                )
                continue

            sim_result = c["similarity_result"]
            similarity = sim_result.get("similarity", "")
            if isinstance(similarity, (int, float)):
                similarity = f"{similarity:.6f}"
            elif similarity is None:
                similarity = "N/A"
            method = sim_result.get("method", "N/A")
            common_steps = sim_result.get("common_steps", 0)
            last_step_diff = c.get("last_step_diff")
            if last_step_diff is not None:
                last_step_diff_str = f"{last_step_diff:.6f}"
            else:
                last_step_diff_str = "N/A"
            f.write(
                f"{c['experiment_name']},{c['baseline_name']},{method},{similarity},{common_steps},{last_step_diff_str},{c['baseline_log_path']}\n"
            )

    print(f"Comparison results saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare experiment results with baseline using similarity metrics"
    )
    parser.add_argument("path", type=str, help="Log file or directory path")
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Baseline log file path (default: log-baseline/run_embodiment.log)",
    )
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
        "--similarity-method",
        type=str,
        default="pearson",
        choices=["pearson", "spearman", "mse", "mae", "cosine", "dtw", "all"],
        help="Similarity calculation method (default: pearson)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Threshold for similarity alert. For correlation methods (pearson, spearman, cosine): alert if value < threshold. For error methods (mse, mae, dtw): alert if value > threshold",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for comparison results (default: baseline_comparison.csv)",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Plot comparison curves and save to specified path",
    )
    parser.add_argument("--no-plot", action="store_true", help="Do not plot curves")
    parser.add_argument(
        "--plot-data",
        type=str,
        default=None,
        help="Save success_once data for each step to CSV file",
    )

    args = parser.parse_args()

    path = Path(args.path)
    baseline_path = Path(args.baseline)

    # Check baseline path
    if not baseline_path.exists():
        print(f"Error: baseline file {args.baseline} does not exist")
        sys.exit(1)

    # Process log files
    if path.is_file():
        results = [process_single_log(str(path), args.step, args.threshold)]
    elif path.is_dir():
        results = process_log_directory(
            str(path), args.step, args.log_filename, args.threshold
        )
    else:
        print(f"Error: {path} does not exist")
        sys.exit(1)

    # Print basic results
    print("\n" + "=" * 60)
    print("Experiment Results")
    print("=" * 60)
    print_results(results)

    # Save each step's data to CSV
    if args.plot_data:
        save_success_once_data(results, args.plot_data)

    # Plot individual curves
    if not args.no_plot and args.plot:
        # Lazy import to avoid circular dependency
        from plot_success_once import plot_single_experiment, plot_success_once_curves

        if len(results) == 1:
            plot_single_experiment(results[0], args.plot)
        else:
            plot_success_once_curves(results, args.plot)

    # Compare with baseline
    print("\n" + "=" * 60)
    print("Baseline Comparison Results")
    print("=" * 60)

    comparison_results = compare_results_with_baseline(
        results, args.baseline, args.similarity_method
    )
    print_comparison_results(comparison_results, args.similarity_threshold)

    # Save comparison results
    save_comparison_results(comparison_results, args.output)

    # Plot comparison charts
    if not args.no_plot:
        for i, result in enumerate(results):
            if args.plot:
                output_path = f"{args.plot.rsplit('.', 1)[0]}_comparison_{i}.png"
            else:
                output_path = f"{result['experiment_name']}_vs_baseline.png"
            try:
                plot_comparison_with_baseline(result, args.baseline, output_path)
            except Exception as e:
                print(f"Failed to plot comparison chart: {e}")


if __name__ == "__main__":
    main()
