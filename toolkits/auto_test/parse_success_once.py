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
Log parsing module for success_once metrics.

This module provides functionality to parse training logs and extract
success_once metrics without any plotting dependencies.

Usage:
    python parse_success_once.py <log_path> [options]

Example:
    # Parse single log file
    python parse_success_once.py logs/experiment1/run_embodiment.log

    # Parse all logs in a directory
    python parse_success_once.py logs/ --log-filename run_embodiment.log

    # Save parsed data to CSV
    python parse_success_once.py logs/ --output results.csv --plot-data success_once.csv
"""

import re
import sys
from pathlib import Path
from typing import Optional

from check import check_global_step


def parse_log_file(log_path: str) -> tuple[str, dict[int, float]]:
    """
    Parse log file to extract experiment name and success_once values for each step

    Args:
        log_path: Log file path

    Returns:
        experiment_name: Experiment name
        step_to_success: Mapping dictionary from step to success_once
    """
    experiment_name = ""
    step_to_success = {}

    # Extract experiment name from path
    # Path format: .../20260413-02:04:24-libero_90_grpo_openvlaoft/run_embodiment.log
    path_obj = Path(log_path)
    parent_dir = path_obj.parent.name
    if parent_dir:
        # Extract the part after date-time as experiment name
        match = re.match(r"\d{8}-\d{2}:\d{2}:\d{2}-(.+)", parent_dir)
        if match:
            experiment_name = match.group(1)
        else:
            experiment_name = parent_dir

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Use regex to match Metric Table blocks
    # Match Global Step: X/1000 and success_once=xxx
    metric_pattern = r"Global Step:\s*(\d+)/\d+.*?success_once=([\d.]+)"
    matches = re.findall(metric_pattern, content, re.DOTALL)

    for match in matches:
        step = int(match[0])
        success_once = float(match[1])
        step_to_success[step] = success_once

    return experiment_name, step_to_success


def get_target_success_once(
    step_to_success: dict[int, float], target_step: int = 100
) -> tuple[int, Optional[float]]:
    """
    Get success_once value for target step, if not exists return the last step's value

    Args:
        step_to_success: Mapping dictionary from step to success_once
        target_step: Target step, default is 100

    Returns:
        actual_step: Actual step used
        success_once: Corresponding success_once value
    """
    if not step_to_success:
        return 0, None

    if target_step in step_to_success:
        return target_step, step_to_success[target_step]

    # If target step doesn't exist, return the last step
    last_step = max(step_to_success.keys())
    return last_step, step_to_success[last_step]


def process_single_log(
    log_path: str, target_step: int = 100, threshold: Optional[float] = None
) -> dict:
    """
    Process a single log file

    Args:
        log_path: Log file path
        target_step: Target step
        threshold: Threshold for check_global_step

    Returns:
        Dictionary containing results
    """
    experiment_name, step_to_success = parse_log_file(log_path)
    actual_step, success_once = get_target_success_once(step_to_success, target_step)

    # Call check_global_step
    reached_threshold, crashed_before_threshold = check_global_step(
        log_path, threshold, verbose=False
    )

    result = {
        "log_path": log_path,
        "experiment_name": experiment_name,
        "target_step": target_step,
        "actual_step": actual_step,
        "success_once": success_once,
        "total_steps_found": len(step_to_success),
        "max_step": max(step_to_success.keys()) if step_to_success else 0,
        "found_target": target_step in step_to_success,
        "reached_threshold": reached_threshold,
        "crashed_before_threshold": crashed_before_threshold,
        "step_to_success": step_to_success,  # Save success_once data for each step
    }

    return result


def process_log_directory(
    log_dir: str,
    target_step: int = 100,
    log_filename: str = "run_embodiment.log",
    threshold: Optional[float] = None,
) -> list:
    """
    Process all log files in directory

    Args:
        log_dir: Log directory path
        target_step: Target step
        log_filename: Log filename
        threshold: Threshold for check_global_step

    Returns:
        List of results
    """
    results = []
    log_dir = Path(log_dir)

    # Find all matching log files
    for log_file in log_dir.rglob(log_filename):
        try:
            result = process_single_log(str(log_file), target_step, threshold)
            results.append(result)
        except Exception as e:
            print(f"Error processing {log_file}: {e}")

    return results


def print_results(results: list):
    """
    Print results

    Args:
        results: List of results
    """
    print("=" * 140)
    print(
        f"{'Experiment Name':<45} {'Step':<8} {'success_once':<14} {'reached':<10} {'crashed':<10} {'Note':<30}"
    )
    print("=" * 140)

    for r in results:
        note = (
            "Target step found"
            if r["found_target"]
            else f"Using last step (total {r['total_steps_found']} steps)"
        )
        success_str = (
            f"{r['success_once']:.6f}" if r["success_once"] is not None else "N/A"
        )
        reached_str = str(r["reached_threshold"])
        crashed_str = str(r["crashed_before_threshold"])
        print(
            f"{r['experiment_name']:<45} {r['actual_step']:<8} {success_str:<14} {reached_str:<10} {crashed_str:<10} {note:<30}"
        )

    print("=" * 140)


def save_success_once_data(results: list, output_path: str | None = None):
    """
    Save success_once data to CSV file

    Args:
        results: List of results, each element contains step_to_success data
        output_path: Output file path
    """
    if not results:
        return

    if output_path is None:
        output_path = "success_once_data.csv"

    with open(output_path, "w") as f:
        f.write("experiment_name,step,success_once\n")

        for r in results:
            step_to_success = r.get("step_to_success", {})
            experiment_name = r["experiment_name"]

            for step in sorted(step_to_success.keys()):
                f.write(f"{experiment_name},{step},{step_to_success[step]}\n")

    print(f"Data saved to path: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse training logs and extract success_once metrics"
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
    parser.add_argument("--output", type=str, help="Output file path (optional)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold for check_global_step, if not specified use 10%% of total steps",
    )
    parser.add_argument(
        "--plot-data",
        type=str,
        default=None,
        help="Save success_once data for each step to CSV file (default: success_once_data.csv)",
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

    # Output to file
    if args.output:
        with open(args.output, "w") as f:
            f.write(
                "experiment_name,actual_step,success_once,reached_threshold,crashed_before_threshold,found_target,total_steps,max_step,log_path\n"
            )
            for r in results:
                f.write(
                    f"{r['experiment_name']},{r['actual_step']},{r['success_once']},{r['reached_threshold']},{r['crashed_before_threshold']},{r['found_target']},{r['total_steps_found']},{r['max_step']},{r['log_path']}\n"
                )
        print(f"\nResults saved to: {args.output}")

    # Save each step's data to CSV
    if args.plot_data:
        save_success_once_data(results, args.plot_data)


if __name__ == "__main__":
    main()
