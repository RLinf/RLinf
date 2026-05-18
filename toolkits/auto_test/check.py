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

import argparse
import os
import re
from pathlib import Path
from typing import Optional


def check_global_step(
    log_file_path: str, threshold: Optional[float] = None, verbose: bool = True
) -> tuple[bool, bool]:
    """
    Find Global Step in log file and check if step >= threshold

    Args:
        log_file_path: Log file path
        threshold: Optional threshold, if None then use 10% of total steps
        verbose: Whether to print verbose information

    Returns:
        Tuple[bool, bool]:
            - First bool: Whether threshold is reached
            - Second bool: Whether error occurred (threshold not reached and program crashed)
    """
    pattern = re.compile(r"Global Step:\s*(\d+)/(\d+)")
    max_step = 0
    total_steps = 0
    has_error = False

    error_patterns = [
        r"Traceback\s*\(most recent call last\)",
        r"Error:",
        r"Exception:",
        r"CRASHED",
        r"Killed",
        r"OOM",
        r"CUDA out of memory",
        r"RuntimeError",
        r"AssertionError",
        r"KeyboardInterrupt",
        r"Segmentation fault",
        r"Aborted",
    ]
    error_regex = re.compile("|".join(error_patterns), re.IGNORECASE)

    with open(log_file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                current_step = int(match.group(1))
                total_steps = int(match.group(2))
                max_step = max(max_step, current_step)

            if error_regex.search(line):
                has_error = True

    if total_steps == 0:
        return False, has_error

    if threshold is None:
        threshold = total_steps * 0.1  # Default: 10% of total steps

    reached_threshold = max_step >= threshold
    crashed_before_threshold = not reached_threshold and has_error

    return reached_threshold, crashed_before_threshold


def check_experiment_logs(
    log_dir: str,
    experiment_name: str,
    threshold: Optional[float] = None,
    verbose: bool = True,
) -> tuple[bool, bool]:
    """
    Check all log directories matching the experiment name.

    Logic:
    - If any log reached threshold: return (True, False)
    - If none reached and any crashed: return (False, True)
    - Otherwise (none reached, none crashed): return (False, False)

    Args:
        log_dir: Base log directory (e.g., /path/to/logs)
        experiment_name: Experiment name to match (e.g., maniskill_ppo_openvlaoft)
        threshold: Optional threshold for checking
        verbose: Whether to print verbose information

    Returns:
        Tuple[bool, bool]: (reached, crashed)
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        if verbose:
            print(f"Log directory not found: {log_dir}")
        return False, False

    # Find all directories matching *-{experiment_name}
    matching_dirs = list(log_path.glob(f"*-{experiment_name}"))
    
    if not matching_dirs:
        if verbose:
            print(f"No matching log directories found for: {experiment_name}")
        return False, False

    any_reached = False
    any_crashed = False

    for log_dir_path in matching_dirs:
        log_file = log_dir_path / "run_embodiment.log"
        if not log_file.exists():
            continue

        if verbose:
            print(f"Checking: {log_file}")

        reached, crashed = check_global_step(str(log_file), threshold, verbose=False)

        if verbose:
            print(f"  - reached={reached}, crashed={crashed}")

        if reached:
            any_reached = True
        if crashed:
            any_crashed = True

    # Apply the logic:
    # - If any reached: return (True, False)
    # - If none reached and any crashed: return (False, True)
    # - Otherwise: return (False, False)
    if any_reached:
        return True, False
    elif any_crashed:
        return False, True
    else:
        return False, False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check Global Step status in experiment logs"
    )
    parser.add_argument(
        "log_path",
        help="Log file path (for single file) or log directory (for experiment check)",
    )
    parser.add_argument(
        "--experiment",
        "-e",
        help="Experiment name to check (when log_path is a directory)",
    )
    parser.add_argument(
        "--format",
        choices=["simple", "verbose"],
        default="verbose",
        help="Output format: simple (only output reached,crashed) or verbose (detailed information)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Custom threshold, if not specified then use 10% of total steps",
    )

    args = parser.parse_args()

    verbose = args.format == "verbose"

    # Determine if we're checking a single file or an experiment
    log_path = Path(args.log_path)
    
    if args.experiment:
        # Check all logs for an experiment in a directory
        reached, crashed = check_experiment_logs(
            args.log_path, args.experiment, args.threshold, verbose=verbose
        )
    elif log_path.is_file():
        # Check single log file
        reached, crashed = check_global_step(args.log_path, args.threshold, verbose=verbose)
    elif log_path.is_dir():
        # Directory provided but no experiment name - check if it's a log directory
        # Try to infer experiment name from directory name (format: timestamp-experiment_name)
        dir_name = log_path.name
        if "-" in dir_name:
            # Extract experiment name (last part after timestamp)
            experiment_name = dir_name.split("-", maxsplit=1)[-1]
            log_file = log_path / "run_embodiment.log"
            if log_file.exists():
                reached, crashed = check_global_step(str(log_file), args.threshold, verbose=verbose)
            else:
                if verbose:
                    print(f"No run_embodiment.log found in {log_path}")
                reached, crashed = False, False
        else:
            if verbose:
                print(f"Cannot infer experiment name from directory: {dir_name}")
            reached, crashed = False, False
    else:
        if verbose:
            print(f"Path not found: {args.log_path}")
        reached, crashed = False, False

    if args.format == "simple":
        print(f"{reached},{crashed}")
    else:
        print(f"\nResult: reached_threshold={reached}, crashed_before_threshold={crashed}")