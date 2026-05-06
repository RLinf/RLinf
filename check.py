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
import re
from typing import Optional


def check_global_step(
    log_file_path: str, threshold: Optional[float] = None, verbose: bool = True
) -> tuple[bool, bool]:
    """
    Find Global Step in log file and check if step >= 10% of total steps

    Args:
        log_file_path: Log file path
        threshold: Optional threshold, if None then use 10% of total steps

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
        # print("No Global Step information found")
        return False, has_error

    if threshold is None:
        threshold = (
            total_steps * 0.1
        )  # Dynamically calculate threshold as 10% of total steps

    reached_threshold = max_step >= threshold

    crashed_before_threshold = not reached_threshold and has_error

    return reached_threshold, crashed_before_threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check Global Step status in log file")
    parser.add_argument("log_file", help="Log file path")
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
    reached, crashed = check_global_step(args.log_file, args.threshold, verbose=verbose)

    if args.format == "simple":
        print(f"{reached},{crashed}")
    else:
        print(
            f"\nResult: reached_threshold={reached}, crashed_before_threshold={crashed}"
        )
