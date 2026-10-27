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

"""Export benchmark tasks from the installed RoboLab package to a TSV file."""

import argparse
import csv
import json
from decimal import Decimal
from importlib.util import find_spec
from pathlib import Path

from rlinf.utils.logging import get_logger


def export_tasks(output: Path) -> int:
    """Write task names, native step limits, and task attributes to a TSV.

    Args:
        output: Destination file. Parent directories are created as needed.

    Returns:
        Number of exported benchmark tasks.
    """
    spec = find_spec("robolab")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(
            "Expected an installed RoboLab package. Activate the environment created "
            "by requirements/install.sh embodied --model openpi --env robolab."
        )
    metadata = Path(spec.origin).parent / "tasks/_metadata/task_metadata.json"
    tasks = [
        task
        for task in json.loads(metadata.read_text())
        if task["filename"].startswith("benchmark/")
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(
            ["task", "max_steps", "difficulty", "attributes", "instruction", "filename"]
        )
        for task in sorted(tasks, key=lambda task: task["task_name"]):
            # DROID uses dt=1/120 s and decimation=8: 15 control steps per second.
            max_steps = int(Decimal(task["episode_s"]) * 15)
            writer.writerow(
                [
                    task["task_name"],
                    max_steps,
                    task["difficulty_label"],
                    task["attributes"],
                    task["instruction"],
                    task["filename"],
                ]
            )
    return len(tasks)


def main() -> None:
    """Export the installed benchmark metadata without starting Isaac Sim."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/robolab_tasks.tsv"),
        help="Output TSV path (default: data/robolab_tasks.tsv).",
    )
    args = parser.parse_args()
    count = export_tasks(args.output)
    get_logger().info("Saved %s benchmark tasks to %s", count, args.output)


if __name__ == "__main__":
    main()
