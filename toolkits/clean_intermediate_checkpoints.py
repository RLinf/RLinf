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

"""Remove intermediate RLinf checkpoints while keeping the latest ones.

The cleanup unit is an entire ``checkpoints/global_step_<N>`` directory. This
keeps the training state and exported model checkpoint from a step together.
The command is a dry run unless ``--execute`` is passed.

Examples:
    Preview cleanup under the default ``logs`` directory::

        python toolkits/clean_intermediate_checkpoints.py

    Keep the latest three checkpoints in every run and delete older ones::

        python toolkits/clean_intermediate_checkpoints.py \
            logs --keep-last 3 --execute
"""

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

_GLOBAL_STEP_PATTERN = re.compile(r"^global_step_(\d+)$")


@dataclass(frozen=True)
class CheckpointCleanupPlan:
    """Checkpoint directories to keep and remove for one run."""

    checkpoints_dir: Path
    keep: tuple[Path, ...]
    remove: tuple[Path, ...]


def discover_checkpoints_dirs(log_root: Path) -> list[Path]:
    """Find checkpoint container directories below a log root.

    A path that points directly to a directory named ``checkpoints`` is also
    accepted. Symbolic-link checkpoint directories are ignored so cleanup does
    not unexpectedly affect files outside the requested tree.

    Args:
        log_root: Log directory, run directory, or checkpoint directory.

    Returns:
        Sorted checkpoint container paths.

    Raises:
        FileNotFoundError: If ``log_root`` does not exist.
        NotADirectoryError: If ``log_root`` is not a directory.
    """
    log_root = log_root.expanduser()
    if not log_root.exists():
        raise FileNotFoundError(f"Log path does not exist: {log_root}")
    if not log_root.is_dir():
        raise NotADirectoryError(f"Log path is not a directory: {log_root}")

    if log_root.name == "checkpoints":
        return [] if log_root.is_symlink() else [log_root]

    checkpoints_dirs = []
    for current_root, dirnames, _ in os.walk(log_root, followlinks=False):
        if "checkpoints" not in dirnames:
            continue

        checkpoints_dir = Path(current_root) / "checkpoints"
        if not checkpoints_dir.is_symlink():
            checkpoints_dirs.append(checkpoints_dir)

        # A checkpoint tree cannot contain another run's checkpoint container.
        # Skipping it also avoids walking large model-state directories.
        dirnames.remove("checkpoints")

    return sorted(checkpoints_dirs)


def plan_checkpoint_cleanup(
    checkpoints_dir: Path, keep_last: int
) -> CheckpointCleanupPlan:
    """Build a cleanup plan for one checkpoint container.

    Args:
        checkpoints_dir: Directory containing ``global_step_<N>`` directories.
        keep_last: Number of checkpoint steps to retain.

    Returns:
        A plan ordered by numeric global step.

    Raises:
        ValueError: If ``keep_last`` is not positive.
    """
    if keep_last <= 0:
        raise ValueError("keep_last must be greater than zero")

    step_dirs = []
    with os.scandir(checkpoints_dir) as entries:
        for entry in entries:
            match = _GLOBAL_STEP_PATTERN.fullmatch(entry.name)
            if match is None or not entry.is_dir(follow_symlinks=False):
                continue
            step_dirs.append((int(match.group(1)), Path(entry.path)))

    step_dirs.sort(key=lambda item: (item[0], item[1].name))
    ordered_paths = tuple(path for _, path in step_dirs)
    split_index = max(0, len(ordered_paths) - keep_last)
    return CheckpointCleanupPlan(
        checkpoints_dir=checkpoints_dir,
        keep=ordered_paths[split_index:],
        remove=ordered_paths[:split_index],
    )


def build_cleanup_plans(
    log_roots: Sequence[Path], keep_last: int
) -> list[CheckpointCleanupPlan]:
    """Build deduplicated cleanup plans for one or more log roots.

    Args:
        log_roots: Log roots to scan recursively.
        keep_last: Number of checkpoint steps to retain per run.

    Returns:
        Cleanup plans sorted by checkpoint container path.
    """
    checkpoints_dirs_by_path = {}
    for log_root in log_roots:
        for checkpoints_dir in discover_checkpoints_dirs(log_root):
            checkpoints_dirs_by_path[checkpoints_dir.resolve()] = checkpoints_dir

    return [
        plan_checkpoint_cleanup(checkpoints_dir, keep_last)
        for checkpoints_dir in sorted(checkpoints_dirs_by_path.values())
    ]


def execute_cleanup(plans: Sequence[CheckpointCleanupPlan]) -> list[Path]:
    """Delete all checkpoint directories marked for removal.

    Args:
        plans: Previously reviewed cleanup plans.

    Returns:
        Paths that were removed.
    """
    removed = []
    for plan in plans:
        for checkpoint_dir in plan.remove:
            # Recheck the target immediately before the destructive operation.
            if checkpoint_dir.parent != plan.checkpoints_dir:
                raise ValueError(
                    f"Refusing to remove path outside {plan.checkpoints_dir}: "
                    f"{checkpoint_dir}"
                )
            if _GLOBAL_STEP_PATTERN.fullmatch(checkpoint_dir.name) is None:
                raise ValueError(
                    f"Refusing to remove unexpected path: {checkpoint_dir}"
                )
            if checkpoint_dir.is_symlink():
                raise ValueError(f"Refusing to remove symbolic link: {checkpoint_dir}")

            shutil.rmtree(checkpoint_dir)
            removed.append(checkpoint_dir)
    return removed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Remove intermediate global_step_<N> directories while keeping the "
            "latest checkpoints and their training states."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Preview: python toolkits/clean_intermediate_checkpoints.py\n"
            "  Delete:  python toolkits/clean_intermediate_checkpoints.py "
            "logs --keep-last 2 --execute"
        ),
    )
    parser.add_argument(
        "log_roots",
        nargs="*",
        type=Path,
        default=[Path("logs")],
        help=("log/run/checkpoints directories to scan recursively (default: logs)"),
    )
    parser.add_argument(
        "--keep-last",
        type=_positive_int,
        default=2,
        metavar="N",
        help="number of latest global steps to keep per run (default: 2)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform deletion; without this flag the command only previews",
    )
    return parser


def _print_plans(plans: Sequence[CheckpointCleanupPlan], *, execute: bool) -> None:
    mode = "execute" if execute else "dry run"
    print(f"Checkpoint cleanup ({mode}):")
    for plan in plans:
        print(f"\n{plan.checkpoints_dir}")
        for path in plan.keep:
            print(f"  KEEP   {path.name}")
        for path in plan.remove:
            print(f"  REMOVE {path.name}")

    kept_count = sum(len(plan.keep) for plan in plans)
    remove_count = sum(len(plan.remove) for plan in plans)
    print(
        f"\nSummary: scanned {len(plans)} checkpoint directories, "
        f"keeping {kept_count} steps, removing {remove_count} steps."
    )
    if not execute and remove_count:
        print("Dry run only; pass --execute to delete the listed directories.")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checkpoint cleanup command."""
    args = _build_parser().parse_args(argv)
    try:
        plans = build_cleanup_plans(args.log_roots, args.keep_last)
        _print_plans(plans, execute=args.execute)
        if args.execute:
            removed = execute_cleanup(plans)
            print(f"Deleted {len(removed)} checkpoint directories.")
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
