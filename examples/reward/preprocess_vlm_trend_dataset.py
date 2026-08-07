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

"""Unified VLM Trend preprocess entrypoint.

Modes:
  terminal_success  Sparse 0/1 terminal-success windows
  potential         Dense potential / progress windows from a state-value teacher
  progress          Legacy state-value delta progress windows
  trend_reward      Classic GAE-delta trend reward windows
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_MODE_FILES = {
    "terminal_success": "vlm_trend/preprocess_terminal_success.py",
    "potential": "vlm_trend/preprocess_potential.py",
    "progress": "vlm_trend/preprocess_progress.py",
    "trend_reward": "vlm_trend/preprocess_reward.py",
}


def _load_mode(mode: str) -> ModuleType:
    path = Path(__file__).resolve().parent / _MODE_FILES[mode]
    spec = importlib.util.spec_from_file_location(
        f"vlm_trend_preprocess_{mode}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load preprocess mode module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_mode(module: ModuleType, args: argparse.Namespace) -> None:
    for name in ("main", "preprocess", "train"):
        fn = getattr(module, name, None)
        if callable(fn):
            fn(args)
            return
    raise RuntimeError(f"No runnable entry found in {module.__name__}")


def parse_known_mode(argv: list[str] | None = None) -> tuple[str, list[str]]:
    """Parse ``--mode`` and leave the remaining argv for the mode-specific parser."""
    pre = argparse.ArgumentParser(
        description="Unified VLM Trend dataset preprocessing.",
        add_help=False,
    )
    pre.add_argument(
        "--mode",
        choices=tuple(_MODE_FILES),
        help="Which VLM Trend label pipeline to run.",
    )
    pre.add_argument("-h", "--help", action="store_true")
    args, remaining = pre.parse_known_args(argv)
    if args.help:
        pre.print_help()
        print(
            "\nMode-specific flags follow --mode. Example:\n"
            "  python examples/reward/preprocess_vlm_trend_dataset.py "
            "--mode terminal_success --help"
        )
        if args.mode is not None:
            module = _load_mode(args.mode)
            module.parse_args(["--help"])
        sys.exit(0)
    if args.mode is None:
        pre.error("--mode is required")
    return args.mode, remaining


def main(argv: list[str] | None = None) -> None:
    mode, remaining = parse_known_mode(argv)
    module = _load_mode(mode)
    args = module.parse_args(remaining)
    _run_mode(module, args)


if __name__ == "__main__":
    main()
