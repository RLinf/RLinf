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

"""Unified VLM Trend recipe stages that are not dataset preprocessing.

Stages:
  train_teacher  Train the state-success-value MLP teacher
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_STAGE_FILES = {
    "train_teacher": "vlm_trend/train_state_success_value.py",
}


def _load_stage(stage: str) -> ModuleType:
    path = Path(__file__).resolve().parent / _STAGE_FILES[stage]
    spec = importlib.util.spec_from_file_location(f"vlm_trend_stage_{stage}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load stage module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> None:
    pre = argparse.ArgumentParser(
        description="Unified VLM Trend non-preprocess recipe stages.",
        add_help=False,
    )
    pre.add_argument(
        "--stage",
        required=True,
        choices=tuple(_STAGE_FILES),
        help="Which recipe stage to run.",
    )
    pre.add_argument("-h", "--help", action="store_true")
    args, remaining = pre.parse_known_args(argv)
    if args.help:
        pre.print_help()
        module = _load_stage(args.stage)
        module.parse_args(["--help"])
        sys.exit(0)
    module = _load_stage(args.stage)
    stage_args = module.parse_args(remaining)
    if hasattr(module, "train"):
        module.train(stage_args)
    else:
        module.main(stage_args)


if __name__ == "__main__":
    main()
