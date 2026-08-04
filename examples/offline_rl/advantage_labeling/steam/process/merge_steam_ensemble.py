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

"""CLI: merge single / ensemble STEAM critic checkpoints into one ensemble checkpoint.

Thin wrapper around
:func:`rlinf.models.embodiment.value_model.steam.checkpoint_merge.merge_ensemble_checkpoints`.
See the STEAM pipeline docs for the full workflow and examples.
"""

import argparse
import logging
import sys
from pathlib import Path

# Make the rlinf package importable regardless of the cwd the user launched from.
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

from rlinf.models.embodiment.value_model.steam.checkpoint_merge import (  # noqa: E402
    merge_ensemble_checkpoints,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--member",
        action="append",
        required=True,
        help=(
            "Output member source. Use PATH for a single-model checkpoint, or "
            "PATH:idx to extract members.idx from an ensemble checkpoint. Repeat "
            "once per output ensemble member, in output order."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output actor directory. The script will create "
            "OUTPUT/model_state_dict/full_weights.pt."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _build_arg_parser().parse_args()
    merge_ensemble_checkpoints(args.member, args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
