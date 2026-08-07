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

"""CLI for building terminal success data from VLM Trend rollout episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlinf.data.datasets.vlm_trend_success import build_rows
from rlinf.utils.logging import get_logger

logger = get_logger()


def main(args: argparse.Namespace) -> None:
    """Write manifests and dataset statistics."""
    rows_by_split, stats = build_rows(args)
    output_dir = Path(args.output_dir)
    for split, rows in rows_by_split.items():
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        with (split_dir / "segments.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
    (output_dir / "dataset_info.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    logger.info("%s", json.dumps(stats, indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for terminal-success dataset preprocessing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-path", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--max-positive", type=int, default=5000)
    parser.add_argument("--negative-positive-ratio", type=float, default=4.0)
    parser.add_argument("--hard-negatives-per-episode", type=int, default=3)
    parser.add_argument("--success-exclusion-steps", type=int, default=10)
    parser.add_argument("--near-terminal-positives-per-episode", type=int, default=1)
    parser.add_argument("--success-positive-lead-steps", type=int, default=4)
    parser.add_argument("--online-interval", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
