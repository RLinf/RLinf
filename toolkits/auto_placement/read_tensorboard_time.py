#!/usr/bin/env python3
"""Extract TensorBoard scalar time metrics under a log directory.

This script recursively searches for TensorBoard event files and extracts all
scalar tags that match a prefix (default: "time/"). It is designed to be used
as a standalone utility for profiling/placement workflows.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ScalarEvent:
    step: int
    time: float
    value: float


def _find_run_dirs(logdir: Path) -> list[Path]:
    event_prefix = "events.out.tfevents"
    run_dirs: list[Path] = []
    for root, _dirs, files in os.walk(logdir):
        if any(f.startswith(event_prefix) for f in files):
            run_dirs.append(Path(root))
    return sorted(set(run_dirs))


def _load_scalars(run_dir: Path, *, tag_prefix: str) -> dict[str, list[ScalarEvent]]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "tensorboard is required. Try `pip install tensorboard`."
        ) from exc

    ea = event_accumulator.EventAccumulator(
        str(run_dir),
        size_guidance={"scalars": 0},
    )
    ea.Reload()

    scalar_tags = ea.Tags().get("scalars", [])
    selected_tags = [t for t in scalar_tags if t.startswith(tag_prefix)]
    selected_tags.sort()

    results: dict[str, list[ScalarEvent]] = {}
    for tag in selected_tags:
        results[tag] = [
            ScalarEvent(step=int(e.step), time=float(e.wall_time), value=float(e.value))
            for e in ea.Scalars(tag)
        ]
    return results


def _to_jsonable(data: dict[str, dict[str, list[ScalarEvent]]]) -> dict[str, Any]:
    return {
        run: {tag: [asdict(e) for e in events] for tag, events in tags.items()}
        for run, tags in data.items()
    }


def _write_csv(
    out_path: Path, data: dict[str, dict[str, list[ScalarEvent]]], *, root_dir: Path
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_dir", "tag", "step", "time", "value"])
        for run_dir, tags in sorted(data.items()):
            rel_run = str(Path(run_dir).relative_to(root_dir))
            for tag, events in sorted(tags.items()):
                for e in events:
                    writer.writerow([rel_run, tag, e.step, e.time, e.value])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract TensorBoard scalar tags under `time/` (or a prefix)."
    )
    parser.add_argument(
        "logdir",
        type=Path,
        help="Root directory to scan (e.g. logs/ or a placement_test_logs/ folder).",
    )
    parser.add_argument(
        "--tag-prefix",
        type=str,
        default="time/",
        help='Only extract scalar tags that start with this prefix (default: "time/").',
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path. Default: <logdir>/time_scalars.csv",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional JSON output path (nested structure).",
    )
    args = parser.parse_args()

    logdir: Path = args.logdir
    if not logdir.exists():
        raise SystemExit(f"logdir not found: {logdir}")

    run_dirs = _find_run_dirs(logdir)
    if not run_dirs:
        raise SystemExit(f"No TensorBoard event files found under: {logdir}")

    out_csv_path = args.out or (logdir / "time_scalars.csv")

    extracted: dict[str, dict[str, list[ScalarEvent]]] = {}
    for run_dir in run_dirs:
        scalars = _load_scalars(run_dir, tag_prefix=args.tag_prefix)
        if not scalars:
            continue
        extracted[str(run_dir)] = scalars

    _write_csv(out_csv_path, extracted, root_dir=logdir)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(_to_jsonable(extracted), indent=2, sort_keys=True)
        )

    print(f"Wrote CSV:  {out_csv_path}")
    if args.json is not None:
        print(f"Wrote JSON: {args.json}")
    print(f"Runs: {len(run_dirs)} (with data: {len(extracted)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
