# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Compute OpenPI Stack Cube normalization from an audited train split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpi.shared import normalize


def calculate_statistics(dataset: Path) -> dict:
    """Save statistics for all H10 windows, clamped at each episode boundary.

    The Stack Cube input transform preserves raw state and relative actions.
    Normalization precedes model padding, so the statistics have 8 state and
    7 action dimensions. A collection audit must explicitly permit training.
    """
    dataset = dataset.expanduser().resolve()
    audit_path = dataset / "meta/collection_audit.json"
    audit = json.loads(audit_path.read_text())
    if audit.get("split") != "train" or not audit.get("normalization_source_allowed"):
        raise ValueError("Normalization requires an explicitly audited train split.")
    output = dataset / "norm_stats.json"
    if output.exists():
        raise FileExistsError(output)
    info = json.loads((dataset / "meta/info.json").read_text())
    if info["fps"] != 20:
        raise ValueError("Stack Cube data must use 20 Hz timestamps.")
    states = normalize.RunningStats()
    actions = normalize.RunningStats()
    sources = []
    frames = 0
    for path in sorted(dataset.glob("data/chunk-*/episode_*.parquet")):
        frame = pd.read_parquet(path)
        state = np.stack(frame["observation.state"]).astype(np.float64)
        action = np.stack(frame["action"]).astype(np.float64)
        if state.shape != (len(frame), 8) or action.shape != (len(frame), 7):
            raise ValueError(f"Unexpected state or action shape: {path}")
        if (
            not len(frame)
            or not np.isfinite(state).all()
            or not np.isfinite(action).all()
        ):
            raise ValueError(f"Empty or non-finite episode: {path}")
        if not np.array_equal(frame["frame_index"], np.arange(len(frame))):
            raise ValueError(f"Non-contiguous episode: {path}")
        indices = np.minimum(
            np.arange(len(frame))[:, None] + np.arange(10), len(frame) - 1
        )
        states.update(state)
        actions.update(action[indices])
        frames += len(frame)
        sources.append(
            {
                "path": str(path.relative_to(dataset)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "frames": len(frame),
            }
        )
    if frames != info["total_frames"] or len(sources) != info["total_episodes"]:
        raise ValueError("Parquet rows or episode count do not match metadata.")
    normalize.save(
        dataset, {"state": states.get_statistics(), "actions": actions.get_statistics()}
    )
    manifest = {
        "dataset": str(dataset),
        "split": "train",
        "frames": frames,
        "action_windows": frames,
        "action_horizon": 10,
        "episode_padding": "repeat_last_action",
        "config_name": "pi05_isaaclab_stack_cube",
        "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "norm_stats_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "sources": sources,
    }
    (dataset / "norm_stats_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    """Compute statistics without decoding video or creating GPU contexts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(calculate_statistics(args.dataset), indent=2))


if __name__ == "__main__":
    main()
