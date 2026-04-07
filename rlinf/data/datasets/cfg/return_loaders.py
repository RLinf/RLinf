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

"""Utility functions for loading return statistics and sidecar data."""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_return_stats_from_dataset(
    dataset_path: str | Path,
) -> tuple[float | None, float | None]:
    """Load return min/max from dataset's stats.json.

    Args:
        dataset_path: Path to LeRobot dataset

    Returns:
        Tuple of (return_min, return_max), or (None, None) if not found
    """
    import json

    stats_path = Path(dataset_path) / "meta" / "stats.json"
    if not stats_path.exists():
        return None, None

    try:
        with open(stats_path, "r") as f:
            stats = json.load(f)
        return_stats = stats.get("return", {})
        return return_stats.get("min"), return_stats.get("max")
    except (json.JSONDecodeError, KeyError):
        return None, None


def load_returns_sidecar(
    dataset_path: str | Path,
    returns_tag: str | None = None,
) -> dict[int, dict[str, np.ndarray]] | None:
    """Load ``meta/returns_{tag}.parquet`` sidecar written by compute_returns.py.

    Falls back to ``meta/returns.parquet`` when *returns_tag* is None.

    Returns:
        ``{episode_index: {"return": np.array, "reward": np.array}}``
        or None if sidecar does not exist **and** no tag was explicitly requested.

    Raises:
        FileNotFoundError: If *returns_tag* is given but the file does not exist.
    """
    import pyarrow.parquet as pq

    dataset_path = Path(dataset_path)
    sidecar_filename = (
        f"returns_{returns_tag}.parquet" if returns_tag else "returns.parquet"
    )
    sidecar_path = dataset_path / "meta" / sidecar_filename
    if not sidecar_path.exists():
        if returns_tag:
            raise FileNotFoundError(
                f"Returns sidecar not found: {sidecar_path}. "
                f"Run compute_returns.py with tag='{returns_tag}' first."
            )
        return None

    table = pq.read_table(str(sidecar_path))
    ep_col = table.column("episode_index").to_numpy()
    frame_col = table.column("frame_index").to_numpy()
    ret_col = table.column("return").to_numpy()
    rew_col = table.column("reward").to_numpy()

    sidecar: dict[int, dict[str, np.ndarray]] = {}
    for ep in np.unique(ep_col):
        mask = ep_col == ep
        frames = frame_col[mask]
        order = np.argsort(frames)
        sidecar[int(ep)] = {
            "return": ret_col[mask][order].astype(np.float32),
            "reward": rew_col[mask][order].astype(np.float32),
        }

    logger.info(f"Loaded returns sidecar: {sidecar_path} ({len(sidecar)} episodes)")
    return sidecar
