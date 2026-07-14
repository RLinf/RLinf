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

"""LeRobot shard I/O helpers for online DAgger datasets."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Parquet index / metadata columns that should not be chunked along the time
# axis — they are scalar per frame and have no meaningful "window" semantics.
_META_KEYS: frozenset[str] = frozenset(
    {"timestamp", "frame_index", "episode_index", "index", "task_index"}
)


def _hf_column_to_numpy_bool_1d(hf_dataset: Any, col: str) -> np.ndarray:
    """Load a per-row boolean column as ``(num_rows,)`` bool numpy array."""
    raw = hf_dataset[col]
    if isinstance(raw, torch.Tensor):
        t = raw
    else:
        t = torch.stack(list(raw))
    return t.reshape(-1).bool().numpy()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _read_lerobot_tasks(path: Path) -> dict[int, str]:
    if not path.is_file():
        return {}
    return {
        int(row["task_index"]): str(row["task"])
        for row in _read_jsonl(path)
        if "task_index" in row and "task" in row
    }


def _episode_task(episode: dict[str, Any]) -> str:
    tasks = episode.get("tasks", [])
    if tasks:
        return str(tasks[0])
    task = episode.get("task", "")
    return str(task) if task is not None else ""


def _episode_parquet_path(
    shard_path: Path,
    info: dict[str, Any],
    episode: dict[str, Any],
) -> Path:
    episode_index = int(episode["episode_index"])
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size
    data_path = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    return shard_path / data_path.format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
    )


def _load_lerobot_episode_frames(
    parquet_path: Path,
    shard_path: Path,
    feature_specs: dict[str, Any],
    task_by_index: dict[int, str],
) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows = pq.read_table(parquet_path).to_pylist()
    return [
        _lerobot_row_to_frame(
            row,
            shard_path,
            feature_specs,
            task_by_index,
        )
        for row in rows
    ]


def _lerobot_row_to_frame(
    row: dict[str, Any],
    shard_path: Path,
    feature_specs: dict[str, Any],
    task_by_index: dict[int, str],
) -> dict[str, Any]:
    task = ""
    if "task" in row and row["task"] is not None:
        task = str(row["task"])
    elif "task_index" in row and row["task_index"] is not None:
        task = task_by_index.get(int(row["task_index"]), "")

    frame: dict[str, Any] = {"task": task}
    for key, value in row.items():
        if key in _META_KEYS or key == "task" or value is None:
            continue
        if _looks_like_lerobot_image(value):
            image = _decode_lerobot_image(value, shard_path)
            if image is not None:
                frame[key] = image
            continue
        frame[key] = _lerobot_value_to_array(
            value,
            feature_specs.get(key, {}).get("dtype"),
        )
    return frame


def _looks_like_lerobot_image(value: Any) -> bool:
    if isinstance(value, dict):
        return "bytes" in value or "path" in value
    return isinstance(value, (bytes, str, Path))


def _decode_lerobot_image(value: Any, shard_path: Path) -> np.ndarray | None:
    import PIL.Image as PILImage

    if isinstance(value, dict):
        raw_bytes = value.get("bytes")
        image_path = value.get("path")
    else:
        raw_bytes = value if isinstance(value, bytes) else None
        image_path = value if isinstance(value, (str, Path)) else None

    if raw_bytes is not None:
        with PILImage.open(io.BytesIO(raw_bytes)) as image:
            return np.asarray(image.convert("RGB"))

    if image_path is None:
        return None

    path = Path(image_path)
    if not path.is_absolute():
        path = shard_path / path
    if not path.is_file():
        return None
    with PILImage.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _lerobot_value_to_array(value: Any, dtype: str | None) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value
    else:
        arr = np.asarray(value)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if dtype and dtype != "image":
        arr = arr.astype(dtype, copy=False)
    return arr
