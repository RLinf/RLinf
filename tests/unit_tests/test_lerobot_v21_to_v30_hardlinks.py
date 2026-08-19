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

import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from toolkits.lerobot.convert_v21_to_v30_hardlinks import (
    convert_v21_to_v30_hardlinks,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def _make_v21_dataset(root: Path) -> None:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [1],
            "names": ["state"],
        },
        "action": {"dtype": "float32", "shape": [1], "names": ["action"]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "test",
        "total_episodes": 2,
        "total_frames": 4,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 10,
        "splits": {"train": "0:2"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": features,
    }
    (root / "meta").mkdir(parents=True)
    with open(root / "meta" / "info.json", "w") as file:
        json.dump(info, file)

    episodes = []
    episode_stats = []
    for episode_index in range(2):
        start = episode_index * 2
        table = pa.table(
            {
                "observation.state": [[float(start)], [float(start + 1)]],
                "action": [[float(start)], [float(start + 1)]],
                "timestamp": [0.0, 0.1],
                "frame_index": [0, 1],
                "episode_index": [episode_index, episode_index],
                "index": [start, start + 1],
                "task_index": [0, 0],
            }
        )
        parquet_path = (
            root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, parquet_path)
        episodes.append(
            {
                "episode_index": episode_index,
                "tasks": ["test task"],
                "length": 2,
            }
        )
        stats = {
            key: {
                "min": [float(start)],
                "max": [float(start + 1)],
                "mean": [float(start) + 0.5],
                "std": [0.5],
                "count": [2],
            }
            for key in ("observation.state", "action")
        }
        episode_stats.append({"episode_index": episode_index, "stats": stats})

    _write_jsonl(root / "meta" / "episodes.jsonl", episodes)
    _write_jsonl(root / "meta" / "episodes_stats.jsonl", episode_stats)
    _write_jsonl(
        root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": "test task"}]
    )


def test_conversion_creates_loadable_v30_hardlink_view(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _make_v21_dataset(source)

    convert_v21_to_v30_hardlinks(source, output)

    metadata = LeRobotDatasetMetadata(output.name, root=output)
    assert metadata.total_episodes == 2
    assert metadata.total_frames == 4
    assert metadata.info["codebase_version"] == "v3.0"
    assert metadata.info["video_path"] is None
    assert (output / "meta" / "stats.json").is_file()
    assert (
        os.stat(source / "data" / "chunk-000" / "episode_000000.parquet").st_ino
        == os.stat(output / "data" / "chunk-000" / "file-000.parquet").st_ino
    )


def test_conversion_refuses_to_overwrite_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _make_v21_dataset(source)
    output.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        convert_v21_to_v30_hardlinks(source, output)
