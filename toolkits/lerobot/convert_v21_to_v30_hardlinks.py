#!/usr/bin/env python3
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

"""Convert a local LeRobot v2.1 dataset to v3.0 with hard-linked data files.

This converter is intended for large, image-in-Parquet datasets where the
official v2.1-to-v3.0 converter's Pandas rewrite would require too much memory
and disk space. It creates a separate v3.0 dataset, keeps the source unchanged,
and hard-links each episode Parquet file into the v3.0 file layout. This is
valid because the official converter does not change data columns; it only
groups episode files and rewrites metadata.

The source and output must be on the same filesystem. Video-backed datasets
are deliberately rejected because their media layout also needs conversion.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Iterator

import jsonlines
import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    cast_stats_to_numpy,
    flatten_dict,
    update_chunk_file_indices,
    write_episodes,
    write_info,
    write_stats,
    write_tasks,
)

logger = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with jsonlines.open(path, "r") as reader:
        return list(reader)


def _source_episode_path(
    source_dir: Path, info: dict[str, Any], episode_index: int
) -> Path:
    data_path = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    chunks_size = int(info.get("chunks_size", DEFAULT_CHUNK_SIZE))
    relative_path = data_path.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )
    return source_dir / relative_path


def _validate_source(source_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    info_path = source_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")

    with open(info_path) as file:
        info = json.load(file)
    if info.get("codebase_version") != "v2.1":
        raise ValueError(
            "Hard-link conversion requires a v2.1 source, got "
            f"{info.get('codebase_version')!r}."
        )

    video_keys = [
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video"
    ]
    if video_keys:
        raise NotImplementedError(
            "Hard-link conversion currently supports embedded images only; "
            f"video features require media conversion: {video_keys}."
        )

    required_metadata = [
        source_dir / "meta" / "episodes.jsonl",
        source_dir / "meta" / "episodes_stats.jsonl",
        source_dir / "meta" / "tasks.jsonl",
    ]
    missing = [str(path) for path in required_metadata if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required v2.1 metadata: {missing}")

    episodes = sorted(
        _read_jsonl(source_dir / "meta" / "episodes.jsonl"),
        key=lambda episode: int(episode["episode_index"]),
    )
    expected_indices = list(range(len(episodes)))
    actual_indices = [int(episode["episode_index"]) for episode in episodes]
    if actual_indices != expected_indices:
        raise ValueError(
            "Episode indices must be contiguous and zero-based: "
            f"expected 0..{len(episodes) - 1}."
        )
    if len(episodes) != int(info["total_episodes"]):
        raise ValueError(
            "episodes.jsonl count does not match info.json: "
            f"{len(episodes)} != {info['total_episodes']}."
        )
    return info, episodes


def _write_v30_info(info: dict[str, Any], output_dir: Path) -> None:
    converted = dict(info)
    converted["codebase_version"] = "v3.0"
    converted.pop("total_chunks", None)
    converted.pop("total_videos", None)
    converted["data_files_size_in_mb"] = DEFAULT_DATA_FILE_SIZE_IN_MB
    converted["video_files_size_in_mb"] = DEFAULT_VIDEO_FILE_SIZE_IN_MB
    converted["data_path"] = DEFAULT_DATA_PATH
    converted["video_path"] = None
    converted["fps"] = int(converted["fps"])
    for feature in converted["features"].values():
        feature["fps"] = converted["fps"]
    write_info(converted, output_dir)


def _write_v30_tasks(source_dir: Path, output_dir: Path) -> None:
    records = sorted(
        _read_jsonl(source_dir / "meta" / "tasks.jsonl"),
        key=lambda task: int(task["task_index"]),
    )
    tasks = pd.DataFrame(
        {"task_index": [int(record["task_index"]) for record in records]},
        index=[record["task"] for record in records],
    )
    write_tasks(tasks, output_dir)


def _load_episode_stats(source_dir: Path) -> dict[int, dict[str, Any]]:
    records = _read_jsonl(source_dir / "meta" / "episodes_stats.jsonl")
    return {
        int(record["episode_index"]): cast_stats_to_numpy(record["stats"])
        for record in records
    }


def _episode_rows(
    episodes: list[dict[str, Any]],
    data_locations: list[dict[str, int]],
    episode_stats: dict[int, dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    for episode, location in zip(episodes, data_locations, strict=True):
        episode_index = int(episode["episode_index"])
        if episode_index not in episode_stats:
            raise ValueError(f"Missing statistics for episode {episode_index}.")
        row = {
            **location,
            **episode,
            **flatten_dict({"stats": episode_stats[episode_index]}),
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        yield row


def _link_data_files(
    source_dir: Path,
    output_dir: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> list[dict[str, int]]:
    chunk_index = 0
    file_index = 0
    global_frame_index = 0
    locations: list[dict[str, int]] = []

    for position, episode in enumerate(episodes):
        episode_index = int(episode["episode_index"])
        source_path = _source_episode_path(source_dir, info, episode_index)
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing episode Parquet: {source_path}")

        num_frames = pq.read_metadata(source_path).num_rows
        expected_frames = int(episode["length"])
        if num_frames != expected_frames:
            raise ValueError(
                f"Episode {episode_index} frame count mismatch: "
                f"Parquet={num_frames}, metadata={expected_frames}."
            )

        relative_output = DEFAULT_DATA_PATH.format(
            chunk_index=chunk_index,
            file_index=file_index,
        )
        output_path = output_dir / relative_output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        os.link(source_path, output_path)

        locations.append(
            {
                "episode_index": episode_index,
                "data/chunk_index": chunk_index,
                "data/file_index": file_index,
                "dataset_from_index": global_frame_index,
                "dataset_to_index": global_frame_index + num_frames,
            }
        )
        global_frame_index += num_frames
        chunk_index, file_index = update_chunk_file_indices(
            chunk_index, file_index, DEFAULT_CHUNK_SIZE
        )
        if (position + 1) % 50 == 0 or position + 1 == len(episodes):
            logger.info("Linked %d/%d episodes", position + 1, len(episodes))

    if global_frame_index != int(info["total_frames"]):
        raise ValueError(
            "Linked frame count does not match info.json: "
            f"{global_frame_index} != {info['total_frames']}."
        )
    return locations


def _validate_output(
    source_dir: Path,
    output_dir: Path,
    episodes: list[dict[str, Any]],
) -> None:
    metadata = LeRobotDatasetMetadata(output_dir.name, root=output_dir)
    if metadata.total_episodes != len(episodes):
        raise ValueError(
            f"Converted metadata has {metadata.total_episodes} episodes; "
            f"expected {len(episodes)}."
        )

    sample_indices = sorted({0, len(episodes) // 2, len(episodes) - 1})
    for episode_index in sample_indices:
        source_info, _ = _validate_source(source_dir)
        source_path = _source_episode_path(source_dir, source_info, episode_index)
        output_path = output_dir / metadata.get_data_file_path(episode_index)
        if source_path.stat().st_ino != output_path.stat().st_ino:
            raise ValueError(
                f"Episode {episode_index} is not hard-linked to its source file."
            )


def convert_v21_to_v30_hardlinks(source_dir: Path, output_dir: Path) -> None:
    """Create a separate LeRobot v3.0 dataset backed by v2.1 hard links.

    Args:
        source_dir: Existing local LeRobot v2.1 dataset.
        output_dir: New path for the LeRobot v3.0 view. It must not exist.
    """
    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    if output_dir == source_dir or source_dir in output_dir.parents:
        raise ValueError("Output must be separate from and outside the source dataset.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if source_dir.stat().st_dev != output_dir.parent.stat().st_dev:
        raise OSError(
            "Source and output must be on the same filesystem for hard links."
        )

    info, episodes = _validate_source(source_dir)
    temporary_dir = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    if temporary_dir.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary_dir}")

    logger.info("Converting %s", source_dir)
    logger.info("Writing v3.0 hard-link view to %s", output_dir)
    temporary_dir.mkdir()
    try:
        _write_v30_info(info, temporary_dir)
        _write_v30_tasks(source_dir, temporary_dir)
        data_locations = _link_data_files(source_dir, temporary_dir, info, episodes)
        episode_stats = _load_episode_stats(source_dir)
        episode_dataset = Dataset.from_generator(
            lambda: _episode_rows(episodes, data_locations, episode_stats)
        )
        write_episodes(episode_dataset, temporary_dir)
        write_stats(aggregate_stats(list(episode_stats.values())), temporary_dir)
        temporary_dir.rename(output_dir)
        _validate_output(source_dir, output_dir, episodes)
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise

    logger.info("Conversion complete: %s", output_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args()
    convert_v21_to_v30_hardlinks(args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()
