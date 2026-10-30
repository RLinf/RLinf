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

"""Load SIMPLE reset states from LeRobot episode metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SimpleLeRobotResetDataset:
    """Load fixed SIMPLE reset states from LeRobot episode metadata."""

    def __init__(
        self,
        path: str | Path,
        *,
        dr_level: int,
        episode_start: int,
        num_episodes: int,
    ):
        dataset_path = Path(path).expanduser()
        if dataset_path.suffix == ".zip":
            raise ValueError("Extract the SIMPLE reset-state zip before loading it.")
        if episode_start < 0 or num_episodes <= 0:
            raise ValueError("episode_start must be >= 0 and num_episodes must be > 0.")

        level_name = f"level-{dr_level}"
        metadata_files = [
            path
            for path in dataset_path.rglob("meta/episodes.jsonl")
            if path.parent.parent.name == level_name
        ]
        if len(metadata_files) != 1:
            raise ValueError(
                f"Expected one {level_name}/meta/episodes.jsonl under "
                f"{dataset_path}, found {len(metadata_files)}."
            )
        with metadata_files[0].open() as file:
            episodes = [json.loads(line) for line in file if line.strip()]
        episodes.sort(key=lambda episode: int(episode["episode_index"]))
        self._episodes = episodes[episode_start : episode_start + num_episodes]
        if len(self._episodes) != num_episodes:
            raise ValueError(
                f"Requested {num_episodes} reset episodes from index {episode_start}, "
                f"but found {len(self._episodes)} in {metadata_files[0]}."
            )

    def __len__(self) -> int:
        return len(self._episodes)

    def load(self, index: int) -> tuple[dict[str, Any], Any]:
        episode = self._episodes[index]
        encoded_config = episode.get("environment_config")
        if not isinstance(encoded_config, str):
            raise ValueError(
                "SIMPLE LeRobot episode metadata is missing environment_config."
            )
        return json.loads(encoded_config), episode
