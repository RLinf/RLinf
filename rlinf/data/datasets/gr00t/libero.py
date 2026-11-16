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
"""Read the OpenPi LIBERO image-Parquet dataset for the GR00T processor."""

import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from PIL import Image


class LiberoEpisodeLoader:
    """Expose embedded images and GR00T's gripper convention without rewriting data."""

    def __init__(self, path: Path, task_names: list[str] | None):
        self.path = path
        self.info = json.loads((path / "meta/info.json").read_text())
        episodes = [
            json.loads(line)
            for line in (path / "meta/episodes.jsonl").read_text().splitlines()
        ]
        self.episodes = [
            episode
            for episode in episodes
            if task_names is None
            or any(task in task_names for task in episode["tasks"])
        ]
        if not self.episodes:
            raise ValueError(f"No LIBERO episodes match the requested tasks in {path}")
        self.episode_lengths = [episode["length"] for episode in self.episodes]
        self.tasks = {
            row["task_index"]: row["task"]
            for row in map(
                json.loads, (path / "meta/tasks.jsonl").read_text().splitlines()
            )
        }

    def episode_path(self, index: int) -> Path:
        episode_id = self.episodes[index]["episode_index"]
        return self.path / self.info["data_path"].format(
            episode_chunk=episode_id // self.info["chunks_size"],
            episode_index=episode_id,
        )

    def get_episode_length(self, index: int) -> int:
        return self.episode_lengths[index]

    def read_numeric(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        frame = pq.read_table(self.episode_path(index), columns=["state", "actions"])
        state = np.asarray(frame["state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(frame["actions"].to_pylist(), dtype=np.float32)
        actions[:, -1] = (1 - actions[:, -1]) / 2
        return state, actions

    def compute_statistics(self) -> dict:
        with ThreadPoolExecutor(max_workers=8) as pool:
            numeric = list(pool.map(self.read_numeric, range(len(self.episodes))))
        result = {}
        keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        for modality, column in (("state", 0), ("action", 1)):
            values = np.concatenate([item[column] for item in numeric])
            stats = {
                "mean": values.mean(axis=0),
                "std": values.std(axis=0),
                "min": values.min(axis=0),
                "max": values.max(axis=0),
                "q01": np.quantile(values, 0.01, axis=0),
                "q99": np.quantile(values, 0.99, axis=0),
            }
            result[modality] = {
                key: {
                    name: value[
                        index : index + (2 if modality == "state" and index == 6 else 1)
                    ].tolist()
                    for name, value in stats.items()
                }
                for index, key in enumerate(keys)
            }
        return result

    def __getitem__(self, index: int) -> pd.DataFrame:
        raw = pd.read_parquet(self.episode_path(index))
        frame = pd.DataFrame(index=raw.index)
        for modality, source in (("state", "state"), ("action", "actions")):
            values = np.stack(raw[source]).astype(np.float32)
            if modality == "action":
                values[:, -1] = (1 - values[:, -1]) / 2
            for column, key in enumerate(
                ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
            ):
                width = 2 if modality == "state" and column == 6 else 1
                frame[f"{modality}.{key}"] = list(values[:, column : column + width])
        for key in ("image", "wrist_image"):
            frame[f"video.{key}"] = raw[key].map(
                lambda image: np.asarray(
                    Image.open(BytesIO(image["bytes"])).convert("RGB")
                )
            )
        frame["language.annotation.human.action.task_description"] = raw[
            "task_index"
        ].map(self.tasks)
        return frame


class LiberoParquetDataset(ShardedSingleStepDataset):
    """Reuse upstream sharding and step extraction with an embedded-image reader."""

    def __init__(
        self,
        path,
        embodiment_tag,
        modality_configs,
        statistics,
        task_names=None,
        shard_size=1024,
        episode_sampling_rate=1.0,
        seed=0,
        allow_padding=False,
    ):
        self.dataset_path = Path(path)
        self.embodiment_tag = embodiment_tag
        self.modality_configs = modality_configs
        self.shard_size = shard_size
        self.episode_sampling_rate = episode_sampling_rate
        self.seed = seed
        self.allow_padding = allow_padding
        self.processor = None
        self.rng = np.random.default_rng(seed)
        deltas = modality_configs["action"].delta_indices
        self.action_horizon = max(deltas) - min(deltas) + 1
        self.episode_loader = LiberoEpisodeLoader(self.dataset_path, task_names)
        self.statistics = statistics
        self.shard_dataset()

    def get_dataset_statistics(self) -> dict:
        return self.statistics
