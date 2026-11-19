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

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.interfaces import ShardedDataset
from gr00t.data.types import EmbodimentTag, MessageType, ModalityConfig
from PIL import Image

_COMPONENT_SLICES = {
    "x": slice(0, 1),
    "y": slice(1, 2),
    "z": slice(2, 3),
    "roll": slice(3, 4),
    "pitch": slice(4, 5),
    "yaw": slice(5, 6),
    "gripper": slice(6, None),
}


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

    @staticmethod
    def _convert_numeric(frame: pd.DataFrame) -> dict[str, np.ndarray]:
        values = {
            "state": np.stack(frame["state"]).astype(np.float32),
            "action": np.stack(frame["actions"]).astype(np.float32),
        }
        values["action"][:, -1] = (1 - values["action"][:, -1]) / 2
        return values

    def read_numeric(self, index: int) -> dict[str, np.ndarray]:
        frame = pd.read_parquet(self.episode_path(index), columns=["state", "actions"])
        return self._convert_numeric(frame)

    def compute_statistics(self) -> dict:
        with ThreadPoolExecutor(max_workers=8) as pool:
            numeric = list(pool.map(self.read_numeric, range(len(self.episodes))))
        result = {}
        for modality in ("state", "action"):
            values = np.concatenate([item[modality] for item in numeric])
            stats = {
                "mean": values.mean(axis=0),
                "std": values.std(axis=0),
                "min": values.min(axis=0),
                "max": values.max(axis=0),
                "q01": np.quantile(values, 0.01, axis=0),
                "q99": np.quantile(values, 0.99, axis=0),
            }
            result[modality] = {
                key: {name: value[indices].tolist() for name, value in stats.items()}
                for key, indices in _COMPONENT_SLICES.items()
            }
        return result

    def statistics_cache_path(self, cache_path: Path) -> Path:
        """Cache entry for this dataset and selected episodes under a base name."""
        identity = [
            str(self.path.resolve()),
            [episode["episode_index"] for episode in self.episodes],
        ]
        digest = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]
        return cache_path.with_name(f"{cache_path.stem}-{digest}{cache_path.suffix}")

    def prepare_statistics(self, cache_path: Path) -> Path:
        """Refresh cached statistics when the selected episode files change."""
        cache_path = self.statistics_cache_path(cache_path)
        sources = []
        for index, episode in enumerate(self.episodes):
            stat = self.episode_path(index).stat()
            sources.append([episode["episode_index"], stat.st_size, stat.st_mtime_ns])
        dataset_path = str(self.path.resolve())
        cached = json.loads(cache_path.read_text()) if cache_path.exists() else None
        if (
            cached is not None
            and cached.get("dataset_path") == dataset_path
            and cached["sources"] == sources
        ):
            return cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "dataset_path": dataset_path,
                    "sources": sources,
                    "statistics": self.compute_statistics(),
                }
            )
        )
        return cache_path

    @staticmethod
    def load_statistics(cache_path: Path) -> dict:
        """Read statistics after the rank preparing the cache has finished."""
        return json.loads(cache_path.read_text())["statistics"]

    def __getitem__(self, index: int) -> pd.DataFrame:
        raw = pd.read_parquet(self.episode_path(index))
        frame = pd.DataFrame(index=raw.index)
        for modality, values in self._convert_numeric(raw).items():
            for key, indices in _COMPONENT_SLICES.items():
                frame[f"{modality}.{key}"] = list(values[:, indices])
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


class LiberoParquetDataset(ShardedDataset):
    """Shard embedded-image episodes with GR00T's balanced timestep schedule."""

    def __init__(
        self,
        episode_loader: LiberoEpisodeLoader,
        embodiment_tag: EmbodimentTag,
        modality_configs: dict[str, ModalityConfig],
        statistics: dict,
        shard_size: int = 1024,
        episode_sampling_rate: float = 1.0,
        seed: int = 0,
        allow_padding: bool = False,
    ):
        super().__init__(episode_loader.path)
        self.episode_loader = episode_loader
        self.embodiment_tag = embodiment_tag
        self.modality_configs = modality_configs
        self.statistics = statistics
        self.allow_padding = allow_padding
        self.processor = None
        deltas = modality_configs["action"].delta_indices
        horizon = max(deltas) - min(deltas) + 1
        lengths = np.maximum(
            0, np.asarray(episode_loader.episode_lengths) - horizon + 1
        )
        num_shards = int(np.ceil(lengths.sum() / shard_size))
        self.sharded_episodes = [[] for _ in range(num_shards)]
        self.shard_lengths = np.zeros(num_shards, dtype=int)
        rng = np.random.default_rng(seed)
        num_splits = int(1 / episode_sampling_rate)
        # Keep the upstream balanced assignment and RNG order for existing runs.
        for episode_index in rng.permutation(len(lengths)):
            steps = np.arange(lengths[episode_index])
            rng.shuffle(steps)
            for split in range(num_splits):
                indices = steps[split::num_splits]
                if not indices.size:
                    continue
                shard = np.argmin(self.shard_lengths)
                self.sharded_episodes[shard].append((episode_index, indices))
                self.shard_lengths[shard] += len(indices)

    def __len__(self) -> int:
        return len(self.shard_lengths)

    def get_shard_length(self, idx: int) -> int:
        return self.shard_lengths[idx]

    def get_shard(self, idx: int) -> list[dict]:
        """Load episode steps and process the action chunks assigned to one shard."""
        datapoints = []
        for episode_index, steps in self.sharded_episodes[idx]:
            episode = self.episode_loader[episode_index]
            for step in steps:
                content = extract_step_data(
                    episode,
                    step,
                    self.modality_configs,
                    self.embodiment_tag,
                    self.allow_padding,
                )
                datapoints.append(
                    self.processor(
                        [{"type": MessageType.EPISODE_STEP.value, "content": content}]
                    )
                )
        return datapoints

    def get_dataset_statistics(self) -> dict:
        return self.statistics
