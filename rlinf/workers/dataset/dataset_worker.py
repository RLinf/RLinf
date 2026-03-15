# Copyright 2025 The RLinf Authors.
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

"""Generic dataset worker for offline training.

This worker owns a rank's dataset shard and exposes a minimal API for actors:
- sample(batch_size)
- get_size()
- get_obs_action_dims()

Built-in dataset types:
- d4rl: load from local hdf5 path via D4RLDataset.from_path (no env creation).

Custom dataset support:
- Set ``env.dataset_cls`` to a fully-qualified class path.
- The class should implement ``from_path(...)`` and ``sample(...)``.
- ``partition(rank, world_size)`` is optional; if missing, full dataset is used on each rank.
"""

import importlib

import numpy as np
from omegaconf import DictConfig, OmegaConf

from rlinf.data.datasets.d4rl import D4RLDataset
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.placement import HybridComponentPlacement


class DatasetWorker(Worker):
    """Worker that owns an offline dataset shard and exposes sample() for actors."""

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        self.cfg = cfg
        self.dataset = None
        self._placement = HybridComponentPlacement(cfg, Cluster())
        self.actor_group_name = str(cfg.actor.group_name)

    def _get_dataset_type(self) -> str:
        # Prefer explicit env.dataset_type; fallback to cfg.data.type if present.
        dataset_type = self.cfg.env.get("dataset_type", None)
        assert dataset_type is not None, (
            "DatasetWorker requires explicit dataset_type in cfg.env.dataset_type "
        )
        return str(dataset_type).lower()

    def _load_dataset_class(self, dataset_type: str):
        if dataset_type == "d4rl":
            return D4RLDataset

        dataset_cls_path = self.cfg.env.get("dataset_cls", None)
        if not dataset_cls_path:
            raise NotImplementedError(
                f"Unsupported dataset_type={dataset_type!r}. "
                "Provide env.dataset_cls for custom dataset classes."
            )

        module_name, class_name = str(dataset_cls_path).rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def init_worker(self) -> None:
        """Build dataset and keep this rank's shard."""
        dataset_type = self._get_dataset_type()
        dataset_cls = self._load_dataset_class(dataset_type)

        env_name = self.cfg.env.get("env_name")
        dataset_path = self.cfg.env.get("dataset_path", None)
        dataset_init_kwargs_cfg = self.cfg.env.get("dataset_init_kwargs", {})
        if OmegaConf.is_config(dataset_init_kwargs_cfg):
            init_kwargs = OmegaConf.to_container(dataset_init_kwargs_cfg, resolve=True)
        else:
            init_kwargs = dict(dataset_init_kwargs_cfg)
        # Keep D4RL backward-compatible defaults, but avoid injecting unrelated
        # kwargs for custom dataset classes.
        if dataset_type == "d4rl":
            init_kwargs.setdefault("dataset_path", dataset_path)
            init_kwargs.setdefault("env_name", env_name)
        if hasattr(dataset_cls, "from_path"):
            full = dataset_cls.from_path(**init_kwargs)
        else:
            raise NotImplementedError(
                f"Dataset class {dataset_cls} must implement from_path(...)."
            )

        if hasattr(full, "partition"):
            self.dataset = full.partition(self._rank, self._world_size)
        else:
            self.dataset = full
        if hasattr(self.dataset, "size"):
            shard_size = int(self.dataset.size)
        elif hasattr(self.dataset, "__len__"):
            shard_size = int(len(self.dataset))
        else:
            raise TypeError(
                f"Dataset object {type(self.dataset)} has neither 'size' nor '__len__'."
            )
        self.log_info(
            f"DatasetWorker(type={dataset_type}) rank {self._rank}: shard size {shard_size}."
        )

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a batch from this rank's shard. Call from actor via remote."""
        if self.dataset is None:
            raise RuntimeError(
                "DatasetWorker.init_worker() must be called before sample()."
            )
        return self.dataset.sample(batch_size)

    def sample_batches(
        self, batch_size: int, num_batches: int
    ) -> list[dict[str, np.ndarray]]:
        """Sample multiple batches from this rank's shard."""
        if self.dataset is None:
            raise RuntimeError(
                "DatasetWorker.init_worker() must be called before sample_batches()."
            )
        return [
            self.dataset.sample(batch_size) for _ in range(max(1, int(num_batches)))
        ]

    def send_batches_to_actor(
        self,
        batch_size: int = 256,
        num_batches: int = 1,
    ) -> None:
        """Sample batches and send to mapped actor ranks via Worker.send()."""
        actor_world_size = int(self._placement.get_world_size("actor"))
        num_batches = max(1, int(num_batches))
        for dst_rank in range(self._rank, actor_world_size, self._world_size):
            batches = self.sample_batches(batch_size, num_batches)
            self.send(
                batches,
                dst_group_name=self.actor_group_name,
                dst_rank=dst_rank,
            )

    def get_size(self) -> int:
        """Return this shard's size (for actor to know dataset size)."""
        if self.dataset is None:
            raise RuntimeError(
                "DatasetWorker.init_worker() must be called before get_size()."
            )
        if hasattr(self.dataset, "size"):
            return int(self.dataset.size)
        if hasattr(self.dataset, "__len__"):
            return int(len(self.dataset))
        raise TypeError(
            f"Dataset object {type(self.dataset)} has neither 'size' nor '__len__'."
        )

    def get_obs_action_dims(self) -> tuple[int, int]:
        """Return (obs_dim, action_dim) for actor model build."""
        if self.dataset is None:
            raise RuntimeError(
                "DatasetWorker.init_worker() must be called before get_obs_action_dims()."
            )
        obs_dim = int(self.dataset.observations.shape[-1])
        action_dim = int(self.dataset.actions.shape[-1])
        return obs_dim, action_dim
