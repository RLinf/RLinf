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

"""Checkpoint map-style training loaders, including prefetched worker RNG."""

import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.data.datasets.common.epoch import forward_set_epoch


class _EpochSampler(Sampler):
    """Bind prefetched indices to the epoch in which they were sampled."""

    def __init__(self, sampler):
        self.sampler = sampler
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.sampler)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def __iter__(self):
        epoch = self.epoch
        return ((epoch, index) for index in self.sampler)


class _StatefulDataset(Dataset):
    """Expose worker RNG and dataset state to torchdata's batch snapshots."""

    def __init__(self, dataset: Dataset, seed: int):
        self.dataset = dataset
        if hasattr(dataset, "set_seed"):
            dataset.set_seed(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def __getitem__(self, key: tuple[int, int]) -> Any:
        epoch, index = key
        if epoch != self.epoch:
            self.set_epoch(epoch)
        return self.dataset[index]

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
            "dataset": self.dataset.state_dict()
            if hasattr(self.dataset, "state_dict")
            else None,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.set_epoch(state["epoch"])
        if state["dataset"] is not None:
            self.dataset.load_state_dict(state["dataset"])
        torch.set_rng_state(state["torch"])
        np.random.set_state(state["numpy"])
        random.setstate(state["random"])


class TrainingDataLoader(StatefulDataLoader):
    """Resume the next consumed batch with the same loader layout and seed.

    Worker snapshots include Python, NumPy and CPU Torch RNG after collation.
    Dataset-specific generators participate through state_dict/load_state_dict.
    Sampler indices carry their epoch to persistent workers.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        seed: int,
        shuffle: bool = False,
        sampler=None,
        **kwargs,
    ):
        wrapped = _StatefulDataset(dataset, seed)
        if sampler is None:
            sampler = DistributedSampler(
                wrapped,
                num_replicas=1,
                rank=0,
                shuffle=shuffle,
                seed=seed,
                drop_last=kwargs.get("drop_last", False),
            )
        generator = torch.Generator().manual_seed(seed)
        self._resume_generator_state = None
        self._settings = {
            "dataset_type": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
            "dataset_length": len(dataset),
            "seed": seed,
            "sampler_type": f"{type(sampler).__module__}.{type(sampler).__qualname__}",
            "sampler": {
                key: getattr(sampler, key, None)
                for key in ("num_replicas", "rank", "seed", "shuffle", "drop_last")
            },
        }
        super().__init__(
            wrapped, sampler=_EpochSampler(sampler), generator=generator, **kwargs
        )
        self._settings.update(
            {
                key: getattr(self, key)
                for key in (
                    "batch_size",
                    "num_workers",
                    "drop_last",
                    "prefetch_factor",
                    "persistent_workers",
                )
            }
        )

    @property
    def epoch(self) -> int:
        """Current epoch used to generate training indices."""
        return self.sampler.epoch

    def set_epoch(self, epoch: int) -> None:
        """Set the sampler and dataset epoch before creating the next iterator."""
        forward_set_epoch(self, epoch)

    def __iter__(self):
        iterator = super().__iter__()
        if self._resume_generator_state is not None:
            self.generator.set_state(self._resume_generator_state)
            self._resume_generator_state = None
        return iterator

    def state_dict(self) -> dict[str, Any]:
        """Snapshot consumed batches and the worker state at that position."""
        loader_state = super().state_dict()
        return {
            "version": 1,
            "settings": self._settings,
            "epoch": self.epoch,
            "generator": self.generator.get_state(),
            "loader": loader_state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a snapshot, rejecting a changed data-loader layout."""
        if state["version"] != 1 or state["settings"] != self._settings:
            raise ValueError(
                "Training data-loader configuration differs from the checkpoint: "
                f"configured {self._settings}, saved {state.get('settings')}."
            )
        self.set_epoch(state["epoch"])
        self.generator.set_state(state["generator"])
        super().load_state_dict(state["loader"])
        self._resume_generator_state = state["generator"]
