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

"""Reuse Isaac-GR00T's processor and distributed sharded dataset."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import IterableDataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader


def build_finetune_config(cfg: DictConfig, data_paths=None):
    """Map RLinf options to the pinned upstream finetuning pipeline."""
    from gr00t.configs.base_config import get_default_config

    options = OmegaConf.to_container(cfg.actor.model.gr00t_sft, resolve=True)
    config = get_default_config()
    # Explicit defaults from upstream launch_finetune.py; user options win.
    for key, value in {
        "load_bf16": False,
        "reproject_vision": False,
        "backbone_trainable_params_fp32": True,
        "use_relative_action": True,
    }.items():
        setattr(config.model, key, value)
    for key, value in options.items():
        if not hasattr(config.model, key):
            raise ValueError(f"Unknown GR00T model option: {key}")
        setattr(config.model, key, value)
    config.training.start_from_checkpoint = cfg.actor.model.model_path
    config.training.transformers_local_files_only = True
    config.training.max_steps = cfg.runner.max_steps
    paths = cfg.data.train_data_paths if data_paths is None else data_paths
    config.load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [paths]
                        if isinstance(paths, str)
                        else list(paths),
                        "mix_ratio": 1.0,
                        "embodiment_tag": cfg.actor.model.embodiment_tag,
                    }
                ],
                **OmegaConf.to_container(cfg.data.gr00t, resolve=True),
            }
        }
    )
    config.validate()
    return config


class Gr00tShardedDataset(IterableDataset):
    """Read the Isaac-GR00T shards with a checkpointable per-worker cursor."""

    def __init__(self, dataset: Any, batch_size: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        return _ShardIterator(self.dataset, self.batch_size)


class _ShardIterator:
    def __init__(self, dataset: Any, batch_size: int) -> None:
        self.dataset = dataset
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        first_shard = dataset.rank * num_workers + worker_id
        self._partition = slice(first_shard, None, dataset.world_size * num_workers)
        self._identity = (
            dataset.rank,
            dataset.world_size,
            worker_id,
            num_workers,
            batch_size,
            dataset.seed,
            dataset.num_shards_per_epoch,
            tuple(dataset.weights),
            tuple(
                (
                    str(shards.dataset_path),
                    len(shards),
                    shards.shard_size,
                    shards.episode_sampling_rate,
                )
                for shards in dataset.datasets
            ),
        )
        self._epoch = 0
        self._schedule = self._worker_schedule()
        self._shard_index = 0
        self._sample_offset = 0
        self._rng = np.random.default_rng(dataset.seed)
        self._rng_before_shard = None
        self._samples = None
        self._order = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._prefetched = None

    def __iter__(self):
        return self

    def _worker_schedule(self):
        if self._epoch == 0:
            schedule = self.dataset.shard_sampling_schedule
        else:
            self.dataset.epoch = self._epoch
            schedule = self.dataset.generate_shard_sampling_schedule()
        schedule = schedule[self._partition]
        if not schedule:
            raise ValueError("GR00T shard schedule has no entries for this data worker")
        return schedule

    def _read_shard(self, shard_index: int):
        dataset_index, index = self._schedule[shard_index]
        return self.dataset.datasets[dataset_index].get_shard(index)

    def _load_current_shard(self):
        if self._shard_index == len(self._schedule):
            self._epoch += 1
            self._schedule = self._worker_schedule()
            self._shard_index = 0

        if self._prefetched is None:
            self._samples = self._read_shard(self._shard_index)
        else:
            self._samples = self._prefetched.result()
            self._prefetched = None

        if not 0 <= self._sample_offset < len(self._samples):
            raise ValueError("GR00T sample cursor falls outside the current shard")

        self._rng_before_shard = self._rng.bit_generator.state
        self._order = np.arange(len(self._samples))
        self._rng.shuffle(self._order)

        next_shard = self._shard_index + 1
        if next_shard < len(self._schedule):
            shard_dataset, shard_index = self._schedule[next_shard]
            self._prefetched = self._executor.submit(
                self.dataset.datasets[shard_dataset].get_shard, shard_index
            )

    def __next__(self):
        if self._samples is None:
            self._load_current_shard()
        sample = self._samples[self._order[self._sample_offset]]
        self._sample_offset += 1
        if self._sample_offset == len(self._samples):
            self._shard_index += 1
            self._sample_offset = 0
            self._samples = None
            self._order = None
            self._rng_before_shard = None
        return sample

    def state_dict(self) -> dict[str, Any]:
        # Recreate the shuffle without saving the permutation or cached samples.
        rng = self._rng_before_shard or self._rng.bit_generator.state
        return {
            "version": 2,
            "identity": self._identity,
            "epoch": self._epoch,
            "shard_index": self._shard_index,
            "sample_offset": self._sample_offset,
            "shuffle_rng": rng,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("version") != 2:
            raise ValueError("Unsupported GR00T shard cursor checkpoint")
        if state["identity"] != self._identity:
            raise ValueError("GR00T dataset, batching or data worker layout changed")
        if self._prefetched is not None:
            self._prefetched.result()
            self._prefetched = None
        self._epoch = state["epoch"]
        self._schedule = self._worker_schedule()
        self._shard_index = state["shard_index"]
        self._sample_offset = state["sample_offset"]
        if not 0 <= self._shard_index <= len(self._schedule) or (
            self._shard_index == len(self._schedule) and self._sample_offset != 0
        ):
            raise ValueError("GR00T shard cursor falls outside its sampling schedule")
        self._rng.bit_generator.state = state["shuffle_rng"]
        self._samples = None
        self._order = None
        self._rng_before_shard = None


class Gr00tSftDataLoader(StatefulDataLoader):
    """Start workers lazily for fresh training and eagerly for checkpoint restore."""

    _defer_iterator_start = True

    def __iter__(self):
        if self._defer_iterator_start:
            return self._iterate_after_seeding()
        return super().__iter__()

    def _iterate_after_seeding(self):
        self._defer_iterator_start = False
        yield from super().__iter__()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Enable iterator startup before the worker restores the training RNG."""
        super().load_state_dict(state_dict)
        self._defer_iterator_start = False


def build_gr00t_sft_dataloader(
    cfg: DictConfig,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
):
    """Build the upstream Spatial training stream without resharding it."""
    if eval_dataset:
        raise ValueError("GR00T sharded SFT does not support validation datasets")
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline

    from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
        redirect_qwen3_backbone_to_local,
    )

    config = build_finetune_config(cfg, data_paths)
    output_dir = (
        Path(cfg.runner.logger.log_path)
        / cfg.runner.logger.experiment_name
        / "gr00t"
        / f"rank_{rank}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = Gr00tN1d7Pipeline(config, output_dir)
    with redirect_qwen3_backbone_to_local(
        "nvidia/Cosmos-Reason2-2B", cfg.actor.model.backbone_model_path
    ):
        dataset, _ = pipeline._create_dataset(output_dir)
    if (dataset.rank, dataset.world_size) != (rank, world_size):
        raise RuntimeError(
            f"GR00T dataset rank/world {(dataset.rank, dataset.world_size)} "
            f"does not match RLinf {(rank, world_size)}"
        )
    if rank == 0:
        pipeline.processor.save_pretrained(output_dir / "processor")
    workers = cfg.data.num_workers
    batch_size = cfg.actor.micro_batch_size
    loader = Gr00tSftDataLoader(
        Gr00tShardedDataset(dataset, batch_size),
        batch_size=batch_size,
        collate_fn=pipeline.processor.collator,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        **({"multiprocessing_context": "fork"} if workers > 0 else {}),
    )
    return loader, None
