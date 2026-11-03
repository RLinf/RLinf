# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adapters for the official OpenPI PyTorch SFT data loader."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.data.storage.lerobot import resolve_lerobot_repo_id
from rlinf.models.embodiment.openpi.transforms.pipeline import (
    norm_stats_path_from_data_kwargs,
    select_openpi_norm_stats,
)


def build_official_openpi_sft_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
) -> tuple[Any, Any]:
    """Build the SFT loader provided by OpenPI for a LeRobot dataset."""
    repo_id = resolve_lerobot_repo_id(data_paths)
    if repo_id is None:
        raise ValueError(
            "OpenPI SFT requires data.train_data_paths to be set to a local "
            "dataset path or LeRobot repo id."
        )
    _validate_local_lerobot_root(repo_id)

    import openpi.training.data_loader as openpi_data_loader

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    model_cfg = cfg.actor.model
    batch_size = cfg.actor.micro_batch_size
    if eval_dataset:
        batch_size = cfg.actor.get("eval_batch_size", batch_size)

    data_kwargs = OmegaConf.select(model_cfg, "openpi_data", default=None)
    if data_kwargs is not None:
        data_kwargs = OmegaConf.to_container(data_kwargs, resolve=True)

    config = get_openpi_config(
        model_cfg.openpi.config_name,
        model_path=model_cfg.model_path,
        batch_size=batch_size * world_size,
        repo_id=repo_id,
        data_kwargs=data_kwargs,
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    select_openpi_norm_stats(
        data_config.norm_stats,
        norm_stats_path=norm_stats_path_from_data_kwargs(data_kwargs),
    )
    config = dataclasses.replace(
        config,
        num_workers=int(
            OmegaConf.select(cfg, "data.num_workers", default=config.num_workers)
        ),
        seed=int(OmegaConf.select(cfg, "actor.seed", default=config.seed)),
    )
    _validate_openpi_model_shape(model_cfg, config)

    if bool(OmegaConf.select(cfg, "data.stateful_loader", default=False)):
        if data_config.rlds_data_dir is not None:
            raise ValueError("data.stateful_loader requires a map-style dataset.")
        dataset = openpi_data_loader.create_torch_dataset(
            data_config, config.model.action_horizon, config.model
        )
        dataset = openpi_data_loader.transform_dataset(dataset, data_config)
        data_loader = StatefulOpenPIDataLoader(
            dataset,
            data_config,
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            num_workers=config.num_workers,
            seed=config.seed,
            shuffle=not eval_dataset,
        )
    else:
        data_loader = openpi_data_loader.create_data_loader(
            config, framework="pytorch", shuffle=not eval_dataset
        )
    return data_loader, data_loader.data_config()


class _WorkerRandomStateDataset(torch.utils.data.Dataset):
    """Capture random transforms at each worker's consumed-data snapshot."""

    def __init__(self, dataset: Any):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]

    def state_dict(self) -> dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])


class StatefulOpenPIDataLoader:
    """Infinite OpenPI batches with resumable worker and sampler state.

    The opt-in path requires at least two ranks; the ordinary single-process
    loader keeps its upstream RandomSampler behavior. Restore requires the same
    rank topology, dataset length, batch size, worker count, and seed.
    Dataset contents and preprocessing must also be unchanged.
    The sampler retains upstream's fixed epoch-zero permutation. This preserves
    the existing training distribution rather than introducing epoch reshuffling.
    """

    def __init__(
        self,
        dataset: Any,
        data_config: Any,
        *,
        batch_size: int,
        world_size: int,
        rank: int,
        num_workers: int,
        seed: int,
        shuffle: bool = True,
    ) -> None:
        from openpi.training.data_loader import _collate_fn, _worker_init_fn
        from torchdata.stateful_dataloader import StatefulDataLoader

        if world_size < 2:
            raise ValueError("Stateful OpenPI loader requires world size >= 2.")
        self._config = data_config
        self._signature = {
            "world_size": world_size,
            "rank": rank,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "seed": seed,
            "dataset_length": len(dataset),
            "shuffle": shuffle,
        }
        self._generator = torch.Generator().manual_seed(seed)
        # Match upstream DistributedSampler's default seed and epoch handling.
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=True,
        )
        self._data_loader = StatefulDataLoader(
            _WorkerRandomStateDataset(dataset),
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context="spawn" if num_workers else None,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=self._generator,
            snapshot_every_n_steps=1,
        )
        if not len(self._data_loader):
            raise ValueError("Each rank must contain at least one full batch.")
        self._restored_generator = None

    def __len__(self) -> int:
        return len(self._data_loader)

    def data_config(self) -> Any:
        """Return the unmodified OpenPI transform configuration."""
        return self._config

    def __iter__(self) -> Iterator[Any]:
        import jax
        from openpi.models.model import Observation

        while True:
            iterator = iter(self._data_loader)
            if self._restored_generator is not None:
                # Reconstructing an iterator consumes a base-seed draw. It has
                # already occurred in the uninterrupted stream represented by
                # the snapshot, so preserve the saved next-epoch generator.
                self._generator.set_state(self._restored_generator)
                self._restored_generator = None
            for batch in iterator:
                batch = jax.tree.map(torch.as_tensor, batch)
                yield Observation.from_dict(batch), batch["actions"]

    def state_dict(self) -> dict[str, Any]:
        """Snapshot consumed batches, including prefetched worker state."""
        return copy.deepcopy(
            {
                "signature": self._signature,
                "loader": self._data_loader.state_dict(),
                "generator": self._generator.get_state(),
            }
        )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a snapshot before creating the training iterator."""
        if state["signature"] != self._signature:
            raise ValueError(
                "OpenPI loader restore requires unchanged rank topology, "
                "dataset length, batch size, workers, seed, and shuffle setting."
            )
        self._data_loader.load_state_dict(copy.deepcopy(state["loader"]))
        self._generator.set_state(state["generator"])
        self._restored_generator = state["generator"].clone()

    def mark_checkpoint_complete(self, path: str, states: list[dict]) -> None:
        """Publish readiness after every rank's data and RNG files are saved."""
        root = Path(path)
        metadata = {
            "version": 1,
            "complete": True,
            "signatures": [state["signature"] for state in states],
            "sha256": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in ("data.pt", "rng.pt")
            },
        }
        temporary = root / "openpi_loader_state.json.tmp"
        temporary.write_text(json.dumps(metadata, sort_keys=True))
        temporary.replace(root / "openpi_loader_state.json")

    def validate_checkpoint(self, path: str) -> None:
        """Reject missing, partial, incompatible, or corrupted loader state."""
        root = Path(path)
        marker = root / "openpi_loader_state.json"
        if not marker.is_file():
            raise ValueError("Checkpoint has no completed OpenPI loader state.")
        metadata = json.loads(marker.read_text())
        signatures = metadata.get("signatures", [])
        if (
            metadata.get("version") != 1
            or not metadata.get("complete")
            or len(signatures) != self._signature["world_size"]
            or signatures[self._signature["rank"]] != self._signature
        ):
            raise ValueError("Checkpoint OpenPI loader topology/configuration differs.")
        for name in ("data.pt", "rng.pt"):
            file = root / name
            if not file.is_file() or hashlib.sha256(
                file.read_bytes()
            ).hexdigest() != metadata["sha256"].get(name):
                raise ValueError(
                    f"Incomplete or corrupted OpenPI loader state: {name}."
                )


def get_official_openpi_sft_num_batches(data_loader: Any) -> int:
    """Return the inner PyTorch ``DataLoader`` length used by OpenPI."""
    if isinstance(data_loader, StatefulOpenPIDataLoader):
        return len(data_loader)
    openpi_loader = getattr(data_loader, "_data_loader", None)
    torch_loader = getattr(openpi_loader, "_data_loader", None) or getattr(
        openpi_loader, "torch_loader", None
    )
    if torch_loader is None:
        raise TypeError(
            "OpenPI dataloader does not expose an inner torch DataLoader; "
            "cannot infer steps per epoch from len()."
        )
    return len(torch_loader)


def is_official_openpi_sft_dataloader(data_loader: Any) -> bool:
    """Return whether ``data_loader`` has OpenPI's loader wrapper layout."""
    return getattr(data_loader, "_data_loader", None) is not None


def _validate_local_lerobot_root(repo_id: str) -> None:
    """Reject a split directory instead of a concrete LeRobot dataset root."""
    path = Path(repo_id).expanduser()
    if not path.exists() or not path.is_dir():
        return
    if not (path / "meta" / "info.json").is_file():
        raise ValueError(
            "OpenPI SFT local data path must contain meta/info.json directly: "
            f"{path}. Pass the concrete LeRobot dataset directory, not its "
            "split parent."
        )


def _validate_openpi_model_shape(model_cfg: Any, openpi_config: Any) -> None:
    """Keep the local Pi0 architecture consistent with the OpenPI config.

    The official loader sizes the SFT action window from
    ``TrainConfig.model.action_horizon``, not ``num_action_chunks`` (that field
    is the env-executed chunk). ``get_model`` already builds the network from
    the same TrainConfig unless YAML overrides ``openpi.action_horizon``.
    """
    yaml_horizon = OmegaConf.select(model_cfg, "openpi.action_horizon", default=None)
    official_horizon = int(openpi_config.model.action_horizon)
    if yaml_horizon is not None and int(yaml_horizon) != official_horizon:
        raise ValueError(
            "openpi SFT data uses TrainConfig.model.action_horizon="
            f"{official_horizon} from {model_cfg.openpi.config_name}; "
            f"openpi.action_horizon={int(yaml_horizon)} would build a different "
            "network. Unset it, or change the TrainConfig."
        )

    local_action_dim = int(model_cfg.openpi.model_action_dim)
    official_action_dim = int(openpi_config.model.action_dim)
    if local_action_dim != official_action_dim:
        raise ValueError(
            "openpi SFT model action dim must match the official OpenPI "
            f"config: actor.model.openpi.model_action_dim={local_action_dim}, "
            f"{model_cfg.openpi.config_name}.model.action_dim="
            f"{official_action_dim}."
        )

    yaml_state_input = OmegaConf.select(
        model_cfg, "openpi.discrete_state_input", default=None
    )
    official_state_input = bool(openpi_config.model.discrete_state_input)
    if yaml_state_input is not None and bool(yaml_state_input) != official_state_input:
        raise ValueError(
            "openpi.discrete_state_input must match the SFT tokenizer config: "
            f"YAML={yaml_state_input}, "
            f"{model_cfg.openpi.config_name}={official_state_input}. "
            "Select a TrainConfig with the intended state conditioning."
        )
