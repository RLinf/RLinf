# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Build an RLinf SFT loader from OpenWAM's native dataset readers."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import ListConfig, OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler


def _normalization_stats_paths(dataset: Any) -> list[str]:
    """Collect stats artifacts from leaf and aggregate native readers."""
    paths: list[str] = []
    visited: set[int] = set()

    def visit(node: Any) -> None:
        if node is None or id(node) in visited:
            return
        visited.add(id(node))
        path = getattr(node, "normalization_stats_path", None)
        if path:
            paths.append(str(path))
        for attr in ("buckets", "_buckets", "datasets", "_datasets", "_sub_datasets"):
            children = getattr(node, attr, None)
            if children:
                for child in children:
                    visit(child)

    visit(dataset)
    return list(dict.fromkeys(paths))


@dataclass(frozen=True)
class ValidationSample:
    """One validation sample, marked when it only keeps ranks synchronized."""

    sample: Any
    is_padding: bool = False


class PaddedValidationDataset(torch.utils.data.Dataset):
    """Pad a validation dataset by repeating its first sample when necessary."""

    def __init__(self, dataset: Any, total_size: int):
        if len(dataset) == 0:
            raise ValueError("Cannot pad an empty validation dataset.")
        self.dataset = dataset
        self.total_size = int(total_size)

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, index: int) -> ValidationSample:
        dataset_size = len(self.dataset)
        is_padding = index >= dataset_size
        return ValidationSample(
            self.dataset[index % dataset_size], is_padding=is_padding
        )


class PaddedValidationSampler(torch.utils.data.Sampler[int]):
    """Shard validation indices evenly, padding only for collective alignment."""

    def __init__(self, dataset: Any, num_replicas: int, rank: int):
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.num_samples = math.ceil(len(dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        return iter(range(self.rank, self.total_size, self.num_replicas))

    def __len__(self) -> int:
        return self.num_samples


class EpochStatefulDistributedSampler(StatefulDistributedSampler):
    """``StatefulDistributedSampler`` that also checkpoints the shuffle epoch.

    torchdata's sampler only records how many indices were yielded; the
    permutation itself depends on ``set_epoch``, which a resumed process would
    otherwise start again from epoch 0 and replay data it has already seen.
    """

    _EPOCH = "epoch"

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state[self._EPOCH] = int(self.epoch)
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        if self._EPOCH in state_dict:
            self.set_epoch(int(state_dict[self._EPOCH]))


def build_openwam_sft_dataloader(
    cfg: Any, world_size: int, rank: int, data_paths: Any, eval_dataset: bool = False
) -> tuple[Any, dict[str, Any]]:
    """Construct a distributed ``StatefulDataLoader`` yielding lists of native samples.

    The loader and its sampler expose ``state_dict``/``load_state_dict``, so the
    SFT worker checkpoints the data position (and the shuffle epoch) next to the
    model and a ``runner.resume_dir`` run continues exactly where it stopped.
    """
    if isinstance(data_paths, (list, tuple, ListConfig)):
        dataset_dirs = [str(path) for path in data_paths if path is not None]
    elif data_paths is None:
        dataset_dirs = []
    else:
        dataset_dirs = [str(data_paths)]
    if not dataset_dirs:
        raise ValueError("OpenWAM SFT requires data.train_data_paths.")

    model_cfg = cfg.actor.model
    model_path = str(model_cfg.model_path)
    config_path = OmegaConf.select(model_cfg, "openwam_config_path", default=None)
    if config_path is None:
        config_path = f"{model_path}/config.yaml"
    native_cfg = OmegaConf.load(str(config_path))
    if not hasattr(native_cfg, "dataloader"):
        raise ValueError(f"OpenWAM config has no dataloader section: {config_path}")
    native_dl = native_cfg.dataloader.copy()
    # LeRobot-style readers select episodes by split. A validation root that only
    # ships a train split (the usual case for a separate held-out dataset) is
    # read with data.openwam_val_split=train.
    val_split = str(OmegaConf.select(cfg, "data.openwam_val_split", default="val"))
    native_dl.split = val_split if eval_dataset else "train"
    overrides = OmegaConf.select(cfg, "data.openwam", default=None)
    if overrides is not None:
        native_dl = OmegaConf.merge(native_dl, overrides)

    from openwam.dataloader.registry import build_dataset

    # Every path is read with the checkpoint's own dataloader settings and the
    # windows are concatenated, so a mixture is sampled in proportion to size.
    datasets = []
    for dataset_dir in dataset_dirs:
        dl_cfg = native_dl.copy()
        dl_cfg.dataset_dir = dataset_dir
        datasets.append(build_dataset(dl_cfg, split=str(native_dl.split)))
    per_dataset = {d: len(ds) for d, ds in zip(dataset_dirs, datasets)}
    stats_paths_by_dataset = [
        _normalization_stats_paths(dataset) for dataset in datasets
    ]
    has_stats = [
        getattr(dataset, "normalization_stats", None) is not None
        for dataset in datasets
    ]
    if any(stats_paths_by_dataset) and not all(stats_paths_by_dataset):
        raise ValueError(
            "OpenWAM SFT dataset roots do not share one normalization stats "
            "artifact; configure normalization_stats_path explicitly for every root."
        )
    if any(
        has_stats[index] and not stats_paths_by_dataset[index]
        for index in range(len(datasets))
    ):
        raise ValueError(
            "OpenWAM reader exposes normalization statistics without a file path; "
            "configure normalization_stats_path so the trained policy can be exported."
        )
    stats_paths = [path for paths in stats_paths_by_dataset for path in paths]
    stats_digest = None
    if stats_paths:
        digests = []
        for stats_path in stats_paths:
            path = Path(stats_path)
            if not path.is_file():
                raise FileNotFoundError(
                    f"OpenWAM normalization stats artifact not found: {path}"
                )
            digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
        if len(set(digests)) != 1:
            raise ValueError(
                "OpenWAM SFT dataset roots use different normalization stats; "
                "use one shared normalization_stats_path before mixing roots."
            )
        stats_digest = digests[0]
    if sum(per_dataset.values()) == 0:
        raise ValueError(
            f"OpenWAM {native_dl.split} dataset is empty: {per_dataset}. "
            + (
                "If the validation root only has a train split, set "
                "data.openwam_val_split=train."
                if eval_dataset
                else "Check data.train_data_paths and the checkpoint's dataloader config."
            )
        )
    dataset = (
        datasets[0] if len(datasets) == 1 else torch.utils.data.ConcatDataset(datasets)
    )
    if eval_dataset:
        sampler = PaddedValidationSampler(
            dataset, num_replicas=int(world_size), rank=int(rank)
        )
        loader_dataset = PaddedValidationDataset(dataset, sampler.total_size)
    else:
        sampler = EpochStatefulDistributedSampler(
            dataset,
            num_replicas=int(world_size),
            rank=int(rank),
            shuffle=True,
            drop_last=True,
            seed=int(OmegaConf.select(cfg, "actor.seed", default=0)),
        )
        loader_dataset = dataset
    batch_size = (
        int(cfg.actor.get("eval_batch_size", cfg.actor.micro_batch_size))
        if eval_dataset
        else int(cfg.actor.micro_batch_size)
    )
    num_workers = int(OmegaConf.select(cfg, "data.num_workers", default=0))
    prefetch_factor = int(OmegaConf.select(cfg, "data.prefetch_factor", default=2))
    loader = StatefulDataLoader(
        loader_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=list,
        pin_memory=True,
        drop_last=not eval_dataset,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    if len(loader) == 0 and not eval_dataset:
        split_name = "training"
        raise ValueError(
            f"OpenWAM {split_name} loader has zero batches: "
            f"num_samples={len(dataset)}, world_size={world_size}, "
            f"batch_size={batch_size}, drop_last=True. "
            "Add more data or reduce the actor world size/batch size."
        )
    return loader, {
        "dataset_type": str(native_dl.type),
        "dataset_dir": dataset_dirs[0] if len(dataset_dirs) == 1 else dataset_dirs,
        "num_samples": len(dataset),
        "num_samples_per_dataset": per_dataset,
        "normalization_stats_path": stats_paths[0] if stats_paths else None,
        "normalization_stats_digest": stats_digest,
    }
