# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Build an RLinf SFT loader from OpenWAM's native dataset readers."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import ListConfig, OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler


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
    sampler = EpochStatefulDistributedSampler(
        dataset,
        num_replicas=int(world_size),
        rank=int(rank),
        shuffle=not eval_dataset,
        drop_last=True,
        seed=int(OmegaConf.select(cfg, "actor.seed", default=0)),
    )
    batch_size = (
        int(cfg.actor.get("eval_batch_size", cfg.actor.micro_batch_size))
        if eval_dataset
        else int(cfg.actor.micro_batch_size)
    )
    num_workers = int(OmegaConf.select(cfg, "data.num_workers", default=0))
    prefetch_factor = int(OmegaConf.select(cfg, "data.prefetch_factor", default=2))
    loader = StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=list,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    return loader, {
        "dataset_type": str(native_dl.type),
        "dataset_dir": dataset_dirs[0] if len(dataset_dirs) == 1 else dataset_dirs,
        "num_samples": len(dataset),
        "num_samples_per_dataset": per_dataset,
    }
