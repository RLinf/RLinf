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


def build_openwam_sft_dataloader(
    cfg: Any, world_size: int, rank: int, data_paths: Any, eval_dataset: bool = False
) -> tuple[Any, dict[str, Any]]:
    """Construct a distributed DataLoader yielding lists of native samples."""
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
    native_dl.split = "val" if eval_dataset else "train"
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
    dataset = (
        datasets[0] if len(datasets) == 1 else torch.utils.data.ConcatDataset(datasets)
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
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
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=list,
        pin_memory=True,
        drop_last=True,
    )
    return loader, {
        "dataset_type": str(native_dl.type),
        "dataset_dir": dataset_dirs[0] if len(dataset_dirs) == 1 else dataset_dirs,
        "num_samples": len(dataset),
        "num_samples_per_dataset": per_dataset,
    }
