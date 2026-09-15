# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""LeRobot v3 loader for SO-101 DAgger intervention frames."""

from __future__ import annotations

import json
import pathlib
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from openpi import transforms
from openpi.models import model as openpi_model
from torch.utils.data.distributed import DistributedSampler

from rlinf.data.so101_sft_data_loader import (
    SO101SftDataConfig,
    _CastModelInputs,
    _ClipNormalizedSO101,
    _collate,
    _NormalizeRawSO101Joints,
    _TransformedDataset,
)


class _IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Any, indices: np.ndarray) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[int(self.indices[index])]

    def __len__(self) -> int:
        return int(self.indices.size)


class SO101DaggerDataLoader:
    def __init__(
        self,
        loader: torch.utils.data.DataLoader,
        data_config: SO101SftDataConfig,
        sampler: DistributedSampler | None,
    ) -> None:
        self._loader = loader
        self._data_config = data_config
        self.sampler = sampler
        self._epoch = 0

    def data_config(self) -> SO101SftDataConfig:
        return self._data_config

    def __iter__(self) -> Iterator[dict[str, Any]]:
        while True:
            if self.sampler is not None:
                self.sampler.set_epoch(self._epoch)
                self._epoch += 1
            for batch in self._loader:
                yield {
                    "observation": openpi_model.Observation.from_dict(batch),
                    "actions": batch["actions"],
                }

    def __len__(self) -> int:
        return len(self._loader)


def build_so101_dagger_dataloader(
    cfg: Any, world_size: int, rank: int, data_path: str, eval_dataset: bool = False
) -> tuple[SO101DaggerDataLoader, SO101SftDataConfig]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from openpi.shared import normalize

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    root = pathlib.Path(data_path).expanduser().resolve()
    info = json.loads((root / "meta" / "info.json").read_text())
    features = info.get("features", {})
    required = {
        "expert_action",
        "policy_action",
        "executed_action",
        "intervene_flag",
        "is_success",
    }
    missing = sorted(required.difference(features))
    if missing:
        raise ValueError(f"DAgger LeRobot dataset missing required fields: {missing}")

    model_cfg = cfg.actor.model
    openpi_cfg = model_cfg.openpi
    train_config = get_openpi_config(
        str(openpi_cfg.config_name),
        model_path=str(model_cfg.model_path),
        batch_size=int(cfg.actor.micro_batch_size) * world_size,
    )
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats_path = pathlib.Path(str(openpi_cfg.norm_stats_path)).expanduser()
    if not norm_stats_path.is_file():
        raise FileNotFoundError(f"SO-101 norm stats not found: {norm_stats_path}")
    norm_stats = normalize.load(norm_stats_path.parent)
    horizon = int(model_cfg.num_action_chunks)
    fps = int(info["fps"])

    dataset = LeRobotDataset(
        repo_id=root.name,
        root=root,
        delta_timestamps={"expert_action": [step / fps for step in range(horizon)]},
        video_backend="pyav",
    )
    dataset._ensure_hf_dataset_loaded()  # noqa: SLF001
    frame_table = dataset.hf_dataset
    success = np.asarray(frame_table["is_success"], dtype=bool).reshape(-1)
    intervene = np.asarray(frame_table["intervene_flag"], dtype=bool).reshape(-1)
    # ``is_success`` is an episode-level outcome and the recorder marks it on
    # the final frame only.  Filter successful episodes first, then keep their
    # intervention frames; intersecting the two columns would usually select
    # nothing because the final frame need not be intervened.
    if "episode_index" in frame_table.column_names:
        episode_index = np.asarray(
            frame_table["episode_index"], dtype=np.int64
        ).reshape(-1)
        successful_episodes = np.unique(episode_index[success])
        selected = np.flatnonzero(
            intervene & np.isin(episode_index, successful_episodes)
        )
    else:
        selected = np.flatnonzero(success & intervene)
    if selected.size == 0:
        raise ValueError("DAgger dataset contains no successful intervention frames")
    selected_dataset = _IndexedDataset(dataset, selected)
    repack = transforms.RepackTransform(
        {
            "observation/image": "observation.images.wrist",
            "observation/state": "observation.state",
            "actions": "expert_action",
            "prompt": "task",
        }
    )
    transformed = _TransformedDataset(
        selected_dataset,
        transforms.compose(
            [
                repack,
                _NormalizeRawSO101Joints(),
                *data_config.data_transforms.inputs,
                transforms.Normalize(
                    norm_stats, use_quantiles=data_config.use_quantile_norm
                ),
                _ClipNormalizedSO101(),
                _CastModelInputs(),
                *data_config.model_transforms.inputs,
            ]
        ),
    )
    sampler = DistributedSampler(
        transformed,
        num_replicas=world_size,
        rank=rank,
        shuffle=not eval_dataset,
        seed=int(cfg.actor.seed),
        drop_last=True,
    )
    loader = torch.utils.data.DataLoader(
        transformed,
        batch_size=int(cfg.actor.get("eval_batch_size", 1))
        if eval_dataset
        else int(cfg.actor.micro_batch_size),
        sampler=sampler,
        shuffle=False,
        num_workers=int(cfg.data.get("num_workers", 0)),
        collate_fn=_collate,
        drop_last=True,
    )
    resolved = SO101SftDataConfig(
        repo_id=str(root),
        asset_id=str(openpi_cfg.asset_id),
        norm_stats=norm_stats,
        action_horizon=horizon,
        action_dim=int(model_cfg.action_dim),
    )
    return SO101DaggerDataLoader(loader, resolved, sampler), resolved
