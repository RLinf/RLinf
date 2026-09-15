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

"""PyTorch data loader for local LeRobot v3 SO-101 demonstrations."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from collections.abc import Iterator
from typing import Any

import jax
import numpy as np
import torch
from openpi import transforms
from openpi.models import model as openpi_model
from torch.utils.data.distributed import DistributedSampler


class _TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Any, transform: Any) -> None:
        self._dataset = dataset
        self._transform = transform

    def __getitem__(self, index: int) -> Any:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class _CastModelInputs(transforms.DataTransformFn):
    """Restore model-facing continuous tensors to float32 after normalization."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        data["state"] = np.asarray(data["state"], dtype=np.float32)
        if "actions" in data:
            data["actions"] = np.asarray(data["actions"], dtype=np.float32)
        return data


class _NormalizeRawSO101Joints(transforms.DataTransformFn):
    """Map LeRobot SO-101 degree-like positions to the [-1, 1] contract."""

    scale: float = 0.01

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        data["observation/state"] = (
            np.asarray(data["observation/state"], dtype=np.float32) * self.scale
        )
        if "actions" in data:
            data["actions"] = np.asarray(data["actions"], dtype=np.float32) * self.scale
        return data


class _ClipNormalizedSO101(transforms.DataTransformFn):
    """Enforce the SO-101 normalized joint contract after quantile scaling."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        data["state"] = np.clip(np.asarray(data["state"], dtype=np.float32), -1.0, 1.0)
        if "actions" in data:
            data["actions"] = np.clip(
                np.asarray(data["actions"], dtype=np.float32), -1.0, 1.0
            )
        return data


def _collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    return jax.tree.map(
        lambda *values: torch.as_tensor(
            np.stack([np.asarray(value) for value in values], axis=0)
        ),
        *items,
    )


@dataclasses.dataclass(frozen=True)
class SO101SftDataConfig:
    """Metadata exposed to the SFT worker and checkpoint logic."""

    repo_id: str
    asset_id: str
    norm_stats: dict[str, Any]
    action_horizon: int
    action_dim: int


class SO101SftDataLoader:
    """Infinite batches of OpenPI observations and padded SO-101 actions."""

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


def build_so101_sft_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    data_path: str,
    eval_dataset: bool = False,
) -> tuple[SO101SftDataLoader, SO101SftDataConfig]:
    """Build an SO-101 loader from a local LeRobot v3 dataset root."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from openpi.shared import normalize

    from rlinf.data.storage.lerobot.paths import resolve_lerobot_dataset_root
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    root = resolve_lerobot_dataset_root(data_path)
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"SO-101 dataset not found: {root}")

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
    fps = int(
        __import__("json").loads((root / "meta" / "info.json").read_text())["fps"]
    )
    raw_features = json.loads((root / "meta" / "info.json").read_text())["features"]
    formal_so101 = "observation.state" in raw_features and "action" in raw_features
    if not formal_so101:
        # The RLinf writer and the historical SO101 recipe use the flat
        # ``state``/``actions`` schema.  Keep that path on OpenPI's official
        # loader; this adapter is only for the cloud LeRobot v3 schema.
        from rlinf.data.datasets.openpi_rlinf import (
            build_official_openpi_sft_dataloader,
        )

        return build_official_openpi_sft_dataloader(
            cfg, world_size, rank, data_path, eval_dataset
        )

    dataset = LeRobotDataset(
        repo_id=root.name,
        root=root,
        delta_timestamps={"action": [step / fps for step in range(horizon)]},
        video_backend="pyav",
    )
    repack = transforms.RepackTransform(
        {
            "observation/image": "observation.images.wrist",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "task",
        }
    )
    joint_scale = _NormalizeRawSO101Joints()
    transformed = _TransformedDataset(
        dataset,
        transforms.compose(
            [
                repack,
                joint_scale,
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
        batch_size=(
            int(cfg.actor.get("eval_batch_size", 1))
            if eval_dataset
            else int(cfg.actor.micro_batch_size)
        ),
        sampler=sampler,
        shuffle=False,
        num_workers=int(cfg.data.get("num_workers", 0)),
        collate_fn=_collate,
        drop_last=True,
    )
    resolved_config = SO101SftDataConfig(
        repo_id=str(root),
        asset_id=str(openpi_cfg.asset_id),
        norm_stats=norm_stats,
        action_horizon=horizon,
        action_dim=int(model_cfg.action_dim),
    )
    wrapped = SO101SftDataLoader(loader, resolved_config, sampler)
    return wrapped, resolved_config
