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

"""LeRobot v3 SO-101 SFT loader for the ``openpi`` model path."""

from __future__ import annotations

import dataclasses
import json
import multiprocessing
import os
import pathlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler

from rlinf.data.datasets.openpi.transform_fn import DataTransformFn, compose
from rlinf.data.storage.lerobot import (
    resolve_lerobot_dataset_root,
    resolve_lerobot_repo_id,
)
from rlinf.models.embodiment.openpi.modules.model import Observation
from rlinf.models.embodiment.openpi.transforms.pipeline import (
    build_openpi_transforms,
)

_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
_RAW_ACTION_DIM = 6
_RAW_JOINT_SCALE = 0.01


@dataclasses.dataclass(frozen=True)
class _RepackSO101(DataTransformFn):
    """Map collected LeRobot fields into the OpenPI SO-101 contract.

    RLinf's real-world writer stores the single camera as ``image``, the
    proprioceptive vector as ``state``, and action chunks as ``actions``.
    OpenPI's SO-101 canonical contract represents all six LeRobot values at
    the 0.01 scale: five arm joints in degree-like units and the gripper in
    its normalized 0..1 range.
    """

    joint_scale: float = _RAW_JOINT_SCALE

    def __call__(self, frame: dict[str, Any]) -> dict[str, Any]:
        aliases = {
            "image": ("image", "observation.images.wrist"),
            "state": ("state", "observation.state"),
            "actions": ("actions", "action"),
        }
        resolved = {
            name: next((key for key in keys if key in frame), None)
            for name, keys in aliases.items()
        }
        missing = [name for name, key in resolved.items() if key is None]
        if missing:
            raise KeyError(
                f"SO-101 SFT frame is missing {missing}; available keys={sorted(frame)}"
            )

        prompt = frame.get("prompt", frame.get("task"))
        if prompt is None:
            raise ValueError("SO-101 SFT frame is missing both 'prompt' and 'task'.")
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        elif not isinstance(prompt, str):
            prompt = prompt.item() if hasattr(prompt, "item") else str(prompt)

        return {
            "observation/image": np.asarray(frame[resolved["image"]]),
            "observation/state": self._scale_joints(frame[resolved["state"]]),
            "actions": self._scale_joints(frame[resolved["actions"]]),
            "prompt": prompt,
        }

    def _scale_joints(self, value: Any) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32).copy()
        if array.shape[-1] != _RAW_ACTION_DIM:
            raise ValueError(
                f"Expected SO-101 vector last dimension {_RAW_ACTION_DIM}, got {array.shape}."
            )
        array *= self.joint_scale
        return array


class _TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Any, transform: Callable[[Any], Any]) -> None:
        self._dataset = dataset
        self._transform = transform

    def __getitem__(self, index: int) -> Any:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


def collate_so101_sft_items(
    items: Sequence[Mapping[str, Any]],
) -> tuple[Observation, torch.Tensor]:
    """Collate transformed samples into the SFT worker's batch contract."""
    if not items:
        raise ValueError("Cannot collate an empty SO-101 SFT batch.")

    images = {
        key: torch.from_numpy(
            np.stack([np.asarray(item["image"][key]) for item in items])
        )
        for key in _IMAGE_KEYS
    }
    image_masks = {
        key: torch.from_numpy(
            np.stack(
                [np.asarray(item["image_mask"][key], dtype=np.bool_) for item in items]
            )
        )
        for key in _IMAGE_KEYS
    }
    observation = Observation.from_dict(
        {
            "image": images,
            "image_mask": image_masks,
            "state": torch.from_numpy(
                np.stack(
                    [np.asarray(item["state"], dtype=np.float32) for item in items]
                )
            ),
            "tokenized_prompt": torch.from_numpy(
                np.stack(
                    [
                        np.asarray(item["tokenized_prompt"], dtype=np.int64)
                        for item in items
                    ]
                )
            ).long(),
            "tokenized_prompt_mask": torch.from_numpy(
                np.stack(
                    [
                        np.asarray(item["tokenized_prompt_mask"], dtype=np.bool_)
                        for item in items
                    ]
                )
            ),
        }
    )
    actions = torch.from_numpy(
        np.stack([np.asarray(item["actions"], dtype=np.float32) for item in items])
    )
    return observation, actions


@dataclasses.dataclass(frozen=True)
class SO101SftDataConfig:
    """Resolved metadata exposed to the SFT worker."""

    repo_id: str
    raw_action_dim: int
    action_dim: int
    action_horizon: int
    max_token_len: int


class SO101SftDataLoader:
    """Infinite SO-101 batches with epoch-aware distributed sampling."""

    def __init__(
        self,
        torch_loader: torch.utils.data.DataLoader,
        data_config: SO101SftDataConfig,
        sampler: DistributedSampler | None,
    ) -> None:
        self._torch_loader = torch_loader
        self._data_config = data_config
        self.sampler = sampler
        self._epoch = 0

    def data_config(self) -> SO101SftDataConfig:
        return self._data_config

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._torch_loader

    def __iter__(self) -> Iterator[tuple[Observation, torch.Tensor]]:
        while True:
            if self.sampler is not None:
                self.sampler.set_epoch(self._epoch)
                self._epoch += 1
            yield from self._torch_loader

    def __len__(self) -> int:
        return len(self._torch_loader)


def _worker_init_fn(worker_id: int) -> None:
    del worker_id
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def _read_dataset_metadata(root: pathlib.Path) -> int:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"SO-101 dataset not found: {root}")
    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    feature_aliases = {
        "state": ("state", "observation.state"),
        "actions": ("actions", "action"),
    }
    for name, aliases in feature_aliases.items():
        key = next((candidate for candidate in aliases if candidate in features), None)
        shape = tuple(features.get(key, {}).get("shape", ())) if key else ()
        if shape != (_RAW_ACTION_DIM,):
            raise ValueError(
                f"SO-101 {name} must have shape [{_RAW_ACTION_DIM}], got {shape}."
            )
    if not any(key in features for key in ("image", "observation.images.wrist")):
        raise ValueError("SO-101 dataset is missing its camera image feature.")
    return int(info["fps"])


def create_so101_sft_data_loader(
    *,
    data_path: str,
    model_path: str,
    config_name: str,
    assets_dir: str,
    asset_id: str,
    raw_action_dim: int,
    action_dim: int,
    action_horizon: int,
    max_token_len: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    dist_rank: int,
    dist_world_size: int,
    data_kwargs: dict[str, Any] | None = None,
) -> SO101SftDataLoader:
    """Build the local LeRobot SO-101 loader and current model transforms."""
    if raw_action_dim != _RAW_ACTION_DIM:
        raise ValueError(
            f"SO-101 raw action dim must be {_RAW_ACTION_DIM}, got {raw_action_dim}."
        )

    root = resolve_lerobot_dataset_root(data_path)
    fps = _read_dataset_metadata(root)
    resolved_data_kwargs = dict(data_kwargs or {})
    configured_stats = resolved_data_kwargs.get("norm_stats_path")
    if configured_stats is None:
        norm_stats_path = pathlib.Path(assets_dir) / asset_id / "norm_stats.json"
        resolved_data_kwargs["norm_stats_path"] = str(norm_stats_path)
    else:
        norm_stats_path = pathlib.Path(str(configured_stats)).expanduser()
        if norm_stats_path.is_dir():
            norm_stats_path /= "norm_stats.json"
    if not norm_stats_path.is_file():
        raise FileNotFoundError(f"SO-101 norm stats not found: {norm_stats_path}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    train_config = get_openpi_config(
        config_name,
        model_path=model_path,
        data_kwargs=resolved_data_kwargs,
    )
    if action_horizon != int(train_config.model.action_horizon):
        raise ValueError(
            "SO-101 action horizon does not match the OpenPI config: "
            f"{action_horizon} != {train_config.model.action_horizon}."
        )
    if action_dim != int(train_config.model.action_dim):
        raise ValueError(
            "SO-101 padded action dim does not match the OpenPI config: "
            f"{action_dim} != {train_config.model.action_dim}."
        )

    dataset = LeRobotDataset(
        repo_id=root.name,
        root=root,
        delta_timestamps={"actions": [step / fps for step in range(action_horizon)]},
        video_backend="pyav",
    )
    input_transforms, _ = build_openpi_transforms(
        model_path,
        config_name,
        data_kwargs=resolved_data_kwargs,
    )
    source = _TransformedDataset(
        dataset,
        compose([_RepackSO101(), *input_transforms]),
    )
    sampler = DistributedSampler(
        source,
        num_replicas=dist_world_size,
        rank=dist_rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=True,
    )
    mp_context = multiprocessing.get_context("spawn") if num_workers > 0 else None
    generator = torch.Generator()
    generator.manual_seed(seed)
    torch_loader = torch.utils.data.DataLoader(
        source,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0,
        collate_fn=collate_so101_sft_items,
        worker_init_fn=_worker_init_fn,
        drop_last=True,
        generator=generator,
    )
    data_config = SO101SftDataConfig(
        repo_id=str(root),
        raw_action_dim=raw_action_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        max_token_len=max_token_len,
    )
    return SO101SftDataLoader(torch_loader, data_config, sampler)


def build_so101_sft_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
) -> tuple[SO101SftDataLoader, SO101SftDataConfig]:
    """Build the SO-101 dataloader selected by the SFT worker registry."""
    from omegaconf import OmegaConf

    data_path = resolve_lerobot_repo_id(data_paths)
    if data_path is None:
        raise ValueError("openpi SO-101 SFT requires data.train_data_paths.")

    model_cfg = cfg.actor.model
    openpi_cfg = model_cfg.openpi
    data_kwargs = OmegaConf.select(model_cfg, "openpi_data", default=None)
    if data_kwargs is not None:
        data_kwargs = OmegaConf.to_container(data_kwargs, resolve=True)

    loader = create_so101_sft_data_loader(
        data_path=str(data_path),
        model_path=str(model_cfg.model_path),
        config_name=str(openpi_cfg.config_name),
        assets_dir=str(openpi_cfg.assets_dir),
        asset_id=str(openpi_cfg.asset_id),
        raw_action_dim=int(model_cfg.action_dim),
        action_dim=int(openpi_cfg.model_action_dim),
        action_horizon=int(model_cfg.num_action_chunks),
        max_token_len=int(openpi_cfg.max_token_len),
        batch_size=(
            int(cfg.actor.get("eval_batch_size", cfg.actor.micro_batch_size))
            if eval_dataset
            else int(cfg.actor.micro_batch_size)
        ),
        num_workers=int(cfg.data.get("num_workers", 0)),
        shuffle=not eval_dataset,
        seed=int(cfg.actor.seed),
        dist_rank=rank,
        dist_world_size=world_size,
        data_kwargs=data_kwargs,
    )
    return loader, loader.data_config()
