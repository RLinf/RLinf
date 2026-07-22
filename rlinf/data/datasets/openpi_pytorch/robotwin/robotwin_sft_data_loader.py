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

"""RoboTwin SFT loader for the JAX-aligned PyTorch OpenPI implementation."""

from __future__ import annotations

import dataclasses
import multiprocessing
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from openpi.transforms import DataTransformFn, compose
from torch.utils.data.distributed import DistributedSampler

from rlinf.data.datasets.openpi_pytorch.robotwin.robotwin_sft_dataset import (
    RobotwinSftDataset,
)
from rlinf.data.lerobot_paths import (
    default_hf_lerobot_home,
    resolve_lerobot_repo_id,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_pytorch.transforms_pipeline import (
    build_openpi_transforms,
)
from rlinf.utils.logging import get_logger

logger = get_logger()

_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


class _Repack(DataTransformFn):
    """Map RoboTwin LeRobot features to the canonical ALOHA input layout."""

    _IMAGE_SOURCES = {
        "cam_high": "observation.images.cam_high",
        "cam_left_wrist": "observation.images.cam_left_wrist",
        "cam_right_wrist": "observation.images.cam_right_wrist",
    }

    def __call__(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        missing = [
            source
            for source in (*self._IMAGE_SOURCES.values(), "observation.state", "action")
            if source not in frame
        ]
        if missing:
            raise KeyError(
                "RoboTwin SFT frame is missing required feature(s): "
                f"{missing}; available keys={sorted(frame)}"
            )

        prompt = frame.get("prompt", frame.get("task"))
        if prompt is None:
            raise ValueError(
                "RoboTwin SFT frame is missing both 'prompt' and 'task'; "
                "the LeRobot dataset must contain task annotations."
            )
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        elif not isinstance(prompt, str):
            prompt = prompt.item() if hasattr(prompt, "item") else str(prompt)

        return {
            "images": {
                name: np.asarray(frame[source])
                for name, source in self._IMAGE_SOURCES.items()
            },
            "state": np.asarray(frame["observation.state"]),
            "actions": np.asarray(frame["action"]),
            "prompt": prompt,
        }


class _TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: RobotwinSftDataset, transform) -> None:
        self._dataset = dataset
        self._transform = transform

    def __getitem__(self, index: int):
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


def collate_robotwin_sft_items(
    items: Sequence[Mapping[str, Any]],
) -> tuple[Observation, torch.Tensor]:
    """Collate transformed RoboTwin items into the PyTorch Pi0 boundary type."""
    if not items:
        raise ValueError("Cannot collate an empty RoboTwin SFT batch.")

    observation = Observation.from_dict(
        {
            "image": {
                key: torch.from_numpy(
                    np.stack([np.asarray(item["image"][key]) for item in items])
                )
                for key in _IMAGE_KEYS
            },
            "image_mask": {
                key: torch.from_numpy(
                    np.stack(
                        [
                            np.asarray(item["image_mask"][key], dtype=np.bool_)
                            for item in items
                        ]
                    )
                )
                for key in _IMAGE_KEYS
            },
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
class RobotwinSftDataConfig:
    """Resolved metadata exposed by :class:`RobotwinSftDataLoader`."""

    repo_id: str
    action_dim: int
    action_horizon: int
    max_token_len: int


class RobotwinSftDataLoader:
    """Infinite distributed ``(Observation, actions)`` RoboTwin loader."""

    def __init__(
        self,
        torch_loader: torch.utils.data.DataLoader,
        data_config: RobotwinSftDataConfig,
        sampler: DistributedSampler,
    ) -> None:
        self._torch_loader = torch_loader
        self._data_config = data_config
        self._sampler = sampler
        self._epoch = 0

    def data_config(self) -> RobotwinSftDataConfig:
        """Return the resolved data-pipeline metadata."""
        return self._data_config

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        """Return the underlying PyTorch data loader."""
        return self._torch_loader

    def __iter__(self):
        while True:
            self._sampler.set_epoch(self._epoch)
            self._epoch += 1
            yield from self._torch_loader

    def __len__(self) -> int:
        return len(self._torch_loader)


def build_robotwin_sft_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
) -> tuple[RobotwinSftDataLoader, RobotwinSftDataConfig]:
    """Build the RoboTwin ``openpi_pytorch`` SFT data loader."""
    from omegaconf import OmegaConf

    data_path = resolve_lerobot_repo_id(data_paths)
    if data_path is None:
        raise ValueError("openpi_pytorch RoboTwin SFT requires data.train_data_paths.")

    data_cfg = cfg.data
    repo_id = OmegaConf.select(data_cfg, "repo_id", default=None)
    if repo_id:
        candidate = default_hf_lerobot_home() / str(repo_id)
        nested = Path(data_path) / str(repo_id)
        if (candidate / "meta" / "info.json").is_file():
            data_path = str(candidate)
        elif (nested / "meta" / "info.json").is_file():
            data_path = str(nested)

    model_cfg = cfg.actor.model
    openpi_cfg = model_cfg.openpi
    data_kwargs = OmegaConf.select(model_cfg, "openpi_data", default=None)
    if data_kwargs is not None:
        data_kwargs = OmegaConf.to_container(data_kwargs, resolve=True)

    fps = OmegaConf.select(data_cfg, "fps", default=None)
    fps = int(fps) if fps is not None else None
    tolerance_s = float(OmegaConf.select(data_cfg, "tolerance_s", default=1e-4))
    batch_size = (
        int(cfg.actor.eval_batch_size)
        if eval_dataset
        else int(cfg.actor.micro_batch_size)
    )

    dataset = RobotwinSftDataset(
        data_path=str(data_path),
        action_horizon=int(model_cfg.num_action_chunks),
        fps=fps,
        tolerance_s=tolerance_s,
    )
    input_transforms, _ = build_openpi_transforms(
        str(model_cfg.model_path),
        str(openpi_cfg.config_name),
        data_kwargs=data_kwargs,
        norm_stats_dir=str(openpi_cfg.assets_dir),
        norm_stats_asset_id=str(openpi_cfg.asset_id),
        input_prefix=[_Repack()],
    )
    source = _TransformedDataset(dataset, compose(input_transforms))

    sampler = DistributedSampler(
        source,
        num_replicas=world_size,
        rank=rank,
        shuffle=not eval_dataset,
        seed=int(cfg.actor.seed),
        drop_last=True,
    )
    num_workers = int(OmegaConf.select(data_cfg, "num_workers", default=0))
    mp_context = multiprocessing.get_context("spawn") if num_workers > 0 else None
    generator = torch.Generator()
    generator.manual_seed(int(cfg.actor.seed))

    torch_loader = torch.utils.data.DataLoader(
        source,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0,
        collate_fn=collate_robotwin_sft_items,
        drop_last=True,
        generator=generator,
        pin_memory=True,
    )
    data_config = RobotwinSftDataConfig(
        repo_id=str(dataset.dataset_root),
        action_dim=int(openpi_cfg.model_action_dim),
        action_horizon=int(model_cfg.num_action_chunks),
        max_token_len=int(openpi_cfg.max_token_len),
    )
    logger.info(
        "RoboTwin openpi_pytorch SFT loader: batch_size=%d workers=%d horizon=%d "
        "config=%s",
        batch_size,
        num_workers,
        data_config.action_horizon,
        openpi_cfg.config_name,
    )
    return RobotwinSftDataLoader(torch_loader, data_config, sampler), data_config
