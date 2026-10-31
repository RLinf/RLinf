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

"""Episode-local visual history for the OpenPI LeRobot SFT pipeline."""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.data.datasets.openpi.official_sft_data_loader import (
    build_official_openpi_sft_config,
)
from rlinf.data.storage.lerobot.compat import index_episode_subset_by_id
from rlinf.models.embodiment.openpi.modules.model import Observation
from rlinf.models.embodiment.openpi.transforms.pipeline import (
    split_short_memory_transforms,
)


class ShortMemoryDataset(torch.utils.data.Dataset):
    """Apply the usual frame transforms, retaining only historical camera inputs.

    LeRobot owns episode clamping and padding masks. State, prompt and action
    targets always belong to the current frame, including for history transforms.
    """

    def __init__(
        self,
        dataset: Any,
        frame_transform: Callable[[dict[str, Any]], dict[str, Any]],
        sample_transform: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        num_frames: int,
    ) -> None:
        self.dataset = dataset
        self.timestamps = dataset.hf_dataset.select_columns("timestamp").with_format(
            None
        )
        self.frame_transform = frame_transform
        self.sample_transform = sample_transform
        self.num_frames = num_frames

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw = self.dataset[index]
        valid = ~np.asarray(raw["index_is_pad"], dtype=np.bool_)
        frames = []
        for position in range(self.num_frames):
            if not valid[position]:
                frames.append(None)
                continue
            frame = dict(raw)
            frame["index"] = raw["index"][position]
            for key in self.dataset.meta.camera_keys:
                # LeRobot squeezes singleton histories for video-backed cameras.
                frame[key] = raw[key][position] if raw[key].ndim == 4 else raw[key]
            frames.append(self.frame_transform(frame))

        current = self.sample_transform(frames[-1])
        current["image"] = {
            key: np.stack(
                [
                    np.zeros_like(image)
                    if frame is None
                    else np.asarray(frame["image"][key])
                    for frame in frames
                ]
            )
            for key, image in current["image"].items()
        }
        current["image_mask"] = {
            key: np.asarray(
                [
                    False if frame is None else frame["image_mask"][key]
                    for frame in frames
                ],
                dtype=np.bool_,
            )
            for key in current["image_mask"]
        }
        # Querying timestamp through delta_timestamps would replace its scalar
        # value, which LeRobot's video decoder needs for the current frame.
        timestamp_indices = raw["index"] - raw["index"][-1] + index
        timestamps = np.asarray(
            self.timestamps[timestamp_indices.tolist()]["timestamp"],
            dtype=np.float32,
        ).reshape(self.num_frames)
        current["history_time_offsets"] = np.where(
            valid, timestamps - timestamps[-1], 0.0
        ).astype(np.float32)
        return current


def collate_short_memory_items(
    items: list[dict[str, Any]],
    *,
    include_history: bool = True,
) -> tuple[Observation, torch.Tensor]:
    """Stack samples, exposing current frames when the model has no memory."""
    from torch.utils.data import default_collate

    batch = default_collate(items)
    if not include_history:
        batch["image"] = {key: image[:, -1] for key, image in batch["image"].items()}
        batch["image_mask"] = {
            key: mask[:, -1] for key, mask in batch["image_mask"].items()
        }
        batch.pop("history_time_offsets")
    return Observation.from_dict(batch), batch["actions"]


def build_short_memory_sft_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
) -> tuple[StatefulDataLoader, Any]:
    """Build a resumable LeRobot loader with configurable past-frame sampling."""
    import openpi.transforms as transforms

    try:
        from lerobot.datasets.lerobot_dataset import (
            LeRobotDataset,
            LeRobotDatasetMetadata,
        )
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import (
            LeRobotDataset,
            LeRobotDatasetMetadata,
        )

    config = build_official_openpi_sft_config(cfg, world_size, data_paths, eval_dataset)
    data_config = config.data.create(config.assets_dirs, config.model)
    memory_config = cfg.data.short_memory
    include_history = bool(cfg.actor.model.openpi.get("short_memory", False))
    num_frames = int(memory_config.num_frames) if include_history else 1
    stride = int(memory_config.stride)
    if num_frames < 1 or stride < 1:
        raise ValueError(
            "Short-memory num_frames and stride must be positive; "
            f"got num_frames={num_frames}, stride={stride}."
        )
    if data_config.rlds_data_dir is not None:
        raise ValueError("Short-memory SFT requires a map-style LeRobot dataset.")
    if data_config.norm_stats is None:
        raise ValueError("Short-memory SFT requires OpenPI normalization statistics.")

    metadata = LeRobotDatasetMetadata(data_config.repo_id)
    history_offsets = [
        position * stride / metadata.fps for position in range(1 - num_frames, 1)
    ]
    delta_timestamps = {
        key: [
            position / metadata.fps for position in range(config.model.action_horizon)
        ]
        for key in data_config.action_sequence_keys
    }
    delta_timestamps.update(
        dict.fromkeys((*metadata.camera_keys, "index"), history_offsets)
    )
    dataset = LeRobotDataset(
        data_config.repo_id,
        delta_timestamps=delta_timestamps,
        episodes=cfg.data.get("episodes"),
    )
    index_episode_subset_by_id(dataset)
    input_transforms = [
        *(
            [transforms.PromptFromLeRobotTask(metadata.tasks)]
            if data_config.prompt_from_task
            else []
        ),
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        transforms.Normalize(
            data_config.norm_stats, use_quantiles=data_config.use_quantile_norm
        ),
        *data_config.model_transforms.inputs,
    ]
    frame_transform, sample_transform = split_short_memory_transforms(input_transforms)
    dataset = ShortMemoryDataset(
        dataset,
        frame_transform,
        sample_transform,
        num_frames=num_frames,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=not eval_dataset,
        seed=config.seed,
        drop_last=True,
    )
    loader = StatefulDataLoader(
        dataset,
        batch_size=config.batch_size // world_size,
        sampler=sampler,
        num_workers=config.num_workers,
        multiprocessing_context=(
            multiprocessing.get_context("spawn") if config.num_workers else None
        ),
        persistent_workers=config.num_workers > 0,
        collate_fn=partial(collate_short_memory_items, include_history=include_history),
        drop_last=True,
        generator=torch.Generator().manual_seed(config.seed + rank),
    )
    if len(loader) == 0:
        raise ValueError(
            f"Short-memory dataset has {len(dataset)} frames, insufficient for "
            f"batch_size={config.batch_size} across {world_size} ranks."
        )
    return loader, data_config
