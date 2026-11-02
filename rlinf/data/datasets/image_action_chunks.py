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

"""Image/state/action-chunk samples for CNN policy SFT.

``toolkits/lerobot/export_image_action_chunks.py`` writes one directory per
LeRobot dataset with memory-mappable arrays: ``main_images`` [N, H, W, 3] and
optional ``extra_view_images`` [N, E, H, W, 3] as uint8, ``states`` [N, S] and
``actions`` [N, C, A] as float32. Items use the env observation keys so the
policy's ``preprocess_env_obs`` applies unchanged.
"""

import json
import os
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, Dataset, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader


class ImageActionChunkDataset(Dataset):
    """Memory-mapped image/state/action chunks from one exported directory.

    Items carry ``main_images``, ``states``, ``action`` and, for multi-camera
    exports, ``extra_view_images``.
    """

    def __init__(self, root: str) -> None:
        self.root = root
        with open(os.path.join(root, "meta.json")) as f:
            self.meta = json.load(f)
        self._open_arrays()

    def _open_arrays(self) -> None:
        self.main_images = np.load(
            os.path.join(self.root, "main_images.npy"), mmap_mode="r"
        )
        self.states = np.load(os.path.join(self.root, "states.npy"), mmap_mode="r")
        self.actions = np.load(os.path.join(self.root, "actions.npy"), mmap_mode="r")
        extra_path = os.path.join(self.root, "extra_view_images.npy")
        self.extra_view_images = (
            np.load(extra_path, mmap_mode="r") if os.path.exists(extra_path) else None
        )
        assert len(self.main_images) == len(self.states) == len(self.actions), (
            f"{self.root}: expected equal sample counts, got images {len(self.main_images)}, "
            f"states {len(self.states)}, actions {len(self.actions)}"
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Workers reopen the files; pickling memmaps would copy their contents.
        for key in ("main_images", "states", "actions", "extra_view_images"):
            del state[key]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._open_arrays()

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            "main_images": torch.from_numpy(np.array(self.main_images[idx])),
            "states": torch.from_numpy(np.array(self.states[idx])),
            "action": torch.from_numpy(np.array(self.actions[idx])),
        }
        if self.extra_view_images is not None:
            item["extra_view_images"] = torch.from_numpy(
                np.array(self.extra_view_images[idx])
            )
        return item


def build_cnn_sft_dataloader(
    cfg: DictConfig,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
) -> tuple[StatefulDataLoader, dict[str, Any]]:
    """Build the distributed CNN SFT dataloader from exported chunk directories.

    Args:
        cfg: Full RLinf config; reads ``actor.seed``, ``actor.micro_batch_size``
            and ``data.num_workers``.
        world_size: Number of distributed sampler replicas.
        rank: Index of this sampler replica.
        data_paths: One exported directory or a list of them.
        eval_dataset: Build the non-shuffled evaluation loader.

    Returns:
        ``(StatefulDataLoader, info)`` with the sample count and export metadata.
    """
    if isinstance(data_paths, str):
        data_paths = [data_paths]
    datasets = [ImageActionChunkDataset(str(path)) for path in data_paths]
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=not eval_dataset,
        seed=cfg.actor.seed,
        drop_last=True,
    )
    num_workers = cfg.data.get("num_workers", 4)
    data_loader = StatefulDataLoader(
        dataset,
        batch_size=cfg.actor.micro_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    info = {"num_samples": len(dataset), "meta": [d.meta for d in datasets]}
    return data_loader, info
