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

"""Value Dataset: loads LeRobot data + returns sidecar for value model SFT."""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from torch.utils.data import Dataset

from rlinf.data.datasets.cfg.lerobot.config import create_data_config_factory
from rlinf.data.datasets.cfg.lerobot.transforms import (
    Normalize,
    RepackTransform,
    compose,
    load_task_descriptions,
)

from .return_loaders import load_returns_sidecar
from .value_transforms import ReturnNormalizer

logger = logging.getLogger(__name__)


class ValueDataset(Dataset):
    """Flat dataset for value model SFT.

    Returns ``{images, prompt, target_values, actions=None}`` per sample.
    """

    def __init__(
        self,
        dataset_path: str,
        robot_type: str,
        model_type: str,
        action_horizon: int = 10,
        action_dim: Optional[int] = None,
        norm_stats_dir: Optional[str] = None,
        asset_id: Optional[str] = None,
        default_prompt: Optional[str] = None,
        return_min: Optional[float] = None,
        return_max: Optional[float] = None,
        normalize_to_minus_one_zero: bool = True,
        extra_delta_transform: bool = False,
        action_norm_skip_dims: Optional[dict[str, list[int]]] = None,
        max_samples: Optional[int] = None,
        tag: Optional[str] = None,
        episode_percentage: Optional[float] = None,
        shuffle_episodes: bool = False,
        episode_seed: int = 42,
        **kwargs,  # accept unused params (gamma, split, etc.)
    ):
        _known_unused = {"gamma", "split", "repo_id"}
        unexpected = set(kwargs) - _known_unused
        if unexpected:
            logger.warning(f"ValueDataset ignoring unexpected kwargs: {unexpected}")

        self.max_samples = max_samples
        local_path = Path(dataset_path).absolute()

        # Metadata + dataset
        self.dataset_meta = LeRobotDatasetMetadata(local_path.name, root=local_path)
        if "action" in self.dataset_meta.features:
            action_key = "action"
        elif "actions" in self.dataset_meta.features:
            action_key = "actions"
        else:
            raise ValueError(
                f"No action key in dataset features: "
                f"{list(self.dataset_meta.features.keys())}"
            )
        delta_timestamps = {
            action_key: [t / self.dataset_meta.fps for t in range(action_horizon)]
        }
        self._base = LeRobotDataset(
            local_path.name,
            root=local_path,
            delta_timestamps=delta_timestamps,
            download_videos=False,
        )

        # Returns sidecar (required for value SFT)
        self._sidecar = load_returns_sidecar(local_path, tag)
        if self._sidecar is None:
            raise FileNotFoundError(
                f"Returns sidecar not found for {dataset_path}. "
                f"Run compute_returns.py first to generate "
                f"meta/returns{'_' + tag if tag else ''}.parquet"
            )

        # Episode filtering
        self._indices = None
        if episode_percentage is not None and episode_percentage < 100:
            if episode_percentage <= 0:
                raise ValueError(
                    f"episode_percentage must be > 0, got {episode_percentage}"
                )
            total = self.dataset_meta.total_episodes
            num = max(1, int(total * episode_percentage / 100.0))
            all_eps = list(range(total))
            if shuffle_episodes:
                rng = np.random.default_rng(episode_seed)
                selected = set(rng.choice(all_eps, size=num, replace=False).tolist())
            else:
                selected = set(all_eps[:num])
            idx = self._base.episode_data_index
            self._indices = [
                i
                for ep in sorted(selected)
                for i in range(idx["from"][ep].item(), idx["to"][ep].item())
            ]

        # Transform pipeline (repack → data → normalize → model)
        factory = create_data_config_factory(
            dataset_path=str(local_path),
            robot_type=robot_type,
            model_type=model_type,
            default_prompt=default_prompt,
            extra_delta_transform=extra_delta_transform,
            norm_stats_dir=norm_stats_dir,
            asset_id=asset_id,
            action_norm_skip_dims=action_norm_skip_dims,
        )
        dc = factory.create(
            action_dim=action_dim or 32,
            skip_norm_stats=(norm_stats_dir is None),
        )
        self._transform = self._make_transform(dc)

        # Task descriptions for prompt injection
        self._tasks = load_task_descriptions(local_path) or (
            self.dataset_meta.tasks if hasattr(self.dataset_meta, "tasks") else None
        )

        # Return normalizer
        self._normalizer = (
            ReturnNormalizer(
                return_min=return_min,
                return_max=return_max,
                normalize_to_minus_one_zero=normalize_to_minus_one_zero,
            )
            if return_min is not None and return_max is not None
            else None
        )

        n = len(self._indices) if self._indices else len(self._base)
        logger.info(f"ValueDataset: {dataset_path}, {min(n, max_samples or n)} samples")

    @staticmethod
    def _make_transform(dc):
        transforms = []
        for t in dc.repack_transforms.inputs:
            if isinstance(t, RepackTransform):
                transforms.append(
                    RepackTransform(t.structure, passthrough_unmapped=True)
                )
            else:
                transforms.append(t)
        transforms.extend(dc.data_transforms.inputs)
        if dc.norm_stats is not None:
            transforms.append(
                Normalize(
                    dc.norm_stats,
                    dc.use_quantile_norm,
                    skip_dims=dc.action_norm_skip_dims,
                )
            )
        transforms.extend(dc.model_transforms.inputs)
        return compose(transforms) if transforms else None

    def __len__(self) -> int:
        n = len(self._indices) if self._indices else len(self._base)
        return min(n, self.max_samples) if self.max_samples else n

    def __getitem__(self, idx: int) -> dict[str, Any]:
        real_idx = self._indices[idx] if self._indices else idx
        sample = self._base[real_idx]

        # Extract episode/frame indices BEFORE transforms (LiberoInputs drops them)
        ep = int(sample.get("episode_index", -1))
        fr = int(sample.get("frame_index", -1))
        if ep < 0 or fr < 0:
            raise KeyError(
                f"LeRobot sample missing episode_index ({ep}) or "
                f"frame_index ({fr}) at real_idx={real_idx}. "
                f"Available keys: {sorted(sample.keys())}"
            )

        # Prompt injection
        if self._tasks and "task_index" in sample:
            ti = sample["task_index"]
            ti = ti.item() if isinstance(ti, torch.Tensor) else int(ti)
            if ti in self._tasks:
                sample = {**sample, "prompt": self._tasks[ti]}

        if self._transform is not None:
            sample = self._transform(sample)

        # Return lookup + normalize (sidecar guaranteed to exist)
        raw = float(self._sidecar[ep]["return"][fr]) if ep in self._sidecar else 0.0
        target_value = (
            self._normalizer.normalize_value(raw) if self._normalizer else raw
        )

        images = sample.get("image", sample.get("images", {}))
        if not isinstance(images, dict):
            images = {}
        masks = sample.get("image_mask", sample.get("image_masks"))

        result: dict[str, Any] = {
            "images": images,
            "prompt": sample.get("prompt", "perform the task"),
            "target_values": target_value,
            "actions": None,
        }
        if isinstance(masks, dict) and masks:
            result["image_masks"] = masks
        return result
