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

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import openpi.models.model as _openpi_model
import openpi.transforms as _openpi_transforms
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from torch.utils.data import Dataset

from rlinf.models.embodiment.openpi.policies import franka_policy, libero_policy

from .return_loaders import load_returns_sidecar
from .value_transforms import ReturnNormalizer

logger = logging.getLogger(__name__)

_MODEL_TYPE_MAP = {
    "pi0": _openpi_model.ModelType.PI0,
    "pi05": _openpi_model.ModelType.PI05,
    "pi0_fast": _openpi_model.ModelType.PI0_FAST,
}

_REPACK_KEYS = {
    "libero": {
        "observation/image": "image",
        "observation/wrist_image": "wrist_image",
        "observation/state": "state",
        "actions": "actions",
        "prompt": "prompt",
    },
    "libero_v2": {
        "observation/image": "observation.images.image",
        "observation/wrist_image": "observation.images.wrist_image",
        "observation/state": "observation.state",
        "actions": "action",
        "prompt": "prompt",
    },
    "franka": {
        "observation/image": "image",
        "observation/state": "state",
        "actions": "actions",
        "prompt": "prompt",
    },
}


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
        default_prompt: Optional[str] = None,
        return_min: Optional[float] = None,
        return_max: Optional[float] = None,
        normalize_to_minus_one_zero: bool = True,
        max_samples: Optional[int] = None,
        tag: Optional[str] = None,
        episode_percentage: Optional[float] = None,
        shuffle_episodes: bool = False,
        episode_seed: int = 42,
        **kwargs,  # accept unused params (gamma, split, etc.)
    ):
        _known_unused = {
            "gamma",
            "split",
            "repo_id",
            "norm_stats_dir",
            "asset_id",
            "extra_delta_transform",
            "action_norm_skip_dims",
        }
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

        # Transform pipeline (repack → data → model)
        self._transform = self._build_transform(
            robot_type=robot_type,
            model_type=model_type,
            action_dim=action_dim or 32,
            default_prompt=default_prompt,
        )

        # Task descriptions for prompt injection
        self._tasks = self._load_tasks(local_path) or (
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
    def _build_transform(robot_type, model_type, action_dim, default_prompt):
        """Build transform pipeline using openpi policies and transforms."""
        model_type_lower = model_type.lower()
        model_type_enum = _MODEL_TYPE_MAP[model_type_lower]
        robot = robot_type.lower()

        transforms_list = []

        # 1. Repack (map dataset keys → standard observation keys)
        repack_keys = _REPACK_KEYS.get(robot)
        if repack_keys is None:
            raise ValueError(
                f"Unknown robot type: {robot_type}. "
                f"Available: {list(_REPACK_KEYS.keys())}"
            )
        transforms_list.append(_openpi_transforms.RepackTransform(repack_keys))

        # 2. Data transforms (robot-specific, from openpi policies)
        if robot in ("libero", "libero_v2"):
            transforms_list.append(
                libero_policy.LiberoInputs(model_type=model_type_enum)
            )
        elif robot == "franka":
            transforms_list.append(
                franka_policy.FrankaEEInputs(
                    action_dim=action_dim,
                    model_type=model_type_enum,
                )
            )

        # 3. Model transforms (without TokenizePrompt and ResizeImages —
        #    value model handles tokenization via ValueProcessor in the
        #    collator, and image resize via ValueImageProcessor)
        transforms_list.append(_openpi_transforms.InjectDefaultPrompt(default_prompt))
        transforms_list.append(_openpi_transforms.PadStatesAndActions(action_dim))

        return _openpi_transforms.compose(transforms_list)

    @staticmethod
    def _load_tasks(dataset_path) -> dict[int, str]:
        """Load task descriptions from dataset meta (tasks.jsonl or tasks.parquet)."""
        meta = Path(dataset_path) / "meta"
        jsonl = meta / "tasks.jsonl"
        if jsonl.exists():
            tasks = {}
            with open(jsonl, "r") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line.strip())
                        tasks[d.get("task_index", len(tasks))] = d.get("task", "")
            return tasks
        parquet = meta / "tasks.parquet"
        if parquet.exists():
            import pandas as pd

            df = pd.read_parquet(parquet)
            if "task_index" in df.columns and "task" in df.columns:
                return {int(r["task_index"]): str(r["task"]) for _, r in df.iterrows()}
        return {}

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
