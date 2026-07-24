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

"""SFT dataloader builder for the Evo-1 (evo1-flash) VLA model.

Follows RLinf's SFT dataset-builder pattern: the dataset
lives in the externally-installed Evo-1 repo, so this builder puts the repo
root on ``sys.path`` (via the shared ``_ensure_evo1_importable`` helper), builds
Evo-1's own ``LeRobotDataset``, and wraps it in a ``StatefulDataLoader`` with a
``DistributedSampler`` so SFT can resume mid-epoch.

The per-sample batch dict emitted by ``evo1_sft_collate_fn`` matches the keys
consumed by ``Evo1ForRLActionPrediction.sft_forward``.
"""

from __future__ import annotations

import os

import torch
import yaml
from torch.utils.data.distributed import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.models.embodiment.evo1 import _ensure_evo1_importable
from rlinf.utils.logging import get_logger

logger = get_logger()


def evo1_sft_collate_fn(batch):
    """Collate Evo-1 ``LeRobotDataset`` samples into a training batch.

    Replicated from Evo-1's ``scripts/train.py::custom_collate_fn`` so we do not
    import that training script (which pulls in accelerate/deepspeed/wandb).
    Output keys match ``Evo1ForRLActionPrediction.sft_forward``.
    """
    return {
        "prompts": [item["prompt"] for item in batch],
        "images": [item["images"] for item in batch],
        "states": torch.stack([item["state"] for item in batch], dim=0),
        "actions": torch.stack([item["action"] for item in batch], dim=0),
        "action_mask": torch.stack([item["action_mask"] for item in batch], dim=0),
        "state_mask": torch.stack([item["state_mask"] for item in batch], dim=0),
        "image_masks": torch.stack([item["image_mask"] for item in batch], dim=0),
        "embodiment_ids": torch.stack(
            [item["embodiment_id"] for item in batch], dim=0
        ),
    }


def _get(evo1_cfg, key, default):
    if evo1_cfg is not None:
        val = getattr(evo1_cfg, key, None)
        if val is not None:
            return val
    return default


def build_evo1_sft_dataloader(cfg, world_size, global_rank, data_paths):
    """Build the Evo-1 SFT dataloader.

    ``data.train_data_paths`` (``data_paths``) is the path to an Evo-1 dataset
    config YAML (the same file Evo-1's own training reads via
    ``config.dataset_config_path``). It defines ``data_groups`` /
    ``max_action_dim`` / ``max_state_dim`` / ``max_views`` etc.

    Returns ``(data_loader, {"num_samples": len(dataset)})``.
    """
    model_cfg = cfg.actor.model
    evo1_cfg = getattr(model_cfg, "evo1", None)

    # Ensure Evo-1's repo root is importable, then load its dataset class.
    _ensure_evo1_importable(model_cfg)
    try:
        from dataset.lerobot_dataset_pretrain_mp import LeRobotDataset  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Evo-1 is required for SFT. Install it and set "
            "cfg.actor.model.evo1.repo_path (or EVO1_REPO_PATH) so that "
            "'import dataset.lerobot_dataset_pretrain_mp' resolves."
        ) from e

    dataset_config_path = data_paths if isinstance(data_paths, str) else data_paths[0]
    if dataset_config_path is None or not os.path.isfile(dataset_config_path):
        raise ValueError(
            "Evo-1 SFT requires data.train_data_paths to point to an Evo-1 "
            f"dataset config YAML. Got: {dataset_config_path}"
        )
    with open(dataset_config_path) as f:
        dataset_config = yaml.safe_load(f)

    image_size = int(_get(evo1_cfg, "image_size", 448))
    action_horizon = int(_get(evo1_cfg, "horizon", 50))
    binarize_gripper = bool(_get(evo1_cfg, "binarize_gripper", True))
    video_backend = _get(evo1_cfg, "video_backend", "av")
    use_augmentation = bool(_get(evo1_cfg, "use_augmentation", False))
    max_samples_per_file = _get(evo1_cfg, "max_samples_per_file", None)
    cache_dir = _get(evo1_cfg, "cache_dir", None)

    dataset = LeRobotDataset(
        config=dataset_config,
        image_size=image_size,
        max_samples_per_file=max_samples_per_file,
        action_horizon=action_horizon,
        binarize_gripper=binarize_gripper,
        use_augmentation=use_augmentation,
        video_backend=video_backend,
        cache_dir=cache_dir,
    )
    logger.info(
        f"Evo-1 SFT dataset: {len(dataset)} samples from {dataset_config_path} "
        f"(image_size={image_size}, action_horizon={action_horizon})"
    )

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=global_rank, shuffle=True
    )
    num_workers = int(_get(evo1_cfg, "num_workers", 4))
    data_loader = StatefulDataLoader(
        dataset,
        batch_size=cfg.actor.micro_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=evo1_sft_collate_fn,
    )

    return data_loader, {"num_samples": len(dataset)}
