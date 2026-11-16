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

"""LeRobot SFT data for LingBot-VLA 2.0, normalized like the RL rollout."""

from typing import Any

from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def _resolve_data_path(data_paths: Any) -> str:
    if isinstance(data_paths, str):
        return data_paths
    if len(data_paths) != 1:
        raise ValueError(
            f"LingBot-VLA V2 SFT expects one LeRobot dataset path, got {data_paths}"
        )
    return str(data_paths[0])


def build_lingbotvlav2_sft_dataloader(
    cfg: DictConfig, world_size: int, global_rank: int, data_paths: Any
) -> tuple[DataLoader, dict[str, int]]:
    """Read a LeRobot dataset through the rollout's FeatureTransform.

    Robot config, norm stats and image size come from ``actor.model``, so SFT
    targets are normalized exactly like the actions PPO/GRPO later decode.
    """
    from lingbotvla.data.vla_data.base_dataset import VLADataset
    from lingbotvla.models import build_processor

    from .lingbotvlav2_action_model import build_feature_transform, build_model_config

    model_cfg = cfg.actor.model
    model_config = build_model_config(model_cfg)
    if int(model_cfg.num_action_chunks) != int(model_config.n_action_steps):
        raise ValueError("num_action_chunks must match the V2 checkpoint horizon")
    processor = build_processor(model_cfg.tokenizer_path)
    transform, image_size = build_feature_transform(model_cfg, model_config, processor)
    dataset = VLADataset(
        repo_id=_resolve_data_path(data_paths),
        # Unused when a feature transform is given; the rollout's config wins.
        data_name="robotwin",
        dataset_config=None,
        robot_config_root=None,
        config=model_config,
        processor=processor,
        chunk_size=int(model_config.n_action_steps),
        image_size=(image_size, image_size),
        feature_transform=transform,
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=global_rank, shuffle=True
    )
    num_workers = int(cfg.data.get("num_workers", 4))
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.actor.micro_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    return data_loader, {"num_samples": len(dataset)}
