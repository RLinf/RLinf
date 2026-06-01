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

"""LingBot-VA SFT dataset and dataloader utilities."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from rlinf.models.embodiment.lingbotva.utils import _extend_import_path


def _lingbotva_sft_collate(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("LingBot-VA SFT collate received an empty batch.")

    max_text_len = max(int(item["text_emb"].shape[0]) for item in batch)
    hidden_size = int(batch[0]["text_emb"].shape[1])
    max_action_len = max(int(item["actions"].shape[2]) for item in batch)
    action_channels = int(batch[0]["actions"].shape[0])
    latent_frames = int(batch[0]["actions"].shape[1])
    action_tail_dim = int(batch[0]["actions"].shape[3])

    text_emb_batch = []
    actions_batch = []
    actions_mask_batch = []
    for item in batch:
        text_emb = item["text_emb"]
        if text_emb.dim() != 2:
            raise ValueError(
                f"LingBot-VA text_emb must be 2D, got shape {tuple(text_emb.shape)}."
            )
        if int(text_emb.shape[1]) != hidden_size:
            raise ValueError(
                "LingBot-VA text_emb hidden sizes must match across batch: "
                f"expected {hidden_size}, got {int(text_emb.shape[1])}."
            )
        if int(text_emb.shape[0]) < max_text_len:
            padding = text_emb.new_zeros(
                max_text_len - int(text_emb.shape[0]), hidden_size
            )
            text_emb = torch.cat([text_emb, padding], dim=0)
        text_emb_batch.append(text_emb)

        actions = item["actions"]
        actions_mask = item["actions_mask"]
        if actions.dim() != 4:
            raise ValueError(
                f"LingBot-VA actions must be 4D, got shape {tuple(actions.shape)}."
            )
        if tuple(actions.shape) != tuple(actions_mask.shape):
            raise ValueError(
                "LingBot-VA actions and actions_mask must share the same shape: "
                f"got {tuple(actions.shape)} vs {tuple(actions_mask.shape)}."
            )
        if (
            int(actions.shape[0]) != action_channels
            or int(actions.shape[1]) != latent_frames
            or int(actions.shape[3]) != action_tail_dim
        ):
            raise ValueError(
                "LingBot-VA actions must agree on channel/frame dimensions across batch."
            )
        pad_action_len = max_action_len - int(actions.shape[2])
        if pad_action_len > 0:
            action_padding = actions.new_zeros(
                action_channels,
                latent_frames,
                pad_action_len,
                action_tail_dim,
            )
            mask_padding = torch.zeros(
                (action_channels, latent_frames, pad_action_len, action_tail_dim),
                dtype=torch.bool,
            )
            actions = torch.cat([actions, action_padding], dim=2)
            actions_mask = torch.cat([actions_mask, mask_padding], dim=2)
        actions_batch.append(actions)
        actions_mask_batch.append(actions_mask)

    return {
        "latents": torch.stack([item["latents"] for item in batch], dim=0),
        "text_emb": torch.stack(text_emb_batch, dim=0),
        "actions": torch.stack(actions_batch, dim=0),
        "actions_mask": torch.stack(actions_mask_batch, dim=0),
    }


def _resolve_dataset_path(data_paths) -> Path | None:
    if data_paths is None:
        return None
    if isinstance(data_paths, (str, Path)):
        return Path(data_paths)

    values = list(data_paths)
    if len(values) != 1:
        raise ValueError("LingBot-VA SFT currently supports exactly one dataset root.")
    return _resolve_dataset_path(values[0])


def build_lingbotva_sft_dataloader(cfg, world_size, global_rank, data_paths):
    lingbotva_cfg = cfg.actor.model.lingbotva
    dataset_path = _resolve_dataset_path(data_paths)
    if dataset_path is None:
        dataset_path = _resolve_dataset_path(
            getattr(lingbotva_cfg, "dataset_path", None)
        )
    if dataset_path is None:
        raise ValueError("LingBot-VA SFT requires a dataset path.")

    repo_path = Path(getattr(lingbotva_cfg, "repo_path"))
    _extend_import_path(repo_path)

    from wan_va.configs import VA_CONFIGS
    from wan_va.dataset import MultiLatentLeRobotDataset

    train_config_name = getattr(lingbotva_cfg, "train_config_name", "robotwin_train")
    dataset_cfg = copy.deepcopy(VA_CONFIGS[train_config_name])
    dataset_cfg.wan22_pretrained_model_name_or_path = str(cfg.actor.model.model_path)
    dataset_cfg.param_dtype = cfg.actor.model.precision
    dataset_cfg.dataset_path = str(dataset_path)

    empty_emb_path = getattr(lingbotva_cfg, "empty_emb_path", None)
    if empty_emb_path is None:
        empty_emb_path = dataset_path / "empty_emb.pt"
    dataset_cfg.empty_emb_path = str(empty_emb_path)
    dataset_cfg.cfg_prob = float(getattr(lingbotva_cfg, "cfg_prob", 0.1))

    dataset = MultiLatentLeRobotDataset(
        config=dataset_cfg,
        num_init_worker=int(getattr(lingbotva_cfg, "num_init_worker", 8)),
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.actor.micro_batch_size,
        sampler=sampler,
        num_workers=int(getattr(lingbotva_cfg, "load_worker", 4)),
        pin_memory=True,
        drop_last=True,
        collate_fn=_lingbotva_sft_collate,
    )
    return data_loader, {}
