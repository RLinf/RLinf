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

"""Thin SFT adapter around the official LingBot-VA transformer."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from rlinf.models.embodiment.lingbotva.utils import _extend_import_path


class LingbotVATrainingAdapter(nn.Module):
    def __init__(self, cfg: Any, torch_dtype: torch.dtype) -> None:
        super().__init__()
        self.cfg = cfg
        self.repo_path = Path(getattr(cfg.lingbotva, "repo_path"))
        self.model_path = Path(cfg.model_path)
        self.torch_dtype = torch_dtype
        _extend_import_path(self.repo_path)

        from wan_va.configs import VA_CONFIGS
        from wan_va.modules.utils import load_transformer
        from wan_va.train import Trainer as WanVATrainer
        from wan_va.utils import FlowMatchScheduler

        train_config_name = getattr(
            cfg.lingbotva, "train_config_name", "robotwin_train"
        )
        self.train_cfg = copy.deepcopy(VA_CONFIGS[train_config_name])
        self.train_cfg.wan22_pretrained_model_name_or_path = str(self.model_path)
        self.train_cfg.param_dtype = torch_dtype

        self.patch_size = tuple(self.train_cfg.patch_size)
        self.device = torch.device("cpu")
        self.dtype = torch_dtype
        self.config = self.train_cfg
        self.gradient_accumulation_steps = 1
        self._add_noise = WanVATrainer._add_noise.__get__(self, type(self))
        self._prepare_input_dict = WanVATrainer._prepare_input_dict.__get__(
            self, type(self)
        )
        self.compute_loss = WanVATrainer.compute_loss.__get__(self, type(self))

        transformer_path = self.model_path / "transformer"
        self.transformer = load_transformer(
            str(transformer_path),
            torch_dtype=torch_dtype,
            torch_device="cpu",
        )
        self.transformer = self.transformer.to(dtype=torch_dtype)
        self._ac_applied = False
        self._apply_official_activation_checkpointing()

        self.train_scheduler_latent = FlowMatchScheduler(
            shift=self.train_cfg.snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(
            shift=self.train_cfg.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.train_scheduler_action.set_timesteps(1000, training=True)

    def _apply_official_activation_checkpointing(self) -> None:
        if self._ac_applied:
            return
        _extend_import_path(self.repo_path)
        from wan_va.distributed.fsdp import apply_ac

        apply_ac(self.transformer)
        self._ac_applied = True

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        del kwargs
        self._apply_official_activation_checkpointing()

    def gradient_checkpointing_disable(self) -> None:
        # Official LingBot-VA training applies activation checkpoint wrappers
        # once before FSDP sharding. There is no lightweight unwrap path, so we
        # intentionally keep this as a no-op.
        return None

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        first_tensor = next(value for value in data.values() if torch.is_tensor(value))
        device = first_tensor.device
        self.device = torch.device(device)
        batch = {
            "latents": data["latents"].to(device=device, dtype=self.torch_dtype),
            "text_emb": data["text_emb"].to(device=device, dtype=self.torch_dtype),
            "actions": data["actions"].to(device=device, dtype=self.torch_dtype),
            "actions_mask": data["actions_mask"].to(device=device),
        }
        input_dict = self._prepare_input_dict(batch)
        pred = self.transformer(input_dict, train_mode=True)
        latent_loss, action_loss = self.compute_loss(input_dict, pred)
        total_loss = latent_loss + action_loss
        return {
            "loss": total_loss,
            "latent_loss": latent_loss,
            "action_loss": action_loss,
        }
