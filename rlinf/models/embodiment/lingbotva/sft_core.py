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

"""SFT support for LingBot-VA."""

from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

_TRANSFORMER_STATE_PREFIXES = (
    "_sft_core.transformer.",
    "transformer.",
)


def _extend_import_path(repo_path: Path) -> None:
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _extract_transformer_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    for prefix in _TRANSFORMER_STATE_PREFIXES:
        extracted = {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if extracted:
            return extracted
    return state_dict


def export_official_transformer_checkpoint(
    *,
    model_path: str | Path,
    state_dict_path: str | Path,
    output_dir: str | Path,
) -> Path:
    model_path = Path(model_path)
    state_dict_path = Path(state_dict_path)
    output_dir = Path(output_dir)

    if not state_dict_path.exists():
        raise FileNotFoundError(
            f"LingBot-VA state dict path does not exist: {state_dict_path}"
        )

    config_src = model_path / "transformer" / "config.json"
    if not config_src.exists():
        raise FileNotFoundError(
            f"LingBot-VA transformer config not found at {config_src}"
        )

    raw_state = torch.load(
        state_dict_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(raw_state, dict):
        raise TypeError(
            "LingBot-VA full_weights checkpoint must deserialize to a dict, got "
            f"{type(raw_state)!r}."
        )

    transformer_state = _extract_transformer_state_dict(raw_state)
    tensor_state = {}
    for key, value in transformer_state.items():
        if not torch.is_tensor(value):
            raise TypeError(
                "LingBot-VA transformer checkpoint values must be tensors, got "
                f"{type(value)!r} for key {key!r}."
            )
        tensor_state[key] = value.detach().cpu().contiguous().to(torch.bfloat16)

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_src, output_dir / "config.json")

    config_data = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    config_data.pop("_name_or_path", None)
    (output_dir / "config.json").write_text(
        json.dumps(config_data, indent=2) + "\n",
        encoding="utf-8",
    )
    save_file(tensor_state, output_dir / "diffusion_pytorch_model.safetensors")
    return output_dir


class LingbotVASFTCore(nn.Module):
    def __init__(self, cfg: Any, torch_dtype: torch.dtype) -> None:
        super().__init__()
        self.cfg = cfg
        self.repo_path = Path(getattr(cfg.lingbotva, "repo_path"))
        self.model_path = Path(cfg.model_path)
        self.torch_dtype = torch_dtype
        _extend_import_path(self.repo_path)

        from wan_va.configs import VA_CONFIGS
        from wan_va.modules.utils import load_transformer
        from wan_va.utils import (
            FlowMatchScheduler,
            data_seq_to_patch,
            get_mesh_id,
            sample_timestep_id,
        )

        train_config_name = getattr(
            cfg.lingbotva, "train_config_name", "robotwin_train"
        )
        self.train_cfg = copy.deepcopy(VA_CONFIGS[train_config_name])
        self.train_cfg.wan22_pretrained_model_name_or_path = str(self.model_path)
        self.train_cfg.param_dtype = torch_dtype

        self.patch_size = tuple(self.train_cfg.patch_size)
        self._data_seq_to_patch = data_seq_to_patch
        self._get_mesh_id = get_mesh_id
        self._sample_timestep_id = sample_timestep_id

        transformer_path = self.model_path / "transformer"
        self.transformer = load_transformer(
            str(transformer_path),
            torch_dtype=torch_dtype,
            torch_device="cpu",
        )
        self.transformer = self.transformer.to(dtype=torch_dtype)

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
        self._ac_applied = False

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

    def _add_noise(
        self,
        latent: torch.Tensor,
        train_scheduler,
        *,
        action_mask: torch.Tensor | None = None,
        action_mode: bool = False,
        noisy_cond_prob: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        _, _, frame_num, _, _ = latent.shape

        timestep_ids = self._sample_timestep_id(
            batch_size=frame_num,
            num_train_timesteps=train_scheduler.num_train_timesteps,
        )
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=latent.device)
        noisy_latents = train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets = train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1

        latent_grid_id = self._get_mesh_id(
            latent.shape[-3] // patch_f,
            latent.shape[-2] // patch_h,
            latent.shape[-1] // patch_w,
            t=1 if action_mode else 0,
            f_w=1,
            f_shift=0,
            action=action_mode,
        ).to(latent.device)
        latent_grid_id = latent_grid_id[None].repeat(latent.shape[0], 1, 1)

        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = self._sample_timestep_id(
                batch_size=frame_num,
                min_timestep_bd=0.5,
                max_timestep_bd=1.0,
                num_train_timesteps=train_scheduler.num_train_timesteps,
            )
            cond_noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(
                device=latent.device
            )
            latent = train_scheduler.add_noise(
                latent, cond_noise, cond_timesteps, t_dim=2
            )
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return {
            "timesteps": timesteps[None].repeat(latent.shape[0], 1),
            "noisy_latents": noisy_latents,
            "targets": targets,
            "latent": latent,
            "cond_timesteps": cond_timesteps[None].repeat(latent.shape[0], 1),
            "grid_id": latent_grid_id,
        }

    def _prepare_input_dict(
        self,
        batch_dict: dict[str, torch.Tensor],
    ) -> dict[str, dict[str, torch.Tensor] | int]:
        latent_dict = self._add_noise(
            latent=batch_dict["latents"],
            train_scheduler=self.train_scheduler_latent,
            action_mode=False,
            noisy_cond_prob=0.5,
        )
        action_dict = self._add_noise(
            latent=batch_dict["actions"],
            train_scheduler=self.train_scheduler_action,
            action_mask=batch_dict["actions_mask"],
            action_mode=True,
            noisy_cond_prob=0.0,
        )

        latent_dict["text_emb"] = batch_dict["text_emb"]
        action_dict["text_emb"] = batch_dict["text_emb"]
        action_dict["actions_mask"] = batch_dict["actions_mask"]

        return {
            "latent_dict": latent_dict,
            "action_dict": action_dict,
            "chunk_size": 1,
            "window_size": int(getattr(self.train_cfg, "attn_window", 72)),
        }

    def compute_loss(
        self,
        input_dict: dict[str, Any],
        pred: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent_pred, action_pred = pred
        action_pred = (
            action_pred.unflatten(
                1,
                (
                    input_dict["action_dict"]["targets"].shape[-3],
                    input_dict["action_dict"]["targets"].shape[-2],
                ),
            )
            .permute(0, 3, 1, 2)
            .unsqueeze(-1)
        )
        latent_pred = self._data_seq_to_patch(
            self.patch_size,
            latent_pred,
            input_dict["latent_dict"]["targets"].shape[-3],
            input_dict["latent_dict"]["targets"].shape[-2],
            input_dict["latent_dict"]["targets"].shape[-1],
            batch_size=latent_pred.shape[0],
        )

        batch_num, frame_num = input_dict["latent_dict"]["timesteps"].shape
        latent_loss_weight = self.train_scheduler_latent.training_weight(
            input_dict["latent_dict"]["timesteps"].flatten()
        ).reshape(batch_num, frame_num)
        action_loss_weight = self.train_scheduler_action.training_weight(
            input_dict["action_dict"]["timesteps"].flatten()
        ).reshape(batch_num, frame_num)

        latent_loss = F.mse_loss(
            latent_pred.float(),
            input_dict["latent_dict"]["targets"].float().detach(),
            reduction="none",
        )
        latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        latent_loss_per_frame = latent_loss.sum(dim=1)
        latent_mask_per_frame = torch.ones_like(latent_loss).sum(dim=1)
        latent_loss = (latent_loss_per_frame / (latent_mask_per_frame + 1e-6)).mean()

        action_loss = F.mse_loss(
            action_pred.float(),
            input_dict["action_dict"]["targets"].float().detach(),
            reduction="none",
        )
        action_loss = action_loss * action_loss_weight[:, None, :, None, None]
        action_loss = action_loss * input_dict["action_dict"]["actions_mask"].float()
        action_loss = action_loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        action_mask = (
            input_dict["action_dict"]["actions_mask"]
            .float()
            .permute(0, 2, 3, 4, 1)
            .flatten(0, 1)
            .flatten(1)
        )
        action_loss_per_frame = action_loss.sum(dim=1)
        action_mask_per_frame = action_mask.sum(dim=1)
        action_loss = (action_loss_per_frame / (action_mask_per_frame + 1e-6)).mean()
        return latent_loss, action_loss

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        first_tensor = next(value for value in data.values() if torch.is_tensor(value))
        device = first_tensor.device
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
