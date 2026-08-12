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

"""STEAM temporal critic operating on frozen RLT Stage 1 features."""

from typing import Any, Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor, nn
from transformers import PretrainedConfig

from .modeling_critic import CriticOutput, SteamCriticModel


class RLTSteamConfig(PretrainedConfig):
    """Configuration for the lightweight RLT-feature STEAM critic."""

    model_type = "steam_value_model"

    def __init__(
        self,
        *,
        backbone_type: str = "rlt_stage1",
        z_dim: int = 2048,
        proprio_dim: int = 9,
        fusion_hidden_dim: int = 512,
        dropout: float = 0.1,
        label_smoothing: float = 0.05,
        num_bins: int = 8,
        stride_k: Optional[int] = None,
        ensemble_size: int = 3,
        ensemble_head_seed_base: Optional[int] = None,
        dtype: str = "float32",
        precision: Optional[str] = None,
        use_gradient_checkpointing: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.backbone_type = str(backbone_type)
        self.z_dim = int(z_dim)
        self.proprio_dim = int(proprio_dim)
        self.fusion_hidden_dim = int(fusion_hidden_dim)
        self.dropout = float(dropout)
        self.label_smoothing = float(label_smoothing)
        self.num_bins = int(num_bins)
        self.stride_k = None if stride_k is None else int(stride_k)
        self.ensemble_size = int(ensemble_size)
        self.ensemble_head_seed_base = (
            None if ensemble_head_seed_base is None else int(ensemble_head_seed_base)
        )
        self.dtype = precision if precision is not None else dtype
        self.precision = self.dtype
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self._validate()

    def _validate(self) -> None:
        if self.backbone_type != "rlt_stage1":
            raise ValueError(
                "RLTSteamConfig.backbone_type must be 'rlt_stage1', got "
                f"{self.backbone_type!r}."
            )
        if self.z_dim <= 0 or self.proprio_dim < 0:
            raise ValueError(
                "z_dim must be positive and proprio_dim must be non-negative"
            )
        if self.fusion_hidden_dim <= 0:
            raise ValueError("fusion_hidden_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.num_bins < 2 or self.num_bins % 2 != 0:
            raise ValueError("num_bins must be an even integer >= 2")
        if self.ensemble_size < 1:
            raise ValueError("ensemble_size must be >= 1")
        if self.dtype not in {"bfloat16", "float32", "float16"}:
            raise ValueError(
                f"dtype must be one of bfloat16/float32/float16, got {self.dtype}"
            )

    def to_diff_dict(self) -> dict[str, Any]:
        return self.to_dict()


class RLTSteamBackbone(nn.Module):
    """Shared per-state projection and ordered temporal-pair head."""

    def __init__(self, config: RLTSteamConfig) -> None:
        super().__init__()
        input_dim = config.z_dim + config.proprio_dim
        hidden_dim = config.fusion_hidden_dim
        self.frame_projector = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.fusion_norm = nn.LayerNorm(2 * hidden_dim)
        self.value_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.num_bins),
        )

    def encode_pair(self, frame_t: Tensor, frame_tk: Tensor) -> Tensor:
        feature_t = self.frame_projector(frame_t)
        feature_tk = self.frame_projector(frame_tk)
        return self.fusion_norm(torch.cat((feature_t, feature_tk), dim=-1))


class RLTSteamCriticModel(SteamCriticModel):
    """STEAM critic consuming ordered pairs of ``z_rl`` and proprio features."""

    def __init__(self, config: RLTSteamConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.model = RLTSteamBackbone(config)
        self.label_smoothing = float(config.label_smoothing)
        self.gradient_checkpointing_enabled = False
        for name, module in self.named_modules():
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

    @property
    def _no_split_modules(self) -> list[str]:
        return ["LayerNorm"]

    @property
    def _no_split_names(self) -> list[str]:
        return ["frame_projector", "fusion_norm", "value_head"]

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_enabled = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing_enabled = False

    def attach_runtime_assets(self, processor, device) -> None:
        del processor
        self._device = device

    def _frame_feature(self, observation: dict[str, Tensor], suffix: str) -> Tensor:
        z_rl = observation[f"z_rl_{suffix}"].to(dtype=torch.float32)
        if z_rl.ndim != 2 or z_rl.shape[-1] != self.config.z_dim:
            raise ValueError(
                f"z_rl_{suffix} must have shape [B, {self.config.z_dim}], "
                f"got {tuple(z_rl.shape)}"
            )
        if self.config.proprio_dim == 0:
            return z_rl
        proprio = observation[f"proprio_{suffix}"].to(dtype=torch.float32)
        if proprio.ndim != 2 or proprio.shape[-1] != self.config.proprio_dim:
            raise ValueError(
                f"proprio_{suffix} must have shape [B, {self.config.proprio_dim}], "
                f"got {tuple(proprio.shape)}"
            )
        return torch.cat((z_rl, proprio), dim=-1)

    def _compute_output(
        self,
        observation: dict[str, Tensor],
        labels: Optional[Tensor],
    ) -> CriticOutput:
        frame_t = self._frame_feature(observation, "t")
        frame_tk = self._frame_feature(observation, "tk")
        projector_dtype = next(self.model.parameters()).dtype
        hidden_states = self.model.encode_pair(
            frame_t.to(dtype=projector_dtype),
            frame_tk.to(dtype=projector_dtype),
        )
        logits = self.model.value_head(hidden_states)
        probs = F.softmax(logits, dim=-1)
        progress_values = self._predicted_signed_value(probs)

        expert_loss = None
        cat_metrics = None
        if labels is not None:
            expert_loss, cat_metrics = self._compute_loss(logits, labels)
        expert_loss_mean = expert_loss.mean() if expert_loss is not None else None
        return CriticOutput(
            loss=expert_loss_mean,
            predicted_values=progress_values,
            logits=logits,
            probs=probs,
            atoms=None,
            expert_loss=expert_loss_mean,
            hidden_states=hidden_states,
            cat_acc_best=cat_metrics["acc_best"] if cat_metrics else None,
            cat_acc_neighbor=cat_metrics["acc_neighbor"] if cat_metrics else None,
            mae=cat_metrics["mae"] if cat_metrics else None,
            progress_values=progress_values,
        )

    def forward(self, observation, labels=None, **kwargs) -> CriticOutput:
        del kwargs
        return self._compute_output(observation, labels)

    @torch.no_grad()
    def predict(self, observation) -> CriticOutput:
        return self._compute_output(observation, None)


_RLT_STEAM_DEFAULTS: dict[str, Any] = {
    "backbone_type": "rlt_stage1",
    "z_dim": 2048,
    "proprio_dim": 9,
    "fusion_hidden_dim": 512,
    "dropout": 0.1,
    "label_smoothing": 0.05,
    "num_bins": 8,
    "stride_k": None,
    "ensemble_size": 3,
    "ensemble_head_seed_base": None,
    "use_gradient_checkpointing": False,
}


def build_rlt_steam_config(cfg: DictConfig) -> RLTSteamConfig:
    """Build an RLT STEAM config from the flat Hydra model config."""
    values = dict(_RLT_STEAM_DEFAULTS)
    for key in values:
        value = cfg.get(key, None)
        if value is not None:
            values[key] = value
    precision = str(cfg.get("precision", "fp32"))
    dtype = {
        "fp32": "float32",
        "bf16": "bfloat16",
        "fp16": "float16",
    }.get(precision, precision)
    return RLTSteamConfig(**values, dtype=dtype)


__all__ = [
    "RLTSteamConfig",
    "RLTSteamCriticModel",
    "build_rlt_steam_config",
]
