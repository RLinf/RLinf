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

"""
Value model configuration.

Provides:
- Config: Gemma action expert configuration (variant configs)
- VLMBaseConfig: Base configuration for VLM-based model architecture
- ValueCriticConfig: Extended configuration for value critic models
"""

import dataclasses
from typing import Literal, Optional

from transformers import PretrainedConfig

ForwardMode = Literal["vla", "vlm", "vlm_vla"]


# =============================================================================
# Gemma Expert Variant Configs
# =============================================================================
#
# Einsum axis naming convention:
#   B: batch, T: query length, S: k/v length, N: num query heads,
#   K: num k/v heads, G: num query heads per k/v head, H: head dim,
#   D: d_model ("features")


@dataclasses.dataclass
class Config:
    """Gemma action expert configuration.

    Parameter count formula (without embeddings, as action expert sets embed_tokens=None):

    Per layer:
      - Attention:
          q_proj: width * (num_heads * head_dim)
          k_proj: width * (num_kv_heads * head_dim)
          v_proj: width * (num_kv_heads * head_dim)
          o_proj: (num_heads * head_dim) * width
      - MLP (GeGLU):
          gate_proj: width * mlp_dim
          up_proj:   width * mlp_dim
          down_proj: mlp_dim * width
      - RMSNorm (weight only):
          input_layernorm:          width
          post_attention_layernorm: width

    Total params = depth * (
        width * (num_heads * head_dim) * 2 +       # q_proj + o_proj
        width * (num_kv_heads * head_dim) * 2 +    # k_proj + v_proj
        width * mlp_dim * 3 +                       # gate + up + down
        width * 2                                   # layernorms
    ) + width                                       # final norm
    """

    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


Variant = Literal[
    "dummy",
    "gemma_1m",
    "gemma_50m",
    "gemma_100m",
    "gemma_150m",
    "gemma_300m",
    "gemma_2b",
]


def get_config(variant: Variant) -> Config:
    """Returns config for specified gemma variant."""
    if variant == "dummy":
        return Config(
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
        )
    if variant == "gemma_1m":
        # ~1.21M params, lightweight 4-layer expert for fast iteration / debugging
        return Config(
            width=128,
            depth=4,
            mlp_dim=448,
            num_heads=1,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_50m":
        # ~56M params (head_dim=256 required for cross-attention with gemma_2b backbone)
        return Config(
            width=384,
            depth=18,
            mlp_dim=1536,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_100m":
        # ~110M params (head_dim=256 required for cross-attention with gemma_2b backbone)
        return Config(
            width=512,
            depth=18,
            mlp_dim=2048,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_150m":
        # ~165M params (head_dim=256 required for cross-attention with gemma_2b backbone)
        return Config(
            width=640,
            depth=18,
            mlp_dim=2560,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_300m":
        # ~311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        # ~1.98B params
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    raise ValueError(f"Unknown variant: {variant}")


# =============================================================================
# PI0.5 Model Configs
# =============================================================================


class VLMBaseConfig(PretrainedConfig):
    """Base configuration for VLM-based value models."""

    model_type = "pi05"

    def __init__(
        self,
        dtype: Optional[str] = None,
        precision: Optional[str] = None,
        action_dim: int = 32,
        action_horizon: int = 50,
        max_token_len: int = 200,
        paligemma_variant: str = "gemma_2b",
        action_expert_variant: str = "gemma_300m",
        freeze_vision_encoder: bool = False,
        freeze_vlm: bool = False,
        train_expert_only: bool = False,
        forward_mode: ForwardMode = "vla",
        max_language_len: int = 50,
        language_temperature: float = 0.0,
        language_loss_weight: float = 1.0,
        action_loss_weight: float = 1.0,
        eos_token_id: int = 1,
        stop_gradient_to_vlm: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if dtype is not None:
            self.dtype = dtype
        elif precision is not None:
            self.dtype = precision
        else:
            self.dtype = "bfloat16"

        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.max_token_len = max_token_len
        self.paligemma_variant = paligemma_variant
        self.action_expert_variant = action_expert_variant
        self.freeze_vision_encoder = freeze_vision_encoder
        self.freeze_vlm = freeze_vlm
        self.train_expert_only = train_expert_only
        self.pi05 = True
        self.forward_mode = forward_mode
        self.max_language_len = max_language_len
        self.language_temperature = language_temperature
        self.language_loss_weight = language_loss_weight
        self.action_loss_weight = action_loss_weight
        self.eos_token_id = eos_token_id
        self.stop_gradient_to_vlm = stop_gradient_to_vlm

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        return cls(**config_dict, **kwargs)

    def to_dict(self):
        output = super().to_dict()
        output.update(
            {
                "dtype": self.dtype,
                "precision": self.dtype,
                "action_dim": self.action_dim,
                "action_horizon": self.action_horizon,
                "max_token_len": self.max_token_len,
                "paligemma_variant": self.paligemma_variant,
                "action_expert_variant": self.action_expert_variant,
                "freeze_vision_encoder": self.freeze_vision_encoder,
                "freeze_vlm": getattr(self, "freeze_vlm", False),
                "train_expert_only": self.train_expert_only,
                "pi05": self.pi05,
                "forward_mode": self.forward_mode,
                "max_language_len": self.max_language_len,
                "language_temperature": self.language_temperature,
                "language_loss_weight": self.language_loss_weight,
                "action_loss_weight": self.action_loss_weight,
                "eos_token_id": self.eos_token_id,
                "stop_gradient_to_vlm": self.stop_gradient_to_vlm,
            }
        )
        return output


class ValueCriticConfig(VLMBaseConfig):
    """Configuration for value critic models (V function)."""

    def __init__(
        self,
        critic_expert_variant: str = "gemma_100m",
        num_bins: int = 201,
        v_min: float = -1.0,
        v_max: float = 0.0,
        backbone_variant: str = "paligemma",
        siglip_path: str | None = None,
        gemma3_path: str | None = None,
        value_dropout: float = 0.0,
        **kwargs,
    ):
        # Accept and ignore legacy parameters for checkpoint compatibility
        super().__init__(**kwargs)
        self.critic_expert_variant = critic_expert_variant
        self.num_bins = num_bins
        self.v_min = v_min
        self.v_max = v_max
        self.backbone_variant = backbone_variant
        self.siglip_path = siglip_path or ""
        self.gemma3_path = gemma3_path or ""
        self.value_dropout = value_dropout
