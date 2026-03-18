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
ValueCriticModel factory for RLinf.
"""

import glob
import logging
import os

import safetensors
import torch
from omegaconf import DictConfig

from .configuration import ValueCriticConfig
from .modeling_critic import CriticOutput, ValueCritic, ValueCriticModel

logger = logging.getLogger(__name__)


def get_value_model(cfg: DictConfig, torch_dtype=None) -> ValueCriticModel:
    """Build a ValueCriticModel.

    Args:
        cfg: Hydra model config. Expected keys:
            - critic_expert_variant: e.g. "gemma_100m", "gemma_300m"
            - num_bins, v_min, v_max
            - action_dim, action_horizon, max_token_len
            - paligemma_variant, freeze_vision_encoder, train_expert_only
            - model_path: checkpoint path (optional)
        torch_dtype: unused, kept for interface compat.

    Returns:
        ValueCriticModel instance.
    """
    # Collect VLMBaseConfig kwargs
    vlm_kwargs = {}

    def _set(key, default=None):
        val = getattr(cfg, key, default)
        if val is not None:
            vlm_kwargs[key] = val

    _set("action_dim", 32)
    _set("action_horizon", 50)
    _set("max_token_len", 200)
    _set("paligemma_variant", "gemma_2b")
    _set("action_expert_variant", "gemma_300m")
    _set("freeze_vision_encoder", False)
    _set("freeze_vlm", False)
    _set("train_expert_only", False)
    _set("forward_mode", "vla")
    _set("max_language_len", 50)
    _set("stop_gradient_to_vlm", False)
    _set("backbone_variant", "paligemma")
    _set("siglip_path", None)
    _set("gemma3_path", None)

    # Handle precision / dtype
    precision = getattr(cfg, "precision", "bf16")
    if precision in ("bf16", "bf16-mixed"):
        vlm_kwargs["dtype"] = "bfloat16"
    elif precision in ("fp16", "16", "16-mixed"):
        vlm_kwargs["dtype"] = "float16"
    else:
        vlm_kwargs["dtype"] = "float32"

    # Critic-specific kwargs
    critic_kwargs = {
        "critic_expert_variant": getattr(cfg, "critic_expert_variant", "gemma_100m"),
        "num_bins": getattr(cfg, "num_bins", 201),
        "v_min": getattr(cfg, "v_min", -1.0),
        "v_max": getattr(cfg, "v_max", 0.0),
        "value_dropout": getattr(cfg, "value_dropout", 0.0),
    }

    config = ValueCriticConfig(**critic_kwargs, **vlm_kwargs)

    # Build model
    model = ValueCriticModel(config)
    logger.info("Created ValueCriticModel (V function)")

    # Load checkpoint if provided
    model_path = getattr(cfg, "model_path", None)
    backbone_variant = getattr(cfg, "backbone_variant", "paligemma")

    # Resolve model_path to the best available weights file/dir
    if model_path is not None:
        full_weights_path = os.path.join(
            model_path, "model_state_dict", "full_weights.pt"
        )
        actor_full_weights_path = os.path.join(
            model_path, "actor", "model_state_dict", "full_weights.pt"
        )
        if os.path.exists(full_weights_path):
            model_path = full_weights_path
        elif os.path.exists(actor_full_weights_path):
            model_path = actor_full_weights_path
    if backbone_variant == "paligemma":
        # PaliGemma: all weights come from a single checkpoint (e.g. PI05)
        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        state_dict = _load_state_dict(model_path)
        if state_dict:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            logger.info(
                "Loaded checkpoint from %s (missing=%d, unexpected=%d)",
                model_path,
                len(missing),
                len(unexpected),
            )
    else:
        # SigLIP-Gemma3: backbone weights already loaded via from_pretrained().
        # model_path is only used for resuming from a fine-tuned checkpoint.
        if model_path and os.path.exists(model_path):
            state_dict = _load_state_dict(model_path)
            if state_dict:
                missing, unexpected = model.load_state_dict(
                    state_dict, strict=False
                )
                logger.info(
                    "Loaded fine-tuned checkpoint from %s (missing=%d, unexpected=%d)",
                    model_path,
                    len(missing),
                    len(unexpected),
                )
        else:
            logger.info(
                "No model_path provided for %s backbone; "
                "using from_pretrained() weights.",
                backbone_variant,
            )

    return model


def _load_state_dict(path: str) -> dict:
    """Load state dict from .safetensors, .pt/.pth, or directory."""
    if path.endswith(".safetensors"):
        return safetensors.torch.load_file(path, device="cpu")
    elif path.endswith((".pt", ".pth")):
        return torch.load(path, map_location="cpu", weights_only=False)
    elif os.path.isdir(path):
        weight_paths = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if not weight_paths:
            weight_paths = sorted(glob.glob(os.path.join(path, "*.pt")))
        sd = {}
        for wp in weight_paths:
            if wp.endswith(".safetensors"):
                sd.update(safetensors.torch.load_file(wp, device="cpu"))
            else:
                sd.update(torch.load(wp, map_location="cpu", weights_only=False))
        return sd
    return {}


__all__ = [
    "get_value_model",
    "ValueCriticModel",
    "ValueCritic",
    "ValueCriticConfig",
    "CriticOutput",
]
