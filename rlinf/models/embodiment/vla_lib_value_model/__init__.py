# Copyright 2025 The RLinf Authors.
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
VLA-Lib ValueCriticModel factory for RLinf.
"""

import glob
import logging
import os

import safetensors
import torch
from omegaconf import DictConfig

from .modeling_critic import CriticOutput, PI05CriticConfig, ValueCriticModel
from .value_policy import ValuePolicy
from .value_policy_config import create_trained_value_policy

logger = logging.getLogger(__name__)


def get_vla_lib_value_model(cfg: DictConfig, torch_dtype=None) -> ValueCriticModel:
    """Build a ValueCriticModel.

    Args:
        cfg: Hydra model config. Expected keys:
            - critic_forward_mode: "vlm" | "expert" | "dual"
            - expert_loss_type: "mse" | "categorical" | "distributional"
            - critic_expert_variant: e.g. "gemma_100m", "gemma_300m"
            - num_bins, v_min, v_max
            - vlm_loss_weight, expert_loss_weight
            - action_dim, action_horizon, max_token_len
            - paligemma_variant, freeze_vision_encoder, train_expert_only
            - model_path: checkpoint path (optional)
        torch_dtype: unused, kept for interface compat.

    Returns:
        ValueCriticModel instance.
    """
    # Collect PI05Config kwargs
    pi05_kwargs = {}

    def _set(key, default=None):
        val = getattr(cfg, key, default)
        if val is not None:
            pi05_kwargs[key] = val

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
    _set("discrete_state_input", False)
    _set("exclude_cot_from_kv_cache", False)

    # Handle precision / dtype
    precision = getattr(cfg, "precision", "bf16")
    if precision in ("bf16", "bf16-mixed"):
        pi05_kwargs["dtype"] = "bfloat16"
    elif precision in ("fp16", "16", "16-mixed"):
        pi05_kwargs["dtype"] = "float16"
    else:
        pi05_kwargs["dtype"] = "float32"

    # Critic-specific kwargs
    critic_kwargs = {
        "critic_expert_variant": getattr(cfg, "critic_expert_variant", "gemma_100m"),
        "critic_forward_mode": getattr(cfg, "critic_forward_mode", "expert"),
        "expert_loss_type": getattr(cfg, "expert_loss_type", "mse"),
        "num_bins": getattr(cfg, "num_bins", 201),
        "v_min": getattr(cfg, "v_min", -1.0),
        "v_max": getattr(cfg, "v_max", 0.0),
        "vlm_loss_weight": getattr(cfg, "vlm_loss_weight", 1.0),
        "expert_loss_weight": getattr(cfg, "expert_loss_weight", 1.0),
    }

    config = PI05CriticConfig(**critic_kwargs, **pi05_kwargs)

    # Build model
    model = ValueCriticModel(config)
    logger.info("Created ValueCriticModel (V function)")

    # Load checkpoint if provided
    model_path = getattr(cfg, "model_path", None)
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

    return model


def _load_state_dict(path: str) -> dict:
    """Load state dict from .safetensors, .pt/.pth, or directory."""
    if path.endswith(".safetensors"):
        return safetensors.torch.load_file(path, device="cpu")
    elif path.endswith((".pt", ".pth")):
        return torch.load(path, map_location="cpu")
    elif os.path.isdir(path):
        weight_paths = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if not weight_paths:
            weight_paths = sorted(glob.glob(os.path.join(path, "*.pt")))
        sd = {}
        for wp in weight_paths:
            if wp.endswith(".safetensors"):
                sd.update(safetensors.torch.load_file(wp, device="cpu"))
            else:
                sd.update(torch.load(wp, map_location="cpu"))
        return sd
    return {}


__all__ = [
    "get_vla_lib_value_model",
    "ValueCriticModel",
    "PI05CriticConfig",
    "CriticOutput",
    "ValuePolicy",
    "create_trained_value_policy",
]
