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

import json
from pathlib import Path

import torch.nn as nn
from groot.vla.data.schema import DatasetMetadata
from groot.vla.data.transform import ComposedModalityTransform
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file

from rlinf.models.embodiment.dreamzero.dreamzero_policy import (
    DreamZeroConfig,
    DreamZeroPolicy,
)
from rlinf.utils.logging import get_logger


def _promote_scalar_params_to_1d(model):
    """FSDP does not support 0-d parameters, so we promote scalar Parameters to shape=[1]."""
    scalar_param_names = [name for name, p in model.named_parameters() if p.ndim == 0]
    for full_name in scalar_param_names:
        if "." in full_name:
            module_name, param_name = full_name.rsplit(".", 1)
            module = model.get_submodule(module_name)
        else:
            module = model
            param_name = full_name

        old_p = getattr(module, param_name)
        new_p = nn.Parameter(
            old_p.detach().reshape(1),
            requires_grad=old_p.requires_grad,
        )
        setattr(module, param_name, new_p)


def get_model(cfg: DictConfig, torch_dtype=None):
    """Load DreamZero policy from checkpoint."""

    model_path = Path(cfg.get("model_path"))
    if not model_path.exists():
        raise FileNotFoundError(f"DreamZero model_path does not exist: {model_path}")

    tokenizer_path = cfg.get("tokenizer_path", "google/umt5-xxl")

    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config_dict = json.load(f)

    dreamzero_config = DreamZeroConfig(**config_dict)

    st = model_path / "model.safetensors"
    st_index = model_path / "model.safetensors.index.json"
    has_full_model_weights = st.exists() or st_index.exists()

    # Disable defer_lora_injection for immediate loading
    if "config" in dreamzero_config.action_head_cfg and isinstance(
        dreamzero_config.action_head_cfg["config"], dict
    ):
        dreamzero_config.action_head_cfg["config"]["defer_lora_injection"] = False
        # If full DreamZero safetensors are absent, fall back to component loading from
        # WAN paths in config.json (diffusion_model_pretrained_path / text / image / vae).
        dreamzero_config.action_head_cfg["config"]["skip_component_loading"] = (
            has_full_model_weights
        )

    dreamzero_config.env_action_dim = cfg.get("action_dim", 7)
    dreamzero_config.gradient_checkpointing = cfg.get("gradient_checkpointing", False)

    exp_cfg_dir = model_path / "experiment_cfg"
    metadata_path = exp_cfg_dir / "metadata.json"
    with open(metadata_path, "r") as f:
        metadatas = json.load(f)

    embodiment_tag = cfg.get("embodiment_tag", "libero_sim")
    metadata = DatasetMetadata.model_validate(metadatas[embodiment_tag])

    train_cfg = OmegaConf.load(exp_cfg_dir / "conf.yaml")
    train_cfg.transforms[embodiment_tag].transforms[-1].tokenizer_path = tokenizer_path
    data_transforms = instantiate(train_cfg.transforms[embodiment_tag])
    assert isinstance(data_transforms, ComposedModalityTransform), f"{data_transforms=}"
    data_transforms.set_metadata(metadata)
    data_transforms.eval()

    dreamzero_config.data_transforms = data_transforms
    dreamzero_config.relative_action = train_cfg.get("relative_action", False)
    dreamzero_config.relative_action_per_horizon = train_cfg.get(
        "relative_action_per_horizon", False
    )
    dreamzero_config.relative_action_keys = train_cfg.get("relative_action_keys", [])

    model = DreamZeroPolicy(
        config=dreamzero_config,
    )

    # Load DreamZero full weights if available; otherwise keep component-initialized model.
    if has_full_model_weights:
        state_dict = {}
        if st_index.exists():
            with open(st_index, "r") as f:
                index = json.load(f)
            for shard_file in sorted(set(index["weight_map"].values())):
                state_dict.update(load_file(str(model_path / shard_file)))
        elif st.exists():
            state_dict.update(load_file(str(st)))
        if any(".base_layer." in k for k in state_dict):
            state_dict = {
                k.replace(".base_layer.", "."): v for k, v in state_dict.items()
            }
        model.load_state_dict(state_dict, strict=False)
    else:
        get_logger().warning(
            "No model.safetensors under %s; initializing DreamZero from component weights "
            "configured in config.json (WAN diffusion/text/image/vae paths).",
            model_path,
        )

    _promote_scalar_params_to_1d(model)
    model = model.to(dtype=torch_dtype)

    return model
