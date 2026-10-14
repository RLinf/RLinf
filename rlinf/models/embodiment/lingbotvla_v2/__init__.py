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
import os
from pathlib import Path

import torch
from omegaconf import DictConfig

from rlinf.utils.logging import get_logger

_WRAPPER_KEY_PREFIX = "vla_model."
_COMPILE_KEY_PREFIX = "_orig_mod."
# An RL value head may be added on top of an SFT checkpoint that never had one.
_OPTIONAL_FULL_WEIGHTS_PREFIXES = ("value_head.",)


@torch.no_grad()
def _load_safetensors_checkpoint(model, checkpoint_dir, cfg):
    from safetensors import safe_open

    checkpoint_dir = Path(checkpoint_dir)
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as index_file:
            filenames = sorted(set(json.load(index_file)["weight_map"].values()))
    else:
        filenames = ["model.safetensors"]
    model.to(device=getattr(cfg, "init_device", "cuda"))
    parameters = dict(model.named_parameters())
    tensors = {**dict(model.named_buffers()), **parameters}
    missing = set(parameters)
    weight_loader = model.get_weight_loader()
    for filename in filenames:
        with safe_open(
            checkpoint_dir / filename, framework="pt", device="cpu"
        ) as shard:
            for checkpoint_key in shard.keys():
                normalized_key = checkpoint_key
                if normalized_key.startswith(_WRAPPER_KEY_PREFIX):
                    normalized_key = normalized_key[len(_WRAPPER_KEY_PREFIX) :]
                name = weight_loader.map_ckpt_key(
                    normalized_key, load_vlm_only=False, post_training=True
                )
                if name is None:
                    continue
                if name not in tensors:
                    raise KeyError(f"Unexpected checkpoint key: {name}")
                tensor = shard.get_tensor(checkpoint_key)
                if tensor.shape != tensors[name].shape:
                    raise ValueError(f"Checkpoint shape mismatch for {name}")
                tensors[name].copy_(tensor)
                missing.discard(name)
    if missing:
        raise KeyError(f"Missing checkpoint parameters: {sorted(missing)}")


def _load_full_weights(model, path):
    state_dict = torch.load(path, map_location="cpu")
    state_dict = {
        key.replace(_COMPILE_KEY_PREFIX, ""): value for key, value in state_dict.items()
    }
    result = model.load_state_dict(state_dict, strict=False)
    missing = [
        key
        for key in result.missing_keys
        if not key.startswith(_OPTIONAL_FULL_WEIGHTS_PREFIXES)
    ]
    if result.unexpected_keys:
        raise KeyError(
            f"Unexpected checkpoint keys in {path}: {sorted(result.unexpected_keys)}"
        )
    if missing:
        raise KeyError(f"Missing checkpoint parameters in {path}: {sorted(missing)}")


def get_model(cfg: DictConfig, torch_dtype=None):
    """Instantiate the LingBot-VLA V2 action model for RLinf."""

    from rlinf.models.embodiment.lingbotvla_v2.lingbotvla_v2_action_model import (
        LingbotVlaV2ActionModel,
    )

    if torch_dtype is None:
        torch_dtype = torch.bfloat16

    model = LingbotVlaV2ActionModel(cfg, torch_dtype=torch_dtype)

    checkpoint_dir = str(cfg.model_path)
    full_weights_path = os.path.join(
        checkpoint_dir, "model_state_dict", "full_weights.pt"
    )
    actor_full_weights_path = os.path.join(
        checkpoint_dir, "actor", "model_state_dict", "full_weights.pt"
    )

    logger = get_logger()
    if os.path.exists(full_weights_path):
        logger.info(
            f"[LingbotVLA-V2] Loading RLinf FSDP weights from {full_weights_path}"
        )
        _load_full_weights(model, full_weights_path)
    elif os.path.exists(actor_full_weights_path):
        logger.info(
            "[LingbotVLA-V2] Loading RLinf FSDP actor weights from "
            f"{actor_full_weights_path}"
        )
        _load_full_weights(model, actor_full_weights_path)
    elif any(
        os.path.isfile(os.path.join(checkpoint_dir, name))
        for name in ("model.safetensors", "model.safetensors.index.json")
    ):
        logger.info(
            "[LingbotVLA-V2] Loading official safetensors checkpoint from "
            f"{checkpoint_dir}"
        )
        _load_safetensors_checkpoint(model.vla_model, checkpoint_dir, cfg)
    else:
        raise FileNotFoundError(
            f"No LingBot-VLA V2 checkpoint found under {checkpoint_dir}: expected "
            "model_state_dict/full_weights.pt, "
            "actor/model_state_dict/full_weights.pt, or model.safetensors[.index.json]."
        )

    return model
