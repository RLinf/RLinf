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

"""Load Peft LoRA adapters from QwenTrend ``full_weights.pt`` checkpoints."""

from __future__ import annotations

import os
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict


def full_weights_pt_path(lora_dir: str) -> str:
    """Return ``.../actor/model_state_dict/full_weights.pt`` under a ckpt dir."""
    return os.path.join(lora_dir, "actor", "model_state_dict", "full_weights.pt")


def load_checkpoint_state(lora_dir: str) -> dict[str, Any]:
    """Load a checkpoint dict from a LoRA training output directory."""
    path = full_weights_pt_path(lora_dir)
    return torch.load(path, map_location="cpu", weights_only=True)


def extract_lora_state(checkpoint_state: dict[str, Any]) -> dict[str, Any]:
    """Keep ``lora_*`` tensors and strip an optional ``module.`` prefix."""
    return {
        key.removeprefix("module."): value
        for key, value in checkpoint_state.items()
        if "lora_" in key
    }


def lora_config_from_state(lora_state: dict[str, Any]) -> LoraConfig:
    """Infer a ``LoraConfig`` from exported adapter tensor shapes/names."""
    if not lora_state:
        raise ValueError("Cannot build LoraConfig from an empty LoRA state dict")
    lora_rank = next(
        int(value.shape[0]) for key, value in lora_state.items() if "lora_A" in key
    )
    target_modules = sorted(
        {key.split(".lora_")[0].split(".")[-1] for key in lora_state if ".lora_" in key}
    )
    return LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        lora_dropout=0.0,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )


def attach_default_lora(model: torch.nn.Module, lora_dir: str) -> torch.nn.Module:
    """Attach the default Peft adapter from ``lora_dir``, or load merged weights."""
    checkpoint_state = load_checkpoint_state(lora_dir)
    lora_state = extract_lora_state(checkpoint_state)
    if lora_state:
        model = get_peft_model(model, lora_config_from_state(lora_state))
        set_peft_model_state_dict(model, lora_state)
        return model
    stripped = {
        key.removeprefix("module."): value for key, value in checkpoint_state.items()
    }
    model.load_state_dict(stripped, strict=False)
    return model


def attach_named_lora_adapter(
    model: torch.nn.Module,
    lora_dir: str,
    adapter_name: str,
) -> str:
    """Add a named Peft adapter from ``lora_dir`` and leave ``default`` active.

    Returns:
        The adapter name that was attached.

    Raises:
        ValueError: If the checkpoint has no LoRA tensors, or the model is not
            already a Peft model that supports ``add_adapter``.
    """
    checkpoint_state = load_checkpoint_state(lora_dir)
    lora_state = extract_lora_state(checkpoint_state)
    if not lora_state:
        raise ValueError(
            f"{adapter_name} LoRA path must point to a checkpoint containing "
            "LoRA weights"
        )
    if not hasattr(model, "add_adapter"):
        raise ValueError(
            f"A {adapter_name} LoRA adapter requires a primary LoRA adapter"
        )
    model.add_adapter(adapter_name, lora_config_from_state(lora_state))
    set_peft_model_state_dict(model, lora_state, adapter_name=adapter_name)
    model.set_adapter("default")
    return adapter_name
