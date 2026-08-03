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

"""Load Peft LoRA adapters from explicit QwenTrend adapter artifacts."""

from __future__ import annotations

import os
from typing import Any

import torch
from peft import LoraConfig, PeftConfig, get_peft_model, set_peft_model_state_dict

from rlinf.workers.sft.lora_checkpoint import (
    ADAPTER_CONFIG_FILENAME,
    ADAPTER_WEIGHTS_FILENAME,
    LORA_ADAPTER_DIRNAME,
)

_ADAPTER_WEIGHT_CANDIDATES = (
    ADAPTER_WEIGHTS_FILENAME,
    "adapter_model.pt",
    "adapter_model.safetensors",
)


def resolve_lora_adapter_dir(lora_path: str) -> str:
    """Resolve a checkpoint root or adapter directory to the adapter artifact.

    Accepts:
      * a Peft adapter directory containing ``adapter_config.json``
      * ``.../actor`` (looks for ``lora_adapter/``)
      * ``.../global_step_*`` (looks for ``actor/lora_adapter/``)
    """
    candidates = [
        lora_path,
        os.path.join(lora_path, LORA_ADAPTER_DIRNAME),
        os.path.join(lora_path, "actor", LORA_ADAPTER_DIRNAME),
    ]
    for candidate in candidates:
        config_path = os.path.join(candidate, ADAPTER_CONFIG_FILENAME)
        if os.path.isfile(config_path):
            return candidate
    raise FileNotFoundError(
        "No LoRA adapter artifact "
        f"({ADAPTER_CONFIG_FILENAME}) found under {lora_path!r}. "
        f"Looked in: {', '.join(candidates)}"
    )


def load_adapter_weights(adapter_dir: str) -> dict[str, Any]:
    """Load adapter tensors from ``adapter_model.bin`` / ``.pt`` / ``.safetensors``."""
    for filename in _ADAPTER_WEIGHT_CANDIDATES:
        path = os.path.join(adapter_dir, filename)
        if not os.path.isfile(path):
            continue
        if filename.endswith(".safetensors"):
            from safetensors.torch import load_file

            return load_file(path)
        return torch.load(path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(
        f"No adapter weights found under {adapter_dir!r}. "
        f"Expected one of: {', '.join(_ADAPTER_WEIGHT_CANDIDATES)}"
    )


def load_lora_adapter_artifacts(
    adapter_dir: str,
) -> tuple[dict[str, Any], LoraConfig]:
    """Load adapter weights and the accompanying Peft ``LoraConfig``."""
    peft_config = PeftConfig.from_pretrained(adapter_dir)
    if not isinstance(peft_config, LoraConfig):
        raise TypeError(
            f"Expected a LoraConfig at {adapter_dir!r}, got {type(peft_config)!r}"
        )
    raw_state = load_adapter_weights(adapter_dir)
    lora_state = {
        key.removeprefix("module."): value for key, value in raw_state.items()
    }
    if not any("lora_" in key for key in lora_state):
        raise ValueError(
            f"Adapter artifact at {adapter_dir!r} does not contain lora_* tensors"
        )
    return lora_state, peft_config


def attach_default_lora(model: torch.nn.Module, lora_path: str) -> torch.nn.Module:
    """Attach the default Peft adapter from an explicit adapter artifact."""
    adapter_dir = resolve_lora_adapter_dir(lora_path)
    lora_state, lora_config = load_lora_adapter_artifacts(adapter_dir)
    model = get_peft_model(model, lora_config)
    set_peft_model_state_dict(model, lora_state)
    return model


def attach_named_lora_adapter(
    model: torch.nn.Module,
    lora_path: str,
    adapter_name: str,
) -> str:
    """Add a named Peft adapter from an explicit adapter artifact.

    Returns:
        The adapter name that was attached.

    Raises:
        ValueError: If the model is not already a Peft model that supports
            ``add_adapter``.
    """
    if not hasattr(model, "add_adapter"):
        raise ValueError(
            f"A {adapter_name} LoRA adapter requires a primary LoRA adapter"
        )
    adapter_dir = resolve_lora_adapter_dir(lora_path)
    lora_state, lora_config = load_lora_adapter_artifacts(adapter_dir)
    model.add_adapter(adapter_name, lora_config)
    set_peft_model_state_dict(model, lora_state, adapter_name=adapter_name)
    model.set_adapter("default")
    return adapter_name
