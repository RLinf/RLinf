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

"""PEFT LoRA loaders for frozen VLM reward models."""

from __future__ import annotations

from pathlib import Path

import torch

ADAPTER_CONFIG_FILENAME = "adapter_config.json"


def _load_lora_state_from_full_weights(path: str) -> dict[str, torch.Tensor]:
    """Load ``lora_*`` tensors from an explicit ``full_weights.pt`` file.

    Args:
        path: Full path to ``full_weights.pt``. Directories are rejected.

    Returns:
        Mapping of LoRA parameter names to tensors.

    Raises:
        FileNotFoundError: If ``path`` is not a file or contains no LoRA tensors.
    """
    weights_path = Path(path)
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Expected the full path to full_weights.pt, got {path}. "
            "Pass the file itself, for example "
            ".../actor/model_state_dict/full_weights.pt."
        )
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    lora_state = {
        key.removeprefix("module."): value
        for key, value in state.items()
        if "lora_" in key
    }
    if not lora_state:
        raise FileNotFoundError(f"{weights_path} contains no lora_* tensors")
    return lora_state


def load_lora_adapter(
    model: torch.nn.Module, path: str, adapter_name: str = "default"
) -> torch.nn.Module:
    """Load one PEFT adapter directory or an RLinf ``full_weights.pt`` LoRA dump.

    Args:
        model: Base model or an existing ``PeftModel``.
        path: Full path to ``full_weights.pt`` (typically
            ``.../actor/model_state_dict/full_weights.pt`` from VLM SFT), or a
            PEFT adapter directory that contains ``adapter_config.json``.
            Parent checkpoint directories are not searched.
        adapter_name: PEFT adapter name to attach.

    Returns:
        The model with the named adapter loaded.

    Raises:
        FileNotFoundError: If ``path`` is neither a PEFT adapter directory nor
            a ``full_weights.pt`` file.
        RuntimeError: If the checkpoint contains unexpected LoRA keys.
    """
    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        set_peft_model_state_dict,
    )

    adapter_dir = Path(path)
    if (adapter_dir / ADAPTER_CONFIG_FILENAME).is_file():
        if isinstance(model, PeftModel):
            model.load_adapter(str(adapter_dir), adapter_name=adapter_name)
            if adapter_name != "default":
                model.set_adapter("default")
            return model
        return PeftModel.from_pretrained(
            model, str(adapter_dir), adapter_name=adapter_name
        )

    if not adapter_dir.is_file():
        raise FileNotFoundError(
            f"No LoRA adapter found at {path}. Pass the full path to "
            "full_weights.pt (typically "
            ".../actor/model_state_dict/full_weights.pt from VLM SFT) or a "
            f"PEFT adapter directory that contains {ADAPTER_CONFIG_FILENAME}."
        )

    state = _load_lora_state_from_full_weights(path)
    rank = next(int(value.shape[0]) for key, value in state.items() if "lora_A" in key)
    targets = sorted(
        {key.split(".lora_")[0].split(".")[-1] for key in state if ".lora_" in key}
    )
    config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        lora_dropout=0.0,
        target_modules=targets,
        init_lora_weights="gaussian",
    )
    if isinstance(model, PeftModel):
        model.add_adapter(adapter_name, config)
    else:
        model = get_peft_model(model, config, adapter_name=adapter_name)

    state = {
        key.replace(".lora_A.default.", ".lora_A.").replace(
            ".lora_B.default.", ".lora_B."
        ): value
        for key, value in state.items()
    }
    result = set_peft_model_state_dict(model, state, adapter_name=adapter_name)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected LoRA checkpoint keys: {result.unexpected_keys}")
    if adapter_name != "default":
        model.set_adapter("default")
    return model
