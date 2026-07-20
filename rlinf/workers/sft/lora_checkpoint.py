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

"""Helpers to export Peft LoRA adapters into FSDP ``full_weights.pt`` files."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import torch
from peft import get_peft_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def find_peft_module(model: torch.nn.Module) -> torch.nn.Module | None:
    """Return the first submodule that exposes ``peft_config``, if any."""
    for module in model.modules():
        if hasattr(module, "peft_config"):
            return module
    return None


def export_lora_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Collect CPU LoRA adapter tensors from an FSDP-wrapped Peft model."""
    peft_model = find_peft_module(model)
    if peft_model is None:
        raise ValueError(
            "Cannot export LoRA adapters: no submodule with peft_config was found"
        )
    with FSDP.summon_full_params(model, writeback=False):
        lora_state = get_peft_model_state_dict(peft_model)
        return {key: value.detach().cpu() for key, value in lora_state.items()}


def full_weights_path(save_path: str) -> str:
    """Resolve ``.../actor/model_state_dict/full_weights.pt`` under ``save_path``."""
    return os.path.join(save_path, "model_state_dict", "full_weights.pt")


def rewrite_full_weights_with_lora_adapters(
    model: torch.nn.Module,
    save_path: str,
    *,
    rank: int,
    log_info: Callable[[str], Any] | None = None,
    log_warning: Callable[[str], Any] | None = None,
) -> bool:
    """Replace merged ``full_weights.pt`` with Peft ``lora_*`` adapter tensors.

    FSDP full-state export merges adapters into base weights and drops ``lora_*``
    keys. Online ``HistoryVLMRewardModel`` loads those keys from
    ``success_lora_path`` / ``lora_path``.

    Returns:
        True when rank 0 wrote a rewritten checkpoint; False when skipped.
    """
    target = full_weights_path(save_path)
    torch.distributed.barrier()
    if not os.path.exists(target):
        if rank == 0 and log_warning is not None:
            log_warning(f"Skip LoRA rewrite: full_weights.pt not found at {target}")
        torch.distributed.barrier()
        return False

    try:
        lora_state = export_lora_state_dict(model)
    except ValueError as error:
        if rank == 0 and log_warning is not None:
            log_warning(f"Skip LoRA rewrite at {save_path}: {error}")
        torch.distributed.barrier()
        return False

    torch.distributed.barrier()
    wrote = False
    if rank == 0:
        if not any("lora_" in key for key in lora_state):
            raise RuntimeError(
                f"Peft export produced no lora_* keys for checkpoint {save_path}"
            )
        torch.save(lora_state, target)
        wrote = True
        if log_info is not None:
            log_info(f"Rewrote {target} with {len(lora_state)} LoRA adapter tensors")
    torch.distributed.barrier()
    return wrote
