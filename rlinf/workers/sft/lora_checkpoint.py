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

"""Export Peft LoRA adapters beside preserved FSDP ``full_weights.pt`` files."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

import torch
from peft import get_peft_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

LORA_ADAPTER_DIRNAME = "lora_adapter"
ADAPTER_WEIGHTS_FILENAME = "adapter_model.bin"
ADAPTER_CONFIG_FILENAME = "adapter_config.json"


def find_peft_module(model: torch.nn.Module) -> torch.nn.Module | None:
    """Return the first submodule that exposes ``peft_config``, if any."""
    for module in model.modules():
        if hasattr(module, "peft_config"):
            return module
    return None


def full_weights_path(save_path: str) -> str:
    """Resolve ``.../actor/model_state_dict/full_weights.pt`` under ``save_path``."""
    return os.path.join(save_path, "model_state_dict", "full_weights.pt")


def lora_adapter_path(save_path: str) -> str:
    """Resolve ``.../actor/lora_adapter`` under an actor ``save_path``."""
    return os.path.join(save_path, LORA_ADAPTER_DIRNAME)


def _jsonable(value: Any) -> Any:
    """Convert Peft config values into JSON-serializable objects."""
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def peft_config_to_jsonable(peft_config: Any) -> dict[str, Any]:
    """Serialize a Peft config (e.g. ``LoraConfig``) for ``adapter_config.json``."""
    return _jsonable(peft_config.to_dict())


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


def get_active_peft_config(peft_model: torch.nn.Module) -> Any:
    """Return the active adapter's Peft config from a Peft-wrapped module."""
    adapter_name = getattr(peft_model, "active_adapter", "default")
    if isinstance(adapter_name, (list, tuple)):
        adapter_name = adapter_name[0]
    return peft_model.peft_config[adapter_name]


def save_lora_adapter_files(
    adapter_dir: str,
    lora_state: dict[str, torch.Tensor],
    peft_config: Any,
) -> None:
    """Write ``adapter_model.bin`` and ``adapter_config.json`` under ``adapter_dir``.

    Raises:
        RuntimeError: If ``lora_state`` contains no ``lora_*`` tensors.
    """
    if not any("lora_" in key for key in lora_state):
        raise RuntimeError(
            f"Peft export produced no lora_* keys for adapter directory {adapter_dir}"
        )
    os.makedirs(adapter_dir, exist_ok=True)
    weights_path = os.path.join(adapter_dir, ADAPTER_WEIGHTS_FILENAME)
    config_path = os.path.join(adapter_dir, ADAPTER_CONFIG_FILENAME)
    torch.save(lora_state, weights_path)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(
            peft_config_to_jsonable(peft_config), handle, indent=2, sort_keys=True
        )
        handle.write("\n")


def _broadcast_rank0_outcome(
    outcome: list[Any],
    *,
    rank: int,
) -> list[Any]:
    """Broadcast ``[kind, detail]`` from rank 0 and return the shared outcome.

    ``kind`` is one of ``"ok"``, ``"skip"``, or ``"error"``.
    """
    if torch.distributed.is_initialized():
        if rank != 0:
            outcome = [None, None]
        torch.distributed.broadcast_object_list(outcome, src=0)
        torch.distributed.barrier()
    return outcome


def export_lora_adapter(
    model: torch.nn.Module,
    save_path: str,
    *,
    rank: int,
    log_info: Callable[[str], Any] | None = None,
    log_warning: Callable[[str], Any] | None = None,
) -> bool:
    """Export Peft LoRA adapters next to a preserved ``full_weights.pt``.

    FSDP full-state export merges adapters into base weights and drops ``lora_*``
    keys. Online ``HistoryVLMRewardModel`` loads the separate adapter artifact
    from ``success_lora_path`` / ``lora_path``. This function never rewrites
    ``model_state_dict/full_weights.pt``.

    Rank-0 I/O failures are caught, broadcast to every rank, and then raised on
    all ranks so no process hangs on a barrier after a silent rank-0 crash.

    Returns:
        True when the adapter artifact was written; False when export was skipped.
    """
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    peft_model = find_peft_module(model)
    lora_state: dict[str, torch.Tensor] | None = None
    peft_config: Any | None = None
    collect_error: Exception | None = None

    if peft_model is not None:
        try:
            lora_state = export_lora_state_dict(model)
            peft_config = get_active_peft_config(peft_model)
        except Exception as error:  # noqa: BLE001 - broadcast a uniform skip/error
            collect_error = error

    outcome: list[Any] = ["skip", None]
    if rank == 0:
        try:
            if peft_model is None:
                outcome = ["skip", "no submodule with peft_config was found"]
            elif collect_error is not None:
                outcome = [
                    "error",
                    f"{type(collect_error).__name__}: {collect_error}",
                ]
            elif lora_state is None or peft_config is None:
                outcome = ["skip", "LoRA state or Peft config was not collected"]
            else:
                adapter_dir = lora_adapter_path(save_path)
                save_lora_adapter_files(adapter_dir, lora_state, peft_config)
                outcome = ["ok", adapter_dir]
        except Exception as error:  # noqa: BLE001 - must not crash before broadcast
            outcome = ["error", f"{type(error).__name__}: {error}"]

    outcome = _broadcast_rank0_outcome(outcome, rank=rank)
    kind, detail = outcome

    if kind == "error":
        raise RuntimeError(f"LoRA adapter export failed on rank 0: {detail}")
    if kind == "skip":
        if rank == 0 and log_warning is not None:
            log_warning(f"Skip LoRA adapter export at {save_path}: {detail}")
        return False
    if rank == 0 and log_info is not None:
        log_info(f"Exported LoRA adapter to {detail}")
    return True
