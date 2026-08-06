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

"""Shared Peft LoRA adapter export and loading for SFT checkpoints.

SFT keeps ``actor/model_state_dict/full_weights.pt`` as the framework full-model
checkpoint and exports the Peft adapter separately via
``peft.PeftModel.save_pretrained`` under ``actor/lora_adapter/``. Consumers
(VLM reward models, feature-extraction scripts) load adapters explicitly with
``peft.PeftModel.from_pretrained`` through this single module; a legacy
``full_weights.pt`` fallback keeps older QwenTrend checkpoints loadable.

Division of labor with ``rlinf.models.apply_lora``:
  * ``apply_lora`` — attach/create LoRA while constructing an actor from a
    training config (``cfg.lora_path`` / ``is_lora``).
  * this module — resolve SFT checkpoint directory layouts
    (``global_step_*/actor/lora_adapter``), export beside preserved
    ``full_weights.pt``, and load those artifacts (or legacy
    ``full_weights.pt``) into frozen reward models.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import torch
from peft import PeftModel, get_peft_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

LORA_ADAPTER_DIRNAME = "lora_adapter"
ADAPTER_CONFIG_FILENAME = "adapter_config.json"

_FULL_WEIGHTS_CANDIDATES = (
    os.path.join("actor", "model_state_dict", "full_weights.pt"),
    os.path.join("model_state_dict", "full_weights.pt"),
    "full_weights.pt",
)


def find_peft_module(model: torch.nn.Module) -> torch.nn.Module | None:
    """Return the Peft-wrapped module, if any.

    Prefers a true ``PeftModel`` instance (needed for ``save_pretrained``), then
    falls back to any submodule exposing ``peft_config``.
    """
    peft_model: PeftModel | None = None
    peft_config_owner: torch.nn.Module | None = None
    for module in model.modules():
        if peft_model is None and isinstance(module, PeftModel):
            peft_model = module
        if peft_config_owner is None and hasattr(module, "peft_config"):
            peft_config_owner = module
        if peft_model is not None and peft_config_owner is not None:
            break
    return peft_model if peft_model is not None else peft_config_owner


def full_weights_path(save_path: str) -> str:
    """Resolve ``.../actor/model_state_dict/full_weights.pt`` under ``save_path``."""
    return os.path.join(save_path, "model_state_dict", "full_weights.pt")


def lora_adapter_path(save_path: str) -> str:
    """Resolve ``.../actor/lora_adapter`` under an actor ``save_path``."""
    return os.path.join(save_path, LORA_ADAPTER_DIRNAME)


def resolve_lora_adapter_dir(lora_path: str) -> str | None:
    """Resolve a checkpoint root or adapter dir to a Peft adapter directory.

    Accepts ``.../lora_adapter``, ``.../actor``, or ``.../global_step_*``.
    Returns ``None`` when no Peft ``adapter_config.json`` is present.
    """
    candidates = [
        lora_path,
        os.path.join(lora_path, LORA_ADAPTER_DIRNAME),
        os.path.join(lora_path, "actor", LORA_ADAPTER_DIRNAME),
    ]
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, ADAPTER_CONFIG_FILENAME)):
            return candidate
    return None


def _full_weights_path_under(lora_path: str) -> str | None:
    for rel in _FULL_WEIGHTS_CANDIDATES:
        path = os.path.join(lora_path, rel)
        if os.path.isfile(path):
            return path
    return None


def extract_lora_state_from_checkpoint(
    checkpoint_state: dict[str, Any],
) -> dict[str, Any]:
    """Keep ``lora_*`` tensors from a full ``full_weights.pt`` state dict."""
    return {
        key.removeprefix("module."): value
        for key, value in checkpoint_state.items()
        if "lora_" in key
    }


def _legacy_lora_config_from_state(lora_state: dict[str, Any]) -> Any:
    """Infer a ``LoraConfig`` from legacy exported adapter tensor shapes/names."""
    from peft import LoraConfig

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


def _attach_lora_from_legacy_full_weights(
    model: torch.nn.Module,
    lora_path: str,
    adapter_name: str = "default",
) -> torch.nn.Module | None:
    """Attach adapters scraped from a legacy ``full_weights.pt``.

    Returns the (possibly re-wrapped) model, or ``None`` when no legacy adapter
    is present. Callers must use the returned module — ``get_peft_model`` yields
    a new wrapper that must not be discarded.
    """
    weights_path = _full_weights_path_under(lora_path)
    if weights_path is None:
        return None
    checkpoint_state = torch.load(weights_path, map_location="cpu", weights_only=True)
    lora_state = extract_lora_state_from_checkpoint(checkpoint_state)
    if not lora_state:
        del checkpoint_state
        return None

    from peft import get_peft_model, set_peft_model_state_dict

    if adapter_name == "default":
        model = get_peft_model(model, _legacy_lora_config_from_state(lora_state))
        set_peft_model_state_dict(model, lora_state)
    else:
        if not hasattr(model, "add_adapter"):
            raise ValueError(
                f"A {adapter_name} LoRA adapter requires a primary LoRA adapter"
            )
        model.add_adapter(adapter_name, _legacy_lora_config_from_state(lora_state))
        set_peft_model_state_dict(model, lora_state, adapter_name=adapter_name)
    del checkpoint_state
    del lora_state
    return model


def load_adapter_onto_model(
    model: torch.nn.Module,
    lora_path: str,
    adapter_name: str = "default",
    is_trainable: bool = False,
) -> torch.nn.Module:
    """Attach a Peft adapter from an explicit artifact or a legacy checkpoint.

    Resolution order:
      1. ``<lora_path>[/actor]/lora_adapter`` saved via Peft ``save_pretrained``
      2. legacy ``full_weights.pt`` containing ``lora_*`` keys (backward compat)
    """
    adapter_dir = resolve_lora_adapter_dir(lora_path)
    if adapter_dir is not None:
        if adapter_name == "default":
            return PeftModel.from_pretrained(
                model, adapter_dir, is_trainable=is_trainable
            )
        if not hasattr(model, "load_adapter"):
            raise ValueError(
                f"A {adapter_name} LoRA adapter requires a primary LoRA adapter"
            )
        model.load_adapter(adapter_dir, adapter_name)
        if hasattr(model, "set_adapter"):
            model.set_adapter("default")
        return model

    legacy_model = _attach_lora_from_legacy_full_weights(
        model, lora_path, adapter_name
    )
    if legacy_model is not None:
        if adapter_name != "default" and hasattr(legacy_model, "set_adapter"):
            legacy_model.set_adapter("default")
        return legacy_model
    raise FileNotFoundError(
        "No LoRA adapter artifact or legacy lora_* full_weights.pt found under "
        f"{lora_path!r}. Expected a Peft adapter directory with "
        f"{ADAPTER_CONFIG_FILENAME}."
    )


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
    """Export the Peft adapter beside a preserved ``full_weights.pt``.

    This function never rewrites ``model_state_dict/full_weights.pt``. All ranks
    enter a single ``FSDP.summon_full_params`` so validation and
    ``save_pretrained`` share one gather; rank-0 I/O failures are caught,
    broadcast to every rank, and then raised on all ranks so no process hangs
    on a barrier after a silent rank-0 crash.

    Returns:
        True when the adapter artifact was written; False when export was skipped.
    """
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    peft_model = find_peft_module(model)
    collect_error: Exception | None = None
    adapter_dir = lora_adapter_path(save_path)

    if peft_model is not None:
        try:
            # One collective summon: validate lora_* keys and let rank 0 save.
            with FSDP.summon_full_params(model, writeback=False):
                lora_state = get_peft_model_state_dict(peft_model)
                if not any("lora_" in key for key in lora_state):
                    raise RuntimeError(
                        f"Peft export produced no lora_* keys for adapter "
                        f"directory {adapter_dir}"
                    )
                if rank == 0:
                    if not isinstance(peft_model, PeftModel):
                        raise RuntimeError(
                            "Cannot save adapter: peft owner is not a PeftModel"
                        )
                    os.makedirs(adapter_dir, exist_ok=True)
                    peft_model.save_pretrained(adapter_dir)
        except Exception as error:  # noqa: BLE001 - broadcast a uniform skip/error
            collect_error = error

    outcome: list[Any] = ["skip", None]
    if rank == 0:
        if peft_model is None:
            outcome = ["skip", "no submodule with peft_config was found"]
        elif collect_error is not None:
            outcome = [
                "error",
                f"{type(collect_error).__name__}: {collect_error}",
            ]
        else:
            outcome = ["ok", adapter_dir]

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
