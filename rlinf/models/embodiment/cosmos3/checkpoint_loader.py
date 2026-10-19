# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load pretrained Cosmos3 weights into the trainable native action model."""

from __future__ import annotations

import os
import re
from pathlib import Path

import torch.distributed.checkpoint as dcp
from torch import nn


def load_pretrained_checkpoint(
    model: nn.Module,
    model_path: str,
    checkpoint_format: str,
    keys_to_skip_loading: list[str],
) -> None:
    """Load a native DCP/VFM checkpoint or a complete Diffusers package.

    ``diffusers`` is a full policy load, including trained action heads. Native
    checkpoints retain the existing warm-start behavior, which skips the keys
    configured for the base model before an RLinf SFT state dict is overlaid.
    """
    checkpoint_format = checkpoint_format.lower()
    if checkpoint_format == "diffusers":
        _load_diffusers_checkpoint(model, Path(model_path).expanduser())
    elif checkpoint_format == "native":
        _load_native_checkpoint(model, model_path, keys_to_skip_loading)
    else:
        raise ValueError(
            f"Unsupported Cosmos3 checkpoint_format={checkpoint_format!r}; "
            "expected 'native' or 'diffusers'."
        )


def _load_diffusers_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    """Use cosmos-framework's checked Diffusers → native tensor mapping."""
    from cosmos_framework.inference.model import (
        _diffusers_weight_map,
        _DiffusersHuggingFaceStorageReader,
        _DiffusersLoadPlanner,
        _is_diffusers_checkpoint,
    )

    if not _is_diffusers_checkpoint(checkpoint_path):
        raise ValueError(
            f"Cosmos3 Diffusers checkpoint is missing model_index.json and "
            f"indexed transformer shards: {checkpoint_path}"
        )
    net = getattr(model, "net", None)
    if net is None:
        raise TypeError("Cosmos3 Diffusers loading requires a model.net network")

    mapped_keys = {
        key.removeprefix("transformer.")
        for key in _diffusers_weight_map(checkpoint_path)
    }
    required_action_keys = {
        "action_modality_embed",
        "action_proj_in.fc.weight",
        "action_proj_out.fc.weight",
    }
    missing_action_keys = required_action_keys - mapped_keys
    if missing_action_keys:
        raise ValueError(
            "Cosmos3 actor checkpoint is missing trained action weights: "
            + ", ".join(sorted(missing_action_keys))
        )

    state_dict = net.state_dict()
    planner = _DiffusersLoadPlanner(checkpoint_path)
    target_state = planner._normalize_target_state_dict(state_dict)
    _, loaded_keys = planner._build_remapped_state_dict(target_state)
    missing_keys = set(target_state) - loaded_keys
    if missing_keys:
        raise ValueError(
            "Cosmos3 full checkpoint must cover every model tensor because random "
            f"initialization is skipped; missing: {sorted(missing_keys)[:10]}"
        )

    # The framework's public from_pretrained_dcp builds a second, full model.
    # Load directly into the native model already constructed for RLinf FSDP.
    # Each rank loads its local CPU replica before FSDP sharding, so no_dist
    # also avoids pickling the framework's regex-based custom load planner.
    dcp.load(
        state_dict=state_dict,
        storage_reader=_DiffusersHuggingFaceStorageReader(checkpoint_path),
        planner=planner,
        no_dist=True,
    )
    net_ema = getattr(model, "net_ema", None)
    if net_ema is not None:
        net_ema.load_state_dict(state_dict)


def _load_native_checkpoint(
    model: nn.Module, model_path: str, keys_to_skip_loading: list[str]
) -> None:
    """Warm-start the native model from Cosmos VFM safetensors or DCP."""
    from cosmos_framework.utils.generator.model_loader import (
        _is_safetensors_checkpoint,
        _load_model,
    )

    if _is_safetensors_checkpoint(model_path, None):
        from cosmos_framework.model.generator.utils.safetensors_loader import (
            load_vfm_model,
        )

        skip_patterns = [f".*{re.escape(key)}.*" for key in keys_to_skip_loading]
        net = getattr(model, "net", model)
        load_vfm_model(
            net,
            model_path,
            credential_path=None,
            parallel_dims=getattr(model, "parallel_dims", None),
            skip_patterns=skip_patterns or None,
        )
        return

    dcp_dir = (
        os.path.join(model_path, "model")
        if os.path.isdir(os.path.join(model_path, "model"))
        else model_path
    )
    _load_model(
        model,
        dcp_dir,
        credential_path=None,
        keys_to_skip_loading=keys_to_skip_loading or None,
    )
