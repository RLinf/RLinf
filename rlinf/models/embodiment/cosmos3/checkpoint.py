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

"""Load native Cosmos base weights and RLinf inference checkpoints."""

from __future__ import annotations

import os
import re

import torch
import torch.nn as nn

from rlinf.models.embodiment.cosmos3.cosmos3_policy import Cosmos3Policy
from rlinf.utils.logging import get_logger

logger = get_logger()


def _load_base_weights(
    cosmos3_model: nn.Module, model_path: str, keys_to_skip: list[str]
) -> None:
    """Warm-start ``cosmos3_model`` from a base checkpoint, skipping action heads + net_ema.

    Reuses cosmos's loaders so the AC prefix and DTensor resharding are handled:
    ``load_vfm_model`` (HF safetensors dir, regex skip on ``cosmos3_model.net``) or
    ``_load_model`` (DCP dir, substring skip via ``CustomLoadPlanner``).
    """
    from cosmos_framework.utils.generator.model_loader import (
        _is_safetensors_checkpoint,
        _load_model,
    )

    if _is_safetensors_checkpoint(model_path, None):
        from cosmos_framework.model.generator.utils.safetensors_loader import (
            load_vfm_model,
        )

        skip_patterns = [f".*{re.escape(s)}.*" for s in keys_to_skip]
        vfm = getattr(cosmos3_model, "net", cosmos3_model)
        logger.info(
            "Cosmos3: loading safetensors base into cosmos3_model.net from %s (skip=%s)",
            model_path,
            skip_patterns,
        )
        load_vfm_model(
            vfm,
            model_path,
            credential_path=None,
            parallel_dims=getattr(cosmos3_model, "parallel_dims", None),
            skip_patterns=skip_patterns or None,
        )
        return

    # DCP dir; accept either model_path or model_path/model layouts.
    dcp_dir = (
        os.path.join(model_path, "model")
        if os.path.isdir(os.path.join(model_path, "model"))
        else model_path
    )
    logger.info(
        "Cosmos3: loading DCP base into cosmos3_model from %s (skip=%s)",
        dcp_dir,
        keys_to_skip,
    )
    _load_model(
        cosmos3_model,
        dcp_dir,
        credential_path=None,
        keys_to_skip_loading=keys_to_skip or None,
    )


def load_inference_checkpoint(model: Cosmos3Policy, checkpoint_path: str) -> None:
    """Load RLinf weights, excluding EMA state when the policy disables EMA."""
    state_dict = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True, mmap=True
    )
    ema_keys = []
    if not model._ema_enabled:
        ema_keys = [key for key in state_dict if key.startswith("omni.net_ema.")]
        for key in ema_keys:
            del state_dict[key]
    model.load_state_dict(state_dict, strict=True)
    logger.info(
        "Cosmos3: loaded inference checkpoint %s (excluded EMA keys=%d)",
        checkpoint_path,
        len(ema_keys),
    )
