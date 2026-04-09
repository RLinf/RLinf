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

"""Backbone execution utilities for the starVLA RLinf wrapper."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from . import vlm_preprocess as vlm_input_utils
from .profile import resolve_vlm_interface

_AUXILIARY_MODEL_INPUT_KEYS = {"dino_features"}
_SUPPORTED_ACTION_HEADS = {"adapter", "dual", "fast", "gr00t", "oft", "pi"}


def compute_values_from_hidden(
    *,
    value_head: Optional[nn.Module],
    hidden: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Compute value predictions from hidden states."""
    if value_head is None:
        return torch.zeros(
            (hidden.shape[0], 1), device=hidden.device, dtype=torch.float32
        )

    if attention_mask is not None:
        idx = attention_mask.long().sum(dim=1) - 1
        feat = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
    else:
        feat = hidden[:, -1]

    feat_cast = feat.to(dtype=next(value_head.parameters()).dtype)
    return value_head(feat_cast).to(dtype=torch.float32)


def run_backbone_pipeline(
    policy,
    action_head_name: Optional[str] = None,
    model_inputs: Optional[dict[str, torch.Tensor]] = None,
    examples: Optional[list[dict[str, Any]]] = None,
    use_cache: bool = False,
    strip_keys: Optional[set[str]] = None,
    input_embedding_hook: Optional[
        Callable[[Any, Any, torch.Tensor], torch.Tensor]
    ] = None,
) -> dict[str, Any]:
    """Run VLM forward and pack standardized backbone outputs."""
    starvla_model = policy.starvla_model
    vlm_interface = resolve_vlm_interface(starvla_model)
    if action_head_name is None:
        action_head_name = getattr(policy, "action_head_type", None)
    action_head_name = str(action_head_name).strip().lower()
    if action_head_name not in _SUPPORTED_ACTION_HEADS:
        raise NotImplementedError(
            "Backbone pipeline only supports action heads "
            f"{sorted(_SUPPORTED_ACTION_HEADS)}, got {action_head_name!r}."
        )

    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else nullcontext()
    )

    if model_inputs is None:
        if examples is None:
            raise ValueError(
                "run_backbone_pipeline requires either 'examples' or 'model_inputs'."
            )
        built_inputs = vlm_input_utils.build_base_vlm_inputs(
            starvla_model,
            examples=examples,
            vlm_type=policy.vlm_type,
            vlm_interface=vlm_interface,
        )
    else:
        built_inputs = dict(model_inputs)

    combined_strip_keys = set(_AUXILIARY_MODEL_INPUT_KEYS)
    if strip_keys:
        combined_strip_keys.update(strip_keys)
    vlm_inputs = dict(built_inputs)
    stripped_inputs: dict[str, torch.Tensor] = {}
    for key in combined_strip_keys:
        if key in vlm_inputs:
            stripped_inputs[key] = vlm_inputs.pop(key)

    hook_handle = None
    if input_embedding_hook is not None:
        hf_model = getattr(vlm_interface, "model", None)
        embedding_layer = None
        for candidate in (hf_model, getattr(hf_model, "model", None)):
            if candidate is None:
                continue
            get_embed = getattr(candidate, "get_input_embeddings", None)
            if not callable(get_embed):
                continue
            embedding_layer = get_embed()
            if embedding_layer is not None:
                break
        if embedding_layer is None:
            raise RuntimeError(
                "Cannot install input embedding hook: no embedding layer found on VLM model."
            )
        hook_handle = embedding_layer.register_forward_hook(input_embedding_hook)

    try:
        with autocast_ctx:
            vlm_outputs = vlm_interface(
                **vlm_inputs,
                use_cache=use_cache,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    hidden_states = getattr(vlm_outputs, "hidden_states", None)
    if hidden_states is None and isinstance(vlm_outputs, dict):
        hidden_states = vlm_outputs.get("hidden_states")
    if hidden_states is None:
        raise RuntimeError("Backbone output does not contain hidden_states.")

    if isinstance(hidden_states, torch.Tensor):
        hidden_layers = (hidden_states,)
    else:
        hidden_layers = tuple(hidden_states)
    last_hidden = hidden_layers[-1]

    if action_head_name == "adapter":
        return {
            "hidden_layers": hidden_layers,
        }

    if action_head_name == "oft":
        return {
            "last_hidden": last_hidden,
        }

    if action_head_name == "fast":
        logits = getattr(vlm_outputs, "logits", None)
        if logits is None and isinstance(vlm_outputs, dict):
            logits = vlm_outputs.get("logits")
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError(
                "FAST action head requires backbone logits, but the VLM output has none."
            )
        stripped_inputs["logits"] = logits
        return {
            "last_hidden": last_hidden,
            "attention_mask": built_inputs.get("attention_mask"),
            "model_inputs": dict(built_inputs),
            "extras": dict(stripped_inputs),
        }

    if action_head_name == "pi":
        return {
            "hidden_layers": hidden_layers,
            "attention_mask": built_inputs.get("attention_mask"),
            "model_inputs": dict(built_inputs),
        }

    if action_head_name == "gr00t":
        return {
            "last_hidden": last_hidden,
            "attention_mask": built_inputs.get("attention_mask"),
            "model_inputs": dict(built_inputs),
        }

    if action_head_name == "dual":
        return {
            "hidden_layers": hidden_layers,
            "last_hidden": last_hidden,
            "attention_mask": built_inputs.get("attention_mask"),
            "model_inputs": dict(built_inputs),
        }

    raise NotImplementedError(
        f"No backbone return layout is defined for action_head_name={action_head_name!r}."
    )
