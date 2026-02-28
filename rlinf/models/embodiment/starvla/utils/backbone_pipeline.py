# Copyright 2025 The RLinf Authors.
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
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

from . import vlm_preprocess as vlm_input_utils
from .profile import infer_vlm_type, resolve_vlm_interface

_SUPPORTED_BACKBONE_FAMILIES = {"qwen", "florence", "cosmos"}
_AUXILIARY_MODEL_INPUT_KEYS = {"dino_features"}
_VLM_FORWARD_MODE_CACHE: dict[type, int] = {}


@dataclass(frozen=True)
class BackboneOutput:
    """Standardized backbone outputs consumed by different action heads."""

    backbone_family: str
    hidden_layers: tuple[torch.Tensor, ...]
    last_hidden: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    model_inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    extras: dict[str, torch.Tensor] = field(default_factory=dict)
    capabilities: frozenset[str] = field(default_factory=frozenset)


def run_vlm_backbone(
    policy,
    *,
    use_cache: bool,
    examples: Optional[list[dict[str, Any]]] = None,
    model_inputs: Optional[dict[str, torch.Tensor]] = None,
    strip_keys: Optional[set[str]] = None,
    input_embedding_hook: Optional[
        Callable[[Any, Any, torch.Tensor], torch.Tensor]
    ] = None,
) -> dict[str, Any]:
    """Run VLM forward once and return unified hidden-state payload."""
    starvla_model = policy.starvla_model
    vlm_family = infer_vlm_type(starvla_model)
    vlm_interface = resolve_vlm_interface(starvla_model)

    if model_inputs is None:
        if examples is None:
            raise ValueError(
                "run_vlm_backbone requires either 'examples' or 'model_inputs'."
            )
        built_inputs = vlm_input_utils.build_base_vlm_inputs(
            starvla_model,
            examples=examples,
            vlm_type=vlm_family,
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

    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else nullcontext()
    )
    hook_handle = None
    if input_embedding_hook is not None:
        hf_model = getattr(vlm_interface, "model", None)
        embedding_layer = None
        if hf_model is not None:
            get_embed = getattr(hf_model, "get_input_embeddings", None)
            if callable(get_embed):
                embedding_layer = get_embed()
            if embedding_layer is None:
                inner_model = getattr(hf_model, "model", None)
                get_embed_inner = getattr(inner_model, "get_input_embeddings", None)
                if callable(get_embed_inner):
                    embedding_layer = get_embed_inner()
        if embedding_layer is None:
            raise RuntimeError(
                "Cannot install input embedding hook: no embedding layer found on VLM model."
            )
        hook_handle = embedding_layer.register_forward_hook(input_embedding_hook)

    try:
        with autocast_ctx:
            vlm_outputs = None
            last_type_error: Optional[TypeError] = None
            iface_type = type(vlm_interface)
            cached_mode = _VLM_FORWARD_MODE_CACHE.get(iface_type)
            candidate_modes: list[int] = []
            if cached_mode is not None:
                candidate_modes.append(int(cached_mode))
            for mode in (0, 1, 2):
                if mode not in candidate_modes:
                    candidate_modes.append(mode)

            for mode in candidate_modes:
                kwargs = dict(vlm_inputs)
                if mode <= 1:
                    kwargs["use_cache"] = use_cache
                if mode == 0:
                    kwargs["output_attentions"] = False
                    kwargs["output_hidden_states"] = True
                    kwargs["return_dict"] = True
                try:
                    vlm_outputs = vlm_interface(**kwargs)
                    _VLM_FORWARD_MODE_CACHE[iface_type] = mode
                    break
                except TypeError as exc:
                    text = str(exc).lower()
                    if (
                        "unexpected keyword argument" not in text
                        and "got an unexpected keyword argument" not in text
                    ):
                        raise
                    last_type_error = exc
            if vlm_outputs is None:
                if last_type_error is not None:
                    raise last_type_error
                raise RuntimeError("VLM forward failed without a captured TypeError.")
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    hidden_states = getattr(vlm_outputs, "hidden_states", None)
    if hidden_states is None and isinstance(vlm_outputs, dict):
        hidden_states = vlm_outputs.get("hidden_states")

    if hidden_states is not None:
        if isinstance(hidden_states, torch.Tensor):
            hidden_layers = (hidden_states,)
        else:
            hidden_layers = tuple(
                x for x in hidden_states if isinstance(x, torch.Tensor)
            )
            if len(hidden_layers) == 0:
                raise RuntimeError(
                    "VLM hidden_states exists but contains no tensor layers."
                )
    else:
        last_hidden = getattr(vlm_outputs, "last_hidden_state", None)
        if last_hidden is None and isinstance(vlm_outputs, dict):
            last_hidden = vlm_outputs.get("last_hidden_state")
        if isinstance(last_hidden, torch.Tensor):
            hidden_layers = (last_hidden,)
        elif isinstance(vlm_outputs, torch.Tensor):
            hidden_layers = (vlm_outputs,)
        else:
            raise RuntimeError(
                "VLM output does not provide 'hidden_states' or 'last_hidden_state'."
            )

    logits = getattr(vlm_outputs, "logits", None)
    if logits is None and isinstance(vlm_outputs, dict):
        logits = vlm_outputs.get("logits")
    if isinstance(logits, torch.Tensor):
        stripped_inputs["logits"] = logits

    return {
        "vlm_family": vlm_family,
        "hidden_layers": hidden_layers,
        "model_inputs": built_inputs,
        "attention_mask": built_inputs.get("attention_mask"),
        "extras": stripped_inputs,
    }


def run_backbone_pipeline(
    policy,
    model_inputs: Optional[dict[str, torch.Tensor]] = None,
    examples: Optional[list[dict[str, Any]]] = None,
    use_cache: bool = False,
    input_embedding_hook: Optional[
        Callable[[Any, Any, torch.Tensor], torch.Tensor]
    ] = None,
) -> BackboneOutput:
    """Validate/pack raw VLM payload into 'BackboneOutput' with capability flags."""
    backbone_payload = run_vlm_backbone(
        policy,
        model_inputs=model_inputs,
        examples=examples,
        use_cache=use_cache,
        input_embedding_hook=input_embedding_hook,
    )

    backbone_family = str(backbone_payload["vlm_family"])
    if backbone_family not in _SUPPORTED_BACKBONE_FAMILIES:
        raise NotImplementedError(
            f"Backbone family '{backbone_family}' is not supported."
        )

    hidden_layers = tuple(backbone_payload["hidden_layers"])
    if len(hidden_layers) == 0:
        raise RuntimeError("Backbone returned no hidden layers.")

    caps: set[str] = {"last_hidden"}
    if len(hidden_layers) > 1:
        caps.add("layerwise_hidden")
    caps.add(
        "single_image_input" if backbone_family == "florence" else "multi_image_input"
    )

    return BackboneOutput(
        backbone_family=backbone_family,
        hidden_layers=hidden_layers,
        last_hidden=hidden_layers[-1],
        attention_mask=backbone_payload.get("attention_mask"),
        model_inputs=dict(backbone_payload.get("model_inputs", {})),
        extras=dict(backbone_payload.get("extras", {})),
        capabilities=frozenset(caps),
    )
