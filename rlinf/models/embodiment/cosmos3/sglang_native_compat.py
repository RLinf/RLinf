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

"""Preserve native Cosmos action captions in SGLang's external-package hook."""

from __future__ import annotations

import functools
import importlib
import json
import math
import os
import sys
from collections.abc import Callable
from typing import Any

_HOOK_MODULE = "rlinf.models.embodiment.cosmos3.sglang_native_compat"
_ACTION_MODULE = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages."
    "model_specific_stages.cosmos3_action"
)
_STAGE_MODULE = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3"
)


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def is_native_action_prompt(value: Any) -> bool:
    """Return whether text contains a complete native action-caption object."""
    if not isinstance(value, str):
        return False
    try:
        prompt = json.loads(value)
    except ValueError:
        return False
    if not isinstance(prompt, dict):
        return False
    framing = prompt.get("cinematography")
    if not isinstance(framing, dict) or not _nonempty_text(framing.get("framing")):
        return False
    actions = prompt.get("actions")
    if not isinstance(actions, list) or not actions:
        return False
    for action in actions:
        if not isinstance(action, dict) or not all(
            _nonempty_text(action.get(key)) for key in ("time", "description")
        ):
            return False
        if "idle_frame" in action and not _nonempty_text(action["idle_frame"]):
            return False
    if not all(_nonempty_text(prompt.get(key)) for key in ("duration", "aspect_ratio")):
        return False
    fps = prompt.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        return False
    try:
        valid_fps = math.isfinite(fps) and fps > 0
    except OverflowError:
        return False
    if not valid_fps:
        return False
    resolution = prompt.get("resolution")
    if not isinstance(resolution, dict):
        return False
    return all(
        type(resolution.get(key)) is int and resolution[key] > 0 for key in ("H", "W")
    )


def make_native_prompt_builder(original: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap SGLang's formatter while preserving its plain-text behavior."""
    if getattr(original, "_rlinf_native_action_prompt", False) is True:
        return original

    @functools.wraps(original)
    def build_action_prompt(description: Any, *args: Any, **kwargs: Any) -> Any:
        if is_native_action_prompt(description):
            return description
        if isinstance(description, (list, tuple)):
            if any(is_native_action_prompt(item) for item in description):
                return [
                    item
                    if is_native_action_prompt(item)
                    else original(item, *args, **kwargs)
                    for item in description
                ]
        return original(description, *args, **kwargs)

    build_action_prompt._rlinf_native_action_prompt = True
    return build_action_prompt


def install_native_prompt_compat() -> None:
    """Install the caption wrapper in this SGLang server process once."""
    action_module = importlib.import_module(_ACTION_MODULE)
    builder = make_native_prompt_builder(action_module.build_action_prompt)
    action_module.build_action_prompt = builder
    # Registry discovery may have imported the stage's function binding already.
    stage_module = sys.modules.get(_STAGE_MODULE)
    if stage_module is not None:
        stage_module.build_action_prompt = builder

    sampling_module = importlib.import_module(
        "sglang.multimodal_gen.configs.sample.cosmos3"
    )
    sampling_class = sampling_module.Cosmos3SamplingParams
    original_adjust = sampling_class._adjust
    if getattr(original_adjust, "_rlinf_native_action_prompt", False) is True:
        return

    @functools.wraps(original_adjust)
    def adjust(params: Any, server_args: Any) -> Any:
        prompts = params.prompt
        if isinstance(prompts, str):
            prompts = [prompts]
        if (
            params.action_mode == "policy"
            and isinstance(prompts, (list, tuple))
            and prompts
            and all(is_native_action_prompt(prompt) for prompt in prompts)
        ):
            # The action endpoint loses explicit-field tracking before _adjust.
            params._explicit_fields = set(
                getattr(params, "_explicit_fields", None) or ()
            ) | {"guidance_scale"}
        return original_adjust(params, server_args)

    adjust._rlinf_native_action_prompt = True
    sampling_class._adjust = adjust


if os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE") == _HOOK_MODULE:
    install_native_prompt_compat()
