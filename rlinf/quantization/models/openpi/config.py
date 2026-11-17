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

"""OpenPI pi0.5 configuration checks without importing models or execution backends."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlinf.quantization.algorithms.correction import CorrectionSpec


def validate_pi05_model_config(
    model: Mapping, correction: CorrectionSpec, scope: str
) -> Mapping[str, object]:
    """Validate pi0.5 RL events and return effective sampling and preprocessing settings."""
    if model.get("pi05", True) is not True:
        raise ValueError(f"quantization requires {scope}.pi05=True")
    openpi = model.get("openpi") or {}
    if openpi.get("task") != "rl":
        raise ValueError(f"quantization requires {scope}.openpi.task='rl'")
    # OpenPI defaults to internal compilation unless explicitly disabled.
    if openpi.get("torch_compile") is not False:
        raise ValueError(f"quantization requires {scope}.openpi.torch_compile=False")
    for field in (
        "is_nft",
        "joint_logprob",
        "rlt",
        "use_rlt",
        "short_memory",
        "pytorch_numerics",
    ):
        if openpi.get(field):
            raise ValueError(f"quantization does not support {scope}.openpi.{field}")
    if openpi.get("noise_method", "flow_ode") not in {
        "flow_ode",
        "flow_sde",
        "flow_cps",
    }:
        raise ValueError(
            f"quantization does not support {scope}.openpi.noise_method with trainable sigma"
        )
    if correction.mode == "importance_sampling":
        if openpi.get("noise_method") != "flow_sde":
            raise ValueError(
                f"importance_sampling requires {scope}.openpi.noise_method='flow_sde'"
            )
        level = openpi.get("noise_level", 0.0)
        if (
            isinstance(level, bool)
            or not isinstance(level, (int, float))
            or not math.isfinite(level)
            or level <= 0
        ):
            raise ValueError(
                f"importance_sampling requires positive finite {scope}.openpi.noise_level"
            )
        if openpi.get("safe_get_logprob"):
            raise ValueError(
                f"importance_sampling does not support {scope}.openpi.safe_get_logprob"
            )
    config_name = openpi.get("config_name")
    if not isinstance(config_name, str) or not config_name:
        raise ValueError(f"quantization requires a nonempty {scope}.openpi.config_name")
    steps = _positive_int(
        _prefer_nested(model, openpi, "num_steps", "num_steps"), f"{scope}.num_steps"
    )
    chunk = _positive_int(
        _prefer_nested(model, openpi, "action_chunk", "num_action_chunks"),
        f"{scope}.action_chunk",
    )
    env_dim = _positive_int(
        openpi.get("action_env_dim", model.get("action_dim")), f"{scope}.action_env_dim"
    )
    model_dim = _positive_int(
        openpi.get("model_action_dim"), f"{scope}.openpi.model_action_dim"
    )
    horizon = openpi.get("action_horizon")
    horizon_source = ("train_config", config_name)
    if horizon is not None:
        horizon = _positive_int(horizon, f"{scope}.openpi.action_horizon")
        if chunk > horizon:
            raise ValueError(
                f"quantization requires {scope}.action_chunk <= action_horizon"
            )
        horizon_source = ("explicit", horizon)
    if env_dim > model_dim:
        raise ValueError(
            f"quantization requires {scope}.action_env_dim <= model_action_dim"
        )
    if openpi.get("ignore_last", False) and steps < 2:
        raise ValueError(
            f"quantization requires {scope}.num_steps >= 2 with ignore_last"
        )
    data = model.get("openpi_data") or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"quantization requires {scope}.openpi_data to be a mapping")
    norm_source = data.get("norm_stats_path") or model.get("model_path")
    max_token_len = openpi.get("max_token_len")
    if max_token_len is None or max_token_len == 48:
        max_token_len = 200
    max_token_len = _positive_int(max_token_len, f"{scope}.openpi.max_token_len")
    discrete_state_input = openpi.get("discrete_state_input")
    if discrete_state_input is None:
        discrete_state_input = True
    return {
        "num_steps": steps,
        "action_chunk": chunk,
        "action_env_dim": env_dim,
        "model_action_dim": model_dim,
        "action_horizon_source": horizon_source,
        "config_name": config_name,
        "noise_method": str(openpi.get("noise_method", "flow_ode")),
        "noise_level": float(openpi.get("noise_level", 0.0)),
        "noise_logvar_range": _plain(openpi.get("noise_logvar_range", [0.08, 0.16])),
        "ignore_last": bool(openpi.get("ignore_last", False)),
        "norm_source": norm_source,
        "openpi_data": _plain(data),
        "max_token_len": max_token_len,
        "discrete_state_input": bool(discrete_state_input),
        "num_images_in_input": openpi.get("num_images_in_input", 2),
        "paligemma_variant": openpi.get("paligemma_variant"),
        "action_expert_variant": openpi.get("action_expert_variant"),
    }


def _prefer_nested(model: Mapping, openpi: Mapping, key: str, fallback: str) -> object:
    value = openpi.get(key)
    return model.get(fallback) if value is None else value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"quantization requires positive integer {name}, found {value!r}"
        )
    return value


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str):
        return tuple(_plain(item) for item in value)
    return value
