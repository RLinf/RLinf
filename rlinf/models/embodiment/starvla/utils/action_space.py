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

"""Action normalization helpers for starVLA-backed RLinf policies."""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from typing import Any, Optional

import numpy as np
import torch


def resolve_action_norm_stats(
    starvla_model: Any,
    unnorm_key: Optional[str],
    action_dim: int,
) -> Optional[dict[str, np.ndarray]]:
    """Resolve action normalization stats from starVLA in one pass."""
    if unnorm_key is None:
        return None

    raw_stats = None

    norm_stats = getattr(starvla_model, "norm_stats", None)
    if isinstance(norm_stats, Mapping):
        raw_stats = norm_stats.get(unnorm_key)

    if raw_stats is None:
        getter = getattr(starvla_model, "get_action_stats", None)
        if callable(getter):
            try:
                raw_stats = getter(unnorm_key)
            except Exception as exc:
                warnings.warn(
                    "starVLA get_action_stats failed; action unnormalization disabled. "
                    f"unnorm_key={unnorm_key!r}, error={exc}",
                    stacklevel=2,
                )
                return None
        else:
            warnings.warn(
                "starVLA model has no get_action_stats; action unnormalization disabled.",
                stacklevel=2,
            )
            return None

    if raw_stats is None:
        return None
    if not isinstance(raw_stats, Mapping):
        warnings.warn(
            "starVLA get_action_stats returned invalid payload; skip action unnormalization.",
            stacklevel=2,
        )
        return None

    stats_payload = raw_stats.get("action", raw_stats)
    if not isinstance(stats_payload, Mapping):
        warnings.warn(
            "starVLA action stats payload is invalid; skip action unnormalization.",
            stacklevel=2,
        )
        return None

    high_src = stats_payload.get("q99", stats_payload.get("max"))
    low_src = stats_payload.get("q01", stats_payload.get("min"))
    if high_src is None or low_src is None:
        warnings.warn(
            "starVLA action norm stats missing q99/q01 (or max/min); "
            "skip action unnormalization.",
            stacklevel=2,
        )
        return None

    try:
        high = np.asarray(high_src, dtype=np.float32).reshape(-1)
        low = np.asarray(low_src, dtype=np.float32).reshape(-1)
    except Exception as exc:
        warnings.warn(
            "starVLA action norm stats are not numeric arrays; "
            f"skip action unnormalization. error={exc}",
            stacklevel=2,
        )
        return None

    if high.shape[0] != action_dim or low.shape[0] != action_dim:
        warnings.warn(
            "starVLA action norm stats dim mismatch with RLinf action_dim; "
            f"stats_dim={high.shape[0]}, action_dim={action_dim}. "
            "Skip unnormalization for env actions.",
            stacklevel=2,
        )
        return None

    try:
        mask = np.asarray(
            stats_payload.get(
                "mask",
                np.ones((action_dim,), dtype=bool),
            ),
            dtype=bool,
        ).reshape(-1)
    except Exception:
        mask = np.ones((action_dim,), dtype=bool)
    if mask.shape[0] != action_dim:
        mask = np.ones((action_dim,), dtype=bool)

    return {
        "high": high,
        "low": low,
        "q99": high,
        "q01": low,
        "mask": mask,
    }


def _gripper_mapping(actions: np.ndarray) -> np.ndarray:
    """Apply platform-specific gripper sign convention mapping."""
    if str(os.environ.get("ROBOT_PLATFORM", "")).upper() != "LIBERO":
        return actions
    if actions.shape[-1] < 7:
        return actions

    g01 = (actions[..., 6] >= 0.5).astype(np.float32)
    # LIBERO execution convention in RLinf: open=-1, close=+1.
    signed = 1.0 - 2.0 * g01
    out = actions.astype(np.float32, copy=True)
    out[..., 6] = signed
    return out


def unnormalize_actions_for_env(
    normalized_actions: np.ndarray,
    action_norm_stats: Optional[dict[str, np.ndarray]],
    warned_missing_action_norm_stats: bool,
) -> tuple[np.ndarray, bool]:
    """Map model normalized actions to env action space via starVLA core only."""
    if action_norm_stats is None:
        if not warned_missing_action_norm_stats:
            warnings.warn(
                "starVLA action norm stats are unavailable; env rollout will use "
                "model normalized actions directly.",
                stacklevel=2,
            )
            warned_missing_action_norm_stats = True
        return _gripper_mapping(
            np.asarray(normalized_actions, dtype=np.float32)
        ), warned_missing_action_norm_stats

    try:
        from starVLA.model.framework.base_framework import baseframework
    except Exception as exc:
        raise ModuleNotFoundError(
            "starVLA is required for action unnormalization but is not importable."
        ) from exc

    actions = np.asarray(normalized_actions, dtype=np.float32)
    flat = actions.reshape(-1, actions.shape[-1]).astype(np.float32, copy=False)
    starvla_stats = {
        "q99": np.asarray(action_norm_stats["q99"], dtype=np.float32),
        "q01": np.asarray(action_norm_stats["q01"], dtype=np.float32),
        "mask": np.asarray(action_norm_stats["mask"], dtype=bool),
    }
    env_flat = baseframework.unnormalize_actions(flat, starvla_stats)
    env_actions = np.asarray(env_flat, dtype=np.float32).reshape(actions.shape)
    return _gripper_mapping(env_actions), warned_missing_action_norm_stats


def normalize_actions_for_model(
    env_actions: torch.Tensor,
    action_norm_stats: Optional[dict[str, np.ndarray]],
) -> torch.Tensor:
    """Map environment actions to model normalized space for training."""
    if (
        str(os.environ.get("ROBOT_PLATFORM", "")).upper() == "LIBERO"
        and torch.is_tensor(env_actions)
        and env_actions.ndim >= 1
        and env_actions.shape[-1] >= 7
    ):
        g = env_actions[..., 6]
        if torch.any(g < 0):
            g01 = (g < 0).to(dtype=env_actions.dtype)
            env_actions = env_actions.clone()
            env_actions[..., 6] = g01

    if action_norm_stats is None:
        return env_actions

    if not torch.is_tensor(env_actions):
        raise TypeError(f"Expected torch.Tensor, got {type(env_actions)}")
    if not env_actions.is_floating_point():
        env_actions = env_actions.float()

    high = torch.as_tensor(
        action_norm_stats["high"], device=env_actions.device, dtype=env_actions.dtype
    ).view(1, 1, -1)
    low = torch.as_tensor(
        action_norm_stats["low"], device=env_actions.device, dtype=env_actions.dtype
    ).view(1, 1, -1)
    mask = torch.as_tensor(
        action_norm_stats["mask"], device=env_actions.device, dtype=torch.bool
    ).view(1, 1, -1)

    denom = high - low
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    normalized = 2.0 * (env_actions - low) / denom - 1.0
    return torch.where(mask, normalized, env_actions)


def clip_actions_for_env(actions: torch.Tensor) -> torch.Tensor:
    """Clip action range and discretize gripper dimension for env stepping."""
    clipped = actions.clamp(-1.0, 1.0)
    if clipped.shape[-1] >= 7:
        clipped[..., 6] = (clipped[..., 6] >= 0.5).to(dtype=clipped.dtype)
    return clipped
