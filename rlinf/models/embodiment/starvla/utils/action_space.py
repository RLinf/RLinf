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
from collections.abc import Mapping
from typing import Any, Optional

import numpy as np
import torch


def resolve_action_norm_stats(
    starvla_model: Any,
    unnorm_key: Optional[str],
    action_dim: int,
) -> Optional[dict[str, np.ndarray]]:
    """Resolve action normalization stats from starVLA.

    Strict contract:
      - If unnorm_key is None, return None (caller opted out of unnormalization).
      - If unnorm_key is not None, return valid stats or raise an exception.
    """
    if unnorm_key is None:
        return None

    raw_stats: Any = None
    norm_stats = getattr(starvla_model, "norm_stats", None)
    if isinstance(norm_stats, Mapping) and unnorm_key in norm_stats:
        raw_stats = norm_stats.get(unnorm_key)
    else:
        getter = getattr(starvla_model, "get_action_stats", None)
        if not callable(getter):
            raise RuntimeError(
                "starVLA action unnormalization requires action norm stats, but the "
                "loaded model provides neither 'norm_stats' nor 'get_action_stats()'. "
                f"unnorm_key={unnorm_key!r}."
            )
        try:
            raw_stats = getter(unnorm_key)
        except Exception as exc:
            raise RuntimeError(
                "starVLA get_action_stats failed; cannot unnormalize actions for env. "
                f"unnorm_key={unnorm_key!r}, error={type(exc).__name__}: {exc}."
            ) from exc

    if raw_stats is None or not isinstance(raw_stats, Mapping):
        raise RuntimeError(
            "starVLA action norm stats payload is missing or invalid; cannot "
            f"unnormalize actions for env. unnorm_key={unnorm_key!r}."
        )

    stats_payload = raw_stats.get("action", raw_stats)
    if not isinstance(stats_payload, Mapping):
        raise RuntimeError(
            "starVLA action stats payload is invalid; cannot unnormalize actions for env. "
            f"unnorm_key={unnorm_key!r}."
        )

    high_src = stats_payload.get("q99", stats_payload.get("max"))
    low_src = stats_payload.get("q01", stats_payload.get("min"))
    if high_src is None or low_src is None:
        raise RuntimeError(
            "starVLA action norm stats missing q99/q01 (or max/min); cannot unnormalize "
            f"actions for env. unnorm_key={unnorm_key!r}."
        )

    try:
        high = np.asarray(high_src, dtype=np.float32).reshape(-1)
        low = np.asarray(low_src, dtype=np.float32).reshape(-1)
    except Exception as exc:
        raise RuntimeError(
            "starVLA action norm stats are not numeric arrays; cannot unnormalize actions "
            f"for env. unnorm_key={unnorm_key!r}, error={type(exc).__name__}: {exc}."
        ) from exc

    if high.shape[0] != action_dim or low.shape[0] != action_dim:
        raise RuntimeError(
            "starVLA action norm stats dim mismatch with RLinf action_dim; cannot "
            f"unnormalize actions for env. stats_dim={high.shape[0]}, action_dim={action_dim}, "
            f"unnorm_key={unnorm_key!r}."
        )

    mask_src = stats_payload.get("mask")
    if mask_src is None:
        mask = np.ones((action_dim,), dtype=bool)
    else:
        try:
            mask = np.asarray(mask_src, dtype=bool).reshape(-1)
        except Exception:
            mask = np.ones((action_dim,), dtype=bool)
        if mask.shape[0] != action_dim:
            mask = np.ones((action_dim,), dtype=bool)

    return {
        "q99": high,
        "q01": low,
        "mask": mask,
    }


def _gripper_mapping(actions: np.ndarray) -> np.ndarray:
    """Apply LIBERO gripper mapping aligned with starVLA eval pipeline."""
    if str(os.environ.get("ROBOT_PLATFORM", "")).upper() != "LIBERO":
        return actions
    if actions.shape[-1] < 7:
        return actions

    out = actions.astype(np.float32, copy=True)
    open_gripper = np.where(out[..., 6] < 0.5, 0.0, 1.0).astype(np.float32)
    out[..., 6] = (1.0 - 2.0 * (open_gripper > 0.5).astype(np.float32)).astype(
        np.float32
    )
    return out


def unnormalize_actions_for_env(
    normalized_actions: np.ndarray,
    action_norm_stats: dict[str, np.ndarray],
) -> np.ndarray:
    """Map model normalized actions to env action space (strict)."""
    if action_norm_stats is None:
        raise RuntimeError(
            "Missing action_norm_stats: cannot unnormalize actions for env. "
            "Set cfg.unnorm_key=None to use normalized actions directly."
        )

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
    return _gripper_mapping(env_actions)


def clip_actions_for_env(actions: torch.Tensor) -> torch.Tensor:
    """Clip action range and discretize gripper dimension for env stepping."""
    clipped = actions.clamp(-1.0, 1.0)
    if clipped.shape[-1] >= 7:
        clipped[..., 6] = torch.where(
            clipped[..., 6] < 0.5,
            torch.zeros_like(clipped[..., 6]),
            torch.ones_like(clipped[..., 6]),
        )
    return clipped
