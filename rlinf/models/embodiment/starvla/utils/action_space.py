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

"""Action normalization and unnormalization helpers for starVLA policies."""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from typing import Any, Optional

import numpy as np
import torch


def _auto_select_unnorm_key(available_keys: list[str]) -> Optional[str]:
    """Pick a likely norm-stats key based on robot platform hint."""
    if not available_keys:
        return None

    keys = [str(k) for k in available_keys]
    platform = str(os.environ.get("ROBOT_PLATFORM", "")).strip().lower()

    token_priority = {
        "libero": ["libero", "franka", "bridge", "fractal"],
        "frankasim": ["franka", "libero", "bridge", "fractal"],
        "bridge": ["bridge", "fractal", "franka"],
        "metaworld": ["metaworld", "franka", "bridge"],
        "robocasa": ["robocasa", "franka", "bridge"],
    }.get(platform, ["libero", "bridge", "franka", "fractal"])

    for token in token_priority:
        matched = [k for k in keys if token in k.lower()]
        if matched:
            return sorted(matched)[0]

    return sorted(keys)[0]


def resolve_action_norm_stats(
    starvla_model: Any,
    unnorm_key: Optional[str],
    action_dim: int,
) -> tuple[Optional[dict[str, np.ndarray]], Optional[str]]:
    """Resolve/validate action normalization statistics from checkpoint."""
    getter = getattr(starvla_model, "get_action_stats", None)

    norm_stats = getattr(starvla_model, "norm_stats", None)
    available_keys: list[str] = []
    if isinstance(norm_stats, Mapping):
        try:
            available_keys = [str(k) for k in norm_stats.keys()]
        except Exception:
            available_keys = []

    def _try_get(key: Optional[str]):
        if isinstance(norm_stats, Mapping) and key is not None:
            entry = None
            try:
                entry = norm_stats.get(key)
            except Exception:
                entry = None

            if isinstance(entry, Mapping):
                action_stats = entry.get("action")
                if isinstance(action_stats, Mapping):
                    return dict(action_stats)
                if isinstance(action_stats, dict):
                    return action_stats
            else:
                action_stats = getattr(entry, "action", None)
                if isinstance(action_stats, Mapping):
                    return dict(action_stats)
                if isinstance(action_stats, dict):
                    return action_stats

        if callable(getter):
            try:
                return getter(key)
            except Exception:
                return None
        return None

    requested_key = None if unnorm_key is None else str(unnorm_key)
    resolved_key = requested_key
    stats = _try_get(resolved_key)
    if stats is None:
        if available_keys and requested_key is not None:
            req_lower = requested_key.lower()
            fuzzy_matches = [
                k
                for k in available_keys
                if req_lower in k.lower() or k.lower() in req_lower
            ]
            if len(fuzzy_matches) == 1:
                resolved_key = fuzzy_matches[0]
                stats = _try_get(resolved_key)
                if stats is not None and resolved_key != requested_key:
                    warnings.warn(
                        "starVLA unnorm_key fuzzy-match: "
                        f"cfg.unnorm_key={requested_key!r} not found, "
                        f"using {resolved_key!r}.",
                        stacklevel=2,
                    )

        if available_keys and len(available_keys) == 1:
            resolved_key = available_keys[0]
            if requested_key is not None and requested_key != resolved_key:
                warnings.warn(
                    "starVLA unnorm_key mismatch: "
                    f"cfg.unnorm_key={requested_key!r} not found, "
                    f"falling back to the only available key {resolved_key!r}.",
                    stacklevel=2,
                )
            stats = _try_get(resolved_key)
        elif (
            available_keys
            and requested_key is None
            and str(os.environ.get("STARVLA_AUTO_UNNORM_KEY", "1")).strip().lower()
            not in {"0", "false", "no"}
        ):
            auto_key = _auto_select_unnorm_key(available_keys)
            if auto_key is not None:
                resolved_key = auto_key
                stats = _try_get(resolved_key)
                if stats is not None:
                    warnings.warn(
                        "starVLA unnorm_key auto-selected: "
                        f"cfg.unnorm_key is None, choosing {resolved_key!r} from {available_keys}. "
                        "Set actor.model.unnorm_key explicitly to avoid ambiguity.",
                        stacklevel=2,
                    )
        elif available_keys and unnorm_key is not None:
            warnings.warn(
                "starVLA unnorm_key mismatch: "
                f"cfg.unnorm_key={requested_key!r} not found in checkpoint norm_stats keys "
                f"{available_keys}. Action unnormalization will be disabled.",
                stacklevel=2,
            )
        elif not available_keys and requested_key is not None:
            warnings.warn(
                "starVLA action norm stats unavailable: "
                f"cfg.unnorm_key={requested_key!r}. "
                "Check that the checkpoint run_dir contains 'dataset_statistics.json'.",
                stacklevel=2,
            )

    platform = str(os.environ.get("ROBOT_PLATFORM", "")).strip().lower()
    if (
        platform == "libero"
        and isinstance(resolved_key, str)
        and ("bridge" in resolved_key.lower() or "fractal" in resolved_key.lower())
    ):
        has_target_like_keys = any(
            ("libero" in k.lower() or "franka" in k.lower()) for k in available_keys
        )
        if has_target_like_keys or unnorm_key is not None:
            warnings.warn(
                "starVLA action stats may be domain-mismatched: "
                f"ROBOT_PLATFORM=LIBERO but resolved unnorm_key={resolved_key!r}. "
                "Bridge/Fractal stats often produce too-small LIBERO arm deltas. "
                "Prefer LIBERO/Franka stats, or set disable_action_unnormalization=True.",
                stacklevel=2,
            )

    if stats is None or not isinstance(stats, dict):
        return None, resolved_key

    high = stats.get("q99", stats.get("max"))
    low = stats.get("q01", stats.get("min"))
    if high is None or low is None:
        return None, resolved_key

    high_np = np.asarray(high, dtype=np.float32)
    low_np = np.asarray(low, dtype=np.float32)
    mask_np = np.asarray(
        stats.get("mask", np.ones_like(high_np, dtype=bool)),
        dtype=bool,
    )
    if high_np.shape[-1] != action_dim or low_np.shape[-1] != action_dim:
        warnings.warn(
            "starVLA action norm stats dim mismatch with RLinf action_dim; "
            f"stats_dim={high_np.shape[-1]}, action_dim={action_dim}. "
            "Skip unnormalization for env actions.",
            stacklevel=2,
        )
        return None, resolved_key
    if mask_np.shape[-1] != action_dim:
        mask_np = np.ones((action_dim,), dtype=bool)

    return {
        "high": high_np,
        "low": low_np,
        "mask": mask_np,
    }, resolved_key


def _gripper_mapping(actions: np.ndarray) -> np.ndarray:
    """Apply platform-specific gripper sign convention mapping."""
    if str(os.environ.get("ROBOT_PLATFORM", "")).upper() != "LIBERO":
        return actions
    if actions.shape[-1] < 7:
        return actions
    mode = (
        str(os.environ.get("STARVLA_LIBERO_GRIPPER_MODE", "open_is_one"))
        .strip()
        .lower()
    )
    if mode not in {"open_is_one", "close_is_one"}:
        warnings.warn(
            f"Invalid STARVLA_LIBERO_GRIPPER_MODE={mode!r}; use 'open_is_one' or 'close_is_one'. "
            "Falling back to 'open_is_one'.",
            stacklevel=2,
        )
        mode = "open_is_one"
    # Keep consistent with starVLA official LIBERO eval:
    # open if value > 0.5, close otherwise.
    g01 = (actions[..., 6] >= 0.5).astype(np.float32)
    signed = 1.0 - 2.0 * g01 if mode == "open_is_one" else 2.0 * g01 - 1.0
    out = actions.astype(np.float32, copy=True)
    out[..., 6] = signed
    return out


def unnormalize_actions_for_env(
    normalized_actions: np.ndarray,
    action_norm_stats: Optional[dict[str, np.ndarray]],
    warned_missing_action_norm_stats: bool,
) -> tuple[np.ndarray, bool]:
    """Map model normalized actions back to environment action space."""
    if action_norm_stats is None:
        if not warned_missing_action_norm_stats:
            warnings.warn(
                "starVLA action norm stats are unavailable; env rollout will use "
                "model normalized actions directly. If motions look abnormal, "
                "check checkpoint norm_stats and actor.model.unnorm_key.",
                stacklevel=2,
            )
            warned_missing_action_norm_stats = True
        return _gripper_mapping(
            np.asarray(normalized_actions, dtype=np.float32)
        ), warned_missing_action_norm_stats

    actions = np.clip(normalized_actions, -1.0, 1.0).astype(np.float32, copy=True)
    if actions.shape[-1] >= 7:
        actions[..., 6] = np.where(actions[..., 6] < 0.5, 0.0, 1.0)

    high = action_norm_stats["high"].reshape(1, 1, -1)
    low = action_norm_stats["low"].reshape(1, 1, -1)
    mask = action_norm_stats["mask"].reshape(1, 1, -1)
    env_actions = np.where(
        mask, 0.5 * (actions + 1.0) * (high - low) + low, actions
    ).astype(
        np.float32,
        copy=False,
    )
    return _gripper_mapping(env_actions), warned_missing_action_norm_stats


def normalize_actions_for_model(
    env_actions: torch.Tensor,
    action_norm_stats: Optional[dict[str, np.ndarray]],
) -> torch.Tensor:
    """Map environment actions to model normalized space for training."""
    mode = (
        str(os.environ.get("STARVLA_LIBERO_GRIPPER_MODE", "open_is_one"))
        .strip()
        .lower()
    )
    mode = mode if mode in {"open_is_one", "close_is_one"} else "open_is_one"

    if (
        str(os.environ.get("ROBOT_PLATFORM", "")).upper() == "LIBERO"
        and torch.is_tensor(env_actions)
        and env_actions.ndim >= 1
        and env_actions.shape[-1] >= 7
    ):
        g = env_actions[..., 6]
        if torch.any(g < 0):
            if mode == "open_is_one":
                g01 = (g < 0).to(dtype=env_actions.dtype)
            else:
                g01 = (g > 0).to(dtype=env_actions.dtype)
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
