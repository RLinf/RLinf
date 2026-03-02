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

"""State preprocessing and dimension adaptation utilities for starVLA."""

from __future__ import annotations

import os
import warnings
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

STATE_ADAPTER_TYPES = {"none", "adapter", "pi", "gr00t", "dual"}


def warn_state_adapt_once(warned_keys: set[str], key: str, message: str) -> None:
    """Emit state-adaptation warning once for each warning key."""
    if os.environ.get("STARVLA_STATE_ADAPT_WARNINGS", "1") in {
        "",
        "0",
        "false",
        "False",
    }:
        warned_keys.add(key)
        return
    if key in warned_keys:
        return
    warnings.warn(message, stacklevel=2)
    warned_keys.add(key)


def infer_expected_state_dim_from_head(head: Optional[nn.Module]) -> Optional[int]:
    """Infer expected state dimension from action head state encoder."""
    if head is None:
        return None
    state_encoder = getattr(head, "state_encoder", None)
    if state_encoder is None:
        return None
    for module in state_encoder.modules():
        if isinstance(module, nn.Linear):
            return int(module.in_features)
    return None


def infer_expected_state_dim_for_state_adapter(
    starvla_model: nn.Module,
    state_adapter_name: str,
    head: Optional[nn.Module] = None,
) -> Optional[int]:
    """Infer target state dim for a given state-adapter mode."""
    if state_adapter_name not in STATE_ADAPTER_TYPES:
        raise ValueError(
            f"Unknown state_adapter_name={state_adapter_name!r}. "
            f"Expected one of {sorted(STATE_ADAPTER_TYPES)}."
        )

    dim_from_head = infer_expected_state_dim_from_head(head)
    if dim_from_head is not None:
        return dim_from_head

    if state_adapter_name == "adapter":
        proprio_projector = getattr(starvla_model, "proprio_projector", None)
        if proprio_projector is not None:
            fc1 = getattr(proprio_projector, "fc1", None)
            if isinstance(fc1, nn.Linear):
                return int(fc1.in_features)
            proprio_dim = getattr(proprio_projector, "proprio_dim", None)
            if proprio_dim is not None:
                try:
                    return int(proprio_dim)
                except (TypeError, ValueError):
                    pass

    cfg = getattr(starvla_model, "config", None)
    framework_cfg = getattr(cfg, "framework", None) if cfg is not None else None
    action_cfg = (
        getattr(framework_cfg, "action_model", None)
        if framework_cfg is not None
        else None
    )
    value = getattr(action_cfg, "state_dim", None) if action_cfg is not None else None
    try:
        dim_from_cfg = int(value) if value is not None else None
    except (TypeError, ValueError):
        dim_from_cfg = None
    if dim_from_cfg is not None:
        return dim_from_cfg

    if state_adapter_name in {"pi", "gr00t", "dual"}:
        return infer_expected_state_dim_from_head(
            getattr(starvla_model, "action_model", None)
        )

    return None


def adapt_state_for_expected_dim(
    state: torch.Tensor,
    expected_dim: int,
    context: str,
    warned_keys: set[str],
) -> torch.Tensor:
    """Adapt state dimension via special-rule/trim/pad to expected dim."""
    current_dim = int(state.shape[-1])
    if current_dim == expected_dim:
        return state

    strict_state_dim = (
        str(os.environ.get("STARVLA_STRICT_STATE_DIM", "0")).strip().lower()
    )
    if strict_state_dim in {"1", "true", "yes", "on"}:
        raise ValueError(
            f"[{context}] State dim mismatch in strict mode: got {current_dim}, expected {expected_dim}. "
            "Align env state preprocessing with the checkpoint."
        )

    if current_dim == 8 and expected_dim == 7:
        adapted = torch.cat(
            [state[..., :6], state[..., 6:8].mean(dim=-1, keepdim=True)], dim=-1
        )
        warn_state_adapt_once(
            warned_keys,
            key=f"{context}:8to7",
            message=(
                f"[{context}] Adapted state dim 8 -> 7 by averaging the 2-dim gripper state. "
                "If your checkpoint used another rule, align env state preprocessing accordingly."
            ),
        )
        return adapted

    if current_dim > expected_dim:
        adapted = state[..., :expected_dim]
        warn_state_adapt_once(
            warned_keys,
            key=f"{context}:{current_dim}to{expected_dim}:truncate",
            message=(
                f"[{context}] Truncated state dim {current_dim} -> {expected_dim}. "
                "Please verify env state definition matches checkpoint training."
            ),
        )
        return adapted

    pad = torch.zeros(
        (*state.shape[:-1], expected_dim - current_dim),
        device=state.device,
        dtype=state.dtype,
    )
    adapted = torch.cat([state, pad], dim=-1)
    warn_state_adapt_once(
        warned_keys,
        key=f"{context}:{current_dim}to{expected_dim}:pad",
        message=(
            f"[{context}] Padded state dim {current_dim} -> {expected_dim} with zeros. "
            "Please verify env state definition matches checkpoint training."
        ),
    )
    return adapted


def prepare_state_tensor(
    state: Any,
    *,
    starvla_model: nn.Module,
    default_state_adapter_name: str,
    warned_keys: set[str],
    state_adapter_name: Optional[str] = None,
    head: Optional[nn.Module] = None,
    expected_dim: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    context: str = "state",
) -> Optional[torch.Tensor]:
    """Normalize state tensor shape/dtype/device and adapt its last dimension."""
    if state is None:
        return None

    if not torch.is_tensor(state):
        state = torch.tensor(np.asarray(state))

    if device is not None or dtype is not None:
        state = state.to(
            device=(device if device is not None else state.device),
            dtype=(dtype if dtype is not None else state.dtype),
        )

    if state.ndim == 1:
        state = state.view(1, 1, -1)
    elif state.ndim == 2:
        state = state.unsqueeze(1)
    elif state.ndim != 3:
        raise ValueError(
            f"[{context}] Expected state with ndim in {{1,2,3}}, got shape {tuple(state.shape)}"
        )

    dim = expected_dim
    if dim is None:
        dim = infer_expected_state_dim_for_state_adapter(
            starvla_model=starvla_model,
            state_adapter_name=(state_adapter_name or default_state_adapter_name),
            head=head,
        )
    if dim is None:
        return state

    return adapt_state_for_expected_dim(
        state,
        expected_dim=dim,
        context=context,
        warned_keys=warned_keys,
    )
