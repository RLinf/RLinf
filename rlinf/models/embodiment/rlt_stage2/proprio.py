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

"""Proprioception selection utilities for RLT Stage 2."""

from __future__ import annotations

from typing import Any, Mapping

import torch

PROPRIO_MODE_TO_DIM = {
    "joint_pos_gripper": 8,
    "joint_pos_vel_gripper": 15,
    "full_state": 34,
}


def _get_cfg_value(cfg: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def resolve_proprio_dim(
    stage2_cfg: Mapping[str, Any] | Any,
    *,
    default_dim: int,
) -> int:
    """Resolve RLT Stage 2 proprio dimension from explicit mode or legacy dim."""

    proprio_mode = _get_cfg_value(stage2_cfg, "proprio_mode", None)
    if proprio_mode is None:
        return int(_get_cfg_value(stage2_cfg, "proprio_dim", default_dim))

    proprio_mode = str(proprio_mode)
    if proprio_mode not in PROPRIO_MODE_TO_DIM:
        valid_modes = ", ".join(sorted(PROPRIO_MODE_TO_DIM))
        raise ValueError(
            f"Unsupported RLT Stage 2 proprio_mode {proprio_mode!r}. "
            f"Expected one of: {valid_modes}."
        )

    resolved_dim = PROPRIO_MODE_TO_DIM[proprio_mode]
    configured_dim = _get_cfg_value(stage2_cfg, "proprio_dim", None)
    if configured_dim is not None and int(configured_dim) != resolved_dim:
        raise ValueError(
            "RLT Stage 2 proprio_dim does not match proprio_mode: "
            f"proprio_mode={proprio_mode!r} implies {resolved_dim}, "
            f"but proprio_dim={configured_dim}."
        )
    return resolved_dim


def select_proprio(
    state: torch.Tensor,
    *,
    proprio_dim: int | None = None,
    proprio_mode: str | None = None,
) -> torch.Tensor:
    """Select proprio state columns for RLT Stage 2.

    Realworld RLT uses contiguous prefixes:
    [gripper, joint_pos, joint_vel, tcp_force, tcp_pose, tcp_torque, tcp_vel].
    ManiSkill configs keep using the legacy ``proprio_dim`` prefix selection.
    """

    if proprio_mode is not None:
        if proprio_mode not in PROPRIO_MODE_TO_DIM:
            valid_modes = ", ".join(sorted(PROPRIO_MODE_TO_DIM))
            raise ValueError(
                f"Unsupported RLT Stage 2 proprio_mode {proprio_mode!r}. "
                f"Expected one of: {valid_modes}."
            )
        resolved_dim = PROPRIO_MODE_TO_DIM[proprio_mode]
        if proprio_dim is not None and int(proprio_dim) != resolved_dim:
            raise ValueError(
                "RLT Stage 2 proprio_dim does not match proprio_mode: "
                f"proprio_mode={proprio_mode!r} implies {resolved_dim}, "
                f"but proprio_dim={proprio_dim}."
            )
        proprio_dim = resolved_dim

    if proprio_dim is None:
        raise ValueError("RLT Stage 2 proprio selection requires proprio_dim or mode.")

    proprio_dim = int(proprio_dim)
    if state.ndim != 2:
        raise ValueError(
            "RLT Stage 2 observation.state must be a 2D tensor [B, state_dim], "
            f"got shape={tuple(state.shape)}."
        )
    if state.shape[1] < proprio_dim:
        raise ValueError(
            "RLT Stage 2 observation.state is too small for proprio selection: "
            f"state_dim={state.shape[1]}, required={proprio_dim}, "
            f"proprio_mode={proprio_mode!r}."
        )
    return state[:, :proprio_dim]
