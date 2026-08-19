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

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def parse_action_mask(cfg: DictConfig) -> list[bool] | None:
    """Parse optional per-action policy mask from env config."""
    mask_cfg = OmegaConf.select(cfg, "action_mask", default=None)
    if mask_cfg is None:
        return None
    if not bool(OmegaConf.select(cfg, "action_mask.enabled", default=True)):
        return None

    mask = OmegaConf.select(cfg, "action_mask.mask", default=None)
    if mask is None:
        action_dim = int(OmegaConf.select(cfg, "action_mask.action_dim", default=23))
        mask = [True] * action_dim
        if bool(OmegaConf.select(cfg, "action_mask.freeze_base", default=False)):
            mask[:3] = [False] * min(3, action_dim)
        if bool(OmegaConf.select(cfg, "action_mask.freeze_trunk", default=False)):
            start, end = 3, min(7, action_dim)
            mask[start:end] = [False] * (end - start)
    else:
        mask = OmegaConf.to_container(mask, resolve=True)

    if not isinstance(mask, (list, tuple)) or not mask:
        raise ValueError("env.action_mask.mask must be a non-empty bool list.")
    return [bool(value) for value in mask]


def parse_first_chunk_action_override(
    cfg: DictConfig,
) -> tuple[bool, list[int], float]:
    """Parse first-action-chunk override config."""
    override_cfg = OmegaConf.select(cfg, "first_chunk_action_override", default=None)
    if override_cfg is None:
        return False, [], -1.0
    enabled = bool(
        OmegaConf.select(cfg, "first_chunk_action_override.enabled", default=False)
    )
    action_ids = OmegaConf.select(
        cfg, "first_chunk_action_override.action_ids", default=[]
    )
    if action_ids is None:
        action_ids = []
    if not isinstance(action_ids, (list, tuple)):
        action_ids = OmegaConf.to_container(action_ids, resolve=True)
    if not isinstance(action_ids, (list, tuple)):
        raise ValueError("env.first_chunk_action_override.action_ids must be a list.")
    action_value = float(
        OmegaConf.select(cfg, "first_chunk_action_override.value", default=-1.0)
    )
    return enabled, [int(action_id) for action_id in action_ids], action_value


def action_like_values(action: Any, values: list[float]) -> Any:
    if torch.is_tensor(action):
        return torch.as_tensor(values, dtype=action.dtype, device=action.device)
    dtype = getattr(action, "dtype", np.float32)
    return np.asarray(values, dtype=dtype)


def robot_joint_positions_list(robot: Any) -> list[float]:
    joint_positions = robot.get_joint_positions()
    if torch.is_tensor(joint_positions):
        return joint_positions.detach().cpu().reshape(-1).tolist()
    return np.asarray(joint_positions).reshape(-1).tolist()


def r1pro_noop_action(robot: Any, action: Any) -> Any:
    """Map current R1Pro joint state to the 23-D no-op action layout."""
    action_dim = int(action.shape[-1])
    values = [0.0] * action_dim
    joint_positions = robot_joint_positions_list(robot)
    if action_dim >= 23 and len(joint_positions) >= 28:
        values[3:7] = joint_positions[6:10]
        values[7:14] = [joint_positions[i] for i in (10, 12, 14, 16, 18, 20, 22)]
        values[14:21] = [joint_positions[i] for i in (11, 13, 15, 17, 19, 21, 23)]
        values[21] = joint_positions[24] + joint_positions[25]
        values[22] = joint_positions[26] + joint_positions[27]
    return action_like_values(action, values)


def apply_action_mask(
    actions: Any,
    action_mask: list[bool] | None,
    child_envs: list[Any],
    get_robot_from_child_env: Callable[[Any], Any],
) -> Any:
    """Replace frozen dimensions with robot no-op action values."""
    if action_mask is None:
        return actions

    action_dim = int(actions.shape[-1])
    if action_dim != len(action_mask):
        raise ValueError(
            f"env.action_mask.mask length {len(action_mask)} does not "
            f"match action_dim {action_dim}."
        )

    masked_actions = actions.clone() if torch.is_tensor(actions) else actions.copy()
    for env_idx in range(int(actions.shape[0])):
        robot = get_robot_from_child_env(child_envs[env_idx])
        if robot is None:
            continue
        noop_action = r1pro_noop_action(robot, actions[env_idx])
        for action_idx, use_policy_action in enumerate(action_mask):
            if not use_policy_action:
                masked_actions[env_idx, action_idx] = noop_action[action_idx]
    return masked_actions


def apply_first_chunk_action_override(
    actions: Any,
    env_mask: np.ndarray,
    enabled: bool,
    action_ids: list[int],
    action_value: float,
) -> Any:
    """Force selected action dimensions on reset's first action chunk."""
    if not enabled or not action_ids or not env_mask.any():
        return actions

    action_dim = int(actions.shape[-1])
    invalid_action_ids = [
        action_id
        for action_id in action_ids
        if action_id < 0 or action_id >= action_dim
    ]
    if invalid_action_ids:
        raise ValueError(
            "env.first_chunk_action_override.action_ids contains invalid "
            f"indices {invalid_action_ids} for action_dim {action_dim}."
        )

    overridden_actions = actions.clone() if torch.is_tensor(actions) else actions.copy()
    env_indices = np.flatnonzero(env_mask).tolist()
    for action_id in action_ids:
        overridden_actions[env_indices, action_id] = action_value
    return overridden_actions
