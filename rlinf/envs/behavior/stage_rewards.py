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

from collections.abc import Sequence
from typing import Any

import torch


def task_reward_from_info(info: dict | None) -> dict:
    """Return the BEHAVIOR task-specific reward dict, if present."""
    if not isinstance(info, dict):
        return {}
    reward_info = info.get("reward", {})
    if not isinstance(reward_info, dict):
        return {}
    task_reward = reward_info.get("task_specific", {})
    return task_reward if isinstance(task_reward, dict) else {}


def completion_bonus_tensor(
    infos: Sequence[dict] | None,
    reward: torch.Tensor,
    reward_coef: float,
) -> torch.Tensor:
    bonuses = [
        float(task_reward_from_info(info).get("completion_bonus", 0.0) or 0.0)
        for info in infos or [{} for _ in range(int(reward.numel()))]
    ]
    return reward_coef * torch.as_tensor(
        bonuses, dtype=reward.dtype, device=reward.device
    )


def is_target_stage_success(
    task_reward: dict,
    success_stage_idx: int | None,
) -> bool:
    if success_stage_idx is None:
        return False
    current_stage_idx = task_reward.get("current_stage_idx", None)
    if current_stage_idx is None:
        return False
    try:
        current_stage_idx = int(current_stage_idx)
    except (TypeError, ValueError):
        return False
    completion_bonus = float(task_reward.get("completion_bonus", 0.0) or 0.0)
    return current_stage_idx == success_stage_idx and completion_bonus != 0.0


def stage_sparse_reward_tensor(
    rewards: Any,
    infos: Sequence[dict] | None,
    reward_coef: float,
    success_stage_idx: int | None,
) -> torch.Tensor:
    reward = torch.as_tensor(rewards)
    bonuses = []
    for info in infos or [{} for _ in range(int(reward.numel()))]:
        task_reward = task_reward_from_info(info)
        bonus = float(task_reward.get("completion_bonus", 0.0) or 0.0)
        bonuses.append(
            bonus if is_target_stage_success(task_reward, success_stage_idx) else 0.0
        )
    return reward_coef * torch.as_tensor(
        bonuses, dtype=reward.dtype, device=reward.device
    )


def stage_weighted_reward_tensor(
    rewards: Any,
    infos: Sequence[dict] | None,
    reward_coef: float,
    stage_reward_weights: list[float],
) -> torch.Tensor:
    reward = torch.as_tensor(rewards)
    bonuses = []
    for info in infos or [{} for _ in range(int(reward.numel()))]:
        task_reward = task_reward_from_info(info)
        stage_idx = int(task_reward.get("current_stage_idx", 0) or 0) - 1
        bonus = float(task_reward.get("completion_bonus", 0.0) or 0.0)
        if 0 <= stage_idx < len(stage_reward_weights):
            bonus *= stage_reward_weights[stage_idx]
        else:
            bonus = 0.0
        bonuses.append(bonus)
    return reward_coef * torch.as_tensor(
        bonuses, dtype=reward.dtype, device=reward.device
    )


def stage_cumulative_reward_tensor(
    rewards: Any,
    infos: Sequence[dict] | None,
    reward_coef: float,
) -> torch.Tensor:
    reward = torch.as_tensor(rewards)
    counts = [
        float(task_reward_from_info(info).get("completed_stage_count", 0.0) or 0.0)
        for info in infos or [{} for _ in range(int(reward.numel()))]
    ]
    return reward_coef * torch.as_tensor(
        counts, dtype=reward.dtype, device=reward.device
    )


def extract_episode_success(
    info: dict | None,
    success_stage_idx: int | None,
) -> bool:
    if not isinstance(info, dict):
        return False
    if success_stage_idx is not None:
        return is_target_stage_success(task_reward_from_info(info), success_stage_idx)
    done_dict = info.get("done", {})
    if isinstance(done_dict, dict):
        return bool(done_dict.get("success", False))
    return bool(info.get("success", False))


def extract_episode_done(
    info: dict | None,
    success_stage_idx: int | None,
    default_done_extractor: Any,
) -> bool:
    if not isinstance(info, dict):
        return False
    if success_stage_idx is not None:
        return extract_episode_success(info, success_stage_idx)
    return default_done_extractor(info)
