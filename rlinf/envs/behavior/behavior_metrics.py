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

import torch

from rlinf.envs.behavior.env_access import (
    get_behavior_robot,
    get_task_reward,
    to_int_or_none,
    unwrap_behavior_env,
)
from rlinf.envs.behavior.stage_rewards import extract_episode_done


def annotate_hold_metrics(child_env, info: dict | None) -> dict:
    info = {} if info is None else info
    base_env = unwrap_behavior_env(child_env)
    reward_info = info.get("reward")
    if not isinstance(reward_info, dict):
        reward_info = {}
        info["reward"] = reward_info
    task_info = reward_info.get("task_specific")
    if not isinstance(task_info, dict):
        task_info = {}
        reward_info["task_specific"] = task_info

    task = getattr(base_env, "task", None)
    activity_instance_id = to_int_or_none(getattr(task, "activity_instance_id", None))
    if activity_instance_id is not None:
        task_info["activity_instance_id"] = activity_instance_id

    try:
        from omnigibson.reward_functions.support_utils import (
            get_stage_objects_by_name,
            is_target_in_hand,
        )

        robot = get_behavior_robot(base_env)
        task_reward = get_task_reward(base_env)
        target_obj = getattr(task_reward, "_radio_obj", None)
        if target_obj is None:
            target_objects = get_stage_objects_by_name(base_env, ("radio_89",))
            target_obj = target_objects[0] if target_objects else None
        if robot is not None and target_obj is not None:
            task_info["held_in_hand"] = bool(is_target_in_hand(robot, target_obj))
    except Exception:
        task_info["held_in_hand_available"] = False

    return info


def extract_behavior_episode_done(
    info: dict | None,
    success_stage_idx: int | None,
    default_done_extractor,
) -> bool:
    return extract_episode_done(info, success_stage_idx, default_done_extractor)


def info_done_tensor(
    infos,
    success_stage_idx: int | None,
    default_done_extractor,
    device=None,
) -> torch.Tensor:
    done_flags = [
        extract_behavior_episode_done(info, success_stage_idx, default_done_extractor)
        for info in infos
    ]
    return torch.as_tensor(done_flags, dtype=torch.bool, device=device)


def apply_info_dones(
    terminations: torch.Tensor,
    truncations: torch.Tensor,
    infos: list[dict],
    *,
    ignore_terminations: bool,
    success_stage_idx: int | None,
    default_done_extractor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    info_dones = info_done_tensor(
        infos,
        success_stage_idx,
        default_done_extractor,
        device=terminations.device,
    )
    if ignore_terminations:
        terminations = torch.zeros_like(terminations, dtype=torch.bool)
        truncations = torch.logical_or(truncations, info_dones)
    else:
        terminations = torch.logical_or(terminations, info_dones)
    dones = torch.logical_or(terminations, truncations)
    return terminations, truncations, dones
