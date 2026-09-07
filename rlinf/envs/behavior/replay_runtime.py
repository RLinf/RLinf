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

from typing import Any

from rlinf.envs.behavior.env_access import (
    get_task_reward,
    to_int_or_none,
    unwrap_behavior_env,
)
from rlinf.envs.behavior.env_access import (
    stage_idx_from_info as stage_idx_from_info,
)
from rlinf.envs.behavior.env_access import (
    stage_idx_from_reward as stage_idx_from_reward,
)
from rlinf.envs.behavior.instance_loader import RLINF_REPLAY_METADATA_KEY


def apply_replay_tro_metadata(child_env: Any, info: dict | None) -> dict:
    """Inject cached replay metadata into reset info and task reward state."""
    child_env = unwrap_behavior_env(child_env)
    scene = getattr(child_env, "scene", None)
    if scene is None or not hasattr(scene, "get_task_metadata"):
        return {} if info is None else info

    metadata = scene.get_task_metadata(key=RLINF_REPLAY_METADATA_KEY)
    if not isinstance(metadata, dict):
        return {} if info is None else info

    info = {} if info is None else info
    stage_idx = to_int_or_none(metadata.get("stage_index"))
    stage_prompts = metadata.get("stage_prompts")
    if not isinstance(stage_prompts, (list, tuple)):
        stage_prompts = []
    stage_prompts = [
        str(prompt).strip() for prompt in stage_prompts if str(prompt).strip()
    ]

    reward_obj = get_task_reward(child_env)
    total_stages = to_int_or_none(getattr(reward_obj, "_total_stages", None))
    if stage_idx is not None and hasattr(reward_obj, "set_active_stage_index"):
        if total_stages is None or 0 <= stage_idx < total_stages:
            reward_obj.set_active_stage_index(stage_idx)

    replay_info = info.get("replay_init")
    if not isinstance(replay_info, dict):
        replay_info = {}
        info["replay_init"] = replay_info
    replay_info.update(metadata)
    replay_info["replay_stage_prompts"] = stage_prompts
    if stage_idx is not None:
        replay_info["replay_stage_idx"] = stage_idx

    reward_info = info.get("reward")
    if not isinstance(reward_info, dict):
        reward_info = {}
        info["reward"] = reward_info
    task_info = reward_info.get("task_specific")
    if not isinstance(task_info, dict):
        task_info = {}
        reward_info["task_specific"] = task_info

    if stage_idx is not None:
        task_info["current_stage_idx"] = stage_idx
        task_info.setdefault("completed_stage_count", stage_idx)
    if total_stages is not None:
        task_info["total_stage_count"] = total_stages
    if stage_idx is not None and 0 <= stage_idx < len(stage_prompts):
        task_info["current_stage_prompt"] = stage_prompts[stage_idx]
    stage_defs = getattr(reward_obj, "_stage_defs", None)
    if (
        isinstance(stage_defs, list)
        and stage_idx is not None
        and 0 <= stage_idx < len(stage_defs)
    ):
        stage_name = stage_defs[stage_idx].get("name")
        if stage_name is not None:
            task_info["current_stage_name"] = stage_name
    task_info.setdefault("completion_bonus", 0.0)
    return info
