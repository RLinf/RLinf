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


def unwrap_behavior_env(env: Any) -> Any:
    """Return the underlying OmniGibson environment."""
    current = env
    seen: set[int] = set()

    while True:
        current_id = id(current)
        if current_id in seen:
            raise RuntimeError("Detected a cycle in OmniGibson wrapper chain.")
        seen.add(current_id)

        inner = getattr(current, "env", None)
        if inner is None or inner is current:
            return current
        current = inner


def get_behavior_robot(env: Any) -> Any | None:
    """Return the task agent or first robot from a BEHAVIOR env."""
    base_env = unwrap_behavior_env(env)

    task = getattr(base_env, "task", None)
    if task is not None and hasattr(task, "get_agent"):
        robot = task.get_agent(base_env)
        if robot is not None:
            return robot

    robots = getattr(base_env, "robots", None)
    if robots:
        return robots[0]

    return None


def to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_task_reward(env: Any) -> Any | None:
    base_env = unwrap_behavior_env(env)
    try:
        return base_env.task.task_reward
    except Exception:
        return None


def stage_idx_from_info(info: dict | None) -> int | None:
    if info is None:
        return None
    task_reward_info = info.get("task_reward", {})
    if isinstance(task_reward_info, dict):
        return to_int_or_none(task_reward_info.get("current_stage_idx"))
    return to_int_or_none(info.get("current_stage_idx"))


def stage_idx_from_reward(env: Any) -> int | None:
    reward_obj = get_task_reward(env)
    if reward_obj is None:
        return None
    return to_int_or_none(getattr(reward_obj, "current_stage_idx", None))
