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

"""Parallel evaluation scheduling helpers for RoboCasa365."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Robocasa365EvalSchedule:
    """Resolved parallel evaluation dimensions."""

    num_tasks: int
    total_num_envs: int
    envs_per_task: int
    episodes_per_task: int
    eval_rollout_epoch: int

    @property
    def total_episodes(self) -> int:
        return self.num_tasks * self.episodes_per_task


def resolve_robocasa365_eval_schedule(
    *,
    num_tasks: int,
    total_num_envs: int,
    episodes_per_task: int,
) -> Robocasa365EvalSchedule:
    """Resolve an exact, balanced parallel evaluation schedule.

    Each evaluation rollout epoch contributes one episode per environment.
    Therefore, if ``total_num_envs / num_tasks`` environments are assigned to
    every task, the required rollout epochs are:

    ``episodes_per_task / (total_num_envs / num_tasks)``.
    """

    if num_tasks <= 0:
        raise ValueError(f"num_tasks must be positive, got {num_tasks}.")
    if total_num_envs <= 0:
        raise ValueError(f"total_num_envs must be positive, got {total_num_envs}.")
    if episodes_per_task <= 0:
        raise ValueError(
            f"episodes_per_task must be positive, got {episodes_per_task}."
        )
    if total_num_envs % num_tasks != 0:
        raise ValueError(
            "RoboCasa365 parallel eval requires total_num_envs to be divisible "
            f"by the selected task count, got total_num_envs={total_num_envs} "
            f"and num_tasks={num_tasks}."
        )

    envs_per_task = total_num_envs // num_tasks
    if envs_per_task > episodes_per_task:
        raise ValueError(
            "RoboCasa365 parallel eval would launch more environments per task "
            "than requested episodes, got "
            f"envs_per_task={envs_per_task} and "
            f"episodes_per_task={episodes_per_task}."
        )
    if episodes_per_task % envs_per_task != 0:
        raise ValueError(
            "RoboCasa365 parallel eval requires episodes_per_task to be "
            "divisible by envs_per_task for an exact episode count, got "
            f"episodes_per_task={episodes_per_task} and "
            f"envs_per_task={envs_per_task}."
        )

    return Robocasa365EvalSchedule(
        num_tasks=num_tasks,
        total_num_envs=total_num_envs,
        envs_per_task=envs_per_task,
        episodes_per_task=episodes_per_task,
        eval_rollout_epoch=episodes_per_task // envs_per_task,
    )
