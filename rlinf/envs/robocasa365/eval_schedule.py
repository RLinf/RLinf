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
from typing import Iterable


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


def resolve_robocasa365_episode_horizons(
    *,
    task_horizons: Iterable[int],
    max_episode_steps: int,
    episode_horizon_source: str,
) -> tuple[int, ...]:
    """Resolve per-environment horizons from the configured source."""

    registry_horizons = tuple(int(horizon) for horizon in task_horizons)
    if not registry_horizons:
        raise ValueError("RoboCasa365 requires at least one episode horizon.")
    if any(horizon <= 0 for horizon in registry_horizons):
        raise ValueError(
            "RoboCasa365 task horizons must be positive, got "
            f"{list(registry_horizons)}."
        )

    source = str(episode_horizon_source)
    if source == "task_horizon":
        return registry_horizons
    if source == "max_episode_steps":
        if max_episode_steps <= 0:
            raise ValueError(
                "env max_episode_steps must be positive when "
                "episode_horizon_source='max_episode_steps', got "
                f"{max_episode_steps}."
            )
        return (int(max_episode_steps),) * len(registry_horizons)
    raise ValueError(
        "RoboCasa365 episode_horizon_source must be one of "
        "{'task_horizon', 'max_episode_steps'}, got "
        f"{source!r}."
    )


def resolve_robocasa365_rollout_budget(
    *,
    episode_horizons: Iterable[int],
    num_action_chunks: int,
) -> int:
    """Resolve the shared rollout budget for the selected episode horizons."""

    horizons = tuple(int(horizon) for horizon in episode_horizons)
    if not horizons:
        raise ValueError("RoboCasa365 requires at least one episode horizon.")
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError(
            f"RoboCasa365 episode horizons must be positive, got {list(horizons)}."
        )
    if num_action_chunks <= 0:
        raise ValueError(
            f"actor.model.num_action_chunks must be positive, got {num_action_chunks}."
        )

    max_horizon = max(horizons)
    return (
        (max_horizon + num_action_chunks - 1) // num_action_chunks
    ) * num_action_chunks


def validate_robocasa365_eval_horizons(
    *,
    episode_horizons: Iterable[int],
    max_steps_per_rollout_epoch: int,
) -> int:
    """Validate that one eval rollout can finish every selected episode.

    Each environment truncates at its resolved horizon. The rollout worker,
    however, advances all evaluation environments for one shared number of
    action chunks, so that shared budget must cover the largest resolved
    horizon.

    Returns:
        The maximum resolved episode horizon.
    """

    horizons = [int(horizon) for horizon in episode_horizons]
    if not horizons:
        raise ValueError("RoboCasa365 eval requires at least one episode horizon.")
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError(
            f"RoboCasa365 episode horizons must be positive, got {horizons}."
        )
    if max_steps_per_rollout_epoch <= 0:
        raise ValueError(
            "env.eval.max_steps_per_rollout_epoch must be positive, got "
            f"{max_steps_per_rollout_epoch}."
        )
    max_episode_horizon = max(horizons)
    if max_steps_per_rollout_epoch < max_episode_horizon:
        raise ValueError(
            "RoboCasa365 eval rollout is shorter than at least one resolved "
            "episode horizon. The shared rollout budget must cover the largest "
            "resolved horizon so failed episodes are counted. "
            f"max_episode_horizon={max_episode_horizon}, "
            f"max_steps_per_rollout_epoch={max_steps_per_rollout_epoch}. Increase "
            "env.eval.max_steps_per_rollout_epoch to at least the maximum resolved "
            "episode horizon."
        )
    return max_episode_horizon


def resolve_robocasa365_eval_schedule(
    *,
    num_tasks: int,
    total_num_envs: int,
    eval_rollout_epoch: int,
    expected_episodes_per_task: int | None = None,
) -> Robocasa365EvalSchedule:
    """Resolve and validate a balanced parallel evaluation schedule.

    Each evaluation rollout epoch contributes one episode per environment.
    Therefore, the number of episodes evaluated for every task is:

    ``(total_num_envs / num_tasks) * eval_rollout_epoch``.

    ``eval_rollout_epoch`` remains an explicit algorithm setting. This helper
    never derives or overrides it.
    """

    if num_tasks <= 0:
        raise ValueError(f"num_tasks must be positive, got {num_tasks}.")
    if total_num_envs <= 0:
        raise ValueError(f"total_num_envs must be positive, got {total_num_envs}.")
    if eval_rollout_epoch <= 0:
        raise ValueError(
            f"eval_rollout_epoch must be positive, got {eval_rollout_epoch}."
        )
    if expected_episodes_per_task is not None and expected_episodes_per_task <= 0:
        raise ValueError(
            "expected_episodes_per_task must be positive when provided, got "
            f"{expected_episodes_per_task}."
        )
    if total_num_envs % num_tasks != 0:
        raise ValueError(
            "RoboCasa365 parallel eval requires total_num_envs to be divisible "
            f"by the selected task count, got total_num_envs={total_num_envs} "
            f"and num_tasks={num_tasks}."
        )

    envs_per_task = total_num_envs // num_tasks
    episodes_per_task = envs_per_task * eval_rollout_epoch
    if (
        expected_episodes_per_task is not None
        and episodes_per_task != expected_episodes_per_task
    ):
        raise ValueError(
            "RoboCasa365 parallel eval episode count does not match the "
            "configured target: "
            f"total_num_envs={total_num_envs}, num_tasks={num_tasks}, "
            f"envs_per_task={envs_per_task}, "
            f"eval_rollout_epoch={eval_rollout_epoch}, "
            f"calculated_episodes_per_task={episodes_per_task}, "
            f"expected_episodes_per_task={expected_episodes_per_task}."
        )

    return Robocasa365EvalSchedule(
        num_tasks=num_tasks,
        total_num_envs=total_num_envs,
        envs_per_task=envs_per_task,
        episodes_per_task=episodes_per_task,
        eval_rollout_epoch=eval_rollout_epoch,
    )
