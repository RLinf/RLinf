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

"""Evaluation horizon helpers for RoboCasa365."""

from __future__ import annotations

from typing import Iterable


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
