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

"""Training schedule helpers for RLT Stage 2 TD3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SKIP_REASON_NONE = 0
SKIP_REASON_BUFFER_NOT_READY = 1
SKIP_REASON_NO_PENDING_UPDATES = 2
SKIP_REASON_UPDATE_RATIO_DISABLED = 3


@dataclass(frozen=True)
class RLTActorLossWeights:
    bc_weight: float
    q_weight: float
    delta_weight: float
    in_warmup: bool
    ramp_progress: float


@dataclass(frozen=True)
class RLTUpdateSchedule:
    warmup_required_updates: int
    update_ratio: int
    max_updates_per_train_step: int
    train_every_transitions: int
    train_every_episodes: int
    desired_total_updates: int
    pending_update_budget: int
    updates_scheduled: int
    updates_to_run: int
    should_train: bool
    skip_reason: int


def resolve_warmup_required_updates(cfg: Any) -> int:
    """Return the number of TD3 updates required before online control."""
    td3_bc_cfg = cfg.algorithm.get("td3_bc", {})
    warmup_required_updates = int(
        td3_bc_cfg.get(
            "warmup_updates",
            cfg.algorithm.get("warmup_post_collect_updates", 0),
        )
    )
    if warmup_required_updates < 0:
        raise ValueError(
            "algorithm.td3_bc.warmup_updates must be >= 0, "
            f"got {warmup_required_updates}."
        )
    return warmup_required_updates


def resolve_actor_loss_weights(cfg: Any, update_step: int) -> RLTActorLossWeights:
    """Resolve BC/Q weights for the current actor update."""
    td3_bc_cfg = cfg.algorithm.get("td3_bc", {})
    stage2_cfg = cfg.actor.model.rlt_stage2
    loss_warmup_updates = int(
        td3_bc_cfg.get(
            "actor_loss_warmup_updates",
            cfg.algorithm.get("actor_loss_warmup_updates", 0),
        )
    )
    in_warmup = int(update_step) < loss_warmup_updates
    warmup_bc_weight = float(
        td3_bc_cfg.get(
            "warmup_bc_weight",
            stage2_cfg.get(
                "warmup_bc_weight",
                stage2_cfg.get("bc_regularizer_beta", 1.0),
            ),
        )
    )
    warmup_q_weight = float(
        td3_bc_cfg.get(
            "warmup_q_weight",
            stage2_cfg.get("warmup_q_weight", 0.1),
        )
    )
    online_bc_weight = float(
        td3_bc_cfg.get(
            "online_bc_weight",
            stage2_cfg.get(
                "online_bc_weight",
                stage2_cfg.get("bc_regularizer_beta", 1.0),
            ),
        )
    )
    online_q_weight = float(
        td3_bc_cfg.get(
            "online_q_weight",
            stage2_cfg.get("online_q_weight", 0.1),
        )
    )
    if in_warmup:
        bc_weight = warmup_bc_weight
        q_weight = warmup_q_weight
        ramp_progress = 0.0
    else:
        ramp_updates = int(
            td3_bc_cfg.get(
                "actor_loss_ramp_updates",
                cfg.algorithm.get("actor_loss_ramp_updates", 0),
            )
        )
        if ramp_updates > 0:
            ramp_progress = min(
                1.0,
                max(
                    0.0,
                    float(int(update_step) - loss_warmup_updates + 1)
                    / float(ramp_updates),
                ),
            )
        else:
            ramp_progress = 1.0
        bc_weight = warmup_bc_weight + ramp_progress * (
            online_bc_weight - warmup_bc_weight
        )
        q_weight = warmup_q_weight + ramp_progress * (
            online_q_weight - warmup_q_weight
        )
    delta_weight = float(
        td3_bc_cfg.get("delta_weight", stage2_cfg.get("delta_weight", 0.0))
    )
    return RLTActorLossWeights(
        bc_weight=bc_weight,
        q_weight=q_weight,
        delta_weight=delta_weight,
        in_warmup=in_warmup,
        ramp_progress=ramp_progress,
    )


def resolve_update_schedule(
    cfg: Any,
    *,
    update_step: int,
    buffer_ready: bool,
    global_total_transitions_added: int,
    global_total_episodes_added: int,
    warmup_ready_total_transitions: int | None,
    warmup_ready_total_episodes: int | None,
) -> RLTUpdateSchedule:
    """Resolve how many TD3 updates should run for this train call."""
    warmup_required_updates = resolve_warmup_required_updates(cfg)
    update_ratio = int(cfg.algorithm.get("update_epoch", 1))
    max_updates_per_train_step = int(
        cfg.algorithm.get("max_updates_per_train_step", 0)
    )
    train_every_transitions = int(cfg.algorithm.get("train_every_transitions", 0))
    train_every_episodes = int(cfg.algorithm.get("train_every_episodes", 0))

    desired_total_updates = 0
    if (
        buffer_ready
        and warmup_ready_total_transitions is not None
        and update_ratio > 0
    ):
        online_transitions_added = max(
            int(global_total_transitions_added) - int(warmup_ready_total_transitions),
            0,
        )
        online_episodes_added = max(
            int(global_total_episodes_added) - int(warmup_ready_total_episodes or 0),
            0,
        )
        transition_cycles = (
            online_transitions_added // train_every_transitions
            if train_every_transitions > 0
            else 0
        )
        episode_cycles = (
            online_episodes_added // train_every_episodes
            if train_every_episodes > 0
            else 0
        )
        if train_every_transitions <= 0 and train_every_episodes <= 0:
            online_update_cycles = online_transitions_added
        else:
            online_update_cycles = max(transition_cycles, episode_cycles)
        desired_total_updates = (
            warmup_required_updates + online_update_cycles * update_ratio
        )

    pending_update_budget = max(int(desired_total_updates) - int(update_step), 0)
    updates_scheduled = int(pending_update_budget)
    should_train = bool(buffer_ready and updates_scheduled > 0)

    skip_reason = SKIP_REASON_NONE
    if update_ratio <= 0:
        skip_reason = SKIP_REASON_UPDATE_RATIO_DISABLED
    elif not buffer_ready:
        skip_reason = SKIP_REASON_BUFFER_NOT_READY
    elif not should_train:
        skip_reason = SKIP_REASON_NO_PENDING_UPDATES

    updates_to_run = updates_scheduled
    if max_updates_per_train_step > 0:
        updates_to_run = min(updates_to_run, max_updates_per_train_step)

    return RLTUpdateSchedule(
        warmup_required_updates=warmup_required_updates,
        update_ratio=update_ratio,
        max_updates_per_train_step=max_updates_per_train_step,
        train_every_transitions=train_every_transitions,
        train_every_episodes=train_every_episodes,
        desired_total_updates=desired_total_updates,
        pending_update_budget=pending_update_budget,
        updates_scheduled=updates_scheduled,
        updates_to_run=updates_to_run,
        should_train=should_train,
        skip_reason=skip_reason,
    )
