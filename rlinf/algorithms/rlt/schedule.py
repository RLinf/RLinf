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

from dataclasses import dataclass
from typing import Any

import torch

from rlinf.utils.distributed import all_reduce_dict


@dataclass(frozen=True)
class RLTUpdatePlan:
    """Update count and metrics for one learner scheduling decision."""

    updates: int
    metrics: dict[str, float]


class RLTUpdateSchedule:
    """Track replay progress and derive RLT learner update budgets."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.schedule_cfg = cfg.algorithm.get("rlt_schedule", {}) or {}
        self.enabled = bool(self.schedule_cfg.get("enable", False))
        self.transitions_since_train = 0
        self.episodes_since_train = 0
        self.total_transitions_added = 0
        self.total_episodes_added = 0
        self._warmup_ready_total_transitions: int | None = None
        self._warmup_ready_total_episodes: int | None = None
        self.pending_update_budget = 0

    def rollout_version(self, *, model_version: int, update_step: int) -> int:
        """Return the version used to gate online actor rollout."""
        return int(update_step if self.enabled else model_version)

    def configure_buffer_dataset(self, buffer_dataset: Any) -> None:
        """Let the schedule, rather than the dataloader, enforce warmup."""
        if self.enabled:
            buffer_dataset.min_replay_buffer_size = 1

    def record_ingest(self, *, transitions: int, completed_episodes: int) -> None:
        """Record newly ingested rollout data."""
        if not self.enabled:
            return
        self.transitions_since_train += transitions
        self.episodes_since_train += completed_episodes
        self.total_transitions_added += transitions
        self.total_episodes_added += completed_episodes

    def plan(
        self,
        *,
        update_step: int,
        replay_size: int,
        demo_size: int,
        replay_metrics: dict[str, float] | None = None,
    ) -> RLTUpdatePlan:
        """Compute the number of learner updates currently due."""
        if not self.enabled:
            raise RuntimeError("RLTUpdateSchedule.plan() requires an enabled schedule.")

        replay_cfg = self.cfg.algorithm.replay_buffer
        min_buffer_size = int(
            self.schedule_cfg.get(
                "warmup_min_size",
                replay_cfg.get("min_buffer_size", 1),
            )
        )
        counters = self._global_counters(
            replay_size=replay_size,
            demo_size=demo_size,
        )
        buffer_ready = counters["min_replay_size"] >= min_buffer_size
        warmup_required_updates = int(
            self.schedule_cfg.get("warmup_post_collect_updates", 0)
        )
        if buffer_ready and self._warmup_ready_total_transitions is None:
            self._warmup_ready_total_transitions = int(
                counters["total_transitions_added"]
            )
            self._warmup_ready_total_episodes = int(counters["total_episodes_added"])

        train_every_transitions = int(
            self.schedule_cfg.get("train_every_transitions", 0)
        )
        train_every_episodes = int(self.schedule_cfg.get("train_every_episodes", 0))
        update_epoch = int(self.cfg.algorithm.get("update_epoch", 1))
        max_updates = int(self.schedule_cfg.get("max_updates_per_train_step", 0))

        updates_to_run = 0
        skip_reason = 0
        desired_total_updates = 0
        pending_updates = 0
        updates_scheduled = 0
        if update_epoch <= 0:
            skip_reason = 3
        elif not buffer_ready:
            skip_reason = 1
        else:
            online_transitions = max(
                int(counters["total_transitions_added"])
                - int(self._warmup_ready_total_transitions or 0),
                0,
            )
            online_episodes = max(
                int(counters["total_episodes_added"])
                - int(self._warmup_ready_total_episodes or 0),
                0,
            )
            if train_every_transitions <= 0 and train_every_episodes <= 0:
                online_cycles = online_transitions
            else:
                transition_cycles = (
                    online_transitions // train_every_transitions
                    if train_every_transitions > 0
                    else 0
                )
                episode_cycles = (
                    online_episodes // train_every_episodes
                    if train_every_episodes > 0
                    else 0
                )
                online_cycles = max(transition_cycles, episode_cycles)
            desired_total_updates = (
                warmup_required_updates + online_cycles * update_epoch
            )
            pending_updates = max(desired_total_updates - int(update_step), 0)
            updates_scheduled = pending_updates
            updates_to_run = pending_updates
            if max_updates > 0:
                updates_to_run = min(updates_to_run, max_updates)
            if updates_to_run <= 0:
                skip_reason = 2
        self.pending_update_budget = int(pending_updates)

        metrics = {
            "rlt/update_step": float(update_step),
            "rlt/ready_for_online": float(update_step >= warmup_required_updates),
            "rlt/warmup_required_updates": float(warmup_required_updates),
            "rlt/update_epoch": float(update_epoch),
            "rlt/max_updates_per_train_step": float(max_updates),
            "rlt/train_every_transitions": float(train_every_transitions),
            "rlt/train_every_episodes": float(train_every_episodes),
            "rlt/desired_total_updates": float(desired_total_updates),
            "rlt/pending_update_budget": float(self.pending_update_budget),
            "rlt/updates_scheduled": float(updates_scheduled),
            "rlt/updates_to_run": float(updates_to_run),
            "rlt/critic_updates_run": 0.0,
            "rlt/actor_updates_run": 0.0,
            "rlt/should_train": float(updates_to_run > 0),
            "rlt/skip_reason": float(skip_reason),
            "rlt/global_min_replay_size": float(counters["min_replay_size"]),
            "rlt/min_replay_buffer_size": float(min_buffer_size),
            "rlt/global_transitions_since_train": float(
                counters["transitions_since_train"]
            ),
            "rlt/global_total_transitions_added": float(
                counters["total_transitions_added"]
            ),
        }
        metrics.update(replay_metrics or {})
        return RLTUpdatePlan(updates=updates_to_run, metrics=metrics)

    def finish_updates(self, updates_run: int) -> None:
        """Consume update budget and reset per-train ingestion counters."""
        self.pending_update_budget = max(
            self.pending_update_budget - int(updates_run),
            0,
        )
        self.transitions_since_train = 0
        self.episodes_since_train = 0

    def _global_counters(
        self,
        *,
        replay_size: int,
        demo_size: int,
    ) -> dict[str, float]:
        summed = all_reduce_dict(
            {
                "transitions_since_train": float(self.transitions_since_train),
                "episodes_since_train": float(self.episodes_since_train),
                "total_transitions_added": float(self.total_transitions_added),
                "total_episodes_added": float(self.total_episodes_added),
            },
            op=torch.distributed.ReduceOp.SUM,
        )
        minimums = all_reduce_dict(
            {
                "min_replay_size": float(replay_size),
                "min_demo_size": float(demo_size),
            },
            op=torch.distributed.ReduceOp.MIN,
        )
        summed.update(minimums)
        return summed
