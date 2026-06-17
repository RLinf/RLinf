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

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


SKIP_REASON_NONE = 0
SKIP_REASON_BUFFER_NOT_READY = 1
SKIP_REASON_NO_PENDING_UPDATES = 2
SKIP_REASON_UPDATE_RATIO_DISABLED = 3

PHASE_WARMUP = "warmup"
PHASE_WARMUP_WAIT_ONLINE = "warmup_wait_online"
PHASE_ONLINE = "online"

PHASE_TO_ID = {
    PHASE_WARMUP: 0,
    PHASE_WARMUP_WAIT_ONLINE: 1,
    PHASE_ONLINE: 2,
}


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp for status payloads."""
    return datetime.now(timezone.utc).isoformat()


def resolve_training_phase(
    *,
    buffer_ready: bool,
    ready_for_online: bool,
) -> str:
    """Resolve the actor-side training phase."""
    if not buffer_ready:
        return PHASE_WARMUP
    if not ready_for_online:
        return PHASE_WARMUP_WAIT_ONLINE
    return PHASE_ONLINE


def resolve_rollout_phase(
    *,
    ready_for_online: bool,
    student_control_rate: float,
) -> str:
    """Resolve the env-side rollout phase from rollout forward inputs."""
    if not ready_for_online:
        return PHASE_WARMUP
    if float(student_control_rate) > 0.0:
        return PHASE_ONLINE
    return PHASE_WARMUP_WAIT_ONLINE


def phase_id(phase: str) -> int:
    """Return the stable numeric id for a phase string."""
    return PHASE_TO_ID.get(phase, -1)


def metric_mean(
    metrics: dict[str, Any],
    key: str,
    *,
    default: float | None = None,
) -> float | None:
    """Return a float mean for a tensor/list/scalar metric."""
    if key not in metrics:
        return default
    value = metrics[key]
    try:
        import torch
    except ModuleNotFoundError:
        torch = None

    if torch is None:
        if value is None:
            return default
        if isinstance(value, list):
            if not value:
                return default
            return float(sum(float(item) for item in value) / len(value))
        return float(value)

    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return float(value.detach().float().mean().cpu().item())
    if isinstance(value, list):
        tensors = [
            item.detach().float().reshape(-1).cpu()
            if isinstance(item, torch.Tensor)
            else torch.as_tensor(item, dtype=torch.float32).reshape(-1)
            for item in value
        ]
        tensors = [tensor for tensor in tensors if tensor.numel() > 0]
        if not tensors:
            return default
        return float(torch.cat(tensors).mean().item())
    if value is None:
        return default
    return float(value)


def write_status_json(path: str, payload: dict[str, Any]) -> None:
    """Atomically write a small JSON status payload."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(tmp_path, path)


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


@dataclass(frozen=True)
class RLTReplayReadiness:
    warmup_min_size: int
    min_demo_buffer_size: int
    demo_ready: bool
    use_demo: bool
    buffer_ready: bool


@dataclass(frozen=True)
class RLTTrainingPlan:
    schedule: RLTUpdateSchedule
    readiness: RLTReplayReadiness
    ready_for_online: bool
    status_phase: str
    pending_update_budget: int


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


def resolve_replay_readiness(
    cfg: Any,
    *,
    has_demo_buffer: bool,
    global_min_replay_size: int,
    global_min_demo_size: int,
    min_replay_buffer_size: int = 0,
    min_demo_buffer_size: int = 0,
) -> RLTReplayReadiness:
    """Resolve whether replay/demo buffers are ready for RLT updates."""
    replay_cfg = cfg.algorithm.get("replay_buffer", {})
    warmup_min_size = int(
        replay_cfg.get(
            "min_buffer_size",
            cfg.algorithm.get("warmup_min_size", 1),
        )
    )
    warmup_min_size = max(warmup_min_size, int(min_replay_buffer_size))
    configured_min_demo_size = (
        int(cfg.algorithm.get("demo_buffer", {}).get("min_buffer_size", 0))
        if has_demo_buffer
        else 0
    )
    min_demo_buffer_size = max(configured_min_demo_size, int(min_demo_buffer_size))
    if has_demo_buffer:
        min_demo_buffer_size = max(min_demo_buffer_size, 1)
    demo_ready = (not has_demo_buffer) or global_min_demo_size >= min_demo_buffer_size
    use_demo = bool(has_demo_buffer and demo_ready)
    buffer_ready = global_min_replay_size >= warmup_min_size and demo_ready
    return RLTReplayReadiness(
        warmup_min_size=warmup_min_size,
        min_demo_buffer_size=min_demo_buffer_size,
        demo_ready=demo_ready,
        use_demo=use_demo,
        buffer_ready=buffer_ready,
    )


class RLTStage2TrainingScheduler:
    """Stateful RLT Stage 2 update scheduler.

    This object owns RLT-specific warmup/update-budget state. The actor worker
    supplies distributed replay counters and executes the returned update plan.
    """

    def __init__(self) -> None:
        self.pending_update_budget = 0
        self.warmup_ready_total_transitions: int | None = None
        self.warmup_ready_total_episodes: int | None = None

    def plan(
        self,
        cfg: Any,
        *,
        update_step: int,
        has_demo_buffer: bool,
        global_counters: dict[str, float],
        global_min_replay_size: int,
        global_min_demo_size: int,
        min_replay_buffer_size: int = 0,
        min_demo_buffer_size: int = 0,
    ) -> RLTTrainingPlan:
        """Resolve the current RLT training plan from distributed counters."""
        readiness = resolve_replay_readiness(
            cfg,
            has_demo_buffer=has_demo_buffer,
            global_min_replay_size=global_min_replay_size,
            global_min_demo_size=global_min_demo_size,
            min_replay_buffer_size=min_replay_buffer_size,
            min_demo_buffer_size=min_demo_buffer_size,
        )
        global_total_transitions_added = int(
            global_counters["total_transitions_added"]
        )
        global_total_episodes_added = int(global_counters["total_episodes_added"])
        if (
            readiness.buffer_ready
            and self.warmup_ready_total_transitions is None
        ):
            self.warmup_ready_total_transitions = global_total_transitions_added
            self.warmup_ready_total_episodes = global_total_episodes_added

        schedule = resolve_update_schedule(
            cfg,
            update_step=update_step,
            buffer_ready=readiness.buffer_ready,
            global_total_transitions_added=global_total_transitions_added,
            global_total_episodes_added=global_total_episodes_added,
            warmup_ready_total_transitions=self.warmup_ready_total_transitions,
            warmup_ready_total_episodes=self.warmup_ready_total_episodes,
        )
        self.pending_update_budget = schedule.pending_update_budget
        ready_for_online = int(update_step) >= schedule.warmup_required_updates
        status_phase = resolve_training_phase(
            buffer_ready=readiness.buffer_ready,
            ready_for_online=ready_for_online,
        )
        return RLTTrainingPlan(
            schedule=schedule,
            readiness=readiness,
            ready_for_online=ready_for_online,
            status_phase=status_phase,
            pending_update_budget=self.pending_update_budget,
        )

    def finish_updates(self, critic_updates_run: int) -> int:
        """Consume scheduled update budget after local critic updates."""
        self.pending_update_budget = max(
            self.pending_update_budget - int(critic_updates_run),
            0,
        )
        return self.pending_update_budget

    @staticmethod
    def ready_for_online(plan: RLTTrainingPlan, update_step: int) -> bool:
        """Return whether the current update step opens online control."""
        return int(update_step) >= plan.schedule.warmup_required_updates

    def status_phase(self, plan: RLTTrainingPlan, update_step: int) -> str:
        """Return the status phase for the current update step."""
        return resolve_training_phase(
            buffer_ready=plan.readiness.buffer_ready,
            ready_for_online=self.ready_for_online(plan, update_step),
        )

    def metrics(
        self,
        *,
        plan: RLTTrainingPlan,
        update_step: int,
        global_counters: dict[str, float],
        global_min_replay_size: int,
        global_min_demo_size: int,
        should_train: bool,
        skip_reason: int,
        critic_updates_run: int = 0,
        actor_updates_run: int = 0,
    ) -> dict[str, float]:
        """Build scalar metrics for RLT schedule/status logging."""
        schedule = plan.schedule
        readiness = plan.readiness
        ready_for_online = self.ready_for_online(plan, update_step)
        status_phase = self.status_phase(plan, update_step)
        return {
            "rlt_stage2/update_step": float(update_step),
            "rlt_stage2/critic_updates_run": float(critic_updates_run),
            "rlt_stage2/actor_updates_run": float(actor_updates_run),
            "rlt_stage2/should_train": float(should_train),
            "rlt_stage2/skip_reason": float(skip_reason),
            "rlt_stage2/ready_for_online": float(ready_for_online),
            "rlt_stage2/status_phase_id": float(phase_id(status_phase)),
            "rlt_stage2/global_min_replay_size": float(global_min_replay_size),
            "rlt_stage2/global_min_demo_size": float(global_min_demo_size),
            "rlt_stage2/min_replay_buffer_size": float(readiness.warmup_min_size),
            "rlt_stage2/min_demo_buffer_size": float(
                readiness.min_demo_buffer_size
            ),
            "rlt_stage2/update_epoch": float(schedule.update_ratio),
            "rlt_stage2/warmup_required_updates": float(
                schedule.warmup_required_updates
            ),
            "rlt_stage2/pending_update_budget": float(
                self.pending_update_budget
            ),
            "rlt_stage2/updates_scheduled": float(schedule.updates_scheduled),
            "rlt_stage2/global_transitions_since_train": float(
                global_counters["transitions_since_train"]
            ),
            "rlt_stage2/global_total_transitions_added": float(
                global_counters["total_transitions_added"]
            ),
        }

    def status_payload(
        self,
        *,
        plan: RLTTrainingPlan,
        rank: int,
        update_step: int,
        global_min_replay_size: int,
        should_train: bool,
        skip_reason: int,
        critic_updates_run: int = 0,
        actor_updates_run: int = 0,
        global_total_transitions_added: int = 0,
        global_total_episodes_added: int = 0,
    ) -> dict[str, Any]:
        """Build the actor-side status JSON payload."""
        ready_for_online = self.ready_for_online(plan, update_step)
        status_phase = self.status_phase(plan, update_step)
        return {
            "timestamp": utc_timestamp(),
            "component": "actor",
            "rank": int(rank),
            "phase": status_phase,
            "phase_id": phase_id(status_phase),
            "ready_for_online": bool(ready_for_online),
            "buffer_ready": bool(plan.readiness.buffer_ready),
            "update_step": int(update_step),
            "warmup_required_updates": int(plan.schedule.warmup_required_updates),
            "global_min_replay_size": int(global_min_replay_size),
            "warmup_min_size": int(plan.readiness.warmup_min_size),
            "pending_update_budget": int(self.pending_update_budget),
            "updates_scheduled": int(plan.schedule.updates_scheduled),
            "critic_updates_run": int(critic_updates_run),
            "actor_updates_run": int(actor_updates_run),
            "should_train": bool(should_train),
            "skip_reason": int(skip_reason),
            "global_total_transitions_added": int(global_total_transitions_added),
            "global_total_episodes_added": int(global_total_episodes_added),
        }

    def state_dict(self) -> dict[str, int | None]:
        """Return checkpointable scheduler state."""
        return {
            "pending_update_budget": self.pending_update_budget,
            "warmup_ready_total_transitions": self.warmup_ready_total_transitions,
            "warmup_ready_total_episodes": self.warmup_ready_total_episodes,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load scheduler state from checkpoint fields."""
        self.pending_update_budget = int(state.get("pending_update_budget", 0))
        warmup_ready_total_transitions = state.get(
            "warmup_ready_total_transitions",
            None,
        )
        self.warmup_ready_total_transitions = (
            None
            if warmup_ready_total_transitions is None
            else int(warmup_ready_total_transitions)
        )
        warmup_ready_total_episodes = state.get(
            "warmup_ready_total_episodes",
            None,
        )
        self.warmup_ready_total_episodes = (
            None
            if warmup_ready_total_episodes is None
            else int(warmup_ready_total_episodes)
        )
