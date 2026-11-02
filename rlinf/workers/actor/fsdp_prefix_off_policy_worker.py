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

import queue
from typing import Any

import torch

from rlinf.algorithms.prefix_off_policy import build_prefix_off_policy_algorithm
from rlinf.algorithms.rlt.replay import RLTReplayService
from rlinf.algorithms.rlt.schedule import RLTUpdatePlan, RLTUpdateSchedule
from rlinf.data.schema.embodied_types import Trajectory
from rlinf.scheduler import Worker
from rlinf.utils.metric_utils import append_to_dict, compute_split_num
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
    AsyncEmbodiedSACFSDPPolicy,
)
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class PrefixOffPolicyRuntime:
    """Connect prefix algorithm components to the shared SAC/FSDP runtime."""

    def _init_prefix_runtime(self, cfg: Any) -> None:
        self.algorithm_component = build_prefix_off_policy_algorithm(cfg)
        self.replay_service = RLTReplayService(cfg, rank=self._rank)
        self.update_schedule = RLTUpdateSchedule(cfg)
        self._last_replay_metrics: dict[str, float] = {}

    def get_rollout_sync_version(self) -> int:
        """Return the model version that controls online actor routing."""
        return self.update_schedule.rollout_version(
            model_version=self.version,
            update_step=self.update_step,
        )

    def setup_sac_components(self) -> None:
        """Initialize shared replay storage, then attach RLT services."""
        super().setup_sac_components()
        self.replay_service.bind_buffers(self.replay_buffer, self.demo_buffer)
        self.update_schedule.configure_buffer_dataset(self.buffer_dataset)

    def before_actor_update(self) -> None:
        """Keep critic optimizer state out of actor gradient accumulation."""
        super().before_actor_update()
        if self.qf_optimizer is not None:
            self.qf_optimizer.zero_grad(set_to_none=True)

    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        """Delegate critic loss construction to the selected algorithm."""
        return self.algorithm_component.critic_loss(self, batch)

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        """Delegate actor loss construction to the selected algorithm."""
        return self.algorithm_component.actor_loss(self, batch)

    @Worker.timer("forward_alpha")
    def forward_alpha(self, batch):
        """Reject entropy-temperature training for prefix algorithms."""
        del batch
        raise NotImplementedError(
            "Prefix off-policy algorithms disable entropy/alpha training. Use "
            "algorithm.entropy_tuning.alpha_type=fixed_alpha."
        )

    def _ingest_rollout_trajectories(
        self,
        trajectories: list[Trajectory],
    ) -> None:
        result = self.replay_service.ingest(trajectories)
        self._last_replay_metrics = result.metrics
        self.update_schedule.record_ingest(
            transitions=result.transitions,
            completed_episodes=result.completed_episodes,
        )

    def _plan_updates(self) -> RLTUpdatePlan:
        demo_size = 0 if self.demo_buffer is None else self.demo_buffer.total_samples
        return self.update_schedule.plan(
            update_step=self.update_step,
            replay_size=self.replay_buffer.total_samples,
            demo_size=demo_size,
            replay_metrics=self._last_replay_metrics,
        )

    def _prepare_scheduled_training(self) -> None:
        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )
        self.model.train()

    def _finish_scheduled_training(
        self,
        *,
        metrics: dict[str, Any],
        plan: RLTUpdatePlan,
        critic_updates_run: int,
        actor_updates_run: int,
    ) -> dict[str, Any]:
        plan.metrics["rlt/critic_updates_run"] = float(critic_updates_run)
        plan.metrics["rlt/actor_updates_run"] = float(actor_updates_run)
        self.update_schedule.finish_updates(critic_updates_run)
        plan.metrics["rlt/pending_update_budget"] = float(
            self.update_schedule.pending_update_budget
        )
        append_to_dict(metrics, plan.metrics)
        mean_metrics = self.process_train_metrics(metrics)
        self._synchronize_after_training()
        return mean_metrics

    def _finish_idle_training(self, plan: RLTUpdatePlan) -> dict[str, Any]:
        mean_metrics = self.process_train_metrics(plan.metrics)
        self._synchronize_after_training()
        return mean_metrics

    @staticmethod
    def _synchronize_after_training() -> None:
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()


class PrefixOffPolicyFSDPPolicy(PrefixOffPolicyRuntime, EmbodiedSACFSDPPolicy):
    """Synchronous FSDP worker for registered prefix off-policy algorithms."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._init_prefix_runtime(cfg)

    @Worker.timer("actor/recv_traj")
    async def recv_rollout_trajectories(self, input_channel) -> None:
        """Receive trajectories and pass them to the configured replay service."""
        clear_memory(sync=False)
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)
        trajectories = [
            await input_channel.get(async_op=True).async_wait()
            for _ in range(split_num)
        ]
        self._ingest_rollout_trajectories(trajectories)

    def run_training(self):
        """Run shared SAC training or an RLT-scheduled update batch."""
        if not self.update_schedule.enabled:
            mean_metrics = super().run_training()
            if self._last_replay_metrics:
                mean_metrics = {**mean_metrics, **self._last_replay_metrics}
            return mean_metrics

        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)
        plan = self._plan_updates()
        if plan.updates <= 0:
            return self._finish_idle_training(plan)

        self._prepare_scheduled_training()
        metrics: dict[str, Any] = {}
        critic_updates_run = 0
        actor_updates_run = 0
        for _ in range(plan.updates):
            update_actor = self.update_step % int(self.critic_actor_ratio) == 0
            append_to_dict(metrics, self.update_one_epoch(train_actor=True))
            self.update_step += 1
            critic_updates_run += 1
            actor_updates_run += int(update_actor)

        return self._finish_scheduled_training(
            metrics=metrics,
            plan=plan,
            critic_updates_run=critic_updates_run,
            actor_updates_run=actor_updates_run,
        )


class AsyncPrefixOffPolicyFSDPPolicy(
    PrefixOffPolicyRuntime,
    AsyncEmbodiedSACFSDPPolicy,
):
    """Asynchronous FSDP worker for registered prefix off-policy algorithms."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._init_prefix_runtime(cfg)

    def _drain_received_trajectories(
        self,
        max_trajectories: int | None = None,
    ) -> None:
        if getattr(self, "_recv_queue", None) is None:
            return
        trajectories = []
        while max_trajectories is None or len(trajectories) < max_trajectories:
            try:
                trajectories.append(self._recv_queue.get_nowait())
            except queue.Empty:
                break
        if trajectories:
            self._ingest_rollout_trajectories(trajectories)

    async def run_training(self):
        """Drain rollout data while running RLT-scheduled learner updates."""
        import asyncio

        if not self.update_schedule.enabled:
            mean_metrics = await super().run_training()
            if self._last_replay_metrics:
                mean_metrics = {**mean_metrics, **self._last_replay_metrics}
            return mean_metrics

        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)
        max_trajectories = self.cfg.actor.get("recv_drain_max_trajectories", 1024)
        self._drain_received_trajectories(max_trajectories=max_trajectories)
        plan = self._plan_updates()
        if plan.updates <= 0:
            return self._finish_idle_training(plan)

        self._prepare_scheduled_training()
        metrics: dict[str, Any] = {}
        critic_updates_run = 0
        actor_updates_run = 0
        for _ in range(plan.updates):
            await asyncio.sleep(0)
            self._drain_received_trajectories(max_trajectories=max_trajectories)
            update_actor = self.update_step % int(self.critic_actor_ratio) == 0
            append_to_dict(metrics, self.update_one_epoch(train_actor=True))
            self.update_step += 1
            critic_updates_run += 1
            actor_updates_run += int(update_actor)

        return self._finish_scheduled_training(
            metrics=metrics,
            plan=plan,
            critic_updates_run=critic_updates_run,
            actor_updates_run=actor_updates_run,
        )
