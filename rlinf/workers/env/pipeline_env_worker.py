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

import torch

from rlinf.algorithms.registry import calculate_adv_and_returns
from rlinf.data.embodied_io_struct import (
    EmbodiedRolloutResult,
    Trajectory,
    convert_trajectories_to_batch,
)
from rlinf.scheduler import Channel, Worker
from rlinf.utils.metric_utils import (
    flatten_embodied_rollout_batch_for_train,
    pack_pipeline_train_batch_for_transport,
    process_embodied_rollout_batch_for_adv,
)
from rlinf.workers.env.env_worker import EnvWorker


class PipelineEnvWorker(EnvWorker):
    def _compute_advantages_and_returns_for_actor(
        self,
        rollout_batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": rollout_batch["rewards"],
            "dones": rollout_batch["dones"],
            "values": rollout_batch.get("prev_values", None),
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": rollout_batch.get("loss_mask", None),
            "loss_mask_sum": rollout_batch.get("loss_mask_sum", None),
            "normalize_advantages": False,
        }
        if self.cfg.algorithm.adv_type == "grpo_dynamic":
            kwargs["idx_to_traj"] = rollout_batch["idx_to_traj"]

        advantages_and_returns = calculate_adv_and_returns(**kwargs)
        rollout_batch.update(advantages_and_returns)
        if kwargs["loss_mask"] is not None:
            rollout_batch["loss_mask"] = kwargs["loss_mask"]
        if kwargs["loss_mask_sum"] is not None:
            rollout_batch["loss_mask_sum"] = kwargs["loss_mask_sum"]
        return rollout_batch

    def _prepare_training_pipeline_batch(
        self,
        trajectory: Trajectory,
    ) -> dict[str, torch.Tensor]:
        rollout_batch = convert_trajectories_to_batch([trajectory])
        rollout_batch = process_embodied_rollout_batch_for_adv(
            rollout_batch,
            rollout_epoch=1,
            auto_reset=self.cfg.env.train.auto_reset,
            ignore_terminations=self.cfg.env.train.ignore_terminations,
            reward_type=self.cfg.algorithm.reward_type,
            filter_rewards=self.cfg.algorithm.get("filter_rewards", False),
            group_size=self.cfg.algorithm.group_size,
            rewards_lower_bound=self.cfg.algorithm.get("rewards_lower_bound", None),
            rewards_upper_bound=self.cfg.algorithm.get("rewards_upper_bound", None),
        )
        if self.cfg.algorithm.adv_type == "grpo_dynamic":
            rollout_batch["idx_to_traj"] = list(
                range(rollout_batch["rewards"].shape[1])
            )
        rollout_batch = self._compute_advantages_and_returns_for_actor(rollout_batch)
        rollout_size = (
            rollout_batch["prev_logprobs"].shape[0]
            * rollout_batch["prev_logprobs"].shape[1]
        )
        return pack_pipeline_train_batch_for_transport(
            flatten_embodied_rollout_batch_for_train(
                rollout_batch,
                torch.arange(rollout_size),
            )
        )

    async def send_rollout_trajectories_pipeline(
        self,
        rollout_result: EmbodiedRolloutResult,
        channel: Channel,
    ) -> None:
        trajectories: list[Trajectory] = rollout_result.to_splited_trajectories(
            self.actor_split_num
        )
        for trajectory in trajectories:
            with self.worker_timer("prepare_pipeline_actor_train_batch"):
                actor_batch = self._prepare_training_pipeline_batch(trajectory)
            with self.worker_timer("send_pipeline_actor_train_batch"):
                channel.put(actor_batch, async_op=True)

    @Worker.timer("interact_pipeline")
    async def interact_pipeline(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel,
    ):
        env_metrics = await self._run_interact_once(
            input_channel,
            rollout_channel,
            reward_channel,
            actor_channel,
            cooperative_yield=False,
            use_training_pipeline=True,
        )

        for env in self.env_list:
            if self.enable_offload and hasattr(env, "offload"):
                env.offload()

        return env_metrics
