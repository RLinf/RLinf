# Copyright 2025 The RLinf Authors.
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

import asyncio
import time
from typing import TYPE_CHECKING

from omegaconf.omegaconf import DictConfig

from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.runner_utils import check_progress

if TYPE_CHECKING:
    from rlinf.workers.actor.async_ppo_fsdp_worker import (
        AsyncPPOEmbodiedFSDPActor,
    )
    from rlinf.workers.env.async_env_worker import AsyncEnvWorker
    from rlinf.workers.rollout.hf.async_huggingface_worker import (
        AsyncMultiStepRolloutWorker,
    )


class AsyncPPOEmbodiedRunner(EmbodiedRunner):
    """Runner for async PPO with long-running env and rollout workers."""

    def __init__(
        self,
        cfg: DictConfig,
        actor: "AsyncPPOEmbodiedFSDPActor",
        rollout: "AsyncMultiStepRolloutWorker",
        env: "AsyncEnvWorker",
        critic=None,
        reward=None,
    ):
        if reward is not None and cfg.get("reward", {}).get("use_output_step", 0) != 0:
            raise ValueError(
                "Async PPO with reward workers only supports reward.use_output_step=0. "
                "Delayed reward activation is not implemented for the long-running "
                "async env/rollout pipeline."
            )

        super().__init__(
            cfg=cfg,
            actor=actor,
            rollout=rollout,
            env=env,
            reward=reward,
            critic=critic,
        )
        self.env_metric_channel = Channel.create("EnvMetric")
        self.rollout_metric_channel = Channel.create("RolloutMetric")
        self.reward_metric_channel = Channel.create("RewardMetric")
        self.recompute_logprobs = bool(
            self.cfg.rollout.get("recompute_logprobs", False)
        )

        if self.cfg.runner.val_check_interval > 0:
            self.logger.warning(
                "Validation check interval is set to a positive value, but validation is not implemented for AsyncPPOEmbodiedRunner, so validation will be skipped."
            )

    def get_rollout_metrics(self) -> tuple[dict, list[dict]]:
        results: list[dict] = []
        while True:
            try:
                result = self.rollout_metric_channel.get_nowait()
                results.append(result)
            except asyncio.QueueEmpty:
                break

        if not results:
            return {}, []

        time_metrics, ranked_time_metrics_list = self._process_ranked_numeric_results(
            results,
            metric_field="time",
            intra_rank_reduction="sum",
            cross_rank_reduction="max",
        )
        return time_metrics, ranked_time_metrics_list

    def get_reward_metrics(self) -> tuple[dict, list[dict], dict]:
        results: list[dict] = []
        while True:
            try:
                result = self.reward_metric_channel.get_nowait()
                results.append(result)
            except asyncio.QueueEmpty:
                break

        if not results:
            return {}, [], {}

        time_metrics, ranked_time_metrics_list = self._process_ranked_numeric_results(
            results,
            metric_field="time",
            intra_rank_reduction="sum",
            cross_rank_reduction="max",
        )
        reward_metric_payloads = [result.get("reward", {}) for result in results]
        reward_metrics = {
            f"reward/{k}": v
            for k, v in self._aggregate_numeric_metrics(reward_metric_payloads).items()
        }
        reward_metrics.update(
            {
                f"reward/{k}": v
                for k, v in self._aggregate_numeric_metrics(
                    [
                        {
                            key: value
                            for key, value in metrics.items()
                            if key.endswith("_max")
                        }
                        for metrics in reward_metric_payloads
                    ],
                    reduction="max",
                ).items()
            }
        )
        return time_metrics, ranked_time_metrics_list, reward_metrics

    def get_env_metrics(self) -> tuple[dict, list[dict], list[dict]]:
        results: list[dict] = []
        while True:
            try:
                result = self.env_metric_channel.get_nowait()
                results.append(result)
            except asyncio.QueueEmpty:
                break

        if not results:
            return {}, [], []

        time_metrics, ranked_time_metrics_list = self._process_ranked_numeric_results(
            results,
            metric_field="time",
            intra_rank_reduction="sum",
            cross_rank_reduction="max",
        )
        env_metrics, ranked_env_metrics_list = self._process_ranked_eval_results(
            results, metric_field="env"
        )
        if not env_metrics:
            return {**time_metrics}, ranked_time_metrics_list, ranked_env_metrics_list

        return (
            {**env_metrics, **time_metrics},
            ranked_time_metrics_list,
            ranked_env_metrics_list,
        )

    def update_rollout_weights(self) -> None:
        rollout_handle = self.rollout.sync_model_from_actor()
        self.actor.sync_model_to_rollout().wait()
        rollout_handle.wait()

    def _start_reward_worker(self) -> Handle | None:
        if self.reward is None:
            return None

        self.logger.info("Activating reward worker at step %s", self.global_step)
        self.reward_channel = Channel.create("Reward")
        self.reward_initialized = True
        return self.reward.compute_rewards_async(
            input_channel=self.reward_channel,
            output_channel=self.env_channel,
            metric_channel=self.reward_metric_channel,
        )

    def run(self) -> None:
        start_step = self.global_step
        start_time = time.time()

        self.actor.set_global_step(self.global_step).wait()
        self.rollout.set_global_step(self.global_step).wait()
        self.update_rollout_weights()
        reward_handle = self._start_reward_worker()

        env_handle: Handle = self.env.interact(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
            reward_channel=self.reward_channel,
            actor_channel=self.actor_channel,
            metric_channel=self.env_metric_channel,
            max_rollouts=self.max_steps - self.global_step,
        )
        rollout_handle: Handle = self.rollout.generate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
            metric_channel=self.rollout_metric_channel,
        )

        actor_handle: Handle = self.actor.recv_rollout_trajectories(
            input_channel=self.actor_channel
        )

        while self.global_step < self.max_steps:
            # Use the step we're ABOUT to run as the profiling key, mirroring
            # ``EmbodiedRunner.run`` which gates before ``self.global_step += 1``.
            profiled_step = (
                self.global_step
                if self._should_profile_step(self.global_step)
                else None
            )
            if profiled_step is not None:
                self._open_profiling_window(profiled_step)
            with self.timer("step"):
                with self.timer("construct_rollout_batch"):
                    rollout_data_metrics = self.actor.construct_rollout_batch().wait()
                if self.recompute_logprobs:
                    raise NotImplementedError

                with self.timer("cal_adv_and_returns"):
                    rollout_metrics_list = (
                        self.actor.compute_advantages_and_returns().wait()
                    )

                with self.timer("actor_training"):
                    actor_training_handle = self.actor.run_training()
                    training_metrics = actor_training_handle.wait()

                self.global_step += 1
                self.actor.set_global_step(self.global_step).wait()
                self.rollout.set_global_step(self.global_step).wait()
                if self.global_step < self.max_steps:
                    with self.timer("update_rollout_weights"):
                        self.update_rollout_weights()

            time_metrics = self.timer.consume_durations()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            actor_time_metrics, actor_time_metrics_per_rank = (
                actor_training_handle.consume_durations(return_per_rank=True)
            )
            actor_time_metrics = {
                f"time/actor/{k}": v for k, v in actor_time_metrics.items()
            }
            time_metrics.update(actor_time_metrics)

            train_metrics = {
                f"train/{k}": v
                for k, v in self._aggregate_numeric_metrics(training_metrics).items()
            }
            rollout_metrics = {
                f"rollout/{k}": v
                for k, v in self._aggregate_numeric_metrics(
                    rollout_metrics_list
                ).items()
            }
            env_metrics, env_time_metrics_per_rank, env_metrics_per_rank = (
                self.get_env_metrics()
            )
            rollout_time_metrics, rollout_time_metrics_per_rank = (
                self.get_rollout_metrics()
            )
            reward_time_metrics, reward_time_metrics_per_rank, reward_metrics = (
                self.get_reward_metrics()
            )
            self.metric_logger.log(train_metrics, self.global_step)
            if env_metrics:
                self.metric_logger.log(env_metrics, self.global_step)
            if rollout_time_metrics:
                self.metric_logger.log(rollout_time_metrics, self.global_step)
            if reward_time_metrics:
                self.metric_logger.log(reward_time_metrics, self.global_step)
            if reward_metrics:
                self.metric_logger.log(reward_metrics, self.global_step)
            self.metric_logger.log(rollout_metrics, self.global_step)
            if rollout_data_metrics:
                data_staleness_metrics = {
                    f"rollout/{k}": v
                    for k, v in self._aggregate_numeric_metrics(
                        rollout_data_metrics
                    ).items()
                }
                self.metric_logger.log(data_staleness_metrics, self.global_step)
            self.metric_logger.log(time_metrics, self.global_step)
            self._log_ranked_metrics(
                metrics_list=training_metrics,
                step=self.global_step,
                prefix="train",
                worker_group_name=self.actor.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=actor_time_metrics_per_rank,
                step=self.global_step,
                prefix="time/actor",
                worker_group_name=self.actor.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=rollout_metrics_list,
                step=self.global_step,
                prefix="rollout",
                worker_group_name=self.actor.worker_group_name,
            )
            self._log_ranked_metrics(
                metrics_list=env_time_metrics_per_rank,
                step=self.global_step,
                prefix="time/env",
                worker_group_name=self.env.worker_group_name,
                add_prefix=False,
            )
            self._log_ranked_metrics(
                metrics_list=env_metrics_per_rank,
                step=self.global_step,
                prefix="env",
                worker_group_name=self.env.worker_group_name,
                add_prefix=False,
            )
            self._log_ranked_metrics(
                metrics_list=rollout_time_metrics_per_rank,
                step=self.global_step,
                prefix="time/rollout",
                worker_group_name=self.rollout.worker_group_name,
                add_prefix=False,
            )
            if self.reward is not None:
                self._log_ranked_metrics(
                    metrics_list=reward_time_metrics_per_rank,
                    step=self.global_step,
                    prefix="time/reward",
                    worker_group_name=self.reward.worker_group_name,
                    add_prefix=False,
                )

            logging_metrics = {**time_metrics, **train_metrics, **rollout_metrics}
            if env_metrics:
                logging_metrics.update(env_metrics)
            if reward_time_metrics:
                logging_metrics.update(reward_time_metrics)
            if reward_metrics:
                logging_metrics.update(reward_metrics)

            self.print_metrics_table_async(
                self.global_step - 1,
                self.max_steps,
                start_time,
                logging_metrics,
                start_step,
            )

            _, save_model, _ = check_progress(
                self.global_step,
                self.max_steps,
                self.cfg.runner.val_check_interval,
                self.cfg.runner.save_interval,
                1.0,
                run_time_exceeded=False,
            )
            if save_model:
                self._save_checkpoint()

            if profiled_step is not None:
                self._close_profiling_window(profiled_step)

        self.metric_logger.finish()

        self.stop_logging = True
        self.log_queue.join()
        self.log_thread.join(timeout=1.0)

        self.env.stop().wait()
        self.rollout.stop().wait()
        if self.reward is not None:
            self.reward.stop().wait()
            reward_handle.wait()

        env_handle.wait()
        rollout_handle.wait()
        actor_handle.wait()
