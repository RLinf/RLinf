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
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.runner_utils import check_progress

if TYPE_CHECKING:
    from rlinf.workers.actor.async_ppo_fsdp_worker import (
        AsyncPPOEmbodiedFSDPActor,
    )
    from rlinf.workers.env.async_ppo_env_worker import AsyncPPOEnvWorker
    from rlinf.workers.rollout.hf.async_ppo_huggingface_worker import (
        AsyncPPOMultiStepRolloutWorker,
    )


class AsyncPPOEmbodiedRunner(EmbodiedRunner):
    """Runner for async PPO with long-running env and rollout workers."""

    def __init__(
        self,
        cfg: DictConfig,
        actor: "AsyncPPOEmbodiedFSDPActor",
        rollout: "AsyncPPOMultiStepRolloutWorker",
        env: "AsyncPPOEnvWorker",
        critic=None,
        reward=None,
        run_timer=None,
    ):
        super().__init__(cfg, actor, rollout, env, critic, reward, run_timer)
        self.env_metrics_channel = Channel.create("EnvMetrics")
        self.recompute_logprobs = bool(self.cfg.rollout.get("recompute_logprobs", True))

        if self.cfg.runner.val_check_interval > 0:
            self.logger.warning(
                "Validation check interval is set to a positive value, but validation is not implemented for AsyncPPOEmbodiedRunner, so validation will be skipped."
            )

    def get_env_metrics(self) -> dict:
        results = []
        while True:
            try:
                result = self.env_metrics_channel.get_nowait()
                results.append(result)
            except asyncio.QueueEmpty:
                break

        if not results:
            return {}

        env_metrics = compute_evaluate_metrics(results)
        env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
        return env_metrics

    def update_rollout_weights(self) -> None:
        self.rollout.pause_generation().wait()

        self.rollout.sync_model_from_actor()
        self.actor.sync_model_to_rollout().wait()
        self.rollout.set_version(self.global_step).wait()
        self.actor.set_version(self.global_step).wait()
        self.rollout.resume_generation().wait()

    def run(self) -> None:
        start_step = self.global_step
        start_time = time.time()

        self.update_rollout_weights()

        env_handle: Handle = self.env.interact(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
            metrics_channel=self.env_metrics_channel,
        )
        rollout_handle: Handle = self.rollout.generate(
            input_channel=self.env_channel,
            output_channel=self.rollout_channel,
            actor_channel=self.actor_channel,
        )

        while self.global_step < self.max_steps:
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)
            with self.timer("step"):
                with self.timer("recv_rollout_trajectories"):
                    self.actor.recv_rollout_trajectories(
                        input_channel=self.actor_channel
                    ).wait()

                if self.recompute_logprobs:
                    with self.timer("recompute_logprobs"):
                        self.actor.compute_proximal_logprobs().wait()

                with self.timer("cal_adv_and_returns"):
                    rollout_metrics = self.actor.compute_advantages_and_returns().wait()

                with self.timer("actor_training"):
                    training_metrics = self.actor.run_training().wait()

                self.global_step += 1
                with self.timer("update_rollout_weights"):
                    self.update_rollout_weights()

            time_metrics = self.timer.consume_durations()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}

            train_metrics = {f"train/{k}": v for k, v in training_metrics[0].items()}
            rollout_metrics = {f"rollout/{k}": v for k, v in rollout_metrics[0].items()}
            env_metrics = self.get_env_metrics()

            self.metric_logger.log(train_metrics, self.global_step)
            if env_metrics:
                self.metric_logger.log(env_metrics, self.global_step)
            self.metric_logger.log(rollout_metrics, self.global_step)
            self.metric_logger.log(time_metrics, self.global_step)

            logging_metrics = {**time_metrics, **train_metrics, **rollout_metrics}
            if env_metrics:
                logging_metrics.update(env_metrics)

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

        self.metric_logger.finish()

        self.stop_logging = True
        self.log_queue.join()
        self.log_thread.join(timeout=1.0)

        self.env.stop().wait()
        self.rollout.stop()

        env_handle.wait()
        rollout_handle.wait()
