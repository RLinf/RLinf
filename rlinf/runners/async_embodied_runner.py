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

from omegaconf.dictconfig import DictConfig

from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.runner_utils import check_progress

if TYPE_CHECKING:
    from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
        AsyncEmbodiedSACFSDPPolicy,
    )
    from rlinf.workers.env.async_env_worker import AsyncEnvWorker
    from rlinf.workers.rollout.hf.async_huggingface_worker import (
        AsyncMultiStepRolloutWorker,
    )


class AsyncEmbodiedRunner(EmbodiedRunner):
    def __init__(
        self,
        cfg: DictConfig,
        actor: "AsyncEmbodiedSACFSDPPolicy",
        rollout: "AsyncMultiStepRolloutWorker",
        env: "AsyncEnvWorker",
        critic=None,
        reward=None,
        run_timer=None,
    ):
        super().__init__(cfg, actor, rollout, env, critic, reward, run_timer)

        # Data channels
        self.env_metric_channel = Channel.create("EnvMetric")
        self.rollout_metric_channel = Channel.create("RolloutMetric")
        self.replay_channel = Channel.create("ReplayBuffer")

    def get_env_metrics(self):
        try:
            result = self.env_metric_channel.get_nowait()
        except asyncio.QueueEmpty:
            return {}

        time_metrics = {}
        env_metrics = {}
        for key, value in result.items():
            if key.startswith("time/"):
                time_metrics[key] = value
            else:
                env_metrics[key] = value

        env_metrics = compute_evaluate_metrics(
            [
                env_metrics,
            ]
        )
        return {**env_metrics, **time_metrics}

    def get_rollout_metrics(self):
        try:
            result = self.rollout_metric_channel.get_nowait()
        except asyncio.QueueEmpty:
            return {}
        return result

    def run(self):
        start_step = self.global_step
        start_time = time.time()
        self.update_rollout_weights()

        env_handle: Handle = self.env.interact(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
            metric_channel=self.env_metric_channel,
        )
        rollout_handle: Handle = self.rollout.generate(
            input_channel=self.env_channel,
            output_channel=self.rollout_channel,
            replay_channel=self.replay_channel,
            metric_channel=self.rollout_metric_channel,
        )
        actor_handle: Handle = self.actor.recv_rollout_trajectories(
            input_channel=self.replay_channel
        )

        while self.global_step < self.max_steps:
            skip_step = False
            with self.timer("step"):
                actor_training_handle: Handle = self.actor.run_training()
                actor_result = actor_training_handle.wait()
                if not actor_result[0]:
                    skip_step = True

                if not skip_step:
                    self.global_step += 1
                    if self.global_step % self.weight_sync_interval == 0:
                        self.update_rollout_weights()

                    training_metrics = {
                        f"train/{k}": v for k, v in actor_result[0].items()
                    }

                    run_val, save_model, _ = check_progress(
                        self.global_step,
                        self.max_steps,
                        self.cfg.runner.val_check_interval,
                        self.cfg.runner.save_interval,
                        1.0,
                        run_time_exceeded=False,
                    )
                    if save_model:
                        self._save_checkpoint()
                    eval_metrics = {}
                    if run_val:
                        with self.timer("eval"):
                            eval_metrics = self.evaluate()
                            eval_metrics = {
                                f"eval/{k}": v for k, v in eval_metrics.items()
                            }

            if skip_step:
                self.timer.consume_durations()
                time.sleep(1.0)
                continue

            time_metrics = self.timer.consume_durations()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            training_metrics["train/replay_channel_qsize"] = self.replay_channel.qsize()
            actor_training_time_metrics = {
                f"time/actor/{k}": v
                for k, v in actor_training_handle.consume_durations().items()
            }
            time_metrics.update(actor_training_time_metrics)
            env_metrics = self.get_env_metrics()
            rollout_metrics = self.get_rollout_metrics()

            self.metric_logger.log(time_metrics, self.global_step)
            self.metric_logger.log(env_metrics, self.global_step)
            self.metric_logger.log(rollout_metrics, self.global_step)
            self.metric_logger.log(training_metrics, self.global_step)
            self.metric_logger.log(eval_metrics, self.global_step)

            logging_metrics = time_metrics
            logging_metrics.update(eval_metrics)
            logging_metrics.update(env_metrics)
            logging_metrics.update(rollout_metrics)
            logging_metrics.update(training_metrics)

            self.print_metrics_table_async(
                self.global_step - 1,
                self.max_steps,
                start_time,
                logging_metrics,
                start_step,
            )

        self.env.stop().wait()
        self.rollout.stop().wait()
        self.actor.stop().wait()
        env_handle.wait()
        rollout_handle.wait()
        actor_handle.wait()

        self._save_checkpoint()
