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


import asyncio
from typing import TYPE_CHECKING

from omegaconf.omegaconf import DictConfig
from tqdm import tqdm

from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.runner_utils import check_progress

if TYPE_CHECKING:
    from rlinf.workers.actor.async_ppo_fsdp_worker import AsyncPPOEmbodiedFSDPActor
    from rlinf.workers.env.async_ppo_env_worker import AsyncPPOEnvWorker
    from rlinf.workers.rollout.hf.async_ppo_hugginface_worker import (
        AsyncPPOMultiStepRolloutWorker,
    )


class AsyncEmbodiedRunner(EmbodiedRunner):
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
        # just make type checker happy
        self.actor: "AsyncPPOEmbodiedFSDPActor"
        self.rollout: "AsyncPPOMultiStepRolloutWorker"
        self.env: "AsyncPPOEnvWorker"

    def get_env_metrics(self):
        results = []
        while True:
            try:
                result = self.env_metrics_channel.get_nowait()
                results.append(result)
            except asyncio.QueueEmpty:
                break
        assert len(results) > 0, "No env metrics received from env workers."
        env_metrics = compute_evaluate_metrics(results)
        env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
        return env_metrics

    def update_rollout_weights(self):
        self.rollout.pause_generation().wait()
        self.rollout.sync_model_from_actor()
        self.actor.sync_model_to_rollout().wait()
        self.rollout.set_version(self.global_step).wait()
        self.actor.set_version(self.global_step).wait()
        self.rollout.resume_generation().wait()

    def run(self):
        global_pbar = tqdm(
            initial=self.global_step,
            total=self.max_steps,
            desc="Global Step",
            dynamic_ncols=True,
        )

        self.update_rollout_weights()

        env_handle: Handle = self.env.start_interacting(
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

            training_metrics = {f"train/{k}": v for k, v in training_metrics[0].items()}
            self.metric_logger.log(training_metrics, self.global_step)

            env_metrics = self.get_env_metrics()
            self.metric_logger.log(env_metrics, self.global_step)

            rollout_metrics = {f"rollout/{k}": v for k, v in rollout_metrics[0].items()}
            self.metric_logger.log(rollout_metrics, self.global_step)

            logging_metrics = time_metrics
            logging_metrics.update(training_metrics)
            # Add other metrics to logging
            logging_metrics.update(env_metrics)
            logging_metrics.update(rollout_metrics)

            global_pbar.set_postfix(logging_metrics, refresh=False)
            global_pbar.update(1)
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
        self.env.stop().wait()
        self.rollout.stop().wait()
        env_handle.wait()
        rollout_handle.wait()
        self._save_checkpoint()
