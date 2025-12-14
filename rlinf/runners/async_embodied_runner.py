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

import os
import time

from omegaconf.dictconfig import DictConfig
from tqdm import tqdm
from typing import Optional
import asyncio

from rlinf.scheduler import Channel
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.runner_utils import check_progress
from rlinf.workers.actor.async_fsdp_actor_worker_sac import AsyncEmbodiedSACFSDPActor
from rlinf.workers.env.async_env_worker import AsyncEnvWorker
from rlinf.workers.rollout.hf.async_huggingface_worker import AsyncMultiStepRolloutWorker
from rlinf.data.replay_buffer import SACReplayBuffer
from rlinf.runners.embodied_runner import EmbodiedRunner

class AsyncEmbodiedRunner(EmbodiedRunner):
    def __init__(
        self,
        cfg: DictConfig,
        actor: AsyncEmbodiedSACFSDPActor,
        rollout: AsyncMultiStepRolloutWorker,
        env: AsyncEnvWorker,
        demo_buffer: Optional[SACReplayBuffer]=None, 
        critic=None,
        reward=None,
        run_timer=None,
    ):
        self.cfg = cfg
        self.actor = actor
        self.rollout = rollout
        self.env = env
        self.critic = critic
        self.reward = reward
        self.demo_buffer = demo_buffer
        self.demo_data_channel = Channel.create("DemoData")

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.consumed_samples = 0
        # the step here is GRPO step
        self.global_step = 0

        # compute `max_steps`
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.metric_logger = MetricLogger(cfg)
        self.env_metric_channel = Channel.create("env_metric_buffer")
        self.rollout_channel = Channel.create("replay_buffer")

    def generate_rollouts(self):
        env_handle = self.env.interact(self.env_metric_channel)
        rollout_handle = self.rollout.generate(self.rollout_channel)
        return env_handle, rollout_handle
    
    def get_env_metrics(self):
        try:
            result = self.env_metric_channel.get_nowait()
        except asyncio.QueueEmpty:
            return None
        env_metrics = compute_evaluate_metrics([result, ])
        return env_metrics

    def run(self):
        start_step = self.global_step
        self.send_demo_buffer()

        env_handle, rollout_handle = self.generate_rollouts()
        self.actor.start_replay_buffer(self.rollout_channel)
        
        train_step = start_step
        while train_step < self.max_steps:
            if (
                self.cfg.runner.val_check_interval > 0
                and train_step % self.cfg.runner.val_check_interval == 0
            ):
                self.update_rollout_weights()
                eval_metrics = self.evaluate()
                eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                self.metric_logger.log(data=eval_metrics, step=train_step)
            
            actor_handle = self.actor.run_training()
            actor_result = actor_handle.wait()
            if actor_result[0] == False:
                time.sleep(1.0)
                continue
            train_step += 1
            self.update_rollout_weights()
            
            training_metrics = {
                f"train/{k}": v for k, v in actor_result[0]["train"].items()
            }
            replay_buffer_metrics = {
                f"replay_buffer/{k}": v for k, v in actor_result[0]["replay_buffer"].items()
            }
            self.metric_logger.log(training_metrics, train_step)
            self.metric_logger.log(replay_buffer_metrics, train_step)

            env_metrics = self.get_env_metrics()
            if env_metrics is not None:
                rollout_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
                self.metric_logger.log(rollout_metrics, train_step)
        
            run_val, save_model, is_train_end = check_progress(
                self.global_step,
                self.max_steps,
                self.cfg.runner.val_check_interval,
                self.cfg.runner.save_interval,
                1.0,
                run_time_exceeded=False,
            )
            if save_model:
                self._save_checkpoint()
        # rollout_handle.wait()
        # env_handle.wait()
