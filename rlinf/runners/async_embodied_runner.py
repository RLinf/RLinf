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

from rlinf.scheduler import Channel
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.runner_utils import check_progress
from rlinf.workers.actor.fsdp_actor_worker_sac import EmbodiedSACFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.async_huggingface_worker import MultiStepRolloutWorker
from rlinf.data.replay_buffer import SACReplayBuffer


class EmbodiedRunner:
    def __init__(
        self,
        cfg: DictConfig,
        actor: EmbodiedSACFSDPActor,
        rollout: MultiStepRolloutWorker,
        env: EnvWorker,
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
        self.rollout_channel = Channel.create("replay_buffer")

    def init_workers(self):
        # create worker in order to decrease the maximum memory usage
        self.actor.init_worker().wait()
        self.rollout.init_worker().wait()
        self.env.init_worker().wait()

    def update_rollout_weights(self):
        rollout_futures = self.rollout.sync_model_from_actor()
        actor_futures = self.actor.sync_model_to_rollout()
        actor_futures.wait()
        rollout_futures.wait()

    def generate_rollouts(self):
        env_handle = self.env.interact()
        rollout_handle = self.rollout.generate(self.rollout_channel)
        return env_handle, rollout_handle
    
    def evaluate(self):
        env_futures = self.env.evaluate()
        rollout_futures = self.rollout.evaluate()
        env_results = env_futures.wait()
        rollout_futures.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def send_demo_buffer(self):
        if self.demo_buffer is not None:
            sub_demo_buffer_ls = self.demo_buffer.split_to_dict(self.actor._world_size)
        
            for sub_demo_buffer in sub_demo_buffer_ls:
                self.demo_data_channel.put(sub_demo_buffer, async_op=True)
            actor_futures = self.actor.recv_demo_data()
            actor_futures.wait()

    def run(self):
        start_step = self.global_step
        self.send_demo_buffer()

        env_handle, rollout_handle = self.generate_rollouts()
        self.actor.start_replay_buffer(self.rollout_channel)
        
        train_step = start_step
        while train_step < self.max_steps:
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
        
        rollout_handle.wait()
        env_handle.wait()
        # TODO: update actor weights

    def _save_checkpoint(self):
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            f"checkpoints/global_step_{self.global_step}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        save_futures = self.actor.save_checkpoint(actor_save_path, self.global_step)
        save_futures.wait()

    def set_max_steps(self):
        self.num_steps_per_epoch = 1
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self):
        return self.global_step // self.num_steps_per_epoch
