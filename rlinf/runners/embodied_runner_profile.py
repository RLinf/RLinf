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

from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedRunnerProfile:
    def __init__(
        self,
        cfg: DictConfig,
        rollout: MultiStepRolloutWorker,
        env: EnvWorker,
        critic=None,
        reward=None,
        run_timer=None,
    ):
        self.cfg = cfg
        self.rollout = rollout
        self.env = env
        self.critic = critic
        self.reward = reward

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.consumed_samples = 0
        # the step here is GRPO step
        self.global_step = 0

        # compute `max_steps`
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.metric_logger = MetricLogger(cfg)

    def init_workers(self):
        # create worker in order to decrease the maximum memory usage
        self.rollout.init_worker().wait()
        self.env.init_worker().wait()

        if self.cfg.runner.get("resume_dir", None) is not None:
            self.global_step = int(self.cfg.runner.resume_dir.split("global_step_")[-1])

    def update_rollout_weights(self):
        rollout_futures = self.rollout.sync_model_from_actor()
        rollout_futures.wait()

    def generate_rollouts(self):
        env_futures = self.env.interact()
        rollout_futures = self.rollout.generate()
        env_results = env_futures.wait()
        rollout_futures.wait()

        env_results_list = [results for results in env_results if results is not None]
        env_metrics = compute_evaluate_metrics(env_results_list)
        return env_metrics

    # def evaluate(self):
    #     env_futures = self.env.evaluate()
    #     rollout_futures = self.rollout.evaluate()
    #     env_results = env_futures.wait()
    #     rollout_futures.wait()
    #     eval_metrics_list = [results for results in env_results if results is not None]
    #     eval_metrics = compute_evaluate_metrics(eval_metrics_list)
    #     return eval_metrics

    def run(self):
        start_step = self.global_step
        global_pbar = tqdm(
            initial=start_step,
            total=self.max_steps,
            desc="Global Step",
            ncols=800,
        )
        for _step in range(start_step, self.max_steps):
            # set global step
            self.rollout.set_global_step(self.global_step)
            eval_metrics = {}

            with self.timer("step"):
                with self.timer("generate_rollouts"):
                    env_metrics = self.generate_rollouts()

                self.global_step += 1

            time_metrics = self.timer.consume_durations()

            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            # 删去rollout，即删去关于adv等信息的metrics
            # rollout_metrics = {
            #     f"rollout/{k}": v for k, v in actor_rollout_metrics[0].items()
            # }
            env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            self.metric_logger.log(env_metrics, _step)
            self.metric_logger.log(time_metrics, _step)

            logging_metrics = time_metrics
            logging_metrics.update(eval_metrics)
            logging_metrics.update(env_metrics)

            global_pbar.set_postfix(logging_metrics)
            global_pbar.update(1)

        self.metric_logger.finish()

    def set_max_steps(self):
        self.num_steps_per_epoch = 1
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self):
        return self.global_step // self.num_steps_per_epoch
