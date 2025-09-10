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

from rlinf.algorithms.embodiment.utils import compute_evaluate_metrics
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.runner_utils import check_progress
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedRunner:
    def __init__(
        self,
        cfg: DictConfig,
        actor: EmbodiedFSDPActor,
        rollout: MultiStepRolloutWorker,
        env: EnvWorker,
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
        self.actor.init_worker().wait()
        self.rollout.init_worker().wait()
        self.env.init_worker().wait()

    def update_rollout_weights(self):
        rollout_futures = self.rollout.sync_model_from_actor()
        actor_futures = self.actor.sync_model_to_rollout()
        actor_futures.wait()
        rollout_futures.wait()

    def generate_rollouts(self):
        env_futures = self.env.interact()
        rollout_futures = self.rollout.generate()
        actor_futures = self.actor.recv_rollout_batch()
        env_futures.wait()
        actor_futures.wait()
        rollout_futures.wait()

    def evaluate(self):
        env_futures = self.env.evaluate()
        rollout_futures = self.rollout.evaluate()
        env_futures.wait()
        rollout_results = rollout_futures.wait()
        eval_metrics_list = [
            results for results in rollout_results if results is not None
        ]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def run(self):
        start_step = self.global_step
        for _step in tqdm(range(start_step, self.max_steps), ncols=120):
            if (
                _step % self.cfg.runner.val_check_interval == 0
                and self.cfg.runner.val_check_interval > 0
            ):
                with self.timer("eval"):
                    self.update_rollout_weights()
                    eval_metrics = self.evaluate()
                    per_task_eval_metrics = {}
                    all_keys = list(eval_metrics.keys())
                    for key in all_keys:
                        if "task_" in key:
                            per_task_eval_metrics[key] = eval_metrics[key]
                            eval_metrics.pop(key)
                    eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                    eval_metrics.update(per_task_eval_metrics)
                    self.metric_logger.log(data=eval_metrics, step=_step)

            with self.timer("step"):
                with self.timer("rollout"):
                    self.update_rollout_weights()
                    self.generate_rollouts()

                # compute advantages and returns.
                with self.timer("cal_adv_and_returns"):
                    actor_futures = self.actor.compute_advantages_and_returns()
                    actor_rollout_metrics = actor_futures.wait()

                # actor training.
                with self.timer("actor_training"):
                    actor_training_futures = self.actor.run_training()
                    actor_training_metrics = actor_training_futures.wait()

                self.global_step += 1

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

            time_metrics = self.timer.consume_durations()

            # rollout_metrics = {
            #     f"rollout/{k}": v for k, v in actor_rollout_metrics[0].items()
            # }
            per_task_metrics = {}
            for i in range(len(actor_rollout_metrics)):
                task_i_keys = [key for key in actor_rollout_metrics[i].keys() if "task_" in key]
                for key in task_i_keys:
                    per_task_metrics[key] = actor_rollout_metrics[i][key]
                    actor_rollout_metrics[i].pop(key)

            actor_rollout_metrics = {
                key: sum(metrics[key] for metrics in actor_rollout_metrics) / len(actor_rollout_metrics)
                for key in actor_rollout_metrics[0].keys()
            }
            actor_rollout_metrics.update(per_task_metrics)
            actor_rollout_metrics.update({"batch_size": self.cfg.actor.global_batch_size})
            per_task_metrics = {}
            for i in range(len(actor_training_metrics)):
                task_i_keys = [key for key in actor_training_metrics[i].keys() if "task_" in key]
                for key in task_i_keys:
                    per_task_metrics[key] = actor_training_metrics[i][key]
                    actor_training_metrics[i].pop(key)
            actor_training_metrics = {
                key: sum(metrics[key] for metrics in actor_training_metrics) / len(actor_training_metrics)
                for key in actor_training_metrics[0].keys()
            }
            actor_training_metrics.update(per_task_metrics)
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            # training_metrics = {
            #     f"train/{k}": v for k, v in actor_training_metrics[0].items()
            # }
            self.metric_logger.log(actor_rollout_metrics, _step)
            self.metric_logger.log(time_metrics, _step)
            self.metric_logger.log(actor_training_metrics, _step)

        self.metric_logger.finish()

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
