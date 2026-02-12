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
import queue
import threading
import time
from typing import TYPE_CHECKING, Union

from omegaconf.dictconfig import DictConfig

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics, print_metrics_table
from rlinf.utils.runner_utils import check_progress

if TYPE_CHECKING:
    from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
        AsyncEmbodiedSACFSDPPolicy,
    )
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
    from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy
    from rlinf.workers.env.async_env_worker import AsyncEnvWorker
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.async_huggingface_worker import (
        AsyncMultiStepRolloutWorker,
    )
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedRunner:
    def __init__(
        self,
        cfg: DictConfig,
        actor: Union[
            "EmbodiedFSDPActor", "EmbodiedSACFSDPPolicy", "AsyncEmbodiedSACFSDPPolicy"
        ],
        rollout: Union["MultiStepRolloutWorker", "AsyncMultiStepRolloutWorker"],
        env: Union["EnvWorker", "AsyncEnvWorker"],
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
        self.weight_sync_interval = self.cfg.runner.weight_sync_interval
        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")
        self.actor_channel = Channel.create("Actor")

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.consumed_samples = 0
        # the step here is GRPO step
        self.global_step = 0

        # compute `max_steps`
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.logger = get_logger()
        self.metric_logger = MetricLogger(cfg)

        # Async logging setup
        self.stop_logging = False
        self.log_queue = queue.Queue()
        self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.log_thread.start()

    def _log_worker(self):
        """Background thread for processing log messages."""
        while not self.stop_logging:
            try:
                # Wait for log message with timeout
                log_func, args = self.log_queue.get(timeout=0.1)
                log_func(*args)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Logging error: {e}")
                continue

    def print_metrics_table_async(
        self,
        step: int,
        total_steps: int,
        start_time: float,
        metrics: dict,
        start_step: int = 0,
    ):
        """Async version that puts table printing in queue."""
        self.log_queue.put(
            (print_metrics_table, (step, total_steps, start_time, metrics, start_step))
        )

    def init_workers(self):
        # create worker in order to decrease the maximum memory usage
        self.actor.init_worker().wait()
        self.rollout.init_worker().wait()
        self.env.init_worker().wait()

        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is None:
            return

        self.logger.info(f"Resuming training from checkpoint directory {resume_dir}.")
        actor_checkpoint_path = os.path.join(resume_dir, "actor")
        assert os.path.exists(actor_checkpoint_path), (
            f"resume_dir {actor_checkpoint_path} does not exist."
        )
        self.actor.load_checkpoint(actor_checkpoint_path).wait()
        self.global_step = int(resume_dir.split("global_step_")[-1])

    def update_rollout_weights(self):
        rollout_handle: Handle = self.rollout.sync_model_from_actor()
        actor_handle: Handle = self.actor.sync_model_to_rollout()
        actor_handle.wait()
        rollout_handle.wait()

    def evaluate(self):
        env_handle: Handle = self.env.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.env_channel,
            output_channel=self.rollout_channel,
        )
        env_results = env_handle.wait()
        rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def run(self):
        start_step = self.global_step
        start_time = time.time()
        for _step in range(start_step, self.max_steps):
            # set global step
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)

            with self.timer("step"):
                with self.timer("sync_weights"):
                    if _step % self.weight_sync_interval == 0:
                        self.update_rollout_weights()
                with self.timer("generate_rollouts"):
                    env_handle: Handle = self.env.interact(
                        input_channel=self.rollout_channel,
                        output_channel=self.env_channel,
                    )
                    rollout_handle: Handle = self.rollout.generate(
                        input_channel=self.env_channel,
                        output_channel=self.rollout_channel,
                        actor_channel=self.actor_channel,
                    )
                    self.actor.recv_rollout_trajectories(
                        input_channel=self.actor_channel
                    ).wait()
                    rollout_handle.wait()

                # compute advantages and returns.
                with self.timer("cal_adv_and_returns"):
                    actor_rollout_metrics = (
                        self.actor.compute_advantages_and_returns().wait()
                    )

                # actor training.
                actor_training_handle: Handle = self.actor.run_training()

                actor_training_metrics = actor_training_handle.wait()

                self.global_step += 1

                run_val, save_model, is_train_end = check_progress(
                    self.global_step,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    1.0,
                    run_time_exceeded=False,
                )

                eval_metrics = {}
                if run_val:
                    with self.timer("eval"):
                        self.update_rollout_weights()
                        eval_metrics = self.evaluate()
                        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                        self.metric_logger.log(data=eval_metrics, step=_step)

                if save_model:
                    self._save_checkpoint()

            time_metrics = self.timer.consume_durations()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            time_metrics.update(
                {f"time/env/{k}": v for k, v in env_handle.consume_durations().items()}
            )
            time_metrics.update(
                {
                    f"time/rollout/{k}": v
                    for k, v in rollout_handle.consume_durations().items()
                }
            )
            time_metrics.update(
                {
                    f"time/actor/{k}": v
                    for k, v in actor_training_handle.consume_durations().items()
                }
            )

            env_results_list = [
                results for results in env_handle.wait() if results is not None
            ]
            env_metrics = compute_evaluate_metrics(env_results_list)
            env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}

            rollout_metrics = {
                f"rollout/{k}": v for k, v in actor_rollout_metrics[0].items()
            }

            training_metrics = {
                f"train/{k}": v for k, v in actor_training_metrics[0].items()
            }

            self.metric_logger.log(env_metrics, _step)
            self.metric_logger.log(rollout_metrics, _step)
            self.metric_logger.log(time_metrics, _step)
            self.metric_logger.log(training_metrics, _step)

            logging_metrics = time_metrics
            logging_metrics.update(eval_metrics)
            logging_metrics.update(env_metrics)
            logging_metrics.update(rollout_metrics)
            logging_metrics.update(training_metrics)

            self.print_metrics_table_async(
                _step, self.max_steps, start_time, logging_metrics, start_step
            )

        self.metric_logger.finish()

        # Stop logging thread
        self.stop_logging = True
        self.log_queue.join()  # Wait for all queued logs to be processed
        self.log_thread.join(timeout=1.0)

    def _save_checkpoint(self):
        self.logger.info(f"Saving checkpoint at step {self.global_step}.")
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            f"checkpoints/global_step_{self.global_step}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)
        self.actor.save_checkpoint(actor_save_path, self.global_step).wait()

    def set_max_steps(self):
        self.num_steps_per_epoch = 1
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self):
        return self.global_step // self.num_steps_per_epoch
