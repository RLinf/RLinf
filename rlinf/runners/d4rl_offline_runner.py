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
from typing import TYPE_CHECKING

from omegaconf.dictconfig import DictConfig

from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import print_metrics_table

if TYPE_CHECKING:
    from rlinf.workers.actor.fsdp_iql_policy_worker import EmbodiedIQLFSDPPolicy


class D4RLOfflineRunner:
    def __init__(self, cfg: DictConfig, actor: "EmbodiedIQLFSDPPolicy"):
        self.cfg = cfg
        self.actor = actor
        self.global_step = 0
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
        # create worker before resume loading
        self.actor.init_worker().wait()
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

    def run(self):
        start_step = self.global_step
        start_time = time.time()
        for _step in range(start_step, self.max_steps):
            next_step = self.global_step + 1
            self.actor.set_global_step(next_step)

            with self.timer("step"):
                actor_training_handle: Handle = self.actor.run_training()
                actor_training_metrics = actor_training_handle.wait()
                self.global_step = next_step

            metrics = (
                actor_training_metrics[0]
                if isinstance(actor_training_metrics, list)
                else actor_training_metrics
            )

            def _is_scalar(v):
                if hasattr(v, "ndim"):
                    return v.ndim == 0
                return isinstance(v, (int, float))

            training_metrics = {
                f"train/{k}": v
                for k, v in metrics.items()
                if not k.startswith("evaluation/") and _is_scalar(v)
            }
            eval_metrics = {
                k: v for k, v in metrics.items() if k.startswith("evaluation/")
            }
            time_metrics = {
                f"time/{k}": v for k, v in self.timer.consume_durations().items()
            }

            self.metric_logger.log(time_metrics, _step)
            self.metric_logger.log(training_metrics, _step)
            if eval_metrics:
                self.metric_logger.log(eval_metrics, _step)

            logging_metrics = dict(time_metrics)
            logging_metrics.update(eval_metrics)
            logging_metrics.update(training_metrics)
            self.print_metrics_table_async(
                _step, self.max_steps, start_time, logging_metrics, start_step
            )

            save_interval = int(self.cfg.runner.get("save_interval", -1))
            if save_interval > 0 and self.global_step % save_interval == 0:
                self._save_checkpoint()
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
