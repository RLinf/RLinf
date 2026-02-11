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

import logging
import os
from typing import Optional

from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.runner_utils import check_progress
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


class SFTRunner:
    def __init__(
        self,
        cfg: DictConfig,
        actor: FSDPSftWorker,
        run_timer: Optional[ScopedTimer] = None,
    ) -> None:
        self.cfg = cfg
        self.actor = actor

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.consumed_samples = 0
        # the step here is GRPO step
        self.global_step = 0

        # compute `max_steps`
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.metric_logger = MetricLogger(cfg)

    def init_workers(self) -> None:
        # create worker in order to decrease the maximum memory usage
        self.actor.init_worker().wait()

        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is None:
            return

        actor_checkpoint_path = os.path.join(resume_dir, "actor")
        assert os.path.exists(actor_checkpoint_path), (
            f"resume_dir {actor_checkpoint_path} does not exist."
        )
        self.actor.load_checkpoint(actor_checkpoint_path).wait()
        self.global_step = int(resume_dir.split("global_step_")[-1])

    def run(self) -> None:
        start_step = self.global_step
        global_pbar = tqdm(
            initial=start_step,
            total=self.max_steps,
            desc="Global Step",
            ncols=800,
        )
        for _step in range(start_step, self.max_steps):
            # set global step
            self.actor.set_global_step(self.global_step)

            # SFt Training
            with self.timer("step"):
                # Actor training.
                actor_handle: Handle = self.actor.run_training()
                actor_metrics = actor_handle.wait()

                self.global_step += 1

                eval_model, save_model, _ = check_progress(
                    self.global_step,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    1.0,
                    run_time_exceeded=False,
                )

                if save_model:
                    self._save_checkpoint()

                if eval_model:
                    eval_handle: Handle = self.actor.run_eval()
                    eval_metrics = eval_handle.wait()

            time_metrics = self.timer.consume_durations()
            time_metrics["training"] = actor_handle.consume_duration()
            if eval_model:
                time_metrics["evaluate"] = eval_handle.consume_duration()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            training_metrics = {f"train/{k}": v for k, v in actor_metrics[0].items()}
            self.metric_logger.log(time_metrics, _step)
            self.metric_logger.log(training_metrics, _step)

            logging_metrics = time_metrics
            logging_metrics.update(training_metrics)

            if eval_model:
                evaluate_metrics = {f"eval/{k}": v for k, v in eval_metrics[0].items()}
                logging_metrics.update(evaluate_metrics)
                self.metric_logger.log(evaluate_metrics, _step)

            global_pbar.set_postfix(logging_metrics, refresh=False)
            global_pbar.update(1)

        self.metric_logger.finish()

    def run_eval(self) -> None:
        with self.timer("evaluate"):
            eval_handle: Handle = self.actor.run_eval()
            eval_metrics = eval_handle.wait()

        time_metrics = self.timer.consume_durations()
        time_metrics["evaluate"] = eval_handle.consume_duration()
        time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}

        raw_eval = (
            eval_metrics[0]
            if isinstance(eval_metrics, (list, tuple)) and len(eval_metrics) > 0
            else {}
        )
        evaluate_metrics = {f"eval/{k}": v for k, v in raw_eval.items()}

        logging_metrics = {}
        logging_metrics.update(time_metrics)
        logging_metrics.update(evaluate_metrics)

        logging.info(f"evaluate the model, metrics: {evaluate_metrics}")
        self.metric_logger.finish()

    def _save_checkpoint(self) -> None:
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            f"checkpoints/global_step_{self.global_step}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)
        self.actor.save_checkpoint(actor_save_path, self.global_step).wait()

    def set_max_steps(self) -> None:
        self.num_steps_per_epoch = 1
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self) -> int:
        return self.global_step // self.num_steps_per_epoch
