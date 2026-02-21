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
from typing import Optional

from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.runner_utils import check_progress
from rlinf.workers.vla_lib_sft.fsdp_value_sft_worker import FSDPValueSftWorker


class VlaLibSFTRunner:
    """SFT Runner for VLA lib value model training with evaluation support."""

    def __init__(
        self,
        cfg: DictConfig,
        actor: FSDPValueSftWorker,
        run_timer: Optional[ScopedTimer] = None,
    ) -> None:
        self.cfg = cfg
        self.actor = actor
        self.run_timer = run_timer

        self.consumed_samples = 0
        self.global_step = 0

        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_logger = MetricLogger(cfg)

    def init_workers(self) -> None:
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

    def evaluate(self) -> dict[str, float]:
        env_handle: Handle = self.env.evaluate(
            input_channel=self.rollout_channel, output_channel=self.env_channel
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.env_channel, output_channel=self.rollout_channel
        )
        env_results = env_handle.wait()
        rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def _collect_eval_metrics(self) -> dict[str, float]:
        """Collect evaluation metrics from actor or env+rollout."""
        eval_metrics = {}
        actor_eval_metrics = None

        if hasattr(self.actor, "run_evaluation"):
            eval_handle: Handle = self.actor.run_evaluation()
            actor_eval_metrics = eval_handle.wait()
        elif hasattr(self, "env") and hasattr(self, "rollout"):
            eval_metrics = {f"eval/{k}": v for k, v in self.evaluate().items()}

        if actor_eval_metrics and actor_eval_metrics[0]:
            eval_metrics = {f"eval/{k}": v for k, v in actor_eval_metrics[0].items()}
        return eval_metrics

    def run(self) -> None:
        start_step = self.global_step
        global_pbar = tqdm(
            initial=start_step,
            total=self.max_steps,
            desc="Global Step",
            ncols=800,
        )
        for _step in range(start_step, self.max_steps):
            self.actor.set_global_step(self.global_step)

            # Training
            with self.timer("step"):
                actor_handle: Handle = self.actor.run_training()
                actor_metrics = actor_handle.wait()

                self.global_step += 1

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

            time_metrics = self.timer.consume_durations()
            time_metrics["training"] = actor_handle.consume_duration()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            training_metrics = {f"train/{k}": v for k, v in actor_metrics[0].items()}

            eval_metrics = {}
            if run_val:
                eval_metrics = self._collect_eval_metrics()

            self.metric_logger.log(time_metrics, _step)
            self.metric_logger.log(training_metrics, _step)
            if eval_metrics:
                self.metric_logger.log(eval_metrics, _step)

            logging_metrics = time_metrics
            logging_metrics.update(training_metrics)
            logging_metrics.update(eval_metrics)

            global_pbar.set_postfix(logging_metrics, refresh=False)
            global_pbar.update(1)

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
