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

from __future__ import annotations

import os
from typing import Any

from omegaconf import DictConfig

from rlinf.utils.checkpoint import parse_global_step_from_checkpoint_path
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger


class SuperMimicRunner:
    """Outer loop: iteration, logging, checkpoint, eval. No Channel."""

    def __init__(self, cfg: DictConfig, worker: Any):
        self.cfg = cfg
        self.worker = worker
        self.logger = None
        self.metric_logger = MetricLogger(cfg)
        self.global_step = 0
        self.max_steps = int(cfg.algorithm.total_updates)

    def _logger(self):
        if self.logger is None:
            self.logger = get_logger()
        return self.logger

    def init_workers(self) -> None:
        self.worker.init_worker().wait()
        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir:
            self._logger().info(
                "Resuming SuperMimic networks and optimizers from "
                f"{resume_dir}. Replay and simulator state are not restored."
            )
            self.global_step = parse_global_step_from_checkpoint_path(resume_dir)
            self.worker.load_checkpoint(resume_dir).wait()

    def evaluate(self) -> dict:
        metrics = self.worker.evaluate().wait()[0]
        return metrics

    def save(self, *, force: bool = False) -> None:
        """Write one checkpoint. ``force`` is the final update.

        ``save_interval <= 0`` skips periodic checkpoints. The last update
        is still written, so a finished run has a model to resume from.
        """
        interval = int(self.cfg.runner.get("save_interval", -1))
        if interval <= 0 and not force:
            return
        directory = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            f"global_step_{self.global_step}",
        )
        self.worker.save_checkpoint(directory).wait()

    def run(self) -> dict:
        if self.cfg.runner.get("only_eval", False):
            metrics = self.evaluate()
            self.metric_logger.log(metrics, step=self.global_step)
            self._logger().info(f"supermimic eval {metrics}")
            return metrics
        # Imported here so only_eval does not load the Ray worker.
        from rlinf.utils.runner_utils import check_progress

        last = {}
        first = True
        while self.global_step < self.max_steps:
            last = self.worker.run_iteration().wait()[0]
            updated = int(last.get("train/global_update", self.global_step))
            if first or updated != self.global_step:
                self._logger().info(f"supermimic metrics {last}")
            first = False
            self.global_step = updated
            self.metric_logger.log(last, step=self.global_step)
            _, save_model, train_end = check_progress(
                self.global_step,
                self.max_steps,
                int(self.cfg.runner.get("val_check_interval", -1)),
                int(self.cfg.runner.get("save_interval", -1)),
                None,
            )
            # One save on an update that is both periodic and final.
            if train_end:
                self.save(force=True)
            elif save_model:
                self.save()
            val_interval = int(self.cfg.runner.get("val_check_interval", -1))
            if val_interval > 0 and self.global_step % val_interval == 0:
                eval_metrics = self.evaluate()
                self.metric_logger.log(eval_metrics, step=self.global_step)
        return last
