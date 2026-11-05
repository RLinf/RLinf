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

import math
import os
from typing import Any

from omegaconf import DictConfig

from rlinf.utils.checkpoint import parse_global_step_from_checkpoint_path
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.workers.supermimic.checkpoint import find_latest_checkpoint

# Per-rank counts that add up to a global count.
_SUM_KEYS = {"env_transitions", "buffer/valid_windows", "sim/episodes_finished"}
# Counters every rank advances in lockstep; rank 0's value is the global one.
_SHARED_KEYS = {
    "train/global_update",
    "sim_vector_steps",
    "wm_optimizer_steps",
    "actor_optimizer_steps",
    "critic_optimizer_steps",
    "train/nonfinite_skips",
    "train/param_divergence",
}
_LOG_EVERY = 100


def report_training_step(step: int, total: int, every: int = _LOG_EVERY) -> bool:
    """Whether this update is printed. Every ``every`` updates, and the last one.

    A run of 278 updates prints 100, 200 and 278. Update 0 stays in TensorBoard
    only.
    """
    if step <= 0:
        return False
    if step >= total:
        return True
    return every > 0 and step % every == 0


def aggregate_rank_metrics(
    per_rank: list[dict[str, float]], per_rank_detail: bool = False
) -> dict[str, float]:
    """Merge one metrics dict per rank into global metrics.

    Counts are summed, lockstep counters taken from rank 0, times and
    memory reported as the slowest or largest rank, and losses and rates
    averaged over the ranks that produced a finite value. Each rank holds
    the same number of envs, so the mean of rates is the global rate.
    """
    if len(per_rank) == 1:
        return dict(per_rank[0])
    keys: list[str] = []
    for metrics in per_rank:
        keys.extend(key for key in metrics if key not in keys)
    merged: dict[str, float] = {}
    for key in keys:
        values = [float(m[key]) for m in per_rank if key in m]
        if key in _SHARED_KEYS:
            merged[key] = float(per_rank[0].get(key, values[0]))
        elif key in _SUM_KEYS:
            merged[key] = sum(values)
        elif key.startswith(("time/", "memory/")):
            merged[key] = max(values)
        else:
            finite = [v for v in values if math.isfinite(v)]
            merged[key] = sum(finite) / len(finite) if finite else float("nan")
    if "memory/host_rss" in keys:
        merged["memory/host_rss_total"] = sum(
            float(m.get("memory/host_rss", 0.0)) for m in per_rank
        )
    merged["train/world_size"] = float(len(per_rank))
    if per_rank_detail:
        for rank, metrics in enumerate(per_rank):
            for key, value in metrics.items():
                merged[f"rank_{rank}/{key}"] = float(value)
    return merged


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

    def checkpoint_root(self) -> str:
        """Directory that holds this run's ``global_step_<N>`` checkpoints."""
        return os.path.join(
            self.cfg.runner.logger.log_path, self.cfg.runner.logger.experiment_name
        )

    def init_workers(self) -> None:
        """Initialize every rank and resume from ``runner.resume_dir`` when set."""
        self.worker.init_worker().wait()
        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir == "auto":
            resume_dir = find_latest_checkpoint(self.checkpoint_root())
            if resume_dir is None:
                self._logger().info(
                    f"resume_dir=auto: no complete checkpoint under "
                    f"{self.checkpoint_root()}, starting from scratch"
                )
        if resume_dir:
            self._logger().info(
                "Resuming SuperMimic networks, optimizers, reset bins and RNG from "
                f"{resume_dir}. Replay and simulator state are not saved; the run "
                "warms a new replay first, so this is not an exact restore."
            )
            self.global_step = parse_global_step_from_checkpoint_path(resume_dir)
            self.worker.load_checkpoint(resume_dir).wait()

    def evaluate(self) -> dict:
        # Ranks hold identical weights; rank 0 covers every clip and the
        # other ranks return nothing.
        """Evaluate on rank 0 and return its metrics."""
        metrics = self.worker.evaluate().wait()[0]
        return metrics

    def _iteration(self) -> dict:
        per_rank = self.worker.run_iteration().wait()
        return aggregate_rank_metrics(
            per_rank, bool(self.cfg.runner.get("log_per_rank_metrics", False))
        )

    def shutdown(self) -> None:
        """Release every rank's resources; a failure is logged, not raised."""
        try:
            self.worker.shutdown().wait()
        except Exception as error:  # a failed rank must not hide the first error
            self._logger().warning(f"SuperMimic worker shutdown failed: {error}")

    def save(self, *, force: bool = False) -> None:
        """Write one checkpoint. ``force`` is the final update.

        ``save_interval <= 0`` skips periodic checkpoints. The last update
        is still written, so a finished run has a model to resume from.
        """
        interval = int(self.cfg.runner.get("save_interval", -1))
        if interval <= 0 and not force:
            return
        directory = os.path.join(
            self.checkpoint_root(), f"global_step_{self.global_step}"
        )
        self.worker.save_checkpoint(directory).wait()

    def run(self) -> dict:
        """Train (or only evaluate) to the end, then shut down and close the logger.

        Returns:
            The metrics of the last iteration or of the evaluation.
        """
        try:
            return self._run()
        finally:
            self.shutdown()
            self.metric_logger.finish()

    def _run(self) -> dict:
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
            last = self._iteration()
            updated = int(last.get("train/global_update", self.global_step))
            # UTD can collect several times before global_update moves.
            # Save and eval run once, on the iteration that advances it.
            # Update 0 is not a completed training update. TensorBoard records
            # it; the terminal line waits for the 100-update marks and the end.
            advanced = updated != self.global_step
            # TensorBoard still records every update. The terminal line is
            # every 100 updates, and also the last update when it is not a
            # multiple of 100.
            show = report_training_step(updated, self.max_steps)
            if (advanced or (first and self.global_step == 0)) and show:
                self._logger().info(f"supermimic metrics {last}")
            if advanced or (first and self.global_step == 0):
                self.metric_logger.log(last, step=updated)
            first = False
            self.global_step = updated
            if not advanced or updated <= 0:
                continue
            # Like the embodied runner: evaluate every val_check_interval
            # updates and on the last one, when evaluation is enabled.
            run_val, save_model, train_end = check_progress(
                self.global_step,
                self.max_steps,
                int(self.cfg.runner.get("val_check_interval", -1)),
                int(self.cfg.runner.get("save_interval", -1)),
                1.0,
            )
            # One save on an update that is both periodic and final.
            if train_end:
                self.save(force=True)
            elif save_model:
                self.save()
            if run_val:
                eval_metrics = self.evaluate()
                self._logger().info(f"supermimic eval {eval_metrics}")
                self.metric_logger.log(eval_metrics, step=self.global_step)
        return last
