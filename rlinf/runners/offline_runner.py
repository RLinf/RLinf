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
from collections import defaultdict
from typing import Any

from omegaconf import open_dict
from omegaconf.dictconfig import DictConfig

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics, print_metrics_table
from rlinf.utils.runner_utils import check_progress


class OfflineRunner:
    """Offline RL runner: drives dataset worker and actor; dataset worker owns data."""

    def __init__(
        self,
        cfg: DictConfig,
        actor: Any,
        dataset,
        env,
        rollout,
    ):
        self.cfg = cfg
        self.actor = actor
        self.dataset = dataset
        self.env = env
        self.rollout = rollout

        # Embodied-style eval channels (env <-> rollout)
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")

        self.global_step = 0

        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.logger = get_logger()
        self.metric_logger = MetricLogger(cfg)
        self.enable_per_worker_metric_log = bool(
            self.cfg.runner.get("per_worker_log", False)
        )

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
                self.logger.error("Logging error: %s", e)
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
        """Initialize workers and build actor model dims from dataset metadata."""
        self.dataset.init_worker().wait()
        obs_action_dims = self.dataset.get_obs_action_dims().wait()
        if isinstance(obs_action_dims, list):
            obs_action_dims = next(
                (dims for dims in obs_action_dims if dims is not None), None
            )
        if not (
            isinstance(obs_action_dims, (tuple, list)) and len(obs_action_dims) == 2
        ):
            raise TypeError(
                f"DatasetWorker.get_obs_action_dims() should return (obs_dim, action_dim), got {obs_action_dims!r}."
            )
        obs_dim, action_dim = int(obs_action_dims[0]), int(obs_action_dims[1])
        with open_dict(self.cfg.actor.model):
            self.cfg.actor.model.obs_dim = obs_dim
            self.cfg.actor.model.action_dim = action_dim
        self.env.init_worker().wait()
        self.rollout.init_worker().wait()
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

    def update_rollout_weights(self):
        rollout_handle: Handle = self.rollout.sync_model_from_actor()
        actor_handle: Handle = self.actor.sync_model_to_rollout()
        actor_handle.wait()
        rollout_handle.wait()

    def update_actor_batches(self, batch_size: int, local_update_steps: int) -> None:
        """Sync dataset batches to actor workers via send/recv (like actor↔rollout)."""
        dataset_handle: Handle = self.dataset.send_batches_to_actor(
            batch_size=batch_size, num_batches=local_update_steps
        )
        actor_recv_handle: Handle = self.actor.recv_dataset_batches()
        actor_recv_handle.wait()
        dataset_handle.wait()

    def evaluate(self):
        """Run embodied-style evaluation and return aggregated metrics."""
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

    def _log_ranked_metrics(
        self,
        metrics_list: list[dict] | None,
        step: int,
        prefix: str,
        worker_group_name: str,
        add_prefix: bool = True,
    ):
        if not self.enable_per_worker_metric_log or not metrics_list:
            return
        for rank, metrics in enumerate(metrics_list):
            if not metrics:
                continue
            metrics_to_log = (
                {f"{prefix}/{k}": v for k, v in metrics.items()}
                if add_prefix
                else metrics
            )
            self.metric_logger.log(
                data=metrics_to_log,
                step=step,
                worker_group_name=worker_group_name,
                rank=rank,
            )

    def _aggregate_numeric_metrics(self, metrics_list: list[dict]) -> dict:
        """Average numeric metrics from a list of metric dictionaries."""
        if not metrics_list:
            return {}
        merged_metrics = defaultdict(list)
        for metrics in metrics_list:
            if not metrics:
                continue
            for key, value in metrics.items():
                merged_metrics[key].append(value)
        return {
            key: (sum(values) / len(values))
            for key, values in merged_metrics.items()
            if values
        }

    def _process_ranked_numeric_results(
        self, results: list[dict], metric_field: str
    ) -> tuple[dict, list[dict]]:
        metric_list: list[dict] = []
        per_rank_metrics: dict[int, list[dict]] = defaultdict(list)
        for result in results:
            metrics = result.get(metric_field, None)
            if not metrics:
                continue
            metric_list.append(metrics)
            rank = result.get("rank", None)
            if rank is not None:
                per_rank_metrics[int(rank)].append(metrics)

        aggregated_metrics = self._aggregate_numeric_metrics(metric_list)
        ranked_metrics_list: list[dict] = []
        if per_rank_metrics:
            max_rank = max(per_rank_metrics.keys())
            ranked_metrics_list = [{} for _ in range(max_rank + 1)]
            for rank, metrics_list in per_rank_metrics.items():
                ranked_metrics_list[rank] = self._aggregate_numeric_metrics(
                    metrics_list
                )
        return aggregated_metrics, ranked_metrics_list

    def _process_ranked_eval_results(
        self, results: list[dict], metric_field: str
    ) -> tuple[dict, list[dict]]:
        metric_list: list[dict] = []
        per_rank_metrics: dict[int, list[dict]] = defaultdict(list)
        for result in results:
            metrics = result.get(metric_field, None)
            if not metrics:
                continue
            metric_list.append(metrics)
            rank = result.get("rank", None)
            if rank is not None:
                per_rank_metrics[int(rank)].append(metrics)

        aggregated_metrics = (
            compute_evaluate_metrics(metric_list) if metric_list else {}
        )
        ranked_metrics_list: list[dict] = []
        if per_rank_metrics:
            max_rank = max(per_rank_metrics.keys())
            ranked_metrics_list = [{} for _ in range(max_rank + 1)]
            for rank, metrics_list in per_rank_metrics.items():
                ranked_metrics_list[rank] = compute_evaluate_metrics(metrics_list)
        return aggregated_metrics, ranked_metrics_list

    def run(self):
        start_step = self.global_step
        start_time = time.time()
        log_interval = int(self.cfg.runner.log_interval)
        if log_interval < 1:
            raise ValueError(f"runner.log_interval must be >= 1, got {log_interval}.")
        worker_step_synced = False
        local_update_steps = int(self.cfg.runner.local_update_steps)
        if local_update_steps < 1:
            raise ValueError(
                f"runner.local_update_steps must be >= 1, got {local_update_steps}."
            )
        batch_size = int(self.cfg.algorithm.batch_size)
        if batch_size < 1:
            raise ValueError(f"algorithm.batch_size must be >= 1, got {batch_size}.")

        while self.global_step < self.max_steps:
            _step = self.global_step
            next_step = _step + 1
            if not worker_step_synced:
                self.actor.set_global_step(_step)

            with self.timer("step"):
                self.update_actor_batches(batch_size, local_update_steps)

                actor_training_handle: Handle = self.actor.run_training()
                actor_metrics_per_rank = actor_training_handle.wait()
                if not isinstance(actor_metrics_per_rank, list):
                    actor_metrics_per_rank = [actor_metrics_per_rank]
                (
                    _actor_time_metrics_agg,
                    actor_time_metrics_per_rank,
                ) = actor_training_handle.consume_durations(return_per_rank=True)

            ranked_actor_training_results = [
                {"rank": rank, "train": rank_metrics}
                for rank, rank_metrics in enumerate(actor_metrics_per_rank)
                if rank_metrics is not None
            ]
            metrics, actor_training_metrics_per_rank = (
                self._process_ranked_numeric_results(
                    ranked_actor_training_results, metric_field="train"
                )
            )
            metrics.pop("__global_step", None)
            global_steps = []
            for rank_metrics in actor_training_metrics_per_rank:
                if "__global_step" in rank_metrics:
                    global_steps.append(int(rank_metrics.pop("__global_step")))

            if global_steps:
                self.global_step = max(global_steps)
                worker_step_synced = True
            else:
                self.global_step = next_step
                worker_step_synced = False
            _step = self.global_step

            run_val, save_model, is_train_end = check_progress(
                self.global_step,
                self.max_steps,
                self.cfg.runner.val_check_interval,
                self.cfg.runner.save_interval,
                1.0,
                run_time_exceeded=False,
            )
            eval_episodes = int(self.cfg.runner.get("eval_episodes", 0))
            eval_metrics: dict[str, Any] = {}
            if run_val and eval_episodes > 0:
                with self.timer("eval"):
                    self.update_rollout_weights()
                    eval_metrics = {f"eval/{k}": v for k, v in self.evaluate().items()}
                    self.metric_logger.log(data=eval_metrics, step=_step)
            if save_model:
                self._save_checkpoint()

            time_metrics = self.timer.consume_durations()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            actor_time_metrics_agg = self._aggregate_numeric_metrics(
                actor_time_metrics_per_rank
            )
            time_metrics.update(
                {f"time/actor/{k}": v for k, v in actor_time_metrics_agg.items()}
            )
            training_metrics = {f"train/{k}": v for k, v in metrics.items()}

            if _step == start_step + 1 or _step % log_interval == 0:
                self.metric_logger.log(time_metrics, _step)
                self.metric_logger.log(training_metrics, _step)
                self._log_ranked_metrics(
                    metrics_list=actor_training_metrics_per_rank,
                    step=_step,
                    prefix="train",
                    worker_group_name=self.actor.worker_group_name,
                )
                self._log_ranked_metrics(
                    metrics_list=actor_time_metrics_per_rank,
                    step=_step,
                    prefix="time/actor",
                    worker_group_name=self.actor.worker_group_name,
                )
                logging_metrics = dict(time_metrics)
                logging_metrics.update(training_metrics)
                logging_metrics.update(eval_metrics)
                self.print_metrics_table_async(
                    _step - 1, self.max_steps, start_time, logging_metrics, start_step
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
