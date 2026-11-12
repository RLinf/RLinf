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

"""Control plane of SuperMimic training.

The env, rollout and actor workers of one rank exchange every control step
point to point (see :class:`SuperMimicEnvWorker`). This runner decides how
many steps to collect before the next update, launches the three workers
for those steps, runs the due updates on the actor, stopping on the next
save or evaluation step, syncs the policy to the rollout workers, and
handles logging, evaluation and checkpoints.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import TYPE_CHECKING

from omegaconf import DictConfig

from rlinf.envs.sim.g1_mimic.eval_metrics import summarize_rows
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.runners.supermimic_eval_runner import (
    eval_export_dir,
    export_eval_results,
    merge_rank_rows,
)
from rlinf.scheduler import Tracer
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.checkpoint import parse_global_step_from_checkpoint_path
from rlinf.utils.runner_utils import check_progress

if TYPE_CHECKING:
    from rlinf.workers.actor.fsdp_supermimic_policy_worker import (
        EmbodiedSuperMimicFSDPPolicy,
    )
    from rlinf.workers.env.supermimic_env_worker import SuperMimicEnvWorker
    from rlinf.workers.rollout.hf.supermimic_rollout_worker import (
        SuperMimicRolloutWorker,
    )

_RUNNER_STATE = "runner_state.json"


def due_updates(step: int, anchor: float | None, ratio: float) -> tuple[int, float]:
    """Updates that fall due after vector step ``step``, and the next anchor.

    The first step makes one update due. After it, an update falls due each
    time ``ratio`` times the steps since ``anchor`` passes a whole number,
    and the anchor moves past the steps those updates paid for.
    """
    if anchor is None:
        return 1, float(step)
    count = int((step - anchor) * ratio)
    return count, anchor + count / ratio


def update_ratio(cfg: DictConfig) -> float:
    """Global updates per vector step.

    Each update draws ``wm_batches`` batches of ``wm_window_batch`` windows
    of ``k`` steps, and the replay ratio is the number of times each
    collected transition is drawn on average.
    """
    alg = cfg.algorithm
    return (
        int(cfg.env.train.total_num_envs)
        / int(alg.wm_batches)
        / int(alg.wm_window_batch)
        / int(alg.k)
        * float(alg.replay_ratio)
    )


def _boundary_distance(step: int, interval: int) -> int | None:
    """Updates from ``step`` until the next positive multiple of ``interval``.

    A finished count of 0 is not a boundary, so the distance from 0 is the
    interval itself. An interval of 0 or less is disabled.
    """
    if interval <= 0:
        return None
    remainder = step % interval
    if remainder == 0:
        return interval
    return interval - remainder


def boundary_update_spans(
    step: int,
    count: int,
    max_steps: int,
    save_interval: int,
    val_check_interval: int,
) -> list[int]:
    """Split ``count`` updates at save and evaluation steps.

    ``check_progress`` acts only when the finished count is an exact multiple
    of the interval, so a piece that would pass the next multiple stops on
    it. The multiple itself is included. The pieces also stop at
    ``max_steps``.
    """
    spans: list[int] = []
    left = min(max(count, 0), max(max_steps - step, 0))
    while left > 0:
        distance = left
        for interval in (save_interval, val_check_interval):
            gap = _boundary_distance(step, interval)
            if gap is not None:
                distance = min(distance, gap)
        distance = min(distance, max_steps - step)
        spans.append(distance)
        step += distance
        left -= distance
    return spans


class SuperMimicRunner(EmbodiedRunner):
    """Collect, update and sync at the replay ratio of :func:`update_ratio`.

    Collection stops at the first vector step that makes updates due while
    the replay holds at least ``k`` steps; updates that fall due before that
    are dropped. The due updates then run, but a batch stops on the next
    save or evaluation step so that step is checkpointed before the
    remaining updates. The new policy goes to the rollout workers before
    collection resumes.
    """

    def __init__(
        self,
        cfg: DictConfig,
        actor: EmbodiedSuperMimicFSDPPolicy,
        rollout: SuperMimicRolloutWorker,
        env: SuperMimicEnvWorker,
        reward=None,
    ):
        if reward is not None:
            raise ValueError("SuperMimic computes its rewards in the env worker.")
        super().__init__(cfg, actor=actor, rollout=rollout, env=env)
        self.ratio = update_ratio(cfg)
        self.k = int(cfg.algorithm.k)
        self.anchor: float | None = None
        self.env_steps = 0
        # Vector steps in the replay; it is not checkpointed.
        self.replay_steps = 0
        self.log_interval = int(cfg.runner.get("log_interval", 100))
        if self.log_interval < 1:
            raise ValueError(
                f"runner.log_interval must be >= 1, got {self.log_interval}."
            )
        self._logged_step = 0

    # ----------------------------------------------------------- checkpoints

    def _checkpoint_root(self) -> str:
        return os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            "checkpoints",
        )

    def _latest_checkpoint(self) -> str | None:
        """Newest ``global_step_<N>`` whose runner state was written last."""
        root = self._checkpoint_root()
        if not os.path.isdir(root):
            return None
        steps = [
            int(match.group(1))
            for name in os.listdir(root)
            if (match := re.fullmatch(r"global_step_(\d+)", name))
            and os.path.isfile(os.path.join(root, name, _RUNNER_STATE))
        ]
        return os.path.join(root, f"global_step_{max(steps)}") if steps else None

    def init_workers(self) -> None:
        rollout_handle = self.rollout.init_worker()
        env_handle = self.env.init_worker()
        rollout_handle.wait()
        env_handle.wait()
        self.actor.init_worker().wait()

        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir == "auto":
            resume_dir = self._latest_checkpoint()
            if resume_dir is None:
                self.logger.info(
                    "resume_dir=auto found no complete checkpoint under "
                    f"{self._checkpoint_root()}; starting from scratch."
                )
        if not resume_dir:
            return
        self.logger.info(
            f"Resuming from {resume_dir}. Networks, optimizers, reset statistics "
            "and RNG are restored; the replay and simulator state are not saved, "
            "so the run refills the replay before its next update."
        )
        self.global_step = parse_global_step_from_checkpoint_path(resume_dir)
        self.actor.load_checkpoint(os.path.join(resume_dir, "actor")).wait()
        self.env.load_checkpoint(os.path.join(resume_dir, "env")).wait()
        with open(os.path.join(resume_dir, _RUNNER_STATE)) as file:
            state = json.load(file)
        self.anchor = state["anchor"]
        self.env_steps = int(state["env_steps"])

    def _save_checkpoint(self) -> None:
        self.logger.info(f"Saving checkpoint at step {self.global_step}.")
        base = os.path.join(self._checkpoint_root(), f"global_step_{self.global_step}")
        actor_path = os.path.join(base, "actor")
        os.makedirs(actor_path, exist_ok=True)
        actor_handle = self.actor.save_checkpoint(actor_path, self.global_step)
        env_handle = self.env.save_checkpoint(os.path.join(base, "env"))
        actor_handle.wait()
        env_handle.wait()
        state = {"anchor": self.anchor, "env_steps": self.env_steps}
        with open(os.path.join(base, _RUNNER_STATE), "w") as file:
            json.dump(state, file, indent=2)

    # -------------------------------------------------------------- training

    def _collect(self, num_steps: int) -> tuple[Handle, Handle]:
        """Run ``num_steps`` control steps on every rank; returns the env and
        rollout handles."""
        env_handle = self.env.collect(num_steps)
        rollout_handle = self.rollout.serve(num_steps)
        self.actor.ingest(num_steps).wait()
        env_handle.wait()
        rollout_handle.wait()
        self.env_steps += num_steps
        self.replay_steps += num_steps
        return env_handle, rollout_handle

    def _steps_until_update(self) -> tuple[int, int, float]:
        """Vector steps to collect until updates fall due with a full enough
        replay, those updates, and the anchor after them."""
        steps, anchor = 0, self.anchor
        while True:
            due, anchor = due_updates(self.env_steps + steps, anchor, self.ratio)
            steps += 1
            if due and self.replay_steps + steps >= self.k:
                return steps, due, anchor

    def evaluate(self) -> dict[str, float]:
        meta = self.env.eval_metadata().wait()[0]
        env_handle = self.env.evaluate(
            input_channel=self.env_channel, rollout_channel=self.rollout_channel
        )
        rollout_handle = self.rollout.evaluate(
            input_channel=self.rollout_channel, output_channel=self.env_channel
        )
        rows = merge_rank_rows(env_handle.wait())
        rollout_handle.wait()
        metrics, tables, nested = summarize_rows(rows, meta)
        out_dir = eval_export_dir(self.cfg, self.global_step)
        if out_dir is not None:
            export_eval_results(out_dir, self.cfg, meta, metrics, tables, nested)
        return metrics

    def _set_global_step(self) -> None:
        """Pass the finished update count to every worker group.

        The actor tags its next weight sync with it as the policy version.
        """
        handles = [
            group.set_global_step(self.global_step)
            for group in (self.actor, self.rollout, self.env)
        ]
        for handle in handles:
            handle.wait()

    def run(self) -> None:
        start_step = self.global_step
        start_time = time.time()
        self._logged_step = start_step
        self._set_global_step()
        self.update_rollout_weights()

        while self.global_step < self.max_steps:
            step = self.global_step
            profiled_step = step if self._should_profile_step(step) else None
            if profiled_step is not None:
                self._open_profiling_window(profiled_step)
            with self.timer("step"):
                with self.timer("generate_rollouts"):
                    steps, due, self.anchor = self._steps_until_update()
                    env_handle, rollout_handle = self._collect(steps)
                # check_progress matches exact multiples, so a batch that would
                # pass a save or evaluation step stops on that step first.
                spans = boundary_update_spans(
                    step,
                    due,
                    self.max_steps,
                    int(self.cfg.runner.save_interval),
                    int(self.cfg.runner.val_check_interval),
                )
                (
                    actor_handle,
                    training_metrics,
                    eval_metrics,
                    last_step,
                ) = self._execute_update_spans(spans)
            if profiled_step is not None:
                self._close_profiling_window(profiled_step)
            self._log_iteration(
                last_step,
                start_time,
                start_step,
                (env_handle, rollout_handle, actor_handle),
                training_metrics,
                eval_metrics,
            )
        self._finish_run()

    def _execute_update_spans(
        self, spans: list[int]
    ) -> tuple[Handle, list[dict], dict[str, float], int]:
        """Run each span, then sync and checkpoint before the next one.

        One collection still pays for every span. The runner timer keeps a
        single measurement per name, so a batch that crosses more than one
        boundary records the summed training and sync time after the loop.
        """
        single = len(spans) == 1
        actor_handle: Handle | None = None
        training_metrics: list[dict] = []
        eval_metrics: dict[str, float] = {}
        measured = {"actor_training": 0.0, "sync_weights": 0.0}
        last_step = self.global_step
        for count in spans:
            actor_handle, training_metrics = self._run_update_count(
                count, single, measured
            )
            self.global_step += count
            self._sync_updated_weights(single, measured)
            last_step = self.global_step - 1
            segment_eval = self._maybe_eval_and_checkpoint(last_step)
            if segment_eval:
                eval_metrics = segment_eval
        if not single:
            for name, duration in measured.items():
                self.timer.add_duration(name, duration)
        if actor_handle is None:
            raise RuntimeError("An update batch produced no actor call.")
        return actor_handle, training_metrics, eval_metrics, last_step

    def _run_update_count(
        self, count: int, single: bool, measured: dict[str, float]
    ) -> tuple[Handle, list[dict]]:
        def run() -> tuple[Handle, list[dict]]:
            handle = self.actor.run_updates(count)
            return handle, handle.wait()

        if single:
            with self.timer("actor_training"):
                return run()
        return self._measure("actor_training", measured, run)

    def _sync_updated_weights(self, single: bool, measured: dict[str, float]) -> None:
        def sync() -> None:
            self._set_global_step()
            self.update_rollout_weights()

        if single:
            with self.timer("sync_weights"):
                sync()
            return
        self._measure("sync_weights", measured, sync)

    def _measure(self, name: str, measured: dict[str, float], fn):
        Tracer.trace_begin(name, cat="runner")
        started = time.perf_counter()
        try:
            return fn()
        finally:
            measured[name] += time.perf_counter() - started
            Tracer.trace_end(name, cat="runner")

    def _maybe_eval_and_checkpoint(self, step: int) -> dict:
        # One iteration can evaluate at more than one boundary. The first
        # uses the parent path. ScopedTimer records each name once, so a
        # later evaluation adds to that record instead of entering the
        # parent's ``with self.timer("eval")`` again.
        if not self.timer.has_duration("eval"):
            return super()._maybe_eval_and_checkpoint(step)
        run_val, save_model, _ = check_progress(
            self.global_step,
            self.max_steps,
            self.cfg.runner.val_check_interval,
            self.cfg.runner.save_interval,
            1.0,
            run_time_exceeded=False,
        )
        eval_metrics: dict[str, float] = {}
        if run_val:
            started = time.perf_counter()
            self.update_rollout_weights()
            evaluated = self.evaluate()
            eval_metrics = {f"eval/{key}": value for key, value in evaluated.items()}
            self.metric_logger.log(data=eval_metrics, step=step)
            self.timer.add_duration("eval", time.perf_counter() - started)
        if save_model:
            self._save_checkpoint()
        return eval_metrics

    def _log_iteration(
        self,
        step: int,
        start_time: float,
        start_step: int,
        handles: tuple[Handle, Handle, Handle],
        training_metrics: list[dict],
        eval_metrics: dict[str, float],
    ) -> None:
        """Record the iteration that finished update ``step`` when a record is due.

        TensorBoard blocks the caller once ten events wait in its queue, so
        recording every scalar costs tens of milliseconds on network storage.
        Records therefore follow ``runner.log_interval``, and the env
        statistics of a record cover every step since the previous one.
        """
        driver_durations = self.timer.consume_durations()
        worker_durations = [
            handle.consume_durations(return_per_rank=True) for handle in handles
        ]
        # One iteration can run several updates, so a record is due whenever
        # the finished count passes a multiple of the interval.
        done = step + 1
        due = (
            bool(eval_metrics)
            or done >= self.max_steps
            or done // self.log_interval != self._logged_step // self.log_interval
        )
        if not due:
            return
        self._logged_step = done

        time_metrics = {f"time/{key}": value for key, value in driver_durations.items()}
        for name, (durations, per_rank), group in zip(
            ("env", "rollout", "actor"),
            worker_durations,
            (self.env, self.rollout, self.actor),
            strict=True,
        ):
            time_metrics.update({f"time/{name}/{k}": v for k, v in durations.items()})
            self._log_ranked_metrics(
                metrics_list=per_rank,
                step=step,
                prefix=f"time/{name}",
                worker_group_name=group.worker_group_name,
            )
        env_metrics = {
            f"env/{key}": value
            for key, value in self.env.consume_metrics().wait()[0].items()
        }
        env_metrics["env/env_steps"] = float(self.env_steps)
        self._log_ranked_metrics(
            metrics_list=training_metrics,
            step=step,
            prefix="train",
            worker_group_name=self.actor.worker_group_name,
        )
        training_metrics = {
            f"train/{key}": value
            for key, value in self._aggregate_numeric_metrics(training_metrics).items()
        }
        self.metric_logger.log(env_metrics, step)
        self.metric_logger.log(time_metrics, step)
        self.metric_logger.log(training_metrics, step)
        self.print_metrics_table_async(
            step,
            self.max_steps,
            start_time,
            {**time_metrics, **eval_metrics, **env_metrics, **training_metrics},
            start_step,
        )
