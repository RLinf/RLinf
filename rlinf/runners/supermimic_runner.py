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
for those steps, runs the owed updates on the actor, syncs the policy to
the rollout workers, and handles logging, evaluation and checkpoints.
"""

from __future__ import annotations

import datetime
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
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.checkpoint import parse_global_step_from_checkpoint_path

if TYPE_CHECKING:
    from rlinf.workers.actor.fsdp_supermimic_policy_worker import (
        EmbodiedSuperMimicFSDPPolicy,
    )
    from rlinf.workers.env.supermimic_env_worker import SuperMimicEnvWorker
    from rlinf.workers.rollout.hf.supermimic_rollout_worker import (
        SuperMimicRolloutWorker,
    )

_RUNNER_STATE = "runner_state.json"


def _rounded(metrics: dict) -> dict:
    return {
        key: float(f"{value:.4g}") if isinstance(value, (int, float)) else value
        for key, value in metrics.items()
    }


def _clock(seconds: float) -> str:
    return str(datetime.timedelta(seconds=round(seconds)))


class SuperMimicRunner(EmbodiedRunner):
    """Collect, update and sync at ``algorithm.updates_per_env_step``.

    Every control step adds ``updates_per_env_step`` owed updates. Collection
    stops at the first step that makes an update due, the due updates run,
    and the new policy goes to the rollout workers before collection resumes.
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
        self.updates_per_env_step = float(cfg.algorithm.updates_per_env_step)
        self.warmup_steps = int(cfg.algorithm.warmup_sim_steps)
        self.owed_updates = 0.0
        self.env_steps = 0
        self.log_interval = int(cfg.runner.get("log_interval", 100))
        if self.log_interval < 1:
            raise ValueError(
                f"runner.log_interval must be >= 1, got {self.log_interval}."
            )
        self._printed_step = 0

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
            f"Resuming from {resume_dir}. Networks, optimizers, reset bins and RNG "
            "are restored; the replay and simulator state are not saved, so the "
            "run refills the replay before its next update."
        )
        self.global_step = parse_global_step_from_checkpoint_path(resume_dir)
        self.actor.load_checkpoint(os.path.join(resume_dir, "actor")).wait()
        self.env.load_checkpoint(os.path.join(resume_dir, "env")).wait()
        with open(os.path.join(resume_dir, _RUNNER_STATE)) as file:
            state = json.load(file)
        self.owed_updates = float(state["owed_updates"])
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
        state = {"owed_updates": self.owed_updates, "env_steps": self.env_steps}
        with open(os.path.join(base, _RUNNER_STATE), "w") as file:
            json.dump(state, file, indent=2)

    # -------------------------------------------------------------- training

    def _collect(self, num_steps: int) -> tuple[float, Handle, Handle, Handle]:
        """Run ``num_steps`` control steps on every rank.

        Returns the replay windows summed over ranks and the three handles.
        """
        env_handle = self.env.collect(num_steps)
        rollout_handle = self.rollout.serve(num_steps)
        actor_handle = self.actor.ingest(num_steps)
        windows = sum(result["buffer/valid_windows"] for result in actor_handle.wait())
        env_handle.wait()
        rollout_handle.wait()
        self.env_steps += num_steps
        return windows, env_handle, rollout_handle, actor_handle

    def _steps_until_update(self) -> tuple[int, float]:
        """Control steps until an update is due, and the owed count by then."""
        steps, owed = 0, self.owed_updates
        while True:
            steps += 1
            owed += self.updates_per_env_step
            if owed >= 1.0:
                return steps, owed

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

    def run(self) -> None:
        start_step = self.global_step
        start_time = time.time()
        self._printed_step = start_step
        self.update_rollout_weights()
        if self.global_step < self.max_steps and self.warmup_steps > 0:
            # Warmup only fills the replay: it owes no updates and is not logged.
            self._collect(self.warmup_steps)
            self.env.consume_metrics().wait()
            for group in (self.env, self.rollout, self.actor):
                group.pop_execution_times().wait()

        while self.global_step < self.max_steps:
            step = self.global_step
            with self.timer("step"):
                with self.timer("generate_rollouts"):
                    ready = 0
                    while ready == 0:
                        steps, self.owed_updates = self._steps_until_update()
                        windows, env_handle, rollout_handle, _ = self._collect(steps)
                        if windows > 0:
                            ready = min(int(self.owed_updates), self.max_steps - step)
                    env_metrics = self.env.consume_metrics().wait()[0]
                with self.timer("actor_training"):
                    actor_handle = self.actor.run_updates(ready, step)
                    training_metrics = actor_handle.wait()
                self.owed_updates -= ready
                self.global_step += ready
                with self.timer("sync_weights"):
                    self.update_rollout_weights()
                last_step = self.global_step - 1
                eval_metrics = self._maybe_eval_and_checkpoint(last_step)
            self._log_iteration(
                last_step,
                start_time,
                start_step,
                (env_handle, rollout_handle, actor_handle),
                env_metrics,
                training_metrics,
                eval_metrics,
            )
        self._finish_run()

    def _log_iteration(
        self,
        step: int,
        start_time: float,
        start_step: int,
        handles: tuple[Handle, Handle, Handle],
        env_metrics: dict[str, float],
        training_metrics: list[dict],
        eval_metrics: dict[str, float],
    ) -> None:
        time_metrics = {
            f"time/{key}": value
            for key, value in self.timer.consume_durations().items()
        }
        for name, handle, group in zip(
            ("env", "rollout", "actor"),
            handles,
            (self.env, self.rollout, self.actor),
            strict=True,
        ):
            durations, per_rank = handle.consume_durations(return_per_rank=True)
            time_metrics.update({f"time/{name}/{k}": v for k, v in durations.items()})
            self._log_ranked_metrics(
                metrics_list=per_rank,
                step=step,
                prefix=f"time/{name}",
                worker_group_name=group.worker_group_name,
            )
        env_metrics = {f"env/{key}": value for key, value in env_metrics.items()}
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
        self._print_progress(
            step + 1,
            start_time,
            start_step,
            {**time_metrics, **env_metrics, **training_metrics},
            eval_metrics,
        )

    def _print_progress(
        self,
        done: int,
        start_time: float,
        start_step: int,
        metrics: dict[str, float],
        eval_metrics: dict[str, float],
    ) -> None:
        """Log the latest metrics every ``runner.log_interval`` updates and at the end.

        Evaluation results get their own line when they arrive. One iteration
        can run several updates, so a line is due whenever ``done`` passes a
        multiple of the interval.
        """
        if eval_metrics:
            self.logger.info(
                f"step {done}/{self.max_steps} eval {_rounded(eval_metrics)}"
            )
        interval = self.log_interval
        if done < self.max_steps and done // interval == self._printed_step // interval:
            return
        self._printed_step = done
        elapsed = time.time() - start_time
        eta = elapsed / max(done - start_step, 1) * (self.max_steps - done)
        self.logger.info(
            f"step {done}/{self.max_steps} elapsed {_clock(elapsed)} "
            f"eta {_clock(eta)} metrics {_rounded(metrics)}"
        )
