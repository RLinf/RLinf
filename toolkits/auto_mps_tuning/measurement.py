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

"""Embodied runner adapters for MPS verification and calibration measurements."""

import json
import os
import statistics
import time
from pathlib import Path

from omegaconf import DictConfig

from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler.worker.worker_group import WorkerGroup


def run_mps(
    cfg: DictConfig,
    actor: WorkerGroup,
    rollout: WorkerGroup,
    env: WorkerGroup,
    reward: WorkerGroup | None = None,
    router: WorkerGroup | None = None,
    server: WorkerGroup | None = None,
) -> None:
    """Run calibration or managed training, owning worker and Ray cleanup.

    Args:
        cfg: Training or internal calibration configuration.
        actor: Launched actor worker group.
        rollout: Launched rollout worker group.
        env: Launched environment worker group.
        reward: Optional launched reward worker group.
        router: Optional reward API router group.
        server: Optional reward API server group.
    """
    import ray

    trial = (cfg.get("mps") or {}).get("trial", False)
    try:
        runner = (
            CalibrationRunner(cfg, actor, rollout, env, reward)
            if trial
            else EmbodiedRunner(cfg, actor, rollout, env, reward)
        )
        runner.init_workers()
        if os.environ.get("RLINF_MPS_SESSION"):
            for group in (rollout, env):
                group.verify_mps_client().wait()
        runner.run()
        for group in (reward, router, server):
            if group is not None:
                group.stop().wait()
    finally:
        ray.shutdown()


class CalibrationRunner(EmbodiedRunner):
    """Reuse embodied worker setup while owning the calibration loop and metrics."""

    def _set_global_step(self) -> None:
        for group in (self.actor, self.rollout, self.env):
            group.set_global_step(self.global_step).wait()

    def run(self) -> None:
        """Measure frozen-weight rollouts and save timings in the trial directory.

        Bootstrap is prepared before timing when overlap is enabled. Service
        times exclude waiting for the peer; window times include trajectory
        receipt and completion of both env and rollout workers.
        """
        settings = self.cfg.mps.auto_tuning
        windows = []

        try:
            self._set_global_step()
            self.update_rollout_weights()
            for _ in range(settings.warmup_windows + settings.measure_windows):
                self._set_global_step()
                if self.overlap_env_bootstrap:
                    self.env.prefetch_train_bootstrap(
                        rollout_channel=self.rollout_channel
                    ).wait()
                for group in (self.actor, self.rollout, self.env):
                    group.pop_execution_times().wait()
                started = time.perf_counter()
                env_handle = self.env.interact(
                    input_channel=self.env_channel,
                    rollout_channel=self.rollout_channel,
                    reward_channel=self.reward_channel,
                )
                rollout_handle = self.rollout.generate(
                    input_channel=self.rollout_channel,
                    output_channel=self.env_channel,
                    actor_channel=self.actor_channel,
                )
                self.actor.recv_rollout_trajectories(
                    input_channel=self.actor_channel
                ).wait()
                rollout_handle.wait()
                env_handle.wait()
                elapsed = time.perf_counter() - started
                env_times = env_handle.consume_durations()
                policy_times = rollout_handle.consume_durations()
                windows.append(
                    {
                        "window_s": elapsed,
                        "service_s": {
                            "rollout": policy_times["predict"],
                            "env": env_times["env_interact_step"],
                        },
                    }
                )
                self.global_step += 1
            measured = windows[settings.warmup_windows :]
            result = {
                "window_s": statistics.median(w["window_s"] for w in measured),
                "service_s": {
                    k: statistics.median(w["service_s"][k] for w in measured)
                    for k in ("rollout", "env")
                },
            }
        finally:
            self._finish_run()
        (Path(self.cfg.runner.logger.log_path) / "measurement.json").write_text(
            json.dumps(result)
        )
