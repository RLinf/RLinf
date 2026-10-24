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

"""Time complete Search-R1 runs, including terminal publication and checkpoints."""

import json
import time
from pathlib import Path

import hydra
from agent.searchr1 import train
from omegaconf import DictConfig, OmegaConf

from rlinf.utils.logging import get_logger


class _BenchmarkRunner:
    """Time the real runner's lifecycle, including terminal weight publication."""

    initialization_started: float

    def init_workers(self) -> None:
        log_dir = Path(self.cfg.runner.logger.log_path)
        log_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.cfg, log_dir / "benchmark_config.yaml", resolve=True)
        super().init_workers()
        self.initialized = time.perf_counter()

    def run(self) -> None:
        super().run()
        cfg = self.cfg
        if cfg.runner.synchronous:
            # The synchronous runner normally publishes at the start of the next step.
            self._sync_weights()
            self.rollout.stop_serverless().wait()
        finished = time.perf_counter()
        if self.global_steps != cfg.runner.max_steps:
            raise RuntimeError(
                f"Expected {cfg.runner.max_steps} learner steps, completed {self.global_steps}."
            )
        measurements = {
            "synchronous": cfg.runner.synchronous,
            "steps": self.global_steps,
            "batch_size": cfg.data.rollout_batch_size,
            "critic_update_epochs": cfg.critic.get("update_epochs", 1),
            "initialization_seconds": self.initialized - self.initialization_started,
            "training_seconds": finished - self.initialized,
            "seconds_per_step": (finished - self.initialized) / self.global_steps,
            "prompts_per_second": self.global_steps
            * cfg.data.rollout_batch_size
            / (finished - self.initialized),
        }
        if not cfg.runner.synchronous:
            measurements.update(
                {
                    "published_version": self.published_version,
                    "actor_updates": self.actor_updates,
                    "critic_updates": self.critic_updates,
                    "max_in_flight_rollouts": cfg.rollout.max_in_flight_rollouts,
                    "rollout_buffer_capacity": cfg.rollout.rollout_buffer_capacity,
                }
            )
        output = Path(cfg.runner.logger.log_path) / "benchmark.json"
        output.write_text(json.dumps(measurements, indent=2) + "\n")
        get_logger().info(f"Completed benchmark: {measurements}")


class _BenchmarkAgentPPORunner(_BenchmarkRunner, train.AgentPPORunner):
    """Measure synchronous PPO using the training entry point."""


class _BenchmarkAsyncAgentPPORunner(_BenchmarkRunner, train.AsyncAgentPPORunner):
    """Measure asynchronous PPO using the training entry point."""


@hydra.main(version_base="1.1")
def main(cfg: DictConfig) -> None:
    """Measure initialization separately from all work needed by learner steps."""
    sync_runner, async_runner = train.AgentPPORunner, train.AsyncAgentPPORunner
    train.AgentPPORunner = _BenchmarkAgentPPORunner
    train.AsyncAgentPPORunner = _BenchmarkAsyncAgentPPORunner
    _BenchmarkRunner.initialization_started = time.perf_counter()
    try:
        train.main(cfg)
    finally:
        train.AgentPPORunner, train.AsyncAgentPPORunner = sync_runner, async_runner


if __name__ == "__main__":
    main()
