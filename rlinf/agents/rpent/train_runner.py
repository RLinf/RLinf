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

"""Policy updates for RPent's synchronous cohorts and asynchronous rollout buffer."""

import json
import time
from pathlib import Path

from rlinf.runners.async_agent_runner import AsyncAgentRunner
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle


class RPentRunner(AsyncAgentRunner):
    """Run RPent training over the session server for both sync and async GRPO."""

    supports_actor_recompute = True

    def _build_dataloader(self, train_dataset, val_dataset, collate_fn=None):
        width = (
            self.cfg.data.sampling_batch_size if self.cfg.runner.synchronous else None
        )
        super()._build_dataloader(
            train_dataset, val_dataset, collate_fn, data_batch_size=width
        )

    def _prepare_update(
        self, release_handle: Handle, watched_handles: list[Handle]
    ) -> None:
        if not self.cfg.runner.synchronous:
            return
        cohort = self._wait_actor(release_handle, watched_handles)[0]
        self.agent_loop.cancel_groups(cohort["remaining"]).wait()
        handle = self.rollout_buffer.admit_cohort(cohort)
        cohort["late_completions"] = self._wait_actor(handle, watched_handles)[0]
        output = (
            Path(self.cfg.runner.output_dir)
            / self.cfg.runner.experiment_name
            / "sampling"
        )
        output.mkdir(parents=True, exist_ok=True)
        (output / f"step-{self.global_steps}.json").write_text(
            json.dumps(cohort, indent=2)
        )

    def _run_actor_step(
        self, payload_channels: list[Channel], watched_handles: list[Handle]
    ) -> tuple[dict | None, list[dict], float, dict[str, float]]:
        if not self.recompute_logprobs:
            return super()._run_actor_step(payload_channels, watched_handles)
        # Synchronous GRPO scores admitted trajectories against the pre-update
        # actor for truncated importance sampling.
        score = self.actor.run_inference(
            input_channel=self.rollout_channel,
            output_channel=self.inference_channel,
            compute_ref_logprobs=self.compute_ref_logprobs,
        )
        self._wait_actor(score, watched_handles)
        train = self.actor.run_training(
            input_channel=self.inference_channel, version=self.global_steps
        )
        train_start = time.perf_counter()
        metrics = self._wait_actor(train, watched_handles)
        actor_wait_time = time.perf_counter() - train_start
        return (
            metrics[0][0],
            metrics[0][1],
            actor_wait_time,
            {
                "training": train.consume_duration(),
                "inference": score.consume_duration(),
            },
        )
