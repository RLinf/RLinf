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

import time

from rlinf.runners.embodied_runner import EmbodiedRunner


class ReFLRunner(EmbodiedRunner):
    """Sample and update ReFL batches; rollout and env workers serve evaluation."""

    def run(self) -> None:
        start_step = self.global_step
        start_time = time.time()
        for step in range(start_step, self.max_steps):
            self.actor.set_global_step(self.global_step).wait()
            self.rollout.set_global_step(self.global_step).wait()
            self.env.set_global_step(self.global_step).wait()
            profiled = self._should_profile_step(self.global_step)
            if profiled:
                self._open_profiling_window(self.global_step)
            with self.timer("step", trace_args={"step_idx": step}):
                with self.timer("actor_training"):
                    handle = self.actor.run_training()
                    results = handle.wait()
                self.global_step += 1
                eval_metrics = self._maybe_eval_and_checkpoint(step)
            if profiled:
                self._close_profiling_window(step)
            metrics = {
                f"train/{key}": value
                for key, value in self._aggregate_numeric_metrics(results).items()
            }
            metrics.update(
                {
                    f"time/{key}": value
                    for key, value in self.timer.consume_durations().items()
                }
            )
            actor_times, ranked_times = handle.consume_durations(return_per_rank=True)
            metrics.update(
                {f"time/actor/{key}": value for key, value in actor_times.items()}
            )
            self.metric_logger.log(metrics, step)
            self._log_ranked_metrics(
                results, step, "train", self.actor.worker_group_name
            )
            self._log_ranked_metrics(
                ranked_times, step, "time/actor", self.actor.worker_group_name
            )
            metrics.update(eval_metrics)
            self.print_metrics_table_async(
                step, self.max_steps, start_time, metrics, start_step
            )
        self._finish_run()
