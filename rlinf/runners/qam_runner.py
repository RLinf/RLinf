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


from pathlib import Path

from rlinf.runners.offline_runner import OfflineRunner
from rlinf.utils.checkpoint import parse_global_step_from_checkpoint_path


class QAMRunner(OfflineRunner):
    """Independent seeds with colocated OGBench collection and evaluation.

    Each actor rank owns a separate replay and optimizer. The inherited run loop
    calls run_training, evaluates and checkpoints at configured step boundaries.
    """

    def init_workers(self):
        self.actor.init_worker().wait()
        resume = self.cfg.runner.get("resume_dir")
        if resume:
            self.global_step = parse_global_step_from_checkpoint_path(resume)
            steps = self.actor.load_checkpoint(str(Path(resume) / "actor")).wait()
            if any(s != self.global_step for s in steps):
                raise ValueError("Seed checkpoint steps must match the runner step")

    def update_rollout_weights(self):
        self.actor.prepare_evaluation().wait()

    def evaluate(self):
        results = self.actor.evaluate().wait()
        self._log_ranked_metrics(
            results, self.global_step, "eval", self.actor.worker_group_name
        )
        return self._aggregate_numeric_metrics(results)

    def run(self):
        if self.cfg.runner.only_eval:
            self.update_rollout_weights()
            result = self.evaluate()
            self._finish_run()
            return result
        return super().run()
