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

from pathlib import Path


def test_agent_runner_has_single_minibatch_logging_loop():
    """Agent runner should not have nested minibatch logging loops."""
    source = Path("rlinf/runners/agent_runner.py").read_text(encoding="utf-8")
    _, _, run_body = source.partition("def run(self):")
    loop_stmt = "for i in range(self.cfg.algorithm.n_minibatches):"
    assert run_body.count(loop_stmt) == 1


def test_agent_runner_handles_none_rollout_metrics():
    """Agent runner should guard rollout metrics update when metrics are None."""
    source = Path("rlinf/runners/agent_runner.py").read_text(encoding="utf-8")
    _, _, run_body = source.partition("def run(self):")
    assert "if actor_rollout_metrics is not None:" in run_body
