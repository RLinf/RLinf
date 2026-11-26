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

"""Evaluate an RPent planner on fixed tasks and report its success rate."""

import json
import os
from collections import defaultdict

from rlinf.runners.reasoning_eval_runner import ReasoningEvalRunner
from rlinf.utils.logging import get_logger
from rlinf.utils.runner_utils import local_mkdir_safe

logger = get_logger()


def summarize_episodes(episodes: list[dict]) -> dict:
    """Return the success rate overall and per LIBERO suite."""
    by_suite = defaultdict(list)
    for episode in episodes:
        by_suite[episode["task"]["suite"]].append(episode["success"])
    return {
        "episodes": len(episodes),
        "success_rate": sum(e["success"] for e in episodes) / max(len(episodes), 1),
        "suites": {
            suite: {
                "episodes": len(results),
                "success_rate": sum(results) / len(results),
            }
            for suite, results in sorted(by_suite.items())
        },
    }


class RPentEvalRunner(ReasoningEvalRunner):
    """Run fixed tasks against the loaded SGLang policy and save native outcomes."""

    def __init__(self, *, agent_loop, **kwargs):
        super().__init__(**kwargs)
        self.agent_loop = agent_loop

    def run(self):
        try:
            urls = [url for url in self.rollout.get_server_url().wait() if url]
            if not urls:
                raise ValueError(
                    "RPent evaluation needs rollout.sglang.backend_type=server"
                )
            episodes = []
            for _ in range(self.cfg.runner.max_epochs):
                if self.global_steps >= self.max_steps:
                    break
                for batch in self.val_dataloader:
                    if self.global_steps >= self.max_steps:
                        break
                    self._put_batch(batch, len(self.agent_loop._workers))
                    for results in self.agent_loop.evaluate_batch(
                        self.dataloader_channel, urls
                    ).wait():
                        episodes.extend(results)
                    self.global_steps += 1
            self._stop_rollouts()
            summary = summarize_episodes(episodes)
            eval_metrics = {
                "eval/episodes": summary["episodes"],
                "eval/success_rate": summary["success_rate"],
                **{
                    f"eval/{suite}/success_rate": stats["success_rate"]
                    for suite, stats in summary["suites"].items()
                },
            }
            self.metric_logger.log(eval_metrics, self.global_steps)
            output_dir = os.path.join(
                self.cfg.runner.output_dir, self.cfg.runner.experiment_name
            )
            local_mkdir_safe(output_dir)
            output_file = os.path.join(output_dir, "eval_results.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({"summary": summary, "episodes": episodes}, f, indent=2)
            logger.info(
                f"RPent success rate {summary['success_rate']:.4f} over "
                f"{summary['episodes']} episodes; results in {output_file}"
            )
        finally:
            self.metric_logger.finish()
