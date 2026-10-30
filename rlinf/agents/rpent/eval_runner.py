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

from rlinf.data.schema.reasoning_results import DynamicRolloutResult
from rlinf.runners.agent_eval_runner import AgentEvalRunner
from rlinf.utils.logging import get_logger
from rlinf.utils.runner_utils import local_mkdir_safe

logger = get_logger()
_EPISODE_FIELDS = (
    "task",
    "success",
    "init_state_id",
    "termination_reason",
    "turns",
    "episode_id",
    "infra_retries",
)


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


class RPentEvalRunner(AgentEvalRunner):
    """Run the validation tasks and write ``eval_results.json``."""

    def pre_process(self) -> dict:
        return {"episodes": []}

    def update(self, context: dict, eval_pbar, input_channel, batch_idx, batch):
        received = 0
        while received < self.total_batch_size:
            result: DynamicRolloutResult = input_channel.get()
            traj = result.extra_fields_traj
            episodes = len(traj["success"])
            context["episodes"].extend(
                {key: traj[key][i] for key in _EPISODE_FIELDS} for i in range(episodes)
            )
            received += episodes
            eval_pbar.update(episodes)

    def update_batch(self, context: dict, eval_pbar, time_metrics):
        summary = summarize_episodes(context["episodes"])
        eval_pbar.set_postfix({"success_rate": f"{summary['success_rate']:.4f}"})

    def post_process(self, context: dict) -> None:
        summary = summarize_episodes(context["episodes"])
        output_dir = os.path.join(
            self.cfg.runner.output_dir, self.cfg.runner.experiment_name
        )
        local_mkdir_safe(output_dir)
        output_file = os.path.join(output_dir, "eval_results.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {"summary": summary, "episodes": context["episodes"]}, f, indent=2
            )
        logger.info(
            f"RPent success rate {summary['success_rate']:.4f} over "
            f"{summary['episodes']} episodes; results in {output_file}"
        )
