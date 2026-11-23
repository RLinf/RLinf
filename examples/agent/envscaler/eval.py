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

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Subset

from rlinf.agents.envscaler.agent_loop import EnvScalerAgentLoopWorker
from rlinf.config import validate_cfg
from rlinf.data.datasets.reasoning import create_reasoning_datasets
from rlinf.data.datasets.reasoning.envscaler import EnvScalerDataset
from rlinf.data.schema.reasoning_results import DynamicRolloutResult
from rlinf.models.tokenization.hf import hf_tokenizer
from rlinf.runners.agent_eval_runner import AgentEvalRunner
from rlinf.scheduler import Cluster, NodePlacementStrategy
from rlinf.utils.logging import get_logger
from rlinf.utils.placement import ModelParallelEvalComponentPlacement
from rlinf.utils.utils import output_redirector
from rlinf.workers.rollout.utils import get_rollout_backend_worker


def sample_tasks(dataset: EnvScalerDataset, count: int, seed: int) -> Subset:
    """Select a stable sample without replacement within each environment."""
    groups = defaultdict(list)
    for index, row in enumerate(dataset.data):
        task = row[dataset.answer_key]
        groups[task["env_id"]].append((task["task_id"], index))
    rng = random.Random(seed)
    indices = []
    for env_id, tasks in sorted(groups.items()):
        if not 1 <= count <= len(tasks):
            raise ValueError(
                f"Requested {count} tasks from {env_id}, which has {len(tasks)}"
            )
        indices.extend(index for _, index in rng.sample(sorted(tasks), count))
    return Subset(dataset, indices)


def trajectory_metrics(result: DynamicRolloutResult) -> list[dict]:
    """Count actual sampled policy tokens and the input seen on each turn."""
    trajectories = [
        {"turns": 0, "input_tokens": 0, "generated_tokens": 0}
        for _ in range(result.group_size)
    ]
    for idx, prompt, response, reward, end in zip(
        result.idx_to_traj,
        result.prompt_lengths,
        result.response_lengths,
        result.rewards,
        result.is_end,
        strict=True,
    ):
        row = trajectories[idx]
        row["turns"] += 1
        row["input_tokens"] += prompt
        row["generated_tokens"] += response
        row["final_context_tokens"] = prompt + response
        row["reward"] = reward
        row["passed"] = bool(end and reward == 1.0)
    return trajectories


def summarize(groups: list[list[dict]], ks: list[int]) -> dict:
    """Compute pass@k per task, then average across the selected tasks."""
    rows = [row for group in groups for row in group]
    summary = {"tasks": len(groups), "trajectories": len(rows)}
    for k in ks:
        estimates = []
        for group in groups:
            n = len(group)
            failures = sum(not row["passed"] for row in group)
            estimates.append(1 - math.comb(failures, k) / math.comb(n, k))
        summary[f"pass@{k}"] = mean(estimates)
    for field in ("turns", "generated_tokens", "input_tokens", "final_context_tokens"):
        values = [row[field] for row in rows]
        summary[field] = {
            "mean": mean(values),
            "median": median(values),
            "max": max(values),
        }
    return summary


class EnvScalerEvalRunner(AgentEvalRunner):
    """Save task-level samples as they finish and summarize the completed run."""

    def pre_process(self) -> dict:
        directory = Path(self.cfg.runner.logger.log_path)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "trajectories.jsonl"
        path.touch(exist_ok=False)
        return {"path": path, "groups": []}

    def update(self, context, eval_pbar, input_channel, batch_idx, batch):
        expected = {(task["env_id"], task["task_id"]) for task in batch["answer"]}
        if len(expected) != len(batch["answer"]):
            raise ValueError(
                "Expected unique EnvScaler tasks within each evaluation batch"
            )
        with context["path"].open("a", encoding="utf-8") as out:
            while expected:
                result = input_channel.get()
                task = result.extra_fields_group
                expected.remove((task["env_id"], task["task_id"]))
                if result.group_size != self.cfg.algorithm.group_size:
                    raise ValueError(
                        f"Unexpected rollout group size {result.group_size}"
                    )
                trajectories = trajectory_metrics(result)
                out.write(json.dumps({**task, "trajectories": trajectories}) + "\n")
                out.flush()
                context["groups"].append(trajectories)
                eval_pbar.update(len(trajectories))

    def post_process(self, context):
        summary = summarize(context["groups"], self.cfg.evaluation.pass_at_k)
        summary["config"] = OmegaConf.to_container(self.cfg, resolve=True)
        path = context["path"].with_name("summary.json")
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        get_logger().info("EnvScaler results saved to %s", path)


@hydra.main(version_base="1.1", config_path="config", config_name="eval_qwen3_8b")
@output_redirector
def main(cfg: DictConfig) -> None:
    """Evaluate the policy on a stratified sample of EnvScaler tasks."""
    if not all(1 <= k <= cfg.algorithm.group_size for k in cfg.evaluation.pass_at_k):
        raise ValueError(
            "Each pass_at_k value must be between 1 and algorithm.group_size"
        )
    cfg = validate_cfg(cfg)
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = ModelParallelEvalComponentPlacement(cfg, cluster)
    tokenizer = hf_tokenizer(cfg.rollout.model.model_path)
    _, dataset = create_reasoning_datasets(cfg, tokenizer)
    samples = sample_tasks(
        dataset, cfg.evaluation.tasks_per_environment, cfg.evaluation.sample_seed
    )
    rollout_strategy = placement.get_strategy("rollout")
    rollout_cls = get_rollout_backend_worker(cfg)
    rollout = rollout_cls.create_group(cfg, placement, weight_reload=None).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_strategy
    )
    agent_nodes = [p.cluster_node_rank for p in rollout_strategy.get_placement(cluster)]
    agents = EnvScalerAgentLoopWorker.create_group(cfg, placement).launch(
        cluster,
        name=cfg.agentloop.group_name,
        placement_strategy=NodePlacementStrategy(agent_nodes),
    )
    runner = EnvScalerEvalRunner(
        cfg, placement, samples, rollout, reward=None, agent_loop=agents
    )
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
