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
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from rlinf.scheduler import Channel, Cluster
from rlinf.utils.placement import ComponentPlacement, ModelParallelComponentPlacement
from rlinf.utils.utils import output_redirector
from rlinf.workers.env.android_reward_worker import AndroidRewardWorker
from rlinf.workers.env.m3a_worker import AndroidAgentWorker
from rlinf.workers.rollout.utils import get_rollout_backend_worker

CONFIG_DIR = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)


def find_project_root(start_path: Path) -> Path:
    """Infer project root by walking up from the given path.

    The project root is defined as the nearest ancestor directory that
    contains the ``RLinf`` package directory. This avoids relying on
    a fixed number of ``..`` traversals, which can be fragile when
    files are moved.

    Args:
        start_path: Path inside the repository (typically ``__file__``).

    Returns:
        The inferred project root path.
    """
    resolved = start_path.resolve()
    for parent in resolved.parents:
        if (parent / "RLinf").is_dir():
            return parent
    # Fallback to the directory two levels above the current file if no marker is found.
    return resolved.parent.parent


def load_resume_state(
    output_path: Path,
    dataset_size: int,
) -> tuple[list[dict[str, Any]], set[int], list[int]]:
    """Restore checkpoint state from existing result file.

    Args:
        output_path: Path to the JSON result file.
        dataset_size: Total number of tasks in the dataset.

    Returns:
        A tuple of (existing_results, finished_indices, task_indices_to_run).
    """
    existing_results: list[dict[str, Any]] = []
    finished_indices: set[int] = set()
    if output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as file:
                prev_data = json.load(file)
            existing_results = prev_data.get("tasks", []) or []
            for item in existing_results:
                idx = item.get("task_idx")
                if isinstance(idx, int):
                    finished_indices.add(idx)
            LOGGER.info(
                "Detected existing results file %s, %d tasks have been completed "
                "and will be skipped.",
                output_path,
                len(finished_indices),
            )
        except Exception:  # pylint: disable=broad-except
            LOGGER.warning(
                "Failed to read existing results file %s, starting from scratch.",
                output_path,
            )
            existing_results = []
            finished_indices = set()
    task_indices_to_run = [i for i in range(dataset_size) if i not in finished_indices]
    return existing_results, finished_indices, task_indices_to_run


def save_checkpoint(
    output_path: Path,
    all_results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    """Write the current results and summary to output_path."""
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            {"summary": summary, "tasks": all_results},
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


@hydra.main(
    version_base="1.1",
    config_path=str(CONFIG_DIR / "config"),
    config_name="qwen3vl-4b-eval",
)
@output_redirector
def main(cfg: DictConfig) -> None:
    """Entry point for AndroidWorld M3A evaluation."""
    LOGGER.info("\n%s", OmegaConf.to_yaml(cfg))

    # Derive project root from __file__ for consistent result directory with eval.sh.
    project_root = find_project_root(Path(__file__))
    LOGGER.info("project_root: %s", project_root)
    output_dir = project_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / "eval_results.json"  # Fixed path for checkpointing

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ComponentPlacement(cfg, cluster)

    reward_group = AndroidRewardWorker.create_group(cfg).launch(
        cluster=cluster,
        placement_strategy=component_placement.get_strategy("reward_worker"),
        name="RewardWorkerGroup",
    )
    reward_group.init_worker().wait()

    mpc_placement = ModelParallelComponentPlacement(cfg, cluster)
    rollout_worker_cls = get_rollout_backend_worker(cfg)
    rollout_group = rollout_worker_cls.create_group(
        cfg,
        mpc_placement,
        weight_reload=None,
    ).launch(
        cluster=cluster,
        name=cfg.rollout.get("group_name", "RolloutGroup"),
        placement_strategy=mpc_placement.get_strategy("rollout"),
    )
    rollout_group.init_worker().wait()

    agent_group = AndroidAgentWorker.create_group(cfg).launch(
        cluster=cluster,
        placement_strategy=component_placement.get_strategy("agent_worker"),
        name="AndroidAgentWorkerGroup",
    )
    channel_a2l = Channel.create("a2l")
    channel_l2a = Channel.create("l2a")
    agent_group.init_with_channels(channel_a2l, channel_l2a).wait()
    agent_group.init_worker().wait()

    rollout_group.vl_generate_serverless(channel_a2l, channel_l2a)

    dataset_size_result = agent_group.execute_on(0).get_dataset_size()
    dataset_size = dataset_size_result.wait()
    if isinstance(dataset_size, list):
        dataset_size = dataset_size[0]
    LOGGER.info(
        "====================== Dataset has %d tasks ======================",
        dataset_size,
    )

    existing_results, _, task_indices_to_run = load_resume_state(
        output_path,
        dataset_size,
    )
    all_results: list[dict[str, Any]] = list(existing_results)

    if not task_indices_to_run:
        LOGGER.info("All tasks have been completed, no need to run again.")
        return

    for task_idx in task_indices_to_run:
        # For each task, first start the reward receiver, then run the agent's
        # send/recv, otherwise the agent send will deadlock if no one receives it.
        reward_handle = reward_group.execute_on(0).compute_reward(
            agent_worker_group_name="AndroidAgentWorkerGroup",
        )
        reward = (
            agent_group.execute_on(0)
            .process_task(
                task_idx=task_idx,
                reward_worker_group_name="RewardWorkerGroup",
            )
            .wait()
        )
        reward_handle.wait()
        all_results.append(
            {
                "task_idx": task_idx,
                "reward": reward,
            }
        )
        total_run = len(all_results)
        LOGGER.info("reward: %s", [result["reward"] for result in all_results])
        successful = sum(1 for result in all_results if result.get("reward", 0)[0] > 0)
        acc = successful / total_run if total_run else 0.0
        save_checkpoint(
            output_path,
            all_results,
            {
                "timestamp": timestamp,
                "total_tasks": total_run,
                "successful_tasks": successful,
                "accuracy": round(acc, 4),
            },
        )

    total = len(all_results)
    successful = sum(1 for result in all_results if result.get("reward", 0)[0] > 0)
    accuracy = successful / total if total else 0.0
    save_checkpoint(
        output_path,
        all_results,
        {
            "timestamp": timestamp,
            "total_tasks": total,
            "successful_tasks": successful,
            "accuracy": round(accuracy, 4),
        },
    )
    LOGGER.info("Results have been saved to %s", output_path)
    LOGGER.info(
        "Total tasks: %d, successful: %d, accuracy: %.2f%%",
        total,
        successful,
        accuracy * 100.0,
    )


if __name__ == "__main__":
    main()
