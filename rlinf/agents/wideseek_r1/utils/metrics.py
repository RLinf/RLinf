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

from typing import Optional

import torch
import torch.distributed

from torch.distributed import ProcessGroup


def _compute_tool_call_metrics(
    rollout_batch: dict,
    idx_to_traj: list[int],
    num_trajectories: int,
) -> dict:
    """Compute tool call metrics at both turn and trajectory levels.

    Returns dict with keys like "trajectory/mean/subtask", "turn/mean/search", etc.
    """

    # Load turn-level counts
    turn_subtask_counts = rollout_batch["turn_subtask_counts"]
    turn_search_counts = rollout_batch["turn_search_counts"]
    turn_access_counts = rollout_batch["turn_access_counts"]

    # Valid turn counts
    num_valid_planner_turns = rollout_batch["num_valid_planner_turns"]
    num_valid_worker_turns = rollout_batch["num_valid_worker_turns"]

    # Aggregate to trajectory level
    traj_subtask_counts = [0 for _ in range(num_trajectories)]
    traj_search_counts = [0 for _ in range(num_trajectories)]
    traj_access_counts = [0 for _ in range(num_trajectories)]

    for turn_idx, traj_idx in enumerate(idx_to_traj):
        traj_subtask_counts[traj_idx] += turn_subtask_counts[turn_idx]
        traj_search_counts[traj_idx] += turn_search_counts[turn_idx]
        traj_access_counts[traj_idx] += turn_access_counts[turn_idx]

    # Compute local sums and maxs
    traj_search_plus_access = [i + j for i, j in zip(traj_search_counts, traj_access_counts)]
    turn_search_plus_access = [i + j for i, j in zip(turn_search_counts, turn_access_counts)]

    # Trajectory sums
    mean_traj_subtask = sum(traj_subtask_counts) / num_trajectories
    mean_traj_search = sum(traj_search_counts) / num_trajectories
    mean_traj_access = sum(traj_access_counts) / num_trajectories
    mean_traj_search_plus_access = sum(traj_search_plus_access) / num_trajectories
    # Turn sums
    mean_turn_subtask = 0
    mean_turn_search = 0
    mean_turn_access = 0
    mean_turn_search_plus_access = 0
    if num_valid_planner_turns > 0:
        mean_turn_subtask = sum(turn_subtask_counts) / num_valid_planner_turns
    if num_valid_worker_turns > 0:
        mean_turn_search = sum(turn_search_counts) / num_valid_worker_turns
        mean_turn_access = sum(turn_access_counts) / num_valid_worker_turns
        mean_turn_search_plus_access = sum(turn_search_plus_access) / num_valid_worker_turns
    # Trajectory maxes
    max_traj_subtask = max(traj_subtask_counts)
    max_traj_search = max(traj_search_counts)
    max_traj_access = max(traj_access_counts)
    max_traj_search_plus_access = max(traj_search_plus_access)
    # Turn maxes
    max_turn_subtask = max(turn_subtask_counts)
    max_turn_search = max(turn_search_counts)
    max_turn_access = max(turn_access_counts)
    max_turn_search_plus_access = max(turn_search_plus_access)

    # Build result dict
    tool_call_metrics = {
        # Trajectory-level means
        "agent/traj/mean/subtask": mean_traj_subtask,
        "agent/traj/mean/search": mean_traj_search,
        "agent/traj/mean/access": mean_traj_access,
        "agent/traj/mean/search+access": mean_traj_search_plus_access,
        # Trajectory-level maxs
        "agent/traj/max/subtask": max_traj_subtask,
        "agent/traj/max/search": max_traj_search,
        "agent/traj/max/access": max_traj_access,
        "agent/traj/max/search+access": max_traj_search_plus_access,
        # Turn-level means
        "agent/turn/mean/subtask": mean_turn_subtask,
        "agent/turn/mean/search": mean_turn_search,
        "agent/turn/mean/access": mean_turn_access,
        "agent/turn/mean/search+access": mean_turn_search_plus_access,
        # Turn-level maxs
        "agent/turn/max/subtask": max_turn_subtask,
        "agent/turn/max/search": max_turn_search,
        "agent/turn/max/access": max_turn_access,
        "agent/turn/max/search+access": max_turn_search_plus_access,
    }

    # Cleanup
    for key in [
        "turn_subtask_counts",
        "turn_search_counts",
        "turn_access_counts",
        "num_valid_planner_turns",
        "num_valid_worker_turns",
    ]:
        rollout_batch.pop(key, None)

    return tool_call_metrics


def _compute_eval_metrics(
    rollout_batch: dict,
    device: torch.device,
    data_parallel_group: Optional[ProcessGroup],
) -> dict:
    """Compute evaluation metrics (EM, F1, LLM, format_score) at trajectory level.

    Handles both simple metrics (single value) and nested metrics (dict with sub-metrics).
    Returns dict with keys like "eval/EM", "eval/F1/score", "eval/format_score/call_search_reward", etc.
    """
    eval_metrics_list = rollout_batch["eval_metrics"]

    # Collect all metric types (CRITICAL: sort to ensure consistent order across DPs)
    all_metric_types = set()
    for eval_metric in eval_metrics_list:
        if eval_metric is not None and isinstance(eval_metric, dict):
            all_metric_types.update(eval_metric.keys())
    all_metric_types = sorted(all_metric_types)

    # Collect all (metric_type, sub_metric_name) pairs and their local sums/counts
    # We batch all reductions together for efficiency
    metric_keys = []  # List of (metric_type, sub_metric_name or None)
    local_sums = []
    local_counts = []

    for metric_type in all_metric_types:
        # Find sample to determine structure
        sample_metric = None
        for eval_metric in eval_metrics_list:
            if eval_metric is not None and metric_type in eval_metric:
                sample_metric = eval_metric[metric_type]
                break

        if sample_metric is None:
            continue

        if isinstance(sample_metric, dict):
            # Nested metric (markdown mode or format_score)
            sub_metric_names = sorted(sample_metric.keys())  # Sort for consistency
            for sub_metric_name in sub_metric_names:
                values = []
                for eval_metric in eval_metrics_list:
                    if eval_metric is not None and metric_type in eval_metric:
                        val = eval_metric[metric_type].get(sub_metric_name, 0.0)
                        if isinstance(val, (int, float)):
                            values.append(val)

                if values:
                    metric_keys.append((metric_type, sub_metric_name))
                    local_sums.append(sum(values))
                    local_counts.append(len(values))

        elif isinstance(sample_metric, (int, float)):
            # Simple metric (non-markdown mode)
            values = []
            for eval_metric in eval_metrics_list:
                if eval_metric is not None and metric_type in eval_metric:
                    val = eval_metric[metric_type]
                    if val is not None and isinstance(val, (int, float)):
                        values.append(val)

            if values:
                metric_keys.append((metric_type, None))
                local_sums.append(sum(values))
                local_counts.append(len(values))

    if not metric_keys:
        rollout_batch.pop("eval_metrics", None)
        return {}

    # Batch all-reduce: [sum1, count1, sum2, count2, ...]
    reduce_data = []
    for s, c in zip(local_sums, local_counts):
        reduce_data.extend([s, c])

    reduce_tensor = torch.as_tensor(reduce_data, device=device, dtype=torch.float32)
    torch.distributed.all_reduce(
        reduce_tensor, torch.distributed.ReduceOp.SUM, group=data_parallel_group
    )
    reduce_list = reduce_tensor.tolist()

    # Build result dict
    eval_metrics_dict = {}
    for i, (metric_type, sub_metric_name) in enumerate(metric_keys):
        global_sum = reduce_list[i * 2]
        global_count = reduce_list[i * 2 + 1]
        avg_value = global_sum / global_count if global_count > 0 else 0.0

        if sub_metric_name is not None:
            eval_metrics_dict[f"eval/{metric_type}/{sub_metric_name}"] = avg_value
        else:
            eval_metrics_dict[f"eval/{metric_type}"] = avg_value

    rollout_batch.pop("eval_metrics", None)
    return eval_metrics_dict


def _compute_mas_turn_metrics(
    rollout_batch: dict,
    device: torch.device,
    data_parallel_group: Optional[ProcessGroup],
) -> dict:
    """Compute MAS (multi-agent system) turn metrics at trajectory level.

    total_turn_list format: [subagent1_turns, subagent2_turns, ..., main_agent_turns]
    Returns dict with keys like "mas/avg_main_agent_turns_per_traj", etc.
    """
    if (
        "total_turn_list_metric" not in rollout_batch
        or rollout_batch["total_turn_list_metric"] is None
    ):
        return {}

    total_turn_list_metric = rollout_batch["total_turn_list_metric"]

    # Compute local sums
    local_sum_main_turns = 0
    local_sum_subagent_turns = 0
    local_sum_num_subagents = 0
    local_num_valid_trajs = 0

    for turn_list in total_turn_list_metric:
        if turn_list is not None and len(turn_list) > 0:
            local_sum_main_turns += turn_list[-1]  # Last element is main agent turns
            subagent_turns_list = turn_list[:-1]  # All except last are subagent turns
            local_sum_subagent_turns += sum(subagent_turns_list)
            local_sum_num_subagents += len(subagent_turns_list)
            local_num_valid_trajs += 1

    # All-reduce
    reduce_tensor = torch.as_tensor(
        [
            local_sum_main_turns,
            local_sum_subagent_turns,
            local_sum_num_subagents,
            local_num_valid_trajs,
        ],
        device=device,
        dtype=torch.long,
    )
    torch.distributed.all_reduce(
        reduce_tensor, torch.distributed.ReduceOp.SUM, group=data_parallel_group
    )
    (
        global_sum_main_turns,
        global_sum_subagent_turns,
        global_sum_num_subagents,
        global_num_valid_trajs,
    ) = reduce_tensor.tolist()

    mas_turn_metrics = {}
    if global_num_valid_trajs > 0:
        mas_turn_metrics["mas/avg_main_agent_turns_per_traj"] = (
            global_sum_main_turns / global_num_valid_trajs
        )
        mas_turn_metrics["mas/avg_subagent_turns_per_traj"] = (
            global_sum_subagent_turns / global_num_valid_trajs
        )
        mas_turn_metrics["mas/avg_turns_per_traj"] = (
            global_sum_main_turns + global_sum_subagent_turns
        ) / global_num_valid_trajs
        mas_turn_metrics["mas/avg_num_subagents_per_traj"] = (
            global_sum_num_subagents / global_num_valid_trajs
        )

    rollout_batch.pop("total_turn_list_metric", None)
    return mas_turn_metrics
