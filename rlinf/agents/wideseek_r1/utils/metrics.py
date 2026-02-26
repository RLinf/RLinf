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


def _safe_max(values: list[int]) -> int:
    return max(values) if values else 0


def _add_weighted_mean_metric(
    metrics: dict[str, float],
    key: str,
    numerator: float,
    denominator: float,
) -> None:
    metrics[key] = float(numerator) / float(denominator) if denominator > 0 else 0.0
    metrics[f"__stat/sum/{key}"] = float(numerator)
    metrics[f"__stat/count/{key}"] = float(denominator)


def _compute_tool_call_metrics(
    rollout_batch: dict,
    idx_to_traj: list[int],
    num_trajectories: int,
) -> dict:
    """Compute tool-call metrics for one question group."""
    if num_trajectories <= 0:
        return {}

    turn_subtask_counts = list(rollout_batch.get("turn_subtask_counts") or [])
    turn_search_counts = list(rollout_batch.get("turn_search_counts") or [])
    turn_access_counts = list(rollout_batch.get("turn_access_counts") or [])
    num_valid_planner_turns = int(rollout_batch.get("num_valid_planner_turns", 0) or 0)
    num_valid_worker_turns = int(rollout_batch.get("num_valid_worker_turns", 0) or 0)

    # idx_to_traj may come from training turns; use common prefix to avoid index mismatch.
    num_mapped_turns = min(
        len(idx_to_traj),
        len(turn_subtask_counts),
        len(turn_search_counts),
        len(turn_access_counts),
    )

    traj_subtask_counts = [0 for _ in range(num_trajectories)]
    traj_search_counts = [0 for _ in range(num_trajectories)]
    traj_access_counts = [0 for _ in range(num_trajectories)]
    for turn_idx in range(num_mapped_turns):
        traj_idx = idx_to_traj[turn_idx]
        if 0 <= traj_idx < num_trajectories:
            traj_subtask_counts[traj_idx] += turn_subtask_counts[turn_idx]
            traj_search_counts[traj_idx] += turn_search_counts[turn_idx]
            traj_access_counts[traj_idx] += turn_access_counts[turn_idx]

    traj_search_plus_access = [
        search + access
        for search, access in zip(traj_search_counts, traj_access_counts)
    ]
    turn_search_plus_access = [
        search + access
        for search, access in zip(turn_search_counts, turn_access_counts)
    ]

    metrics: dict[str, float] = {}
    _add_weighted_mean_metric(
        metrics,
        "wideseek_r1/traj/mean/subtask",
        sum(traj_subtask_counts),
        num_trajectories,
    )
    _add_weighted_mean_metric(
        metrics,
        "wideseek_r1/traj/mean/search",
        sum(traj_search_counts),
        num_trajectories,
    )
    _add_weighted_mean_metric(
        metrics,
        "wideseek_r1/traj/mean/access",
        sum(traj_access_counts),
        num_trajectories,
    )
    _add_weighted_mean_metric(
        metrics,
        "wideseek_r1/turn/mean/subtask",
        sum(turn_subtask_counts),
        num_valid_planner_turns,
    )
    _add_weighted_mean_metric(
        metrics,
        "wideseek_r1/turn/mean/search",
        sum(turn_search_counts),
        num_valid_worker_turns,
    )
    _add_weighted_mean_metric(
        metrics,
        "wideseek_r1/turn/mean/access",
        sum(turn_access_counts),
        num_valid_worker_turns,
    )

    return metrics


def _compute_mas_turn_metrics(rollout_batch: dict) -> dict:
    """Compute MAS turn metrics for one question group."""
    total_turn_list_metric = rollout_batch.get("total_turn_list_metric")
    if total_turn_list_metric is None:
        return {}

    sum_main_turns = 0
    sum_subagent_turns = 0
    sum_num_subagents = 0
    num_valid_trajs = 0
    for turn_list in total_turn_list_metric:
        if turn_list is None or len(turn_list) == 0:
            continue
        sum_main_turns += turn_list[-1]
        subagent_turns_list = turn_list[:-1]
        sum_subagent_turns += sum(subagent_turns_list)
        sum_num_subagents += len(subagent_turns_list)
        num_valid_trajs += 1

    metrics: dict[str, float] = {}
    _add_weighted_mean_metric(
        metrics,
        "wideseek_r1/avg_main_agent_turns_per_traj",
        sum_main_turns,
        num_valid_trajs,
    )
    _add_weighted_mean_metric(
        metrics,
        "wideseek_r1/avg_subagent_turns_per_traj",
        sum_subagent_turns,
        num_valid_trajs,
    )
    _add_weighted_mean_metric(
        metrics,
        "wideseek_r1/avg_num_subagents_per_traj",
        sum_num_subagents,
        num_valid_trajs,
    )
    return metrics


def _compute_final_answer_format_metrics(rollout_batch: dict) -> dict:
    """Compute trajectory-level final-answer-format metrics."""
    final_answer_format = rollout_batch.get("final_answer_format")
    if final_answer_format is None:
        return {}

    values = [float(v) for v in final_answer_format]
    metrics: dict[str, float] = {}
    _add_weighted_mean_metric(
        metrics,
        "wideseek_r1/traj/mean/final_answer_format",
        sum(values),
        len(values),
    )
    return metrics


def _compute_rollout_metrics(
    rollout_batch: dict,
    idx_to_traj: list[int],
    num_trajectories: int,
) -> dict:
    """Compute all wideseek agent metrics for one question group."""
    metrics = {}
    metrics.update(
        _compute_tool_call_metrics(
            rollout_batch=rollout_batch,
            idx_to_traj=idx_to_traj,
            num_trajectories=num_trajectories,
        )
    )
    metrics.update(_compute_mas_turn_metrics(rollout_batch=rollout_batch))
    metrics.update(_compute_final_answer_format_metrics(rollout_batch=rollout_batch))
    return metrics
