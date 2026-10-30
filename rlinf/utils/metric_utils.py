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

import math
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed

if TYPE_CHECKING:
    from rlinf.data.schema.embodied_types import Trajectory


@dataclass
class VersionedTrajectoryRecord:
    """Per-trajectory weight-version record for staleness statistics.

    One record per trajectory in a training step; groups are identified by
    `group_index` (unique within the step after the caller offsets it by rank).
    """

    group_index: int
    earliest_version: int
    completion_version: int
    generated_tokens: int
    call_count: int
    cross_version_calls: int
    reward: float
    token_weighted_version_start_sum: int


def compute_staleness_metrics(
    records: Sequence[VersionedTrajectoryRecord], version: int
) -> dict[str, float]:
    """Compute weight-staleness and scheduling metrics for one training step.

    `version` is the actor weight version at step start. Group earliest
    version is the minimum earliest_version in the group, group completion
    version the maximum completion_version; group staleness is
    `version - group earliest version` and decomposes into
    `version_span + queue_steps` (completion - earliest, version - completion).
    """
    metrics: dict[str, float] = {}
    if not records:
        return metrics

    group_earliest: dict[int, int] = {}
    group_completion: dict[int, int] = {}
    group_rewards: dict[int, list[float]] = {}
    for record in records:
        group_earliest[record.group_index] = min(
            group_earliest.get(record.group_index, record.earliest_version),
            record.earliest_version,
        )
        group_completion[record.group_index] = max(
            group_completion.get(record.group_index, record.completion_version),
            record.completion_version,
        )
        group_rewards.setdefault(record.group_index, []).append(record.reward)

    num_groups = len(group_earliest)
    group_staleness = {
        group_index: version - earliest
        for group_index, earliest in group_earliest.items()
    }
    version_spans = [
        group_completion[group_index] - group_earliest[group_index]
        for group_index in group_earliest
    ]
    queue_steps = [
        version - group_completion[group_index] for group_index in group_earliest
    ]

    total_generated_tokens = sum(record.generated_tokens for record in records)
    # Keys follow the embodied `data_staleness_{diff}/ratio` naming; one
    # entry per occurring staleness value, counted in groups not records.
    staleness_group_counts: dict[int, int] = {}
    staleness_tokens: dict[int, int] = {}
    for staleness in group_staleness.values():
        staleness_group_counts[staleness] = staleness_group_counts.get(staleness, 0) + 1
    for record in records:
        staleness_tokens[group_staleness[record.group_index]] = staleness_tokens.get(
            group_staleness[record.group_index], 0
        ) + (record.generated_tokens)
    for k, group_count in sorted(staleness_group_counts.items()):
        metrics[f"data_staleness_{k}/ratio"] = group_count / num_groups
    if total_generated_tokens > 0:
        for k, token_count in sorted(staleness_tokens.items()):
            metrics[f"data_staleness_{k}/token_ratio"] = token_count / (
                total_generated_tokens
            )

    metrics["data_staleness/mean"] = sum(group_staleness.values()) / num_groups
    metrics["data_staleness/max"] = max(group_staleness.values())
    # Token-weighted average version of the calls that produced the data,
    # for contrast with the earliest-version based staleness above; with no
    # generated tokens it is undefined and left out, like the token ratios.
    if total_generated_tokens > 0:
        token_weighted_version = (
            sum(record.token_weighted_version_start_sum for record in records)
            / total_generated_tokens
        )
        metrics["data_staleness/token_mean"] = version - token_weighted_version
    metrics["version_span/mean"] = sum(version_spans) / num_groups
    metrics["version_span/max"] = max(version_spans)
    metrics["queue_steps/mean"] = sum(queue_steps) / num_groups
    metrics["queue_steps/max"] = max(queue_steps)

    total_calls = sum(record.call_count for record in records)
    cross_version_calls = sum(record.cross_version_calls for record in records)
    metrics["cross_version_call_ratio"] = (
        cross_version_calls / total_calls if total_calls else 0.0
    )

    if len(records) >= 4:
        # Quartiles over trajectories by generated tokens (q1 shortest).
        sorted_records = sorted(records, key=lambda record: record.generated_tokens)
        base, remainder = divmod(len(sorted_records), 4)
        quartile_sizes = [base + (offset < remainder) for offset in range(4)]
        quartile_start = 0
        for quartile, size in enumerate(quartile_sizes, start=1):
            quartile_records = sorted_records[quartile_start : quartile_start + size]
            metrics[f"data_staleness/length_q{quartile}"] = sum(
                version - record.earliest_version for record in quartile_records
            ) / max(len(quartile_records), 1)
            quartile_start += size

    metrics["equal_reward_group_ratio"] = (
        sum(1 for rewards in group_rewards.values() if len(set(rewards)) == 1)
        / num_groups
    )
    metrics["zero_reward_group_ratio"] = (
        sum(1 for rewards in group_rewards.values() if all(r == 0 for r in rewards))
        / num_groups
    )

    return metrics


def mean_bool_tensor_rate(
    tensors: Sequence[torch.Tensor | None],
    *,
    sum_key: str,
    count_key: str,
    reducer: Callable[[dict[str, float]], dict[str, float]] | None = None,
) -> float | None:
    """Mean of flattened bool-like tensors, optionally reduced across ranks."""
    shards = [
        tensor.detach().reshape(-1).to(torch.float32)
        for tensor in tensors
        if isinstance(tensor, torch.Tensor) and tensor.numel() > 0
    ]
    if not shards:
        return None

    local_values = torch.cat(shards, dim=0)
    reduced = {
        sum_key: float(local_values.sum().item()),
        count_key: float(local_values.numel()),
    }
    if reducer is not None:
        reduced = reducer(reduced)
    if reduced[count_key] <= 0:
        return 0.0
    return reduced[sum_key] / reduced[count_key]


def mean_bool_tensor_rate_from_trajectories(
    trajectories: Sequence["Trajectory"],
    tensor_getter: Callable[["Trajectory"], torch.Tensor | None],
    *,
    sum_key: str,
    count_key: str,
    reducer: Callable[[dict[str, float]], dict[str, float]] | None = None,
) -> float | None:
    return mean_bool_tensor_rate(
        [tensor_getter(trajectory) for trajectory in trajectories],
        sum_key=sum_key,
        count_key=count_key,
        reducer=reducer,
    )


def trajectory_forward_input_tensor(
    trajectory: "Trajectory", key: str
) -> torch.Tensor | None:
    forward_inputs = trajectory.forward_inputs
    if not isinstance(forward_inputs, dict):
        return None
    value = forward_inputs.get(key)
    return value if isinstance(value, torch.Tensor) else None


def trajectory_has_bool_tensor(tensor: torch.Tensor | None) -> bool:
    return bool(
        isinstance(tensor, torch.Tensor) and tensor.detach().to(torch.bool).any()
    )


def collect_trajectory_replay_metrics(
    trajectories: Sequence["Trajectory"],
    *,
    reducer: Callable[[dict[str, float]], dict[str, float]] | None = None,
) -> dict[str, float]:
    """Replay-route diagnostics aggregated from received trajectories."""
    metrics: dict[str, float] = {}
    rate_specs = (
        (
            "replay/record_transition_rate",
            lambda trajectory: trajectory_forward_input_tensor(
                trajectory, "record_transition"
            ),
            "record_transition_sum",
            "record_transition_count",
        ),
        (
            "replay/actor_switch_rate",
            lambda trajectory: trajectory_forward_input_tensor(
                trajectory, "actor_switch"
            ),
            "actor_switch_sum",
            "actor_switch_count",
        ),
        (
            "replay/intervention_requested_rate",
            lambda trajectory: trajectory_forward_input_tensor(
                trajectory, "intervention_requested"
            ),
            "intervention_requested_sum",
            "intervention_requested_count",
        ),
        (
            "replay/intervention_rate",
            lambda trajectory: trajectory.intervene_flags,
            "intervention_sum",
            "intervention_count",
        ),
    )
    for metric_key, tensor_getter, sum_key, count_key in rate_specs:
        rate = mean_bool_tensor_rate_from_trajectories(
            trajectories,
            tensor_getter,
            sum_key=sum_key,
            count_key=count_key,
            reducer=reducer,
        )
        if rate is not None:
            metrics[metric_key] = rate
    return metrics


METRIC_SUM_PREFIX = "__sum__/"

CRITIC_EXPLAINED_VARIANCE_KEY = "critic/explained_variance"
CRITIC_EXPLAINED_VARIANCE_STATS_PREFIX = (
    f"{METRIC_SUM_PREFIX}_critic_explained_variance/"
)
CRITIC_EXPLAINED_VARIANCE_COUNT_KEY = f"{CRITIC_EXPLAINED_VARIANCE_STATS_PREFIX}count"
CRITIC_EXPLAINED_VARIANCE_RETURNS_SUM_KEY = (
    f"{CRITIC_EXPLAINED_VARIANCE_STATS_PREFIX}returns_sum"
)
CRITIC_EXPLAINED_VARIANCE_RETURNS_SQ_SUM_KEY = (
    f"{CRITIC_EXPLAINED_VARIANCE_STATS_PREFIX}returns_sq_sum"
)
CRITIC_EXPLAINED_VARIANCE_ERRORS_SUM_KEY = (
    f"{CRITIC_EXPLAINED_VARIANCE_STATS_PREFIX}errors_sum"
)
CRITIC_EXPLAINED_VARIANCE_ERRORS_SQ_SUM_KEY = (
    f"{CRITIC_EXPLAINED_VARIANCE_STATS_PREFIX}errors_sq_sum"
)
CRITIC_EXPLAINED_VARIANCE_STAT_KEYS = (
    CRITIC_EXPLAINED_VARIANCE_COUNT_KEY,
    CRITIC_EXPLAINED_VARIANCE_RETURNS_SUM_KEY,
    CRITIC_EXPLAINED_VARIANCE_RETURNS_SQ_SUM_KEY,
    CRITIC_EXPLAINED_VARIANCE_ERRORS_SUM_KEY,
    CRITIC_EXPLAINED_VARIANCE_ERRORS_SQ_SUM_KEY,
)


INTERACT_DELAY_METRIC_KEYS = {"interact_delay"}


def is_interact_delay_metric_key(key: str) -> bool:
    return any(
        key == metric_key or key.endswith(f"/{metric_key}")
        for metric_key in INTERACT_DELAY_METRIC_KEYS
    )


def _build_interact_delay_stat_key(key: str, stat_name: str) -> str:
    matched_metric_key = next(
        metric_key
        for metric_key in INTERACT_DELAY_METRIC_KEYS
        if key == metric_key or key.endswith(f"/{metric_key}")
    )
    metric_prefix = key[: -len(matched_metric_key)]
    return f"{metric_prefix}{stat_name}"


def compute_delay_stats(key: str, stacked: "torch.Tensor") -> dict:
    """Compute average, median, max, min delay stats from delay samples."""
    if stacked.numel() > 0:
        return {
            _build_interact_delay_stat_key(key, "average_delay"): stacked.mean()
            .detach()
            .cpu()
            .numpy(),
            _build_interact_delay_stat_key(key, "median_delay"): torch.quantile(
                stacked, 0.5
            )
            .detach()
            .cpu()
            .numpy(),
            _build_interact_delay_stat_key(key, "max_delay"): stacked.max()
            .detach()
            .cpu()
            .numpy(),
            _build_interact_delay_stat_key(key, "min_delay"): stacked.min()
            .detach()
            .cpu()
            .numpy(),
        }
    return {
        _build_interact_delay_stat_key(key, "average_delay"): np.asarray(
            0.0, dtype=np.float64
        ),
        _build_interact_delay_stat_key(key, "median_delay"): np.asarray(
            0.0, dtype=np.float64
        ),
        _build_interact_delay_stat_key(key, "max_delay"): np.asarray(
            0.0, dtype=np.float64
        ),
        _build_interact_delay_stat_key(key, "min_delay"): np.asarray(
            0.0, dtype=np.float64
        ),
    }


def compute_split_num(num, split_num):
    return math.lcm(num, split_num) // split_num


def compute_group_success_metrics(
    rewards: torch.Tensor,
    *,
    group_size: int,
    loss_mask: torch.Tensor | None = None,
    success_threshold: float = 0.5,
    rewards_lower_bound: float | None = None,
    rewards_upper_bound: float | None = None,
) -> dict[str, float]:
    """Summarize per-group binary outcomes before GRPO reward filtering."""
    if rewards.ndim < 2:
        raise ValueError(
            f"rewards must have at least two dimensions, got {rewards.shape}"
        )
    batch_size = rewards.shape[1]
    if batch_size % group_size != 0:
        raise ValueError(
            f"reward batch size {batch_size} is not divisible by group_size {group_size}"
        )

    masked_rewards = rewards
    if loss_mask is not None:
        mask = torch.broadcast_to(
            loss_mask.to(device=rewards.device, dtype=torch.bool), rewards.shape
        )
        masked_rewards = torch.where(mask, rewards, torch.zeros_like(rewards))

    trajectory_rewards = masked_rewards.transpose(0, 1).reshape(batch_size, -1).sum(-1)
    grouped_rewards = trajectory_rewards.reshape(-1, group_size)
    successes_per_group = (grouped_rewards >= success_threshold).sum(dim=-1)
    histogram = torch.bincount(successes_per_group, minlength=group_size + 1).float()
    histogram /= max(successes_per_group.numel(), 1)

    metrics = {
        f"group_success_{success_count}_of_{group_size}_fraction": histogram[
            success_count
        ].item()
        for success_count in range(group_size + 1)
    }
    metrics["group_mixed_fraction"] = (
        1.0 - histogram[0] - histogram[group_size]
    ).item()
    if rewards_lower_bound is not None and rewards_upper_bound is not None:
        mean_reward_per_group = grouped_rewards.mean(dim=-1)
        keep_mask = (mean_reward_per_group >= rewards_lower_bound) & (
            mean_reward_per_group <= rewards_upper_bound
        )
        metrics["group_filter_keep_fraction"] = keep_mask.float().mean().item()
    return metrics


def compute_critic_explained_variance_stats(
    returns: torch.Tensor,
    values: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute sufficient statistics for critic explained variance."""
    returns = returns.detach().float()
    values = values.detach().float()
    if loss_mask is not None:
        mask = loss_mask.to(device=returns.device, dtype=torch.bool)
        if mask.shape != returns.shape:
            mask = torch.broadcast_to(mask, returns.shape)
        returns = returns[mask]
        values = values[mask]
    else:
        returns = returns.reshape(-1)
        values = values.reshape(-1)

    errors = returns - values
    count = torch.tensor(float(returns.numel()), device=returns.device)
    return {
        CRITIC_EXPLAINED_VARIANCE_COUNT_KEY: count,
        CRITIC_EXPLAINED_VARIANCE_RETURNS_SUM_KEY: returns.sum(),
        CRITIC_EXPLAINED_VARIANCE_RETURNS_SQ_SUM_KEY: (returns * returns).sum(),
        CRITIC_EXPLAINED_VARIANCE_ERRORS_SUM_KEY: errors.sum(),
        CRITIC_EXPLAINED_VARIANCE_ERRORS_SQ_SUM_KEY: (errors * errors).sum(),
    }


def compute_critic_explained_variance_from_stats(
    stats: dict[str, float | torch.Tensor],
) -> torch.Tensor:
    """Compute critic explained variance from summed sufficient statistics."""
    tensor_value = next(
        (v for v in stats.values() if isinstance(v, torch.Tensor)), None
    )
    device = tensor_value.device if tensor_value is not None else torch.device("cpu")

    def as_tensor(key: str) -> torch.Tensor:
        return torch.as_tensor(stats[key], dtype=torch.float32, device=device)

    count = as_tensor(CRITIC_EXPLAINED_VARIANCE_COUNT_KEY)
    returns_sum = as_tensor(CRITIC_EXPLAINED_VARIANCE_RETURNS_SUM_KEY)
    returns_sq_sum = as_tensor(CRITIC_EXPLAINED_VARIANCE_RETURNS_SQ_SUM_KEY)
    errors_sum = as_tensor(CRITIC_EXPLAINED_VARIANCE_ERRORS_SUM_KEY)
    errors_sq_sum = as_tensor(CRITIC_EXPLAINED_VARIANCE_ERRORS_SQ_SUM_KEY)

    nan = torch.tensor(float("nan"), device=device)
    if count < 2:
        return nan

    returns_centered_sq_sum = returns_sq_sum - returns_sum * returns_sum / count
    if torch.isnan(returns_centered_sq_sum) or returns_centered_sq_sum == 0:
        return nan

    errors_centered_sq_sum = errors_sq_sum - errors_sum * errors_sum / count
    if torch.isnan(errors_centered_sq_sum):
        return nan
    return 1 - errors_centered_sq_sum / returns_centered_sq_sum


def pop_critic_explained_variance_stats(
    metrics: dict[str, object],
) -> dict[str, torch.Tensor]:
    """Pop hidden critic explained-variance stats and sum list values."""

    def sum_metric_values(value: object) -> torch.Tensor:
        if isinstance(value, list):
            if not value:
                return torch.tensor(0.0)
            tensors = [
                item.detach()
                if isinstance(item, torch.Tensor)
                else torch.as_tensor(item)
                for item in value
            ]
            return torch.stack([tensor.float() for tensor in tensors]).sum()
        if isinstance(value, torch.Tensor):
            return value.detach().float()
        return torch.as_tensor(value, dtype=torch.float32)

    stats = {}
    for key in CRITIC_EXPLAINED_VARIANCE_STAT_KEYS:
        if key in metrics:
            stats[key] = sum_metric_values(metrics.pop(key))
    if stats:
        metrics.pop(CRITIC_EXPLAINED_VARIANCE_KEY, None)
    return stats


def _normalize_metric_shard(shard: object) -> torch.Tensor:
    """One rank's metric -> 1D float tensor on CPU."""
    if shard is None:
        return torch.tensor([], dtype=torch.float32)
    if isinstance(shard, torch.Tensor):
        return shard.detach().cpu().reshape(-1).float()
    if isinstance(shard, list):
        if not shard:
            return torch.tensor([], dtype=torch.float32)
        return torch.cat([x.detach().cpu().reshape(-1).float() for x in shard], dim=0)
    return torch.as_tensor(shard, dtype=torch.float32).cpu().reshape(-1)


def count_trajectories(metrics_dict):
    """
    Count the total number of trajectories from metrics dictionary.

    Args:
        metrics_dict: Dictionary of metrics where each value is a tensor after concatenation.
                     Each tensor's first dimension represents the number of trajectories.

    Returns:
        int: Total number of trajectories. If metrics_dict is empty, returns 0.
    """
    if not metrics_dict:
        return 0

    # Use a trajectory-shaped metric tensor to get the trajectory count.
    # Some metrics, such as interact delay samples, are auxiliary distributions and
    # should not define the trajectory count.
    valid_metric_keys = [
        key for key in metrics_dict.keys() if not is_interact_delay_metric_key(key)
    ]
    if not valid_metric_keys:
        return 0

    first_key = valid_metric_keys[0]
    first_tensor = metrics_dict[first_key]

    if isinstance(first_tensor, torch.Tensor):
        return first_tensor.shape[0]
    elif isinstance(first_tensor, list):
        # If it's a list of tensors, sum up all trajectory counts
        return sum(
            t.shape[0] if isinstance(t, torch.Tensor) else len(t) for t in first_tensor
        )
    else:
        raise TypeError(f"Unsupported tensor type: {type(first_tensor)}")


def compute_evaluate_metrics(eval_metrics_list):
    """
    List of evaluate metrics, list length stands for rollout process

    Returns:
        dict: Aggregated metrics with mean values and trajectory count
    """
    if not eval_metrics_list:
        return {}

    all_eval_metrics = {}
    env_info_keys: set[str] = set()
    for eval_metrics in eval_metrics_list:
        env_info_keys.update(eval_metrics.keys())

    # Count trajectories from each process
    trajectory_counts = []
    for eval_metrics in eval_metrics_list:
        count = count_trajectories(eval_metrics)
        trajectory_counts.append(count)

    for env_info_key in env_info_keys:
        metric = [
            eval_metrics[env_info_key]
            for eval_metrics in eval_metrics_list
            if env_info_key in eval_metrics
        ]
        if metric:
            all_eval_metrics[env_info_key] = metric

    aggregated_eval_metrics = {}
    for key in all_eval_metrics:
        shards = [_normalize_metric_shard(s) for s in all_eval_metrics[key]]
        stacked = torch.concat(shards).float()
        if is_interact_delay_metric_key(key):
            aggregated_eval_metrics.update(compute_delay_stats(key, stacked))
            continue

        aggregated_eval_metrics[key] = (
            stacked.mean().detach().cpu().numpy()
            if stacked.numel() > 0
            else np.asarray(0.0, dtype=np.float64)
        )

    # Add total trajectory count to metrics
    aggregated_eval_metrics["num_trajectories"] = sum(trajectory_counts)

    return aggregated_eval_metrics


def compute_rollout_metrics(data_buffer: dict) -> dict:
    rollout_metrics = {}
    loss_mask = data_buffer.get("loss_mask", None)

    def reduce_metrics(values: torch.Tensor) -> tuple[float, float, float]:
        from rlinf.scheduler.worker.worker import Worker

        device = Worker.torch_platform.current_device()

        if values.numel() == 0:
            count = torch.tensor(0.0, device=device, dtype=torch.float32)
            values_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
            min_value = float("inf")
            max_value = float("-inf")
        else:
            values = values.to(device)
            count = torch.tensor(
                values.numel(), device=values.device, dtype=torch.float32
            )
            values_sum = values.to(dtype=torch.float32).sum()
            max_value = torch.max(values).detach().item()
            min_value = torch.min(values).detach().item()

        reduce_sum_count = torch.stack([values_sum, count])
        reduce_min_max = torch.as_tensor(
            [-min_value, max_value],
            device=device,
            dtype=torch.float32,
        )
        torch.distributed.all_reduce(
            reduce_sum_count, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(reduce_min_max, op=torch.distributed.ReduceOp.MAX)
        reduced_sum, reduced_count = reduce_sum_count.tolist()
        reduced_min, reduced_max = reduce_min_max.tolist()

        if reduced_count <= 0:
            return float("nan"), float("nan"), float("nan")
        return reduced_sum / reduced_count, -reduced_min, reduced_max

    def valid_values(values: torch.Tensor) -> torch.Tensor:
        if loss_mask is None:
            return values.reshape(-1)
        mask = loss_mask.to(device=values.device, dtype=torch.bool)
        if mask.ndim == values.ndim - 1:
            mask = mask.unsqueeze(-1)
        if mask.shape != values.shape:
            mask = torch.broadcast_to(mask, values.shape)
        return values[mask]

    if loss_mask is not None:
        # Every metric below is taken over the mask, and filter_rewards can mask
        # out whole groups, so they can read as clean simply because little
        # survived. Report the share of the batch that did.
        mean_coverage, _, _ = reduce_metrics(loss_mask.float().reshape(-1))
        rollout_metrics["loss_mask_fraction"] = mean_coverage

    if "rewards" in data_buffer:
        rewards = data_buffer["rewards"]
        rewards = valid_values(rewards)
        mean_rewards, _, _ = reduce_metrics(rewards)

        rewards_metrics = {
            "rewards": mean_rewards,
        }
        rollout_metrics.update(rewards_metrics)

    if "advantages" in data_buffer:
        advantages = data_buffer["advantages"]
        advantages = valid_values(advantages)
        mean_adv, min_adv, max_adv = reduce_metrics(advantages)

        advantages_metrics = {
            "advantages_mean": mean_adv,
            "advantages_max": max_adv,
            "advantages_min": min_adv,
        }
        rollout_metrics.update(advantages_metrics)

    if data_buffer.get("returns", None) is not None:
        returns = data_buffer["returns"]
        returns = valid_values(returns)
        mean_ret, min_ret, max_ret = reduce_metrics(returns)

        returns_metrics = {
            "returns_mean": mean_ret,
            "returns_max": max_ret,
            "returns_min": min_ret,
        }
        rollout_metrics.update(returns_metrics)

    return rollout_metrics


def append_to_dict(data, new_data):
    for key, val in new_data.items():
        if key not in data:
            data[key] = []
        data[key].append(val)


def compute_loss_mask(dones):
    _, actual_bsz, num_action_chunks = dones.shape
    n_chunk_step = dones.shape[0] - 1
    flattened_dones = dones.transpose(1, 2).reshape(
        -1, actual_bsz
    )  # [(n_chunk_step + 1) * num_action_chunks, rollout_epoch x bsz]
    flattened_dones = flattened_dones[
        -(n_chunk_step * num_action_chunks + 1) :
    ]  # [n_steps+1, actual-bsz]
    flattened_loss_mask = (flattened_dones.cumsum(dim=0) == 0)[
        :-1
    ]  # [n_steps, actual-bsz]

    loss_mask = flattened_loss_mask.reshape(n_chunk_step, num_action_chunks, actual_bsz)
    loss_mask = loss_mask.transpose(
        1, 2
    )  # [n_chunk_step, actual_bsz, num_action_chunks]

    loss_mask_sum = loss_mask.sum(dim=(0, 2), keepdim=True)  # [1, bsz, 1]
    loss_mask_sum = loss_mask_sum.expand_as(loss_mask)

    return loss_mask, loss_mask_sum


def print_metrics_table(
    step: int,
    total_steps: int,
    start_time: float,
    metrics: dict,
    start_step: int = 0,
    log_path: str | None = None,
):
    """Print training metrics in a simple, fast formatted table.

    The rendered table is written to stdout and, when ``log_path`` is given,
    also appended to ``<log_path>/metrics.log``.
    """
    # Accumulate the table into lines so the exact same rendering goes to both
    # stdout and the log file.
    lines: list[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)

    # Calculate progress info
    progress = (step + 1) / total_steps * 100
    elapsed_time = time.time() - start_time
    steps_done = step + 1 - start_step
    eta_seconds = (
        elapsed_time / steps_done * (total_steps - step - 1) if steps_done > 0 else 0
    )

    def format_time(seconds):
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"

    # Format elapsed time and ETA
    elapsed_str = format_time(elapsed_time)
    eta_str = format_time(eta_seconds)

    # Create progress bar
    bar_width = 40
    filled = int(bar_width * progress / 100)
    bar = "█" * filled + "░" * (bar_width - filled)

    # Print header with progress
    total_width = 120

    def _fit_line(text: str, width: int) -> str:
        if len(text) <= width:
            return text + (" " * (width - len(text)))
        if width <= 1:
            return text[:width]
        return text[: width - 1] + "…"

    def _fit_cell(text: str, width: int) -> str:
        return _fit_line(text, width)

    def _print_section_title(title: str) -> None:
        title_text = f" {title} "
        padding = total_width - 2 - len(title_text)
        left = padding // 2
        right = padding - left
        emit(f"├{'─' * left}{title_text}{'─' * right}┤")

    emit(f"\n╭{'─' * (total_width - 2)}╮")
    _print_section_title("Metric Table")

    # First line: Global Step and Progress
    step_str = f"Global Step: {step + 1:4d}/{total_steps}"
    progress_str = f"Progress: {bar} │ {progress:5.1f}%"
    line1 = f"│ {step_str} │ {progress_str}"
    line1 = _fit_line(line1, total_width - 2)
    emit(f"{line1} │")

    # Second line: Time information
    elapsed_str_formatted = f"Elapsed: {elapsed_str}"
    eta_str_formatted = f"ETA: {eta_str}"
    step_time_str = f"Step Time: {elapsed_time / steps_done:.3f}s"
    line2 = f"│ {elapsed_str_formatted} │ {eta_str_formatted} │ {step_time_str}"
    line2 = _fit_line(line2, total_width - 2)
    emit(f"{line2} │")

    # Group metrics by category
    categories = {
        "Time": {},
        "Environment": {},
        "Rollout": {},
        "Evaluation": {},
        "Replay Buffer": {},
        "Training/Actor": {},
        "Training/Critic": {},
        "Training/Other": {},
    }

    for key, value in metrics.items():
        if "/" in key:
            category, metric_name = key.split("/", 1)
            category_map = {
                "time": "Time",
                "env": "Environment",
                "rollout": "Rollout",
                "eval": "Evaluation",
                "replay_buffer": "Replay Buffer",
            }
            if category in category_map:
                categories[category_map[category]][metric_name] = value
            elif category == "train":
                if metric_name.startswith("actor/"):
                    categories["Training/Actor"][metric_name] = value
                elif metric_name.startswith("critic/"):
                    categories["Training/Critic"][metric_name] = value
                elif metric_name.startswith("replay_buffer/"):
                    categories["Replay Buffer"][
                        metric_name.replace("replay_buffer/", "")
                    ] = value
                else:
                    categories["Training/Other"][metric_name] = value

    # Print metrics by category - 3 metrics per row
    table_width = total_width  # Match header width
    base_col_width = (table_width - 4) // 3
    remainder = (table_width - 4) - (base_col_width * 3)
    col_widths = [
        base_col_width + (1 if remainder > 0 else 0),
        base_col_width + (1 if remainder > 1 else 0),
        base_col_width,
    ]

    for category_name, category_metrics in categories.items():
        if category_metrics:
            _print_section_title(category_name)
            # Blank line before metrics (except Global Step section, which is separate)
            emit(f"│{' ' * (table_width - 2)}│")

            # Sort metrics for consistent output
            sorted_metrics = sorted(category_metrics.items())

            # Print in 3-column layout
            for i in range(0, len(sorted_metrics), 3):
                # Get up to 3 metrics for this row
                row_metrics = []
                for j in range(3):
                    if i + j < len(sorted_metrics):
                        metric_name, metric_value = sorted_metrics[i + j]

                        # Format value
                        if isinstance(metric_value, float):
                            if abs(metric_value) < 0.001 and metric_value != 0:
                                formatted_value = f"{metric_value:.2e}"
                            elif abs(metric_value) < 0.01:
                                formatted_value = f"{metric_value:.4f}"
                            elif abs(metric_value) > 10000:
                                formatted_value = f"{metric_value:.2e}"
                            elif abs(metric_value) > 100:
                                formatted_value = f"{metric_value:.1f}"
                            else:
                                formatted_value = f"{metric_value:.3f}"
                        else:
                            formatted_value = str(metric_value)

                        display = f"{metric_name}={formatted_value}"
                        row_metrics.append(display)
                    else:
                        row_metrics.append("")

                # Create the line with exactly 3 columns
                line = (
                    f"│{_fit_cell(row_metrics[0], col_widths[0])}"
                    f"│{_fit_cell(row_metrics[1], col_widths[1])}"
                    f"│{_fit_cell(row_metrics[2], col_widths[2])}│"
                )
                emit(line)

            # Section separator (minimal)
            emit(f"│{' ' * (table_width - 2)}│")

    # Bottom border
    emit(f"╰{'─' * (table_width - 2)}╯")

    emit()

    table = "\n".join(lines)
    print(table)
    if log_path:
        os.makedirs(log_path, exist_ok=True)
        with open(os.path.join(log_path, "metrics.log"), "a") as metrics_file:
            metrics_file.write(table + "\n")
