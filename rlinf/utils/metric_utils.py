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
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed

if TYPE_CHECKING:
    from rlinf.data.embodied_io_struct import Trajectory


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


def _is_interact_delay_metric_key(key: str) -> bool:
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


def compute_split_num(num, split_num):
    return math.lcm(num, split_num) // split_num


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
        key for key in metrics_dict.keys() if not _is_interact_delay_metric_key(key)
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
        if _is_interact_delay_metric_key(key):
            if stacked.numel() > 0:
                aggregated_eval_metrics[
                    _build_interact_delay_stat_key(key, "average_delay")
                ] = stacked.mean().detach().cpu().numpy()
                aggregated_eval_metrics[
                    _build_interact_delay_stat_key(key, "medium_delay")
                ] = torch.quantile(stacked, 0.5).detach().cpu().numpy()
                aggregated_eval_metrics[
                    _build_interact_delay_stat_key(key, "max_delay")
                ] = stacked.max().detach().cpu().numpy()
                aggregated_eval_metrics[
                    _build_interact_delay_stat_key(key, "min_delay")
                ] = stacked.min().detach().cpu().numpy()
            else:
                aggregated_eval_metrics[
                    _build_interact_delay_stat_key(key, "average_delay")
                ] = np.asarray(0.0, dtype=np.float64)
                aggregated_eval_metrics[
                    _build_interact_delay_stat_key(key, "medium_delay")
                ] = np.asarray(0.0, dtype=np.float64)
                aggregated_eval_metrics[
                    _build_interact_delay_stat_key(key, "max_delay")
                ] = np.asarray(0.0, dtype=np.float64)
                aggregated_eval_metrics[
                    _build_interact_delay_stat_key(key, "min_delay")
                ] = np.asarray(0.0, dtype=np.float64)
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
        if mask.shape != values.shape:
            mask = torch.broadcast_to(mask, values.shape)
        return values[mask]

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
