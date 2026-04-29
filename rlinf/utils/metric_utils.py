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
import time

import numpy as np
import torch
import torch.distributed

from rlinf.scheduler import Worker


def compute_split_num(num, split_num):
    return math.lcm(num, split_num) // split_num


def compute_pipeline_micro_batch_env_size(
    micro_batch_size: int,
    rollout_epoch: int,
    n_train_chunk_steps: int,
    group_size: int = 1,
    rollout_epochs_per_flush: int | None = None,
) -> int:
    """Compute the env batch size of one pipeline local micro batch."""
    if rollout_epochs_per_flush is None:
        rollout_epochs_per_flush = rollout_epoch
    assert rollout_epoch % rollout_epochs_per_flush == 0, (
        f"rollout_epoch ({rollout_epoch}) must be divisible by "
        f"rollout_epochs_per_flush ({rollout_epochs_per_flush})."
    )
    flat_samples_per_env = rollout_epochs_per_flush * n_train_chunk_steps
    assert micro_batch_size % flat_samples_per_env == 0, (
        f"actor.micro_batch_size ({micro_batch_size}) must be divisible by "
        f"rollout_epochs_per_flush * n_train_chunk_steps ({flat_samples_per_env})."
    )
    env_batch_size = micro_batch_size // flat_samples_per_env
    assert env_batch_size % group_size == 0, (
        f"pipeline env batch size ({env_batch_size}) must be divisible by "
        f"group_size ({group_size}) to keep group-aligned splits."
    )
    return env_batch_size


def compute_pipeline_actor_split_num(
    train_num_envs_per_stage: int,
    micro_batch_size: int,
    rollout_epoch: int,
    n_train_chunk_steps: int,
    group_size: int = 1,
    rollout_epochs_per_flush: int | None = None,
) -> int:
    """Compute how many pipeline splits one env stage should emit per flush."""
    env_batch_size = compute_pipeline_micro_batch_env_size(
        micro_batch_size=micro_batch_size,
        rollout_epoch=rollout_epoch,
        n_train_chunk_steps=n_train_chunk_steps,
        group_size=group_size,
        rollout_epochs_per_flush=rollout_epochs_per_flush,
    )
    assert train_num_envs_per_stage % env_batch_size == 0, (
        f"train_num_envs_per_stage ({train_num_envs_per_stage}) must be divisible by "
        f"pipeline env micro-batch size ({env_batch_size})."
    )
    return train_num_envs_per_stage // env_batch_size


def compute_pipeline_expected_actor_recv_num(
    total_num_envs: int,
    actor_world_size: int,
    micro_batch_size: int,
    rollout_epoch: int,
    n_train_chunk_steps: int,
) -> int:
    """Compute how many pipeline local micro batches each actor rank should receive."""
    total_rollout_samples = total_num_envs * rollout_epoch * n_train_chunk_steps
    per_rank_micro_batch_samples = actor_world_size * micro_batch_size
    assert total_rollout_samples % per_rank_micro_batch_samples == 0, (
        f"Total flattened rollout samples ({total_rollout_samples}) must be "
        f"divisible by actor_world_size * micro_batch_size "
        f"({per_rank_micro_batch_samples})."
    )
    return total_rollout_samples // per_rank_micro_batch_samples


def reshape_embodied_rollout_batch_for_adv(
    nested_dict: dict,
    rollout_epoch: int,
) -> dict:
    """Reshape embodied rollout tensors for advantage computation."""
    ret_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, torch.Tensor):
            new_value = value.reshape(rollout_epoch, -1, *value.shape[1:])
            new_value = new_value.transpose(0, 1)
            new_value = new_value.reshape(new_value.shape[0], -1, *new_value.shape[3:])
            ret_dict[key] = new_value
        elif isinstance(value, dict):
            ret_dict[key] = reshape_embodied_rollout_batch_for_adv(value, rollout_epoch)
    return ret_dict


def process_embodied_rollout_batch_for_adv(
    rollout_batch: dict[str, torch.Tensor],
    *,
    rollout_epoch: int,
    auto_reset: bool,
    ignore_terminations: bool,
    reward_type: str,
    filter_rewards: bool,
    group_size: int,
    rewards_lower_bound: float | None = None,
    rewards_upper_bound: float | None = None,
) -> dict[str, torch.Tensor]:
    """Prepare an embodied rollout batch for advantage computation."""
    rollout_batch = reshape_embodied_rollout_batch_for_adv(rollout_batch, rollout_epoch)

    if not auto_reset and not ignore_terminations:
        dones = rollout_batch["dones"]
        loss_mask, loss_mask_sum = compute_loss_mask(dones)

        if reward_type == "chunk_level":
            loss_mask = loss_mask.any(dim=-1, keepdim=True)
            loss_mask_sum = loss_mask_sum[..., -1:]

        rollout_batch["loss_mask"] = loss_mask
        rollout_batch["loss_mask_sum"] = loss_mask_sum

    if filter_rewards:
        rewards = rollout_batch["rewards"]
        if rollout_batch.get("loss_mask", None) is not None:
            rewards = rewards * rollout_batch["loss_mask"]
        n_chunk_step, batch_size, _ = rewards.shape

        assert batch_size % group_size == 0, (
            f"batch {batch_size} not divisible by group_size {group_size}"
        )
        n_prompts = batch_size // group_size

        rewards = rewards.transpose(0, 1).reshape(rewards.shape[1], -1)
        reward_matrix = rewards.reshape(n_prompts, group_size, rewards.shape[-1])
        reward_matrix = reward_matrix.sum(dim=-1)
        mean_reward_in_group = reward_matrix.mean(dim=1)

        reward_filter_mask = (mean_reward_in_group >= rewards_lower_bound) & (
            mean_reward_in_group <= rewards_upper_bound
        )
        reward_filter_mask = reward_filter_mask.repeat_interleave(group_size)
        reward_filter_mask = (
            reward_filter_mask.unsqueeze(0).expand(n_chunk_step, -1).unsqueeze(-1)
        )

        if rollout_batch.get("loss_mask", None) is not None:
            rollout_batch["loss_mask"] = reward_filter_mask & rollout_batch["loss_mask"]
        else:
            rollout_batch["loss_mask"] = reward_filter_mask

    return rollout_batch


def flatten_embodied_rollout_batch_for_train(
    nested_dict: dict,
    shuffle_id: torch.Tensor,
) -> dict:
    """Flatten an embodied rollout batch into train-order samples."""
    ret_dict = {}
    for key, value in nested_dict.items():
        if key in ["dones", "terminations", "truncations", "prev_values"]:
            value = value[:-1]
        if "env_info" in key:
            raise NotImplementedError
        if value is None:
            ret_dict[key] = None
        elif isinstance(value, torch.Tensor):
            ret_dict[key] = value.reshape(-1, *value.shape[2:])[shuffle_id]
        elif isinstance(value, dict):
            ret_dict[key] = flatten_embodied_rollout_batch_for_train(value, shuffle_id)
    return ret_dict


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

    # Use the first metric tensor to get the trajectory count
    # All metrics should have the same first dimension (number of trajectories)
    first_key = next(iter(metrics_dict.keys()))
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

    for key in all_eval_metrics:
        shards = [_normalize_metric_shard(s) for s in all_eval_metrics[key]]
        stacked = torch.concat(shards).float()
        all_eval_metrics[key] = (
            stacked.mean().detach().cpu().numpy()
            if stacked.numel() > 0
            else np.asarray(0.0, dtype=np.float64)
        )

    # Add total trajectory count to metrics
    all_eval_metrics["num_trajectories"] = sum(trajectory_counts)

    return all_eval_metrics


def compute_rollout_metrics(data_buffer: dict) -> dict:
    rollout_metrics = {}

    if "rewards" in data_buffer:
        rewards = data_buffer["rewards"].clone()
        mean_rewards = torch.mean(rewards).to(Worker.torch_platform.current_device())
        torch.distributed.all_reduce(mean_rewards, op=torch.distributed.ReduceOp.AVG)

        rewards_metrics = {
            "rewards": mean_rewards.item(),
        }
        rollout_metrics.update(rewards_metrics)

    if "advantages" in data_buffer:
        advantages = data_buffer["advantages"]
        mean_adv = torch.mean(advantages).to(Worker.torch_platform.current_device())
        torch.distributed.all_reduce(mean_adv, op=torch.distributed.ReduceOp.AVG)
        max_adv = torch.max(advantages).detach().item()
        min_adv = torch.min(advantages).detach().item()
        reduce_adv_tensor = torch.as_tensor(
            [-min_adv, max_adv],
            device=Worker.torch_platform.current_device(),
            dtype=torch.float32,
        )
        torch.distributed.all_reduce(
            reduce_adv_tensor, op=torch.distributed.ReduceOp.MAX
        )
        min_adv, max_adv = reduce_adv_tensor.tolist()

        advantages_metrics = {
            "advantages_mean": mean_adv.item(),
            "advantages_max": max_adv,
            "advantages_min": -min_adv,
        }
        rollout_metrics.update(advantages_metrics)

    if data_buffer.get("returns", None) is not None:
        returns = data_buffer["returns"]
        mean_ret = torch.mean(returns).to(Worker.torch_platform.current_device())
        torch.distributed.all_reduce(mean_ret, op=torch.distributed.ReduceOp.AVG)
        max_ret = torch.max(returns).detach().item()
        min_ret = torch.min(returns).detach().item()
        reduce_ret_tensor = torch.as_tensor(
            [-min_ret, max_ret],
            device=Worker.torch_platform.current_device(),
            dtype=torch.float32,
        )
        torch.distributed.all_reduce(
            reduce_ret_tensor, op=torch.distributed.ReduceOp.MAX
        )
        min_ret, max_ret = reduce_ret_tensor.tolist()

        returns_metrics = {
            "returns_mean": mean_ret.item(),
            "returns_max": max_ret,
            "returns_min": -min_ret,
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
    step: int, total_steps: int, start_time: float, metrics: dict, start_step: int = 0
):
    """Print training metrics in a simple, fast formatted table."""
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
        print(f"├{'─' * left}{title_text}{'─' * right}┤")

    print(f"\n╭{'─' * (total_width - 2)}╮")
    _print_section_title("Metric Table")

    # First line: Global Step and Progress
    step_str = f"Global Step: {step + 1:4d}/{total_steps}"
    progress_str = f"Progress: {bar} │ {progress:5.1f}%"
    line1 = f"│ {step_str} │ {progress_str}"
    line1 = _fit_line(line1, total_width - 2)
    print(f"{line1} │")

    # Second line: Time information
    elapsed_str_formatted = f"Elapsed: {elapsed_str}"
    eta_str_formatted = f"ETA: {eta_str}"
    step_time_str = f"Step Time: {elapsed_time / steps_done:.3f}s"
    line2 = f"│ {elapsed_str_formatted} │ {eta_str_formatted} │ {step_time_str}"
    line2 = _fit_line(line2, total_width - 2)
    print(f"{line2} │")

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
            print(f"│{' ' * (table_width - 2)}│")

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
                print(line)

            # Section separator (minimal)
            print(f"│{' ' * (table_width - 2)}│")

    # Bottom border
    print(f"╰{'─' * (table_width - 2)}╯")

    print()
