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
from typing import Dict

import torch
import torch.distributed


def compute_split_num(num, split_num):
    return math.lcm(num, split_num) // split_num


def compute_evaluate_metrics(eval_metrics_list):
    """
    List of evaluate metrics, list length stands for rollout process
    """
    all_eval_metrics = {}
    env_info_keys = eval_metrics_list[0].keys()

    for env_info_key in env_info_keys:
        all_eval_metrics[env_info_key] = [
            eval_metrics[env_info_key] for eval_metrics in eval_metrics_list
        ]

    for key in all_eval_metrics:
        all_eval_metrics[key] = (
            torch.concat(all_eval_metrics[key]).float().mean().numpy()
        )

    return all_eval_metrics


def compute_rollout_metrics(data_buffer: Dict) -> Dict:
    rollout_metrics = {}

    if "rewards" in data_buffer:
        rewards = data_buffer["rewards"].clone()
        mean_rewards = torch.mean(rewards).to(torch.cuda.current_device())
        torch.distributed.all_reduce(mean_rewards, op=torch.distributed.ReduceOp.AVG)

        rewards_metrics = {
            "rewards": mean_rewards.item(),
        }
        rollout_metrics.update(rewards_metrics)

    if "advantages" in data_buffer:
        advantages = data_buffer["advantages"]
        mean_adv = torch.mean(advantages).to(torch.cuda.current_device())
        torch.distributed.all_reduce(mean_adv, op=torch.distributed.ReduceOp.AVG)
        max_adv = torch.max(advantages).detach().item()
        min_adv = torch.min(advantages).detach().item()
        reduce_adv_tensor = torch.as_tensor(
            [-min_adv, max_adv], device=torch.cuda.current_device(), dtype=torch.float32
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

    if "returns" in data_buffer:
        returns = data_buffer["returns"]
        mean_ret = torch.mean(returns).to(torch.cuda.current_device())
        torch.distributed.all_reduce(mean_ret, op=torch.distributed.ReduceOp.AVG)
        max_ret = torch.max(returns).detach().item()
        min_ret = torch.min(returns).detach().item()
        reduce_ret_tensor = torch.as_tensor(
            [-min_ret, max_ret], device=torch.cuda.current_device(), dtype=torch.float32
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
    )  # [n_chunk_step + 1, rollout_epoch x bsz]
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
