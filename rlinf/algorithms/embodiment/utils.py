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
from typing import Dict, Optional, Tuple

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
        all_eval_metrics[key] = torch.concat(all_eval_metrics[key]).float().mean().numpy()

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
        advantages = data_buffer['advantages']
        mean_adv = torch.mean(advantages).to(torch.cuda.current_device())
        torch.distributed.all_reduce(
            mean_adv, op=torch.distributed.ReduceOp.AVG
        )
        max_adv = torch.max(advantages).detach().item()
        min_adv = torch.min(advantages).detach().item()
        reduce_adv_tensor = torch.as_tensor(
            [-min_adv, max_adv], device=torch.cuda.current_device(), dtype=torch.float32
        )
        torch.distributed.all_reduce(reduce_adv_tensor, op=torch.distributed.ReduceOp.MAX)
        min_adv, max_adv = reduce_adv_tensor.tolist()

        advantages_metrics = {
            "advantages_mean": mean_adv.item(),
            "advantages_max": max_adv,
            "advantages_min": -min_adv,
        }
        rollout_metrics.update(advantages_metrics)

    if "returns" in data_buffer and data_buffer["returns"] is not None:
        returns = data_buffer["returns"]
        mean_ret = torch.mean(returns).to(torch.cuda.current_device())
        torch.distributed.all_reduce(mean_ret, op=torch.distributed.ReduceOp.AVG)
        max_ret = torch.max(returns).detach().item()
        min_ret= torch.min(returns).detach().item()
        reduce_ret_tensor = torch.as_tensor(
            [-min_ret, max_ret], device=torch.cuda.current_device(), dtype=torch.float32
        )
        torch.distributed.all_reduce(reduce_ret_tensor, op=torch.distributed.ReduceOp.MAX)
        min_ret, max_ret = reduce_ret_tensor.tolist()

        returns_metrics = {
            "returns_mean": mean_ret.item(),
            "returns_max": max_ret,
            "returns_min": -min_ret,
        }
        rollout_metrics.update(returns_metrics)


    env_info_keys = [key for key in data_buffer if key.startswith("env_info/")]
    for env_info_key in env_info_keys:
        value = data_buffer.pop(env_info_key)
        value = value.float().mean().to(torch.cuda.current_device())
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.AVG)
        rollout_metrics[env_info_key] = value.item()

    return rollout_metrics

def huber_loss(error: torch.Tensor, delta: float) -> torch.Tensor:
    return torch.where(
        error.abs() < delta,
        0.5 * error ** 2,
        delta * (error.abs() - 0.5 * delta)
    )

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


def calculate_advantages_and_returns(
    adv_type,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    dones_terminated: torch.Tensor,
    normalize_advantages: bool = True,
    values: torch.Tensor = None,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    num_group_envs: int= 1,
    group_size: int= 2,
    num_action_chunks: int = 1,
    reward_type: Optional[str] = None,
    loss_mask: torch.Tensor = None
):
    # TODO: reward max to solve the bug on the action chunk
    if reward_type == "chunk_level":
        rewards = rewards.max(dim=-1, keepdim=True)[0]
        dones = dones[..., -1:]
        loss_mask = loss_mask[..., -1:]

    # TODO: HERE FOR SIMPLE TEST
    # TODO: ADV CALCULATION BUG & DONE REWARD MATCHING
    if adv_type == "grpo":
        from infini_rl.algorithms.embodiment.grpo_functions import (
            compute_advantages_with_loss_mask,
        )
        advantages = compute_advantages_with_loss_mask(
            rewards=rewards,
            dones=dones,
            num_group_envs=num_group_envs,
            group_size=group_size,
            normalize_advantages=normalize_advantages,
            loss_mask=loss_mask
        )
        return advantages, None
        # rewards_max = rewards[:,:,0].max(dim=0)[0]
        # advs = rewards_max.reshape(2,8)
        # adv_mean = advs.mean(dim=1, keepdim=True)
        # adv_std = advs.std(dim=1, keepdim=True)
        # advs = (advs - adv_mean) / (adv_std + 1e-6)
        # return advs, None

    elif adv_type == "ppo":
        from infini_rl.algorithms.embodiment.ppo_functions import (
            compute_advantages_and_returns,
        )
        advantages, returns = compute_advantages_and_returns(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=gamma,
            gae_lambda=gae_lambda,
            normalize_advantages=normalize_advantages
        )
        return advantages, returns
    else:
        raise ValueError(f"Advantage type {adv_type} not supported")

def actor_loss_fn(
    loss_type,
    logprob_type,
    entropy_type,
    single_action_dim,
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    values: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    prev_values: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    value_clip: float,
    huber_delta: float,
    entropy_bonus: float,
    loss_mask: torch.Tensor,
    loss_mask_sum: torch.Tensor
) -> Tuple[torch.Tensor, Dict]:
    if loss_type == "grpo":
        from infini_rl.algorithms.embodiment.grpo_functions import (
            actor_loss_fn_with_loss_mask,
        )
        return actor_loss_fn_with_loss_mask(
            log_probs=logprobs,
            old_log_prob=old_logprobs,
            advantages=advantages,
            clip_ratio_high=clip_ratio_high,
            clip_ratio_low=clip_ratio_low,
            loss_mask=loss_mask,
            loss_mask_sum=loss_mask_sum
        )
    elif loss_type == "ppo":
        from infini_rl.algorithms.embodiment.ppo_functions import actor_critic_loss_fn
        # TODO: @zanghongzhi @guozhen decoupled critic loss calculation
        bsz = logprobs.shape[0]

        # logprobs.shape: [bsz, token-len]
        # advantage.shape: [bsz, chunk-step]
        if logprob_type == "token_level":
            logprobs = logprobs.reshape(bsz, -1, single_action_dim)
            old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim)
            advantages = advantages.unsqueeze(-1)
        elif logprob_type == "action_level":
            logprobs = logprobs.reshape(bsz, -1, single_action_dim).sum(dim=-1)
            old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim).sum(dim=-1)
        elif logprob_type == "chunk_level":
            logprobs = logprobs.sum(dim=-1)
            old_logprobs = old_logprobs.sum(dim=-1)
            advantages = advantages.sum(dim=-1)

        if entropy_type == "token_level":
            pass
        elif entropy_type == "action_level":
            entropy = entropy.reshape(bsz, -1, single_action_dim).sum(dim=-1)
        elif entropy_type == "chunk_level":
            entropy = entropy.sum(dim=-1)

        return actor_critic_loss_fn(
            logprobs=logprobs,
            entropy=entropy,
            values=values,
            old_logprobs=old_logprobs,
            advantages=advantages,
            returns=returns,
            prev_values=prev_values,
            clip_ratio_high=clip_ratio_high,
            clip_ratio_low=clip_ratio_low,
            value_clip=value_clip,
            huber_delta=huber_delta,
            entropy_bonus=entropy_bonus
        )
