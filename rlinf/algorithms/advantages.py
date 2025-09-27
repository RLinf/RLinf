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

from typing import Tuple

import torch

from rlinf.algorithms.registry import register_advantage


@register_advantage("gae")
def compute_gae_advantages_and_returns(
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate advantages and returns for Proximal Policy Optimization (PPO).
    NOTE: currently this function does not support auto-reset.

    This function implements Generalized Advantage Estimation (GAE) to compute
    advantages and returns for PPO training. The advantages are normalized
    using mean and standard deviation for stable training.

    Args:
        rewards (torch.Tensor): Reward tensor of shape [num-chunk, bsz, chunk-size]
        values (torch.Tensor): Value predictions of shape [num-chunk + 1, bsz, chunk-size]
        dones (torch.Tensor): Done flag tensor of shape [num-chunk + 1, bsz, chunk-size]
        gamma (float): Discount factor
        gae_lambda (float): GAE lambda parameter
        normalize_advantages (bool): Whether to normalize advantages

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (advantages, returns) tensors
    """

    rewards = kwargs["rewards"]
    values = kwargs["values"]
    dones = kwargs["dones"]
    gamma = kwargs.get("gamma", 1.0)
    gae_lambda = kwargs.get("gae_lambda", 1.0)
    normalize_advantages = kwargs.get("normalize_advantages", True)
    normalize_returns = kwargs.get("normalize_returns", False)

    returns = torch.zeros_like(rewards)
    gae = 0
    for step in reversed(range(rewards.shape[0])):
        vt1 = values[step + 1]
        vt = values[step]

        delta = rewards[step] + gamma * vt1 * (~dones[step + 1]) - vt
        gae = delta + gamma * gae_lambda * (~dones[step + 1]) * gae
        returns[step] = gae + vt

    # calc adv
    advantages = returns - values[:-1]

    if normalize_advantages:
        mean_advantages = advantages.mean()
        std_advantages = advantages.std(correction=0)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
    if normalize_returns:
        mean_returns = returns.mean()
        std_retuns = returns.std(correction=0)
        returns = (returns - mean_returns) / (std_retuns + 1e-5)

    kwargs.update(
        {
            "advantages": advantages,
            "returns": returns,
        }
    )

    return kwargs


@register_advantage("grpo")
def compute_grpo_advantages(**kwargs):
    reward_scores = kwargs["reward_scores"]
    mask = kwargs["loss_mask"]
    group_size = kwargs["group_size"]

    grouped_rewards = reward_scores.view(-1, group_size)

    grouped_reward_mean = grouped_rewards.mean(dim=1).repeat_interleave(
        group_size, dim=0
    )
    grouped_reward_std = grouped_rewards.std(dim=1).repeat_interleave(group_size, dim=0)
    advantages = reward_scores - grouped_reward_mean
    advantages = advantages / (grouped_reward_std + 1e-6)

    advantages = (torch.zeros_like(mask) + advantages.view(-1, 1)) * mask

    kwargs.update({"advantages": advantages})

    return kwargs


@register_advantage("reinpp")
def compute_reinpp_advantages(**kwargs):
    reward_scores = kwargs["reward_scores"]
    mask = kwargs["loss_mask"]
    group_size = kwargs["group_size"]
    use_reinpp_baseline = kwargs.get("use_reinpp_baseline", False)

    # first group baseline for reinforce++ baseline
    if use_reinpp_baseline:
        grouped_rewards = reward_scores.view(-1, group_size)  # [num_prompt, group_size]
        grouped_rewards -= grouped_rewards.mean(dim=1, keepdims=True)
        reward_scores = grouped_rewards.view(-1)  # [B]

    # build the reward matrix
    r_matrix = torch.zeros_like(mask)  # [B, L]
    seq_length = mask.size(1)
    mask_flipped = mask.long().fliplr()
    eos_positions = mask_flipped.argmax(
        dim=1, keepdim=True
    )  # position of last True in original mask
    eos_indices = seq_length - 1 - eos_positions  # [B, 1]

    r_matrix = r_matrix.scatter_(
        dim=1, index=eos_indices, src=reward_scores.unsqueeze(1)
    )  # [B, L]

    # compute return
    ret_matrix = torch.cumsum(r_matrix.flip(dims=[1]), dim=1).flip(dims=[1])

    # normalize
    advantages = ret_matrix.clone()

    mean = (advantages * mask).sum() / mask.sum()
    var = ((advantages - mean).pow(2) * mask).sum() / mask.sum()
    rstd = var.clamp(min=1e-8).rsqrt()

    advantages = (advantages - mean) * rstd

    kwargs.update({"advantages": advantages})

    return kwargs


if __name__ == "__main__":
    from rlinf.algorithms.utils import (
        calculate_scores,
        postprocess_advantages_outputs,
        preprocess_advantages_inputs,
    )

    # test ppo adv for embodiment
    torch.manual_seed(0)
    rewards = torch.randn(4, 2, 3)
    values = torch.randn(5, 2, 3)
    dones = torch.zeros(5, 2, 3).bool()
    dones[-1] = 1
    kwargs = {
        "rewards": rewards,
        "values": values,
        "dones": dones,
        "gamma": 0.99,
        "gae_lambda": 0.95,
    }
    kwargs = preprocess_advantages_inputs(**kwargs)
    kwargs = compute_gae_advantages_and_returns(**kwargs)
    advantages, returns = postprocess_advantages_outputs(**kwargs)
    print(advantages.mean(), advantages.shape)
    print(returns.mean(), returns.shape)

    # test grpo adv for embodiment
    torch.manual_seed(0)
    rewards = torch.randn(4, 4, 3)  # num_chunk, bsz, chunk_size
    dones = torch.zeros(5, 4, 3).bool()
    loss_mask = torch.rand_like(rewards) > 0.5
    dones[-1] = 1
    kwargs = {
        "rewards": rewards,
        "dones": dones,
        "loss_mask": loss_mask,
        "group_size": 2,
    }
    kwargs = preprocess_advantages_inputs(**kwargs)
    kwargs = calculate_scores(**kwargs)
    kwargs = compute_grpo_advantages(**kwargs)
    advantages, _ = postprocess_advantages_outputs(**kwargs)
    print(advantages.mean(), advantages.shape)

    # test grpo for math
    torch.manual_seed(0)
    rewards = torch.randn(4)
    loss_mask = torch.rand(4, 2) > 0.3
    group_size = 2
    kwargs = compute_grpo_advantages(
        reward_scores=rewards,
        loss_mask=loss_mask,
        group_size=group_size,
    )
    advantages = kwargs["advantages"]
    print(advantages.mean(), advantages.std(), advantages.shape)
