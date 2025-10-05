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
from rlinf.algorithms.utils import kl_penalty
from rlinf.utils.utils import masked_mean


@register_advantage("gae")
def compute_gae_advantages_and_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    normalize_advantages: bool = True,
    normalize_returns: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate advantages and returns for Proximal Policy Optimization (PPO).
    NOTE: currently this function does not support auto-reset.

    This function implements Generalized Advantage Estimation (GAE) to compute
    advantages and returns for PPO training. The advantages are normalized
    using mean and standard deviation for stable training.

    Args:
        rewards (torch.Tensor): Rewards per timestep.
        values (torch.Tensor): Value function estimates.
        dones (torch.Tensor): Done flags (1 if episode ended, else 0).
        gamma (float, optional): Discount factor. Defaults to 1.0.
        gae_lambda (float, optional): GAE smoothing factor. Defaults to 1.0.
        normalize_advantages (bool, optional): Whether to normalize advantages. Defaults to True.
        normalize_returns (bool, optional): Whether to normalize returns. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (advantages, returns)
    """
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

    return advantages, returns


@register_advantage("grpo")
def compute_grpo_advantages(
    reward_scores: torch.Tensor,
    mask: torch.Tensor,
    group_size: int,
):
    """
    Compute GRPO advantages.

    Args:
        reward_scores (torch.Tensor): Reward or score values.
        mask (torch.Tensor): Loss mask for valid entries.
        group_size (int): Group size for advantage computation.

    Returns:
        torch.Tensor: advantages
    """
    grouped_rewards = reward_scores.view(-1, group_size)

    grouped_reward_mean = grouped_rewards.mean(dim=1).repeat_interleave(
        group_size, dim=0
    )
    grouped_reward_std = grouped_rewards.std(dim=1).repeat_interleave(group_size, dim=0)
    advantages = reward_scores - grouped_reward_mean
    advantages = advantages / (grouped_reward_std + 1e-6)

    advantages = (torch.zeros_like(mask) + advantages.view(-1, 1)) * mask

    return advantages, None


@register_advantage("reinpp")
def compute_reinpp_advantages(
    reward_scores: torch.Tensor,
    mask: torch.Tensor,
    group_size: int,
    use_reinpp_baseline: bool = False,
    kl_beta: float = 0.0,
    logprob=None,
    ref_logprob=None,
    kl_penalty_type: str = "",
):
    """
    Compute advantages for reinforce++ and reinforce++ baseline.

    Args:
        reward_scores (torch.Tensor): The reward or score values.
        mask (torch.Tensor): The loss mask for valid entries.
        group_size (int): The group size for advantage computation.
        use_reinpp_baseline (bool, optional): Whether to use reinforce++ baseline.
        kl_beta (float, optional): KL penalty coefficient.
        logprob (optional): Log probability of current policy.
        ref_logprob (optional): Log probability of reference policy.
        kl_penalty_type (str, optional): Type of KL penalty.

    Returns:
        torch.Tensor: advantages
    """
    # first group baseline for reinforce++ baseline
    if use_reinpp_baseline:
        grouped_rewards = reward_scores.view(-1, group_size)  # [num_prompt, group_size]
        grouped_rewards -= grouped_rewards.mean(dim=1, keepdims=True)
        reward_scores = grouped_rewards.view(-1)  # [B]

    # build the reward matrix
    r_matrix = torch.zeros_like(mask).float()  # [B, L]
    seq_length = mask.size(1)
    mask_flipped = mask.long().fliplr()
    eos_positions = mask_flipped.argmax(
        dim=1, keepdim=True
    )  # position of last True in original mask
    eos_indices = seq_length - 1 - eos_positions  # [B, 1]

    r_matrix = r_matrix.scatter_(
        dim=1, index=eos_indices, src=reward_scores.unsqueeze(1)
    )  # [B, L]

    # add kl penalty
    if kl_beta > 0:
        kld = kl_penalty(logprob, ref_logprob, kl_penalty=kl_penalty_type)  # [B, L]
        r_matrix -= kl_beta * kld

    # compute return
    ret_matrix = torch.cumsum(r_matrix.flip(dims=[1]), dim=1).flip(dims=[1])

    # normalize
    advantages = ret_matrix.clone()

    mean = masked_mean(advantages, mask)
    var = masked_mean((advantages - mean).pow(2), mask)
    rstd = var.clamp(min=1e-8).rsqrt()

    advantages = (advantages - mean) * rstd

    return advantages, None


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

    # test reinforce++ for math
    torch.manual_seed(0)
    rewards = torch.randn(4)
    loss_mask = torch.zeros(4, 2).bool()
    loss_mask[:, :-1] = 1
    group_size = 2
    kwargs = compute_reinpp_advantages(
        reward_scores=rewards,
        loss_mask=loss_mask,
        group_size=group_size,
        use_reinpp_baseline=True,
    )
    advantages = kwargs["advantages"]
    print(advantages.mean(), advantages.std(), advantages.shape)
