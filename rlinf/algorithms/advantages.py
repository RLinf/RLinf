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
    **kwargs,
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
    loss_mask: torch.Tensor,
    group_size: int,
    **kwargs,
):
    """
    Compute GRPO advantages.

    Args:
        reward_scores (torch.Tensor): Reward or score values.
        loss_mask (torch.Tensor): Loss mask for valid entries.
        group_size (int): Group size for advantage computation.

    Returns:
        torch.Tensor: advantages
    """
    grouped_rewards = reward_scores.view(-1, group_size)

    grouped_reward_mean = grouped_rewards.mean(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )
    grouped_reward_std = grouped_rewards.std(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )

    advantages = grouped_rewards - grouped_reward_mean
    advantages = advantages / (grouped_reward_std + 1e-6)

    advantages = (torch.zeros_like(loss_mask) + advantages.view(-1, 1)) * loss_mask

    return advantages, None


@register_advantage("reinpp")
def compute_reinpp_advantages(
    reward_scores: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    use_reinpp_baseline: bool = False,
    kl_beta: float = 0.0,
    logprob=None,
    ref_logprob=None,
    kl_penalty_type: str = "",
    **kwargs,
):
    """
    Compute advantages for reinforce++ and reinforce++ baseline.

    Args:
        reward_scores (torch.Tensor): The reward or score values.
        loss_mask (torch.Tensor): The loss mask for valid entries.
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
    r_matrix = torch.zeros_like(loss_mask).float()  # [B, L]
    seq_length = loss_mask.size(1)
    mask_flipped = loss_mask.long().fliplr()
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

    mean = masked_mean(advantages, loss_mask)
    var = masked_mean((advantages - mean).pow(2), loss_mask)
    rstd = var.clamp(min=1e-8).rsqrt()

    advantages = (advantages - mean) * rstd

    return advantages, None


@register_advantage("math_gae_no_critic")
def compute_math_gae_no_critic_advantages_and_returns(**kwargs):
    """
    Calculate advantages and returns for math tasks using GAE without critic model.

    This function implements a simplified advantage estimation for math tasks
    without requiring a value function, similar to AReaL's disable_head approach.

    Args:
        reward_scores (torch.Tensor): Reward scores for math responses
        mask (torch.Tensor): Attention mask of shape [bsz, seq_len] or [bsz, max_seq_len]
        gamma (float): Discount factor
        gae_lambda (float): GAE lambda parameter
        normalize_advantages (bool): Whether to normalize advantages
        normalize_returns (bool): Whether to normalize returns

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (advantages, returns) tensors
    """
    reward_scores = kwargs["reward_scores"]
    mask = kwargs["mask"]
    gamma = kwargs.get("gamma", 1.0)
    normalize_advantages = kwargs.get("normalize_advantages", True)
    normalize_returns = kwargs.get("normalize_returns", False)

    # For math tasks without critic, we use reward-to-go as baseline
    bsz, seq_len = mask.shape

    # Create reward structure: reward at the end of sequence
    rewards = torch.zeros_like(mask, dtype=torch.float32)
    rewards[:, -1] = reward_scores  # Put reward at the end of sequence

    # Create done flags (episode ends at the last token)
    dones = torch.zeros_like(mask, dtype=torch.bool)
    dones[:, -1] = True

    # Compute reward-to-go (cumulative discounted rewards)
    returns = torch.zeros_like(mask, dtype=torch.float32)
    cumulative_reward = 0

    for t in reversed(range(seq_len)):
        cumulative_reward = rewards[:, t] + gamma * cumulative_reward * (~dones[:, t])
        returns[:, t] = cumulative_reward

    # For no-critic setup, advantages are computed using reward-to-go
    # with a simple baseline subtraction
    advantages = returns.clone()

    # Apply mask
    advantages = advantages * mask
    returns = returns * mask

    # Simple baseline subtraction (mean of valid advantages)
    if normalize_advantages:
        valid_advantages = advantages[mask.bool()]
        if len(valid_advantages) > 0:
            mean_advantages = valid_advantages.mean()
            std_advantages = valid_advantages.std()
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

    # Normalize returns if requested
    if normalize_returns:
        valid_returns = returns[mask.bool()]
        if len(valid_returns) > 0:
            mean_returns = valid_returns.mean()
            std_returns = valid_returns.std()
            returns = (returns - mean_returns) / (std_returns + 1e-5)

    return advantages, returns


@register_advantage("math_gae")
def compute_math_gae_advantages_and_returns(**kwargs):
    """
    Calculate advantages and returns for math tasks using GAE.

    This function implements Generalized Advantage Estimation (GAE) specifically
    designed for math tasks, which may have different data structures compared
    to embodied tasks.

    Args:
        reward_scores (torch.Tensor): Reward scores for math responses
        values (torch.Tensor): Value predictions of shape [bsz, seq_len] or [bsz, max_seq_len]
        mask (torch.Tensor): Attention mask of shape [bsz, seq_len] or [bsz, max_seq_len]
        gamma (float): Discount factor
        gae_lambda (float): GAE lambda parameter
        normalize_advantages (bool): Whether to normalize advantages
        normalize_returns (bool): Whether to normalize returns

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (advantages, returns) tensors
    """
    reward_scores = kwargs["reward_scores"]
    values = kwargs["values"]
    mask = kwargs["mask"]
    gamma = kwargs.get("gamma", 1.0)
    gae_lambda = kwargs.get("gae_lambda", 1.0)
    normalize_advantages = kwargs.get("normalize_advantages", True)
    normalize_returns = kwargs.get("normalize_returns", False)

    # For math tasks, we typically have [bsz, seq_len] tensors
    bsz, seq_len = values.shape

    # Create a simple reward structure for math tasks
    # The reward is typically given at the end of the sequence
    rewards = torch.zeros_like(values)
    rewards[:, -1] = reward_scores  # Put reward at the end of sequence

    # Create done flags (episode ends at the last token)
    dones = torch.zeros_like(values, dtype=torch.bool)
    dones[:, -1] = True

    # Add bootstrap value for the next state (after the sequence)
    next_values = torch.zeros(bsz, 1, device=values.device, dtype=values.dtype)

    # Compute GAE advantages
    advantages = torch.zeros_like(values)
    returns = torch.zeros_like(values)

    gae = 0
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            # Last timestep
            delta = (
                rewards[:, t]
                + gamma * next_values[:, 0] * (~dones[:, t])
                - values[:, t]
            )
        else:
            # Regular timestep
            delta = (
                rewards[:, t] + gamma * values[:, t + 1] * (~dones[:, t]) - values[:, t]
            )

        gae = delta + gamma * gae_lambda * (~dones[:, t]) * gae
        advantages[:, t] = gae
        returns[:, t] = gae + values[:, t]

    # Apply mask to advantages and returns
    advantages = advantages * mask
    returns = returns * mask

    # Normalize advantages if requested
    if normalize_advantages:
        # Only normalize over valid (masked) positions
        valid_advantages = advantages[mask.bool()]
        if len(valid_advantages) > 0:
            mean_advantages = valid_advantages.mean()
            std_advantages = valid_advantages.std()
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

    # Normalize returns if requested
    if normalize_returns:
        valid_returns = returns[mask.bool()]
        if len(valid_returns) > 0:
            mean_returns = valid_returns.mean()
            std_returns = valid_returns.std()
            returns = (returns - mean_returns) / (std_returns + 1e-5)

    return advantages, returns


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
        "adv_type": "gae",
    }
    kwargs = preprocess_advantages_inputs(**kwargs)
    advantages, returns = compute_gae_advantages_and_returns(**kwargs)
    kwargs.update({"advantages": advantages, "returns": returns})
    kwargs = postprocess_advantages_outputs(**kwargs)
    advantages, returns = kwargs["advantages"], kwargs.get("returns", None)
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
        "adv_type": "grpo",
    }
    kwargs = preprocess_advantages_inputs(**kwargs)
    kwargs = calculate_scores(**kwargs)
    advantages, returns = compute_grpo_advantages(**kwargs)
    kwargs.update({"advantages": advantages, "returns": returns})
    kwargs = postprocess_advantages_outputs(**kwargs)
    advantages, returns = kwargs["advantages"], kwargs.get("returns", None)
    print(advantages.mean(), advantages.shape, advantages.min(), advantages.max())

    # test grpo for math
    torch.manual_seed(0)
    rewards = torch.randn(4)
    loss_mask = torch.rand(4, 2) > 0.3
    group_size = 2
    advantages, _ = compute_grpo_advantages(
        reward_scores=rewards,
        loss_mask=loss_mask,
        group_size=group_size,
    )
    print(advantages.mean(), advantages.std(), advantages.shape)

    # test reinforce++ for math
    torch.manual_seed(0)
    rewards = torch.randn(4)
    loss_mask = torch.zeros(4, 2).bool()
    loss_mask[:, :-1] = 1
    group_size = 2
    advantages, _ = compute_reinpp_advantages(
        reward_scores=rewards,
        loss_mask=loss_mask,
        group_size=group_size,
        use_reinpp_baseline=True,
    )
    print(advantages.mean(), advantages.std(), advantages.shape)
