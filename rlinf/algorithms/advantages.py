# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0

from typing import Callable, Dict, Optional, Tuple
import torch
from rlinf.algorithms.registry import register_advantage

@register_advantage("ppo")
def compute_advantages_and_returns(**kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
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

    num_chunk, bsz, chunk_size = rewards.shape
    flattened_rewards = rewards.transpose(1, 2).reshape(num_chunk * chunk_size, -1)
    flattened_values = values.transpose(1, 2).reshape((num_chunk + 1) * chunk_size, -1)
    flattened_values = flattened_values[
        : num_chunk * chunk_size + 1
    ]  # [n_steps+1, bsz]
    flattened_dones = dones.transpose(1, 2).reshape((num_chunk + 1) * chunk_size, -1)[
        -(num_chunk * chunk_size + 1) :
    ]

    flattened_returns = torch.zeros_like(flattened_rewards)

    gae = 0
    for step in reversed(range(flattened_rewards.shape[0])):
        vt1 = flattened_values[step + 1]
        vt = flattened_values[step]

        delta = (
            flattened_rewards[step] + gamma * vt1 * (~flattened_dones[step + 1]) - vt
        )
        gae = delta + gamma * gae_lambda * (~flattened_dones[step + 1]) * gae
        flattened_returns[step] = gae + vt

    # calc adv
    flattened_advantages = flattened_returns - flattened_values[:-1]

    if normalize_advantages:
        mean_advantages = flattened_advantages.mean()
        std_advantages = flattened_advantages.std(correction=0)
        flattened_advantages = (flattened_advantages - mean_advantages) / (
            std_advantages + 1e-5
        )
    if normalize_returns:
        mean_returns = flattened_returns.mean()
        std_retuns = flattened_returns.std(correction=0)
        flattened_returns = (flattened_returns - mean_returns) / (std_retuns + 1e-5)

    advantages = flattened_advantages.reshape(num_chunk, chunk_size, -1).transpose(1, 2)
    returns = flattened_returns.reshape(num_chunk, chunk_size, -1).transpose(1, 2)

    return advantages, returns

@register_advantage("grpo")
def compute_grpo_advantages_and_returns(**kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Group Relative Policy Optimization (GRPO) advantages.
    Returns is set to `None` since GRPO only needs normalized group rewards.
    """

    rewards = kwargs["rewards"]
    dones = kwargs["dones"]
    num_group_envs = kwargs.get("num_group_envs", 1)
    group_size = kwargs.get("group_size", 2)
    normalize_advantages = kwargs.get("normalize_advantages", True)
    loss_mask = kwargs.get("loss_mask", None)
    rollout_epoch = kwargs.get("rollout_epoch", 1)
    epsilon = kwargs.get("epsilon", 1e-6)

    # if loss_mask is None:
    #     loss_mask = torch.ones_like(rewards)

    n_chunk_step, actual_bsz, num_action_chunks = rewards.shape
    flattened_rewards = rewards.transpose(1, 2).reshape(n_chunk_step * num_action_chunks, -1)

    flattened_dones = dones.transpose(1, 2).reshape((n_chunk_step + 1) * num_action_chunks, -1)
    flattened_dones = flattened_dones[-(n_chunk_step * num_action_chunks + 1) :]

    # loss mask
    flattened_loss_mask = None
    if loss_mask is not None:
        flattened_loss_mask = loss_mask.transpose(1, 2).reshape(n_chunk_step * num_action_chunks, -1)

    n_steps = flattened_rewards.shape[0]
    scores = torch.zeros(actual_bsz)
    for step in reversed(range(n_steps)):
        scores = scores * ~flattened_dones[step + 1]
        scores += flattened_rewards[step]

    if normalize_advantages:
        scores = scores.reshape(rollout_epoch * num_group_envs, group_size)
        mean, std = scores.mean(dim=-1, keepdim=True), scores.std(dim=-1, keepdim=True)
        flattened_advantages = (scores - mean) / (std + epsilon)
        flattened_advantages = flattened_advantages.reshape(1, -1)
    else:
        flattened_advantages = scores.reshape(1, -1)

    if flattened_loss_mask is not None:
        flattened_advantages = flattened_advantages.tile([n_steps, 1]) * flattened_loss_mask
    else:
        flattened_advantages = flattened_advantages.tile([n_steps, 1])

    advantages = flattened_advantages.reshape(n_chunk_step, num_action_chunks, actual_bsz).transpose(1, 2)
    return advantages, advantages

@register_advantage("grpo-math")
def compute_grpo_advantages(**kwargs):
    reward_scores, mask, num_responses = kwargs["reward_scores"], kwargs["mask"], kwargs["num_responses"]

    grouped_rewards = reward_scores.view(-1, num_responses)
    # compute median
    grouped_reward_mean = grouped_rewards.mean(dim=1).repeat_interleave(
        num_responses, dim=0
    )
    grouped_reward_std = grouped_rewards.std(dim=1).repeat_interleave(
        num_responses, dim=0
    )

    advantages = reward_scores - grouped_reward_mean
    advantages = advantages / (grouped_reward_std + 1e-6)
    device = mask.device
    advantages = advantages.to(device)

    advantages = (torch.zeros_like(mask) + advantages.view(-1, 1)) * mask

    return advantages
