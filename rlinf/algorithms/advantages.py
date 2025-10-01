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

from typing import Optional, Tuple

import torch

from rlinf.algorithms.registry import register_advantage


@register_advantage("embodied_gae")
def compute_embodied_gae_advantages_and_returns(
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

    loss_mask = kwargs.get("loss_mask", None)
    num_chunk, bsz, chunk_size = rewards.shape
    flattened_rewards = rewards.transpose(1, 2).reshape(num_chunk * chunk_size, -1)
    flattened_values = values.transpose(1, 2).reshape((num_chunk + 1) * chunk_size, -1)
    flattened_values = flattened_values[
        : num_chunk * chunk_size + 1
    ]  # [n_steps+1, bsz]
    flattened_dones = dones.transpose(1, 2).reshape((num_chunk + 1) * chunk_size, -1)[
        -(num_chunk * chunk_size + 1) :
    ]
    flattened_loss_mask = loss_mask.transpose(1, 2).reshape(num_chunk * chunk_size, -1) if loss_mask is not None else None
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
        if flattened_loss_mask is None:
            mean_advantages = flattened_advantages.mean()
            std_advantages = flattened_advantages.std(correction=0)
            flattened_advantages = (flattened_advantages - mean_advantages) / (
                std_advantages + 1e-8
            )
        else:
            mean_advantages = flattened_advantages[flattened_loss_mask].mean()
            std_advantages = flattened_advantages[flattened_loss_mask].std(correction=0)
            if std_advantages < 1e-8 or torch.isinf(std_advantages).any() or torch.isnan(std_advantages).any():
                flattened_advantages = torch.zeros_like(flattened_advantages)
            else:
                flattened_advantages = (flattened_advantages - mean_advantages) / (std_advantages + 1e-8)

    if normalize_returns:
        if flattened_loss_mask is None:
            mean_returns = flattened_returns.mean()
            std_retuns = flattened_returns.std(correction=0)
            flattened_returns = (flattened_returns - mean_returns) / (std_retuns + 1e-5)
        else:
            mean_returns = flattened_returns[flattened_loss_mask].mean()
            std_retuns = flattened_returns[flattened_loss_mask].std(correction=0)
            if std_retuns < 1e-8 or torch.isinf(std_retuns).any() or torch.isnan(std_retuns).any():
                flattened_returns = torch.zeros_like(flattened_returns)
            else:
                flattened_returns = (flattened_returns - mean_returns) / (std_retuns + 1e-8)

    advantages = flattened_advantages.reshape(num_chunk, chunk_size, -1).transpose(1, 2)
    returns = flattened_returns.reshape(num_chunk, chunk_size, -1).transpose(1, 2)

    return advantages, returns


@register_advantage("embodied_grpo")
def compute_embodied_grpo_advantages(
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Group Relative Policy Optimization (GRPO) advantages.
    """

    rewards = kwargs["rewards"]
    dones = kwargs["dones"]
    num_group_envs = kwargs.get("num_group_envs", 1)
    group_size = kwargs.get("group_size", 2)
    normalize_advantages = kwargs.get("normalize_advantages", True)
    loss_mask = kwargs.get("loss_mask", None)
    rollout_epoch = kwargs.get("rollout_epoch", 1)
    epsilon = kwargs.get("epsilon", 1e-6)

    n_chunk_step, actual_bsz, num_action_chunks = rewards.shape
    flattened_rewards = rewards.transpose(1, 2).reshape(
        n_chunk_step * num_action_chunks, -1
    )

    flattened_dones = dones.transpose(1, 2).reshape(
        (n_chunk_step + 1) * num_action_chunks, -1
    )
    flattened_dones = flattened_dones[-(n_chunk_step * num_action_chunks + 1) :]

    # loss mask
    flattened_loss_mask = None
    if loss_mask is not None:
        flattened_loss_mask = loss_mask.transpose(1, 2).reshape(
            n_chunk_step * num_action_chunks, -1
        )

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
        flattened_advantages = (
            flattened_advantages.tile([n_steps, 1]) * flattened_loss_mask
        )
    else:
        flattened_advantages = flattened_advantages.tile([n_steps, 1])

    advantages = flattened_advantages.reshape(
        n_chunk_step, num_action_chunks, actual_bsz
    ).transpose(1, 2)
    return advantages, advantages


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


@register_advantage("math_grpo")
def compute_math_grpo_advantages(**kwargs):
    reward_scores = kwargs["reward_scores"]
    mask = kwargs["mask"]
    num_responses = kwargs["num_responses"]

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

    return advantages, None


if __name__ == "__main__":
    # test compute_ppo_advantages_and_returns
    torch.manual_seed(0)
    rewards = torch.randn(4, 2, 3)
    values = torch.randn(5, 2, 3)
    dones = torch.zeros(5, 2, 3).bool()
    dones[-1] = 1
    advantages, returns = compute_embodied_gae_advantages_and_returns(
        rewards=rewards, values=values, dones=dones, gamma=0.99, gae_lambda=0.95
    )
    print(advantages.mean())
    print(returns.mean())

    # test compute_grpo_advantages_and_returns
    torch.manual_seed(0)
    rewards = torch.randn(4, 4, 3)
    dones = torch.zeros(5, 4, 3).bool()
    loss_mask = torch.rand_like(rewards) > 0.5
    dones[-1] = 1
    advantages, _ = compute_embodied_grpo_advantages(
        rewards=rewards,
        dones=dones,
        loss_mask=loss_mask,
        num_group_envs=2,
        group_size=2,
        normalize_advantages=False,
    )
    print(advantages)
