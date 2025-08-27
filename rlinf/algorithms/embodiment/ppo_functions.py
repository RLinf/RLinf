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

from typing import Dict, Tuple

import torch

from .utils import huber_loss


def compute_advantages_and_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    normalize_advantages: bool = True,
    normalize_returns: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate advantages and returns for Proximal Policy Optimization (PPO).
    NOTE: currently not support auto-reset

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


def actor_critic_loss_fn(
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
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute PPO actor loss function.

    Args:
        logprobs (torch.Tensor): Log probabilities of actions
        entropy (torch.Tensor): Entropy values
        values (torch.Tensor): Current value predictions
        old_log_prob (torch.Tensor): Previous log probabilities
        advantages (torch.Tensor): Advantage values
        returns (torch.Tensor): Return values
        prev_values (torch.Tensor): Previous value predictions
        clip_ratio_low (float): Lower clipping ratio for PPO
        clip_ratio_high (float): Upper clipping ratio for PPO
        value_clip (float): Value clipping threshold
        huber_delta (float): Huber loss delta parameter
        entropy_bonus (float): Entropy bonus coefficient

    Returns:
        Tuple[torch.Tensor, Dict]: Loss and metrics dictionary
    """
    logratio = logprobs - old_logprobs
    ratio = torch.exp(logratio)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    if torch.isnan(policy_loss):
        print("Policy loss is NaN")
        print(f"{logratio=}")
        print(f"{logratio.shape}, {advantages.shape}")
        raise NotImplementedError

    # Value loss
    value_pred_clipped = prev_values + (values - prev_values).clamp(
        -value_clip, value_clip
    )  # [bsz, ] | [bsz, chunk-step]
    error_clipped = returns - value_pred_clipped  # [bsz, ] | [bsz, chunk-step]
    error_original = returns - values  # [bsz, ] | [bsz, chunk-step]
    value_loss_clipped = huber_loss(error_clipped, huber_delta)
    value_loss_original = huber_loss(error_original, huber_delta)
    value_loss = torch.max(value_loss_original, value_loss_clipped)

    value_clip_indicator = (value_pred_clipped - prev_values).abs() > value_clip
    value_clip_ratio = value_clip_indicator.float().mean()

    value_loss = value_loss.mean()

    # Entropy loss
    entropy_loss = entropy.mean()

    loss = policy_loss + value_loss - entropy_bonus * entropy_loss

    # Metrics
    metrics_data = {
        "actor/raw_loss": loss.detach().item(),
        "actor/policy_loss": policy_loss.detach().item(),
        "actor/value_loss": value_loss.detach().item(),
        "actor/ratio": ratio.mean().detach().item(),
        "actor/value_clip_ratio": value_clip_ratio.detach().item(),
        "actor/entropy_loss": entropy_loss.detach().item(),
    }

    return loss, metrics_data


def actor_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    entropy_bonus: float,
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute PPO actor loss function.

    Args:
        logprobs (torch.Tensor): Log probabilities of actions
        entropy (torch.Tensor): Entropy values
        values (torch.Tensor): Current value predictions
        old_log_prob (torch.Tensor): Previous log probabilities
        advantages (torch.Tensor): Advantage values
        returns (torch.Tensor): Return values
        prev_values (torch.Tensor): Previous value predictions
        clip_ratio_low (float): Lower clipping ratio for PPO
        clip_ratio_high (float): Upper clipping ratio for PPO
        value_clip (float): Value clipping threshold
        huber_delta (float): Huber loss delta parameter
        entropy_bonus (float): Entropy bonus coefficient

    Returns:
        Tuple[torch.Tensor, Dict]: Loss and metrics dictionary
    """
    ratio = torch.exp(logprobs - old_logprobs)  # [bsz, ] | [bsz, token-len]
    if len(ratio.shape) == 1:
        ratio = ratio.unsqueeze(-1)

    assert len(advantages.shape) == 2

    surr1 = ratio * advantages  # [bsz, token-len]
    surr2 = (
        torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high) * advantages
    )  # [bsz, token-len]
    policy_loss = -torch.min(surr1, surr2).mean()
    if torch.isnan(policy_loss):
        print("Policy loss is NaN")
        raise NotImplementedError

    # Entropy loss
    entropy_loss = entropy.mean()

    loss = policy_loss - entropy_bonus * entropy_loss

    # Metrics
    metrics_data = {
        "actor/policy_loss": policy_loss.detach().item(),
        "actor/ratio": ratio.mean().detach().item(),
        "actor/entropy_loss": entropy_loss.detach().item(),
    }

    return loss, metrics_data


def critic_loss_fn(
    values: torch.Tensor,
    returns: torch.Tensor,
    prev_values: torch.Tensor,
    value_clip: float,
    huber_delta: float,
):
    # Value loss
    value_pred_clipped = prev_values + (values - prev_values).clamp(
        -value_clip, value_clip
    )  # [bsz, ] | [bsz, chunk-step]
    error_clipped = returns - value_pred_clipped  # [bsz, ] | [bsz, chunk-step]
    error_original = returns - values  # [bsz, ] | [bsz, chunk-step]
    value_loss_clipped = huber_loss(error_clipped, huber_delta)
    value_loss_original = huber_loss(error_original, huber_delta)
    value_loss = torch.max(value_loss_original, value_loss_clipped)

    value_clip_indicator = (value_pred_clipped - prev_values).abs() > value_clip
    value_clip_ratio = value_clip_indicator.float().mean()

    value_loss = value_loss.mean()

    metrics_data = {
        "critic/value_loss": value_loss.detach().item(),
        "critic/value_clip_ratio": value_clip_ratio.detach().item(),
    }
    return value_loss, metrics_data
