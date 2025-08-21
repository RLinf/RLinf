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

from typing import Dict, Optional, Tuple

import torch

from rlinf.utils.utils import masked_sum


def compute_advantages(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    num_group_envs: int,
    group_size: int,
    normalize_advantages: bool = True,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_epoch: int = 1,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Compute advantages for Group Relative Policy Optimization (GRPO).

    This function implements the GRPO algorithm which normalizes rewards within groups
    of responses to the same prompt. The advantages are computed by normalizing
    trajectory returns within each group and then computing cumulative returns.

    Args:
        rewards (torch.Tensor): Reward tensor of shape [n_chunk_step, rollout_epoch * batch_size, token]
        dones (torch.Tensor): Done flag tensor of shape [n_chunk_step + 1, rollout_epoch * batch_size, token]
        num_group_envs (int): Number of rollout samples per group
        group_size (int): Number of responses per prompt
        normalize_advantages (bool): Whether to normalize advantages
        rollout_epoch: the epoch of rollout

    Returns:
        torch.Tensor: Computed advantages tensor
    """
    n_chunk_step, actual_bsz, num_action_chunks = rewards.shape

    flattened_rewards = rewards.transpose(1, 2)
    flattened_rewards = flattened_rewards.reshape(
        n_chunk_step * num_action_chunks, -1
    )  # [n_steps, actual_bsz]

    # process dones
    flattened_dones = dones.transpose(
        1, 2
    )  # [n_chunk_step+1, num_action_chunks, actual_bsz]
    flattened_dones = flattened_dones.reshape(
        (n_chunk_step + 1) * num_action_chunks, -1
    )
    flattened_dones = flattened_dones[
        -(n_chunk_step * num_action_chunks + 1) :
    ]  # [n_steps+1, actual_bsz]

    advantages = torch.zeros_like(flattened_rewards)
    cum_returns = 0
    returns = torch.zeros_like(flattened_rewards)

    # Compute cumulative returns by iterating backwards through the trajectory
    for step in reversed(range(flattened_rewards.shape[0])):
        # Add current reward and continue cumulative returns if not done
        cum_returns = (
            flattened_rewards[step] + (~flattened_dones[step + 1]) * cum_returns
        )
        returns[step] = cum_returns

    # Create mask for first step of each trajectory
    first_step_mask = flattened_dones[
        :-1
    ].clone()  # NOTE: deep copy because we need dones later in actor_loss_fn
    first_step_mask[0] = 1

    # Extract trajectory returns using first step mask
    trajectory_return = returns * first_step_mask

    # Reshape to group responses by prompt:
    # [n_steps, actual_bsz] = [n_steps, (rollout_epoch x num_group_envs x group_size)] reshape to -> [n_steps, rollout_epoch x num_group_envs, group_size]
    trajectory_return = trajectory_return.reshape(
        -1, rollout_epoch * num_group_envs, group_size
    )
    first_step_mask = first_step_mask.reshape(
        -1, rollout_epoch * num_group_envs, group_size
    )
    # Normalize trajectory returns within each group if enabled
    # here we use E(x) to compute std
    cnt = first_step_mask.sum((0, 2), keepdim=True)  # [1, num_group_envs, 1]
    cnt_safe = cnt.clamp_min(1)  # prevent 0

    group_return_mean = (
        trajectory_return.sum(dim=(0, 2), keepdim=True) / cnt_safe
    )  # now shape = [1, rollout_epoch x num_group_envs, 1]

    # here we use E(x^2) to compute std
    group_return_square_mean = (
        trajectory_return.square().sum(dim=(0, 2), keepdim=True) / cnt_safe
    )  # [1, rollout_epoch x num_group_envs, 1]

    # std = sqrt(E(x^2) - E(x)^2)
    group_return_std = (
        (group_return_square_mean - group_return_mean.square())
        .sqrt()
        .clamp_min(epsilon)
    )  # [1, num_group_envs, 1]

    group_mean = trajectory_return - group_return_mean
    norm_trajectory_return = group_mean / group_return_std

    # Reshape back to original format: [n_steps, bsz]
    # [n_steps, num_group_envs, group_size] reshape to -> [n_steps, bsz] = [n_steps, (num_group_envs x group_size)]
    norm_trajectory_return = norm_trajectory_return.reshape(
        -1, rollout_epoch * num_group_envs * group_size
    )
    first_step_mask = first_step_mask.reshape(
        -1, rollout_epoch * num_group_envs * group_size
    )
    # Compute cumulative normalized returns
    for step in range(1, norm_trajectory_return.shape[0]):
        norm_trajectory_return[step] = (
            norm_trajectory_return[step - 1] * ~flattened_dones[step]
            + norm_trajectory_return[step] * flattened_dones[step]
        )

    # Set advantages to the normalized trajectory
    advantages = norm_trajectory_return.reshape(
        n_chunk_step, num_action_chunks, -1
    ).transpose(1, 2)

    return advantages


def compute_advantages_with_loss_mask(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    num_group_envs: int,
    group_size: int,
    normalize_advantages: bool = True,
    loss_mask: torch.Tensor = None,
    rollout_epoch: int = 1,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Compute advantages for Group Relative Policy Optimization (GRPO).

    This function implements the GRPO algorithm which normalizes rewards within groups
    of responses to the same prompt. The advantages are computed by normalizing
    trajectory returns within each group and then computing cumulative returns.

    Args:
        rewards (torch.Tensor): Reward tensor of shape [n_chunk_step, rollout_epoch*batch_size, num_action_chunks(might be reduced)]
        dones (torch.Tensor): Done flag tensor of shape [n_chunk_step+1, rollout_epoch*batch_size, num_action_chunks(might be reduced)]
        num_group_envs (int): Number of rollout samples per group
        group_size (int): Number of responses per prompt
        normalize_advantages (bool): Whether to normalize advantages

    Returns:
        torch.Tensor: Computed advantages tensor
    """

    n_chunk_step, actual_bsz, num_action_chunks = rewards.shape
    flattened_rewards = rewards.transpose(
        1, 2
    )  # [n_chunk_step, num_action_chunks, actual_bsz]
    flattened_rewards = flattened_rewards.reshape(
        n_chunk_step * num_action_chunks, -1
    )  # [n_steps, actual_bsz]

    # process dones
    flattened_dones = dones.transpose(
        1, 2
    )  # [n_chunk_step+1, num_action_chunks, actual_bsz]
    flattened_dones = flattened_dones.reshape(
        (n_chunk_step + 1) * num_action_chunks, -1
    )
    flattened_dones = flattened_dones[
        -(n_chunk_step * num_action_chunks + 1) :
    ]  # [n_steps+1, actual_bsz]

    # process mask
    flattened_loss_mask = loss_mask.transpose(
        1, 2
    )  # [n_chunk_step, num_action_chunks, actual_bsz]
    flattened_loss_mask = flattened_loss_mask.reshape(
        n_chunk_step * num_action_chunks, -1
    )

    n_steps = flattened_rewards.shape[0]
    # culmulate rewards as scores
    scores = torch.zeros(actual_bsz)  # [actual_bsz,]
    for step in reversed(range(n_steps)):
        scores = scores * ~flattened_dones[step + 1]
        scores += flattened_rewards[step]

    # split into groups
    if normalize_advantages:
        scores = scores.reshape(rollout_epoch * num_group_envs, group_size)

        # calculate the mean and std within group
        scores_mean = torch.mean(scores, dim=-1, keepdim=True)
        scores_std = torch.std(scores, dim=-1, keepdim=True)

        # calculate the adv
        flattened_advantages = (scores - scores_mean) / (
            scores_std + epsilon
        )  # [rollout_epoch*num_group_envs, group_size]

        # reshape the advantages and take mask
        flattened_advantages = flattened_advantages.reshape(
            1, -1
        )  # [1, rollout_epoch*num_group_envs*group_size]
    else:
        flattened_advantages = scores.reshape(1, -1)

    flattened_advantages = (
        flattened_advantages.tile([n_steps, 1]) * flattened_loss_mask
    )  # [n_steps, rollout_epoch*num_group_envs*group_size]

    # reshape for train
    advantages = flattened_advantages.reshape(
        n_chunk_step, num_action_chunks, actual_bsz
    )
    advantages = advantages.transpose(1, 2)

    return advantages


def actor_loss_fn(
    log_probs: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio_high: float,
    clip_ratio_low: float,
    loss_mask: Optional[torch.Tensor] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute actor loss for Group Relative Policy Optimization (GRPO).

    This function implements the PPO-style actor loss with clipping for GRPO.
    Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppotrainer.py#L1122

    Args:
        log_prob (torch.Tensor): Current log probabilities of shape (bs,) or (bs, action-token-length)
        old_log_prob (torch.Tensor): Previous log probabilities (will be repeated to match current shape)
        advantages (torch.Tensor): Advantage values of shape (bs,)
        clip_ratio_high (float): Upper clipping ratio for PPO
        clip_ratio_low (float): Lower clipping ratio for PPO

    Returns:
        Tuple[torch.Tensor, Dict]: Policy gradient loss and metrics dictionary containing:
            - actor/loss: Total actor loss
            - actor/pg_loss: Policy gradient loss
            - actor/pg_clipfrac: Fraction of clipped policy gradient loss
            - actor/ppo_kl: Approximate KL divergence
    """
    # Compute approximate KL divergence
    negative_approx_kl = log_probs - old_log_prob  # [bs, ] | [bs, len]

    # Compute probability ratio for PPO clipping
    ratio = torch.exp(negative_approx_kl)  # [bs, ] | [bs, len]
    if len(ratio.shape) == 1:
        ratio = ratio.unsqueeze(-1)

    # Compute clipped and unclipped policy gradient losses
    pg_losses = -advantages * ratio  # [bs, len]
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high
    )  # [bs, len]

    if loss_mask is not None:
        # Take the maximum of clipped and unclipped losses
        pg_loss = masked_sum(
            torch.max(pg_losses, pg_losses2) / loss_mask_sum, loss_mask
        )  # float
        pg_clipfrac = masked_sum(
            torch.gt(pg_losses2, pg_losses).float() / loss_mask_sum, loss_mask
        )  # float
    else:
        # Take the maximum of clipped and unclipped losses
        pg_loss = torch.max(pg_losses, pg_losses2).mean()  # float
        pg_clipfrac = torch.gt(pg_losses2, pg_losses).float().mean()  # float

    # Compile metrics for logging
    metrics_data = {
        "actor/loss": pg_loss.detach().item(),
        "actor/pg_loss": pg_loss.detach().item(),
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
    }
    return pg_loss, metrics_data
