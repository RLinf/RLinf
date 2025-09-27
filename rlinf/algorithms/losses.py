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

from rlinf.algorithms.registry import register_policy_loss
from rlinf.algorithms.utils import huber_loss
from rlinf.utils.utils import masked_mean, masked_mean_ratio, raw_mean


def compute_ppo_actor_loss(**kwargs) -> Tuple[torch.Tensor, Dict]:
    """
    Compute PPO actor loss function.

    There is no shape requirements for the inputs, but they must have the same shape.
    Either [bs, max_seqlen] for batch padded inputs or [tot_seqlen] for padded inputs.

    Args:
        logprobs (torch.FloatTensor): Log probabilities of actions.
        old_logprobs (torch.FloatTensor): Old log probabilities of actions.
        advantages (torch.FloatTensor): GAE (normalized) advantages.
        eps_clip (float): Clip ratio of PPO.
        loss_mask (Optional[torch.BoolTensor], optional): Mask for loss computation.
            1 if valid else 0. Defaults to None.
        max_episode_steps (Optional[int]): Maximum episode steps for normalization

    Returns:
        Tuple[torch.Tensor, Dict]: Scalar loss and statistics.
    """

    logprobs = kwargs["logprobs"]
    old_logprobs = kwargs["old_logprobs"]
    clip_ratio_low = kwargs["clip_ratio_low"]
    clip_ratio_high = kwargs["clip_ratio_high"]
    advantages = kwargs["advantages"]
    loss_mask = kwargs.get("loss_mask", None)
    c_clip = kwargs.get("c_clip", None)
    loss_agg_func = kwargs.get("loss_agg_func", masked_mean)
    max_episode_steps = kwargs.get("max_episode_steps", None)

    if max_episode_steps is not None and loss_mask is not None:
        loss_agg_func = masked_mean_ratio

    if loss_mask is None:
        loss_mask = torch.ones_like(logprobs).bool()
        loss_agg_func = raw_mean

    assert logprobs.dtype == torch.float32
    assert old_logprobs.dtype == torch.float32
    assert advantages.dtype == torch.float32

    loss_mask_count = loss_mask.count_nonzero() or 1
    # For numerical stability.
    ratio = torch.where(loss_mask, torch.exp(logprobs - old_logprobs), 0)
    approx_kl = torch.where(loss_mask, (logprobs - old_logprobs).detach(), 0.0)

    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    policy_loss1 = -advantages * ratio
    policy_loss2 = -advantages * clipped_ratio

    clip_mask = policy_loss1.detach() < policy_loss2.detach()

    policy_loss = torch.max(policy_loss1, policy_loss2)
    if c_clip is not None:
        assert c_clip > 1.0, c_clip
        policy_loss3 = torch.sign(advantages) * c_clip * advantages
        dual_clip_mask = policy_loss3.detach() < policy_loss.detach()
        policy_loss = torch.min(policy_loss, policy_loss3)
    else:
        dual_clip_mask = torch.zeros_like(clip_mask)

    policy_loss = loss_agg_func(
        policy_loss, loss_mask, max_episode_steps
    )  # default max_episode_steps is None

    clip_mask = policy_loss1.detach() < policy_loss2.detach()
    dual_clip_mask.logical_and_(loss_mask)

    clip_fraction = clip_mask.logical_and_(loss_mask).count_nonzero() / loss_mask_count
    approx_kl = -approx_kl.sum() / loss_mask_count

    dual_cliped_ratio = torch.where(dual_clip_mask, ratio, 0)

    # Compile metrics for logging
    metrics_data = {
        "actor/policy_loss": loss_agg_func(policy_loss.detach(), loss_mask),
        "actor/ratio": loss_agg_func(ratio.detach(), loss_mask),
        "actor/clipped_ratio": loss_agg_func(clipped_ratio.detach(), loss_mask),
        "actor/dual_cliped_ratio": loss_agg_func(dual_cliped_ratio.detach(), loss_mask),
        "actor/approx_kl": approx_kl.detach(),
        "actor/clip_fraction": clip_fraction.detach(),
    }
    return policy_loss, metrics_data


def compute_ppo_critic_loss(**kwargs) -> Tuple[torch.Tensor, Dict]:
    """
    Compute PPO critic loss function.

    Args:
        values (torch.Tensor): Current value predictions
        advantages (torch.Tensor): Advantage values
        returns (torch.Tensor): Return values
        prev_values (torch.Tensor): Previous value predictions
        value_clip (float): Value clipping threshold
        huber_delta (float): Huber loss delta parameter

    Returns:
        Tuple[torch.Tensor, Dict]: Loss and metrics dictionary
    """

    values = kwargs["values"]
    returns = kwargs["returns"]
    prev_values = kwargs["prev_values"]
    value_clip = kwargs["value_clip"]
    huber_delta = kwargs["huber_delta"]

    value_pred_clipped = prev_values + (values - prev_values).clamp(
        -value_clip, value_clip
    )

    value_loss_original = huber_loss(returns - values, huber_delta)
    value_loss_clipped = huber_loss(returns - value_pred_clipped, huber_delta)
    value_loss = torch.max(value_loss_original, value_loss_clipped).mean()

    value_clip_indicator = (value_pred_clipped - prev_values).abs() > value_clip
    value_clip_ratio = value_clip_indicator.float().mean()

    # Compile metrics for logging
    metrics_data = {
        "critic/value_loss": value_loss.detach().item(),
        "critic/value_clip_ratio": value_clip_ratio.detach().item(),
    }
    return value_loss, metrics_data


@register_policy_loss("ppo")
def compute_ppo_actor_critic_loss(**kwargs) -> Tuple[torch.Tensor, Dict]:
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
    metrics_data = {}
    actor_loss, actor_metrics_data = compute_ppo_actor_loss(**kwargs)
    critic_loss, critic_metrics_data = compute_ppo_critic_loss(**kwargs)

    # Entropy loss
    entropy = kwargs["entropy"]
    entropy_bonus = kwargs["entropy_bonus"]
    entropy_loss = entropy.mean()

    loss = actor_loss + critic_loss - entropy_bonus * entropy_loss
    metrics_data.update(actor_metrics_data)
    metrics_data.update(critic_metrics_data)
    metrics_data.update(
        {
            "actor/raw_loss": loss.detach().item(),
        }
    )

    return loss, metrics_data


@register_policy_loss("grpo")
def compute_grpo_actor_loss_fn(**kwargs) -> Tuple[torch.Tensor, Dict]:
    """
    Compute actor loss for Group Relative Policy Optimization (GRPO).

    This function implements the PPO-style actor loss with clipping for GRPO.
    Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppotrainer.py#L1122

    Args:
        log_prob (torch.Tensor): Current log probabilities
        old_log_prob (torch.Tensor): Previous log probabilities
        advantages (torch.Tensor): Advantage values of shape
        clip_ratio_high (float): Upper clipping ratio for PPO
        clip_ratio_low (float): Lower clipping ratio for PPO
        loss_mask (Optional[torch.Tensor]): Mask tensor of shape to apply to the loss

    Returns:
        Tuple[torch.Tensor, Dict]: Policy gradient loss and metrics dictionary containing:
            - actor/loss: Total actor loss
            - actor/policy_loss: Policy gradient loss
            - actor/clip_fraction: Fraction of clipped policy gradient loss
            - actor/ppo_kl: Approximate KL divergence
    """
    metrics_data = {}
    actor_loss, actor_metrics_data = compute_ppo_actor_loss(**kwargs)
    metrics_data.update(actor_metrics_data)

    return actor_loss, metrics_data


if __name__ == "__main__":
    # test grpo for math
    torch.manual_seed(0)
    bsz = 4
    max_seqlen = 8
    logprobs = torch.randn(bsz, max_seqlen)
    old_logprobs = logprobs + torch.randn(bsz, max_seqlen) * 0.1
    advantages = torch.randn(bsz, max_seqlen)
    loss_mask = torch.randint(0, 2, (bsz, max_seqlen)).bool()
    clip_ratio_low = 0.2
    clip_ratio_high = 0.2
    kwargs = {
        "logprobs": logprobs,
        "old_logprobs": old_logprobs,
        "advantages": advantages,
        "clip_ratio_low": clip_ratio_low,
        "clip_ratio_high": clip_ratio_high,
        "loss_mask": loss_mask,
        "loss_agg_func": masked_mean,
    }
    loss, metrics_data = compute_grpo_actor_loss_fn(**kwargs)
    print(f"{loss=}, {metrics_data=}")

    # test grpo for embodiment
    torch.manual_seed(0)
    bsz = 4
    max_seqlen = 8
    logprobs = torch.randn(5, bsz, max_seqlen)
    old_logprobs = logprobs + torch.randn(5, bsz, max_seqlen) * 0.1
    advantages = torch.randn(5, bsz, max_seqlen)
    loss_mask = torch.randint(0, 2, (5, bsz, max_seqlen)).bool()
    clip_ratio_low = 0.2
    clip_ratio_high = 0.2
    max_episode_steps = 30
    kwargs = {
        "logprobs": logprobs,
        "old_logprobs": old_logprobs,
        "advantages": advantages,
        "clip_ratio_low": clip_ratio_low,
        "clip_ratio_high": clip_ratio_high,
        "loss_mask": loss_mask,
        "max_episode_steps": max_episode_steps,
    }
    loss, metrics_data = compute_grpo_actor_loss_fn(**kwargs)
    print(f"{loss=}, {metrics_data=}")

    # test ppo for embodiment
    torch.manual_seed(0)
    bsz = 4
    max_seqlen = 8
    values = torch.randn(5, bsz, max_seqlen)
    prev_values = values + torch.randn(5, bsz, max_seqlen) * 0.1
    returns = values + advantages + torch.randn(5, bsz, max_seqlen)
    entropy = torch.randn(5, bsz, max_seqlen)
    clip_ratio_low = 0.2
    clip_ratio_high = 0.2
    value_clip = 0.2
    huber_delta = 1.0
    entropy_bonus = 0.01
    kwargs = {
        "logprobs": logprobs,
        "old_logprobs": old_logprobs,
        "advantages": advantages,
        "values": values,
        "prev_values": prev_values,
        "returns": returns,
        "entropy": entropy,
        "clip_ratio_low": clip_ratio_low,
        "clip_ratio_high": clip_ratio_high,
        "value_clip": value_clip,
        "huber_delta": huber_delta,
        "entropy_bonus": entropy_bonus,
    }
    loss, metrics_data = compute_ppo_actor_critic_loss(**kwargs)
    print(f"{loss=}, {metrics_data=}")
