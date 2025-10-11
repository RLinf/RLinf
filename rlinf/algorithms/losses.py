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
import torch.nn.functional as F

from rlinf.algorithms.registry import register_policy_loss
from rlinf.algorithms.utils import huber_loss
from rlinf.utils.utils import masked_mean


@register_policy_loss("embodied_ppo")
def compute_embodied_ppo_actor_critic_loss(**kwargs) -> Tuple[torch.Tensor, Dict]:
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
    logprobs = kwargs["logprobs"]
    entropy = kwargs["entropy"]
    values = kwargs["values"]
    old_logprobs = kwargs["old_logprobs"]
    advantages = kwargs["advantages"]
    returns = kwargs["returns"]
    prev_values = kwargs["prev_values"]
    clip_ratio_low = kwargs["clip_ratio_low"]
    clip_ratio_high = kwargs["clip_ratio_high"]
    value_clip = kwargs["value_clip"]
    huber_delta = kwargs["huber_delta"]
    entropy_bonus = kwargs["entropy_bonus"]

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
        "actor/ratio": ratio.mean().detach().item(),
        "critic/value_loss": value_loss.detach().item(),
        "critic/value_clip_ratio": value_clip_ratio.detach().item(),
        "actor/entropy_loss": entropy_loss.detach().item(),
    }

    return loss, metrics_data


@register_policy_loss("embodied_grpo")
def compute_embodied_grpo_actor_loss_fn(**kwargs) -> Tuple[torch.Tensor, Dict]:
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
        loss_mask_sum (Optional[torch.Tensor]): Calculate ratio tensor for normalizing the loss when using a mask
        max_episode_steps (Optional[int]): Maximum episode steps for normalization

    Returns:
        Tuple[torch.Tensor, Dict]: Policy gradient loss and metrics dictionary containing:
            - actor/loss: Total actor loss
            - actor/policy_loss: Policy gradient loss
            - actor/clip_fraction: Fraction of clipped policy gradient loss
            - actor/ppo_kl: Approximate KL divergence
    """
    log_probs = kwargs["logprobs"]
    old_log_prob = kwargs["old_logprobs"]
    advantages = kwargs["advantages"]
    clip_ratio_low = kwargs["clip_ratio_low"]
    clip_ratio_high = kwargs["clip_ratio_high"]
    loss_mask = kwargs.get("loss_mask", None)
    loss_mask_sum = kwargs.get("loss_mask_sum", None)
    max_episode_steps = kwargs.get("max_episode_steps", None)

    loss_mask_ratio = (
        (loss_mask_sum * 1.0) / max_episode_steps if loss_mask is not None else None
    )

    logratio = log_probs - old_log_prob
    ratio = torch.exp(logratio)

    # Compute clipped and unclipped policy gradient losses
    policy_loss = -advantages * ratio
    policy_loss2 = -advantages * torch.clamp(
        ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high
    )

    if loss_mask is not None:
        # Take the maximum of clipped and unclipped losses
        policy_loss = (
            torch.max(policy_loss, policy_loss2) / loss_mask_ratio
        ) * loss_mask
        policy_loss = policy_loss.mean()
        clip_fraction = torch.gt(policy_loss2, policy_loss).float() * loss_mask
        clip_fraction = clip_fraction.mean()
        ppo_kl = (-logratio * loss_mask).mean()
    else:
        # Take the maximum of clipped and unclipped losses
        policy_loss = torch.max(policy_loss, policy_loss2).mean()  # float
        clip_fraction = torch.gt(policy_loss2, policy_loss).float().mean()  # float
        ppo_kl = (-logratio).mean()

    # Compile metrics for logging
    metrics_data = {
        "actor/raw_loss": policy_loss.detach().item(),
        "actor/policy_loss": policy_loss.detach().item(),
        "actor/policy_clipfrac": clip_fraction.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
    }
    return policy_loss, metrics_data


@register_policy_loss("math_ppo_actor")
def compute_math_ppo_actor_loss(**kwargs):
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

    Returns:
        Tuple[torch.Tensor, Dict]: Scalar loss and statistics.
    """
    loss_agg_func = kwargs["loss_agg_func"]
    logprobs = kwargs["logprobs"]
    old_logprobs = kwargs["old_logprobs"]
    eps_clip = kwargs["eps_clip"]
    advantages = kwargs["advantages"]
    loss_mask = kwargs.get("loss_mask", None)
    c_clip = kwargs.get("c_clip", None)

    assert logprobs.dtype == torch.float32
    assert old_logprobs.dtype == torch.float32
    assert advantages.dtype == torch.float32

    assert loss_mask is not None

    loss_mask_count = loss_mask.count_nonzero() or 1
    # For numerical stability.
    ratio = torch.where(loss_mask, torch.exp(logprobs - old_logprobs), 0)
    approx_kl = torch.where(loss_mask, (logprobs - old_logprobs).detach(), 0.0)

    clipped_ratio = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip)
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

    policy_loss = loss_agg_func(policy_loss, loss_mask)

    clip_mask = policy_loss1.detach() < policy_loss2.detach()
    dual_clip_mask.logical_and_(loss_mask)

    clip_fraction = clip_mask.logical_and_(loss_mask).count_nonzero() / loss_mask_count
    approx_kl = approx_kl.sum() / loss_mask_count

    dual_cliped_ratio = torch.where(dual_clip_mask, ratio, 0)

    # Compile metrics for logging
    metrics_data = {
        "policy_loss": masked_mean(policy_loss.detach(), loss_mask),
        "ratio": masked_mean(ratio.detach(), loss_mask),
        "clipped_ratio": masked_mean(clipped_ratio.detach(), loss_mask),
        "dual_cliped_ratio": masked_mean(dual_cliped_ratio.detach(), loss_mask),
        "approx_kl": approx_kl.detach(),
        "clip_fraction": clip_fraction.detach(),
    }
    return policy_loss, metrics_data


@register_policy_loss("embodied_sac_gumbel")
def compute_embodied_sac_gumbel_loss(**kwargs) -> Tuple[torch.Tensor, Dict]:
    """
    Compute SAC loss with Gumbel-Softmax for differentiable discrete actions.
    
    This version uses continuous action probabilities from Gumbel-Softmax
    to enable gradient flow through discrete action sampling.
    
    Args:
        q1_values (torch.Tensor): Q1 network predictions
        q2_values (torch.Tensor): Q2 network predictions
        target_q_values (torch.Tensor): Target Q-values
        action_probs (torch.Tensor): Continuous action probabilities from Gumbel-Softmax [B, action_dim, vocab_size]
        logprobs (torch.Tensor): Current policy log probabilities
        entropy (torch.Tensor): Policy entropy
        rewards (torch.Tensor): Rewards
        dones (torch.Tensor): Done flags
        gamma (float): Discount factor
        alpha (float): Temperature parameter for entropy regularization
        loss_mask (torch.Tensor, optional): Mask for valid loss computation
        
    Returns:
        Tuple[torch.Tensor, Dict]: Total loss and metrics dictionary
    """
    # Extract arguments
    q1_values = kwargs["q1_values"]
    q2_values = kwargs["q2_values"] 
    target_q_values = kwargs["target_q_values"]
    action_probs = kwargs.get("action_probs", None)  # Gumbel-Softmax probabilities
    logprobs = kwargs["logprobs"]
    entropy = kwargs["entropy"]
    rewards = kwargs["rewards"]
    dones = kwargs["dones"]
    gamma = kwargs.get("gamma", 0.99)
    alpha = kwargs.get("alpha", 0.2)
    loss_mask = kwargs.get("loss_mask", None)
    
    # Compute target Q-values for critic loss
    with torch.no_grad():
        next_q_values = target_q_values - alpha * logprobs
        target_q = rewards + gamma * (1 - dones.float()) * next_q_values
    
    # Critic loss (MSE between Q-values and targets)
    q1_loss = F.mse_loss(q1_values, target_q, reduction='none')
    q2_loss = F.mse_loss(q2_values, target_q, reduction='none')
    
    if loss_mask is not None:
        q1_loss = q1_loss * loss_mask
        q2_loss = q2_loss * loss_mask
        critic_loss = (q1_loss.sum() + q2_loss.sum()) / (2 * loss_mask.sum())
    else:
        critic_loss = (q1_loss.mean() + q2_loss.mean()) / 2
    
    # Actor loss with Gumbel-Softmax
    min_q_values = torch.min(q1_values, q2_values)
    
    if action_probs is not None:
        # Use continuous probabilities for gradient flow
        # Expected Q-value under the policy distribution
        # This enables gradients to flow through the discrete sampling
        batch_size, action_dim, vocab_size = action_probs.shape
        
        # For each action dimension, compute expected Q-value
        expected_q_values = []
        for i in range(action_dim):
            # Get action probabilities for this dimension
            probs_i = action_probs[:, i, :]  # [B, vocab_size]
            
            # Compute expected Q-value for this action dimension
            # This is a weighted sum of Q-values by action probabilities
            # Note: This is a simplified approach - in practice, you might need
            # to map action tokens back to their corresponding Q-values
            expected_q_i = (probs_i * min_q_values.unsqueeze(-1)).sum(dim=-1)  # [B]
            expected_q_values.append(expected_q_i)
        
        expected_q = torch.stack(expected_q_values, dim=1).mean(dim=1)  # [B]
        
        # Actor loss: maximize expected Q-value and entropy
        actor_loss = alpha * logprobs - expected_q
    else:
        # Fallback to standard SAC actor loss
        actor_loss = alpha * logprobs - min_q_values
    
    if loss_mask is not None:
        actor_loss = (actor_loss * loss_mask).sum() / loss_mask.sum()
    else:
        actor_loss = actor_loss.mean()
    
    # Total loss
    total_loss = critic_loss + actor_loss
    
    # Compute metrics
    metrics = {
        "sac_gumbel/critic_loss": critic_loss.detach().item(),
        "sac_gumbel/actor_loss": actor_loss.detach().item(),
        "sac_gumbel/q1_mean": q1_values.detach().mean().item(),
        "sac_gumbel/q2_mean": q2_values.detach().mean().item(),
        "sac_gumbel/target_q_mean": target_q.detach().mean().item(),
        "sac_gumbel/entropy_mean": entropy.detach().mean().item(),
        "sac_gumbel/alpha": alpha if isinstance(alpha, float) else alpha.detach().item(),
        "sac_gumbel/logprob_mean": logprobs.detach().mean().item(),
    }
    
    if action_probs is not None:
        metrics["sac_gumbel/action_probs_entropy"] = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean().item()
    
    return total_loss, metrics


@register_policy_loss("embodied_sac")
def compute_embodied_sac_loss(**kwargs) -> Tuple[torch.Tensor, Dict]:
    """
    Compute SAC (Soft Actor-Critic) loss for embodied RL.
    
    Args:
        q1_values (torch.Tensor): Q1 network predictions
        q2_values (torch.Tensor): Q2 network predictions
        target_q_values (torch.Tensor): Target Q-values
        logprobs (torch.Tensor): Current policy log probabilities
        old_logprobs (torch.Tensor): Previous policy log probabilities (for comparison)
        entropy (torch.Tensor): Policy entropy
        rewards (torch.Tensor): Rewards
        dones (torch.Tensor): Done flags
        gamma (float): Discount factor
        tau (float): Soft update coefficient for target networks
        alpha (float): Temperature parameter for entropy regularization
        target_entropy (float): Target entropy for automatic temperature tuning
        loss_mask (torch.Tensor, optional): Mask for valid loss computation
        
    Returns:
        Tuple[torch.Tensor, Dict]: Total loss and metrics dictionary
    """
    # Extract arguments
    q1_values = kwargs["q1_values"]
    q2_values = kwargs["q2_values"] 
    target_q_values = kwargs["target_q_values"]
    logprobs = kwargs["logprobs"]
    entropy = kwargs["entropy"]
    rewards = kwargs["rewards"]
    dones = kwargs["dones"]
    gamma = kwargs.get("gamma", 0.99)
    alpha = kwargs.get("alpha", 0.2)
    target_entropy = kwargs.get("target_entropy", None)
    loss_mask = kwargs.get("loss_mask", None)
    
    # Ensure gamma and alpha are tensors with correct dtype
    if not isinstance(gamma, torch.Tensor):
        gamma = torch.tensor(gamma, dtype=q1_values.dtype, device=q1_values.device)
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.tensor(alpha, dtype=q1_values.dtype, device=q1_values.device)
    
    # Ensure all input tensors have correct dtype
    rewards = rewards.to(dtype=q1_values.dtype)
    logprobs = logprobs.to(dtype=q1_values.dtype)
    dones = dones.to(dtype=q1_values.dtype)
    
    # Debug: print shapes and dtypes
    # print(f"SAC Loss - rewards shape: {rewards.shape}, dtype: {rewards.dtype}")
    # print(f"SAC Loss - target_q_values shape: {target_q_values.shape}, dtype: {target_q_values.dtype}")
    # print(f"SAC Loss - logprobs shape: {logprobs.shape}, dtype: {logprobs.dtype}")
    # print(f"SAC Loss - dones shape: {dones.shape}, dtype: {dones.dtype}")
    # print(f"SAC Loss - alpha shape: {alpha.shape if hasattr(alpha, 'shape') else 'scalar'}, dtype: {alpha.dtype if hasattr(alpha, 'dtype') else type(alpha)}")
    # print(f"SAC Loss - gamma shape: {gamma.shape if hasattr(gamma, 'shape') else 'scalar'}, dtype: {gamma.dtype if hasattr(gamma, 'dtype') else type(gamma)}")
    # (EmbodiedSACFSDPActor pid=1531607) SAC Loss - rewards shape: torch.Size([32]), dtype: torch.float32
    # (EmbodiedSACFSDPActor pid=1531607) SAC Loss - target_q_values shape: torch.Size([32, 1]), dtype: torch.bfloat16
    # (EmbodiedSACFSDPActor pid=1531607) SAC Loss - logprobs shape: torch.Size([32, 1]), dtype: torch.float32
    # (EmbodiedSACFSDPActor pid=1531607) SAC Loss - dones shape: torch.Size([32]), dtype: torch.bool
    # (EmbodiedSACFSDPActor pid=1531607) SAC Loss - alpha shape: torch.Size([]), dtype: torch.bfloat16
    # (EmbodiedSACFSDPActor pid=1531607) SAC Loss - gamma shape: torch.Size([]), dtype: torch.bfloat16
    # (EmbodiedSACFSDPActor pid=1531607) SAC Loss - next_q_values shape: torch.Size([32, 1])
    # Compute target Q-values for critic loss
    with torch.no_grad():
        # Now logprobs should be [batch_size, 1] (entire action sequence probability)
        next_q_values = target_q_values - alpha * logprobs
        print(f"SAC Loss - next_q_values shape: {next_q_values.shape}")
        
        # Ensure rewards and dones match next_q_values shape
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)  # [batch_size] -> [batch_size, 1]
        if dones.dim() == 1:
            dones = dones.unsqueeze(1)  # [batch_size] -> [batch_size, 1]
            
        target_q = rewards + gamma * (1 - dones.to(dtype=gamma.dtype)) * next_q_values
    
    # Critic loss (MSE between Q-values and targets)
    q1_loss = F.mse_loss(q1_values, target_q, reduction='none')
    q2_loss = F.mse_loss(q2_values, target_q, reduction='none')
    
    if loss_mask is not None:
        q1_loss = q1_loss * loss_mask
        q2_loss = q2_loss * loss_mask
        critic_loss = (q1_loss.sum() + q2_loss.sum()) / (2 * loss_mask.sum())
    else:
        critic_loss = (q1_loss.mean() + q2_loss.mean()) / 2
    
    # Actor loss (maximize Q-value and entropy)
    min_q_values = torch.min(q1_values, q2_values)
    actor_loss = alpha * logprobs - min_q_values
    
    if loss_mask is not None:
        actor_loss = (actor_loss * loss_mask).sum() / loss_mask.sum()
    else:
        actor_loss = actor_loss.mean()
    
    # Total loss
    total_loss = critic_loss + actor_loss
    
    # Compute metrics
    metrics = {
        "sac/critic_loss": critic_loss.detach().item(),
        "sac/actor_loss": actor_loss.detach().item(),
        "sac/q1_mean": q1_values.detach().mean().item(),
        "sac/q2_mean": q2_values.detach().mean().item(),
        "sac/target_q_mean": target_q.detach().mean().item(),
        "sac/entropy_mean": entropy.detach().mean().item(),
        "sac/alpha": alpha if isinstance(alpha, float) else alpha.detach().item(),
        "sac/logprob_mean": logprobs.detach().mean().item(),
    }
    
    return total_loss, metrics


@register_policy_loss("embodied_sac_actor")
def compute_embodied_sac_actor_loss(**kwargs) -> Tuple[torch.Tensor, Dict]:
    """
    Compute SAC actor loss only.
    
    Args:
        q1_values (torch.Tensor): Q1 network predictions for current actions
        q2_values (torch.Tensor): Q2 network predictions for current actions
        logprobs (torch.Tensor): Log probabilities of current actions
        entropy (torch.Tensor): Policy entropy
        alpha (torch.Tensor or float): Temperature parameter
        loss_mask (torch.Tensor, optional): Mask for valid loss computation
        
    Returns:
        Tuple[torch.Tensor, Dict]: Actor loss and metrics
    """
    q1_values = kwargs["q1_values"]
    q2_values = kwargs["q2_values"]
    logprobs = kwargs["logprobs"]
    entropy = kwargs["entropy"]
    alpha = kwargs.get("alpha", 0.2)
    loss_mask = kwargs.get("loss_mask", None)
    
    # Use minimum Q-value to reduce overestimation bias
    min_q_values = torch.min(q1_values, q2_values)
    
    # Actor loss: maximize Q-value and entropy
    # Equivalent to minimizing: alpha * log_prob - Q(s,a)
    actor_loss = alpha * logprobs - min_q_values
    
    if loss_mask is not None:
        actor_loss = (actor_loss * loss_mask).sum() / loss_mask.sum()
    else:
        actor_loss = actor_loss.mean()
    
    metrics = {
        "sac/actor_loss": actor_loss.detach().item(),
        "sac/q_mean": min_q_values.detach().mean().item(),
        "sac/entropy_mean": entropy.detach().mean().item(),
        "sac/logprob_mean": logprobs.detach().mean().item(),
        "sac/alpha": alpha if isinstance(alpha, float) else alpha.detach().item(),
    }
    
    return actor_loss, metrics


@register_policy_loss("embodied_sac_critic")
def compute_embodied_sac_critic_loss(**kwargs) -> Tuple[torch.Tensor, Dict]:
    """
    Compute SAC critic loss only.
    
    Args:
        q1_values (torch.Tensor): Q1 network predictions
        q2_values (torch.Tensor): Q2 network predictions
        target_q_values (torch.Tensor): Target Q-values
        rewards (torch.Tensor): Rewards
        dones (torch.Tensor): Done flags
        gamma (float): Discount factor
        loss_mask (torch.Tensor, optional): Mask for valid loss computation
        
    Returns:
        Tuple[torch.Tensor, Dict]: Critic loss and metrics
    """
    q1_values = kwargs["q1_values"]
    q2_values = kwargs["q2_values"]
    target_q_values = kwargs["target_q_values"]
    rewards = kwargs["rewards"]
    dones = kwargs["dones"]
    gamma = kwargs.get("gamma", 0.99)
    loss_mask = kwargs.get("loss_mask", None)
    
    # Compute target Q-values
    with torch.no_grad():
        target_q = rewards + gamma * (1 - dones.float()) * target_q_values
    
    # Critic loss (MSE)
    q1_loss = F.mse_loss(q1_values, target_q, reduction='none')
    q2_loss = F.mse_loss(q2_values, target_q, reduction='none')
    
    if loss_mask is not None:
        q1_loss = q1_loss * loss_mask
        q2_loss = q2_loss * loss_mask
        critic_loss = (q1_loss.sum() + q2_loss.sum()) / (2 * loss_mask.sum())
    else:
        critic_loss = (q1_loss.mean() + q2_loss.mean()) / 2
    
    metrics = {
        "sac/critic_loss": critic_loss.detach().item(),
        "sac/q1_mean": q1_values.detach().mean().item(),
        "sac/q2_mean": q2_values.detach().mean().item(),
        "sac/target_q_mean": target_q.detach().mean().item(),
        "sac/q1_loss": q1_loss.detach().mean().item(),
        "sac/q2_loss": q2_loss.detach().mean().item(),
    }
    
    return critic_loss, metrics


def compute_sac_temperature_loss(
    log_alpha: torch.Tensor,
    logprobs: torch.Tensor,
    target_entropy: float,
    loss_mask: torch.Tensor = None
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute temperature (alpha) loss for automatic entropy tuning in SAC.
    
    Args:
        log_alpha (torch.Tensor): Log of temperature parameter
        logprobs (torch.Tensor): Log probabilities from policy
        target_entropy (float): Target entropy value
        loss_mask (torch.Tensor, optional): Mask for valid loss computation
        
    Returns:
        Tuple[torch.Tensor, Dict]: Temperature loss and metrics
    """
    # Temperature loss: minimize difference between current and target entropy
    entropy_diff = -logprobs - target_entropy
    
    if loss_mask is not None:
        entropy_diff = entropy_diff * loss_mask
        alpha_loss = (log_alpha * entropy_diff.detach()).sum() / loss_mask.sum()
    else:
        alpha_loss = (log_alpha * entropy_diff.detach()).mean()
    
    metrics = {
        "sac/alpha_loss": alpha_loss.detach().item(),
        "sac/log_alpha": log_alpha.detach().item(),
        "sac/alpha": log_alpha.exp().detach().item(),
        "sac/entropy": -logprobs.detach().mean().item(),
        "sac/target_entropy": target_entropy,
    }
    
    return alpha_loss, metrics


if __name__ == "__main__":
    # test math_actor_loss_fn
    torch.manual_seed(0)
    bsz = 4
    max_seqlen = 8
    logprobs = torch.randn(bsz, max_seqlen)
    old_logprobs = logprobs + torch.randn(bsz, max_seqlen) * 0.1
    advantages = torch.randn(bsz, max_seqlen)
    loss_mask = torch.randint(0, 2, (bsz, max_seqlen)).bool()
    eps_clip = 0.2
    kwargs = {
        "logprobs": logprobs,
        "old_logprobs": old_logprobs,
        "advantages": advantages,
        "eps_clip": eps_clip,
        "loss_mask": loss_mask,
        "loss_agg_func": lambda x, mask: (x * mask).sum() / (mask.sum() or 1),
    }
    (
        loss,
        clip_fraction,
        approx_kl,
        ratio,
        clipped_ratio,
        dual_cliped_ratio,
    ) = compute_math_ppo_actor_loss(**kwargs)
    print(f"{loss=}, {clip_fraction=}, {approx_kl=}")
    print(f"{ratio=}")
    print(f"{clipped_ratio=}")
    print(f"{dual_cliped_ratio=}")

    # test grpo_actor_loss_fn
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
        "loss_mask_sum": loss_mask.sum(),
    }
    loss, metrics_data = compute_embodied_grpo_actor_loss_fn(**kwargs)
    print(f"{loss=}, {metrics_data=}")

    # test ppo_actor_critic_loss_fn
    torch.manual_seed(0)
    bsz = 4
    max_seqlen = 8
    logprobs = torch.randn(bsz, max_seqlen)
    old_logprobs = logprobs + torch.randn(bsz, max_seqlen) * 0.1
    advantages = torch.randn(bsz, max_seqlen)
    values = torch.randn(bsz, max_seqlen)
    prev_values = values + torch.randn(bsz, max_seqlen) * 0.1
    returns = values + advantages + torch.randn(bsz, max_seqlen)
    entropy = torch.randn(bsz, max_seqlen)
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
    loss, metrics_data = compute_embodied_ppo_actor_critic_loss(**kwargs)
    print(f"{loss=}, {metrics_data=}")