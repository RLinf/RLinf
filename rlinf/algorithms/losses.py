# losses.py
import torch
from typing import Optional, Dict, Tuple, Callable
from rlinf.algorithms.utils import huber_loss
from rlinf.utils.utils import masked_sum
from rlinf.algorithms.registry import register_policy_loss

@register_policy_loss("ppo_actor_critic")
def actor_critic_loss_fn(**kwargs) -> Tuple[torch.Tensor, Dict]:
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
        "actor/value_loss": value_loss.detach().item(),
        "actor/ratio": ratio.mean().detach().item(),
        "actor/value_clip_ratio": value_clip_ratio.detach().item(),
        "actor/entropy_loss": entropy_loss.detach().item(),
    }

    return loss, metrics_data


@register_policy_loss("grpo_actor")
def actor_loss_fn(**kwargs) -> Tuple[torch.Tensor, Dict]:
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
    log_probs = kwargs["logprobs"]
    loss_mask = kwargs.get("loss_mask", None)
    loss_mask_sum = kwargs.get("loss_mask_sum", None)
    old_log_prob = kwargs["old_logprobs"]
    advantages = kwargs["advantages"]
    clip_ratio_low = kwargs["clip_ratio_low"]
    clip_ratio_high = kwargs["clip_ratio_high"]

    bsz = log_probs.shape[0]
    logratio = log_probs - old_log_prob
    ratio = torch.exp(logratio)

    # Compute clipped and unclipped policy gradient losses
    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high
    )

    if loss_mask is not None:
        # Take the maximum of clipped and unclipped losses
        pg_loss = (
            masked_sum(torch.max(pg_losses, pg_losses2) / loss_mask_sum, loss_mask)
            / bsz
        )  # float
        pg_clipfrac = (
            masked_sum(
                torch.gt(pg_losses2, pg_losses).float() / loss_mask_sum, loss_mask
            )
            / bsz
        )  # float
    else:
        # Take the maximum of clipped and unclipped losses
        pg_loss = torch.max(pg_losses, pg_losses2).mean()  # float
        pg_clipfrac = torch.gt(pg_losses2, pg_losses).float().mean()  # float

    # Compile metrics for logging
    metrics_data = {
        "actor/raw_loss": pg_loss.detach().item(),
        "actor/policy_loss": pg_loss.detach().item(),
        "actor/policy_clipfrac": pg_clipfrac.detach().item(),
    }
    return pg_loss, metrics_data

@register_policy_loss("math_actor")
def math_actor_loss_fn(**kwargs):
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
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio

    clip_mask = pg_loss1.detach() < pg_loss2.detach()

    pg_loss = torch.max(pg_loss1, pg_loss2)
    if c_clip is not None:
        assert c_clip > 1.0, c_clip
        pg_loss3 = torch.sign(advantages) * c_clip * advantages
        dual_clip_mask = pg_loss3.detach() < pg_loss.detach()
        pg_loss = torch.min(pg_loss, pg_loss3)
    else:
        dual_clip_mask = torch.zeros_like(clip_mask)

    pg_loss = loss_agg_func(pg_loss, loss_mask)

    clip_mask = pg_loss1.detach() < pg_loss2.detach()
    dual_clip_mask.logical_and_(loss_mask)

    proportion_clipped = (
        clip_mask.logical_and_(loss_mask).count_nonzero() / loss_mask_count
    )
    approx_kl = approx_kl.sum() / loss_mask_count

    dual_cliped_ratio = torch.where(dual_clip_mask, ratio, 0)

    return (
        pg_loss,
        proportion_clipped,
        approx_kl,
        ratio,
        clipped_ratio,
        dual_cliped_ratio,
    )
