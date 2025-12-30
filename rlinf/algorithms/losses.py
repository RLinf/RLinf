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

from typing import Callable, Optional

import torch

from rlinf.algorithms.registry import register_policy_loss
from rlinf.algorithms.utils import huber_loss
from rlinf.utils.utils import masked_mean, masked_mean_ratio


def compute_decoupled_ppo_actor_loss(
    logprobs: torch.Tensor,
    proximal_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    advantages: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    c_clip: Optional[float] = None,
    loss_agg_func: Callable[..., torch.Tensor] = masked_mean,
    max_episode_steps: Optional[int] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    critic_warmup: Optional[bool] = False,
    behave_weight_threshold: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    r"""
    Compute the decoupled PPO actor loss with dual-clip and behavior
    importance weighting.

    This implements a three-policy, decoupled PPO objective:

    - Current policy:        :math:`\pi_\theta`      (``logprobs``)
    - Proximal policy:       :math:`\pi_{\text{prox}}` (``proximal_logprobs``)
    - Behavior policy:      :math:`\pi_{\text{behav}}` (``old_logprobs``)

    All per-token tensors are shaped ``[batch_size, seq_len]``.

    **1. Proximal ratio and PPO clipping**

    For each valid token (masked by ``loss_mask``), define

    .. math::

        r_{\text{prox}} &= \exp(\log \pi_\theta - \log \pi_{\text{prox}}), \\
        \tilde{r}_{\text{prox}} &= \mathrm{clip}\big(
            r_{\text{prox}},
            1 - \text{clip\_ratio\_low},
            1 + \text{clip\_ratio\_high}
        \big).

    where:

    - ``clip_ratio_low``  = :math:`\epsilon_{\text{low}}`
    - ``clip_ratio_high`` = :math:`\epsilon_{\text{high}}`

    The unclipped and clipped PPO losses (per token) are

    .. math::

        \ell_1 &= - A \cdot r_{\text{prox}}, \\
        \ell_2 &= - A \cdot \tilde{r}_{\text{prox}}, \\[2mm]
        \ell_{\text{ppo}} &= \max(\ell_1, \ell_2),

    where :math:`A` is the per-token advantage.

    **2. Dual-clip (optional)**

    If ``c_clip`` is provided (and must satisfy ``c_clip > 1.0``), a
    dual-clip bound is applied to further limit the magnitude of the loss,
    especially for negative advantages and large ratios:

    .. math::

        \ell_{\text{dual}} &= c_{\text{clip}} \cdot |A|
            = c_{\text{clip}} \cdot \big|\text{advantages}\big|, \\[1mm]
        \ell_{\text{final}} &= \min\big(\ell_{\text{ppo}}, \ell_{\text{dual}}\big).

    In code this is implemented as

    .. code-block:: python

        pg_loss1 = -advantages * proximal_ratio
        pg_loss2 = -advantages * clipped_proximal_ratio
        pg_loss  = torch.max(pg_loss1, pg_loss2)       # \ell_{\text{ppo}}

        if c_clip is not None:
            pg_loss3 = torch.sign(advantages) * c_clip * advantages  # c_clip * |A|
            pg_loss  = torch.min(pg_loss, pg_loss3)                  # \ell_{\text{final}}
    When :math:`A > 0`, dual-clip is effectively inactive; when
    :math:`A < 0` and :math:`r_{\text{prox}}` is large, the per-token loss
    is capped by :math:`c_{\text{clip}} \cdot |A|`.

    **3. Behavior importance weights and filtering**

    To decouple the Behavior and proximal policies, an importance weight

    .. math::

        w_{\text{behav}} = \exp(
            \log \pi_{\text{prox}} - \log \pi_{\text{behav}}
        ) = \frac{\pi_{\text{prox}}}{\pi_{\text{behav}}}

    is computed per token:

    .. code-block:: python

        behav_weight = torch.exp(proximal_logprobs - old_logprobs)

    If ``behave_weight_threshold`` is not ``None``, tokens whose
    importance weight exceeds this threshold are **filtered out**:

    .. code-block:: python

        behav_mask = (
            (behav_weight <= behave_weight_threshold).logical_and_(loss_mask)
            if behave_weight_threshold is not None
            else loss_mask
        )
        behav_weight = torch.where(behav_mask, behav_weight, 0.0)

    That is, only tokens with
    :math:`w_{\text{behav}} \le \text{behave\_weight\_threshold}` and
    ``loss_mask == True`` contribute to the loss; other tokens are dropped
    (weight set to 0, mask set to False).

    **4. Aggregation over tokens**

    The per-token policy loss after dual-clip is scaled by the Behavior
    weight and then aggregated with a masked reduction:

    .. math::

        L_{\text{actor}} =
        \text{loss\_agg\_func}\big(
          \ell_{\text{final}} \cdot w_{\text{behav}},
          \text{behav\_mask},
          \text{loss\_mask\_ratio}
        \big).

    The default ``loss_agg_func`` is :func:`masked_mean`, i.e. a masked
    arithmetic mean over valid tokens. If ``max_episode_steps`` and
    ``loss_mask_sum`` are provided, ``loss_agg_func`` is switched to
    :func:`masked_mean_ratio` and

    .. math::

        \text{loss\_mask\_ratio} =
          \frac{\text{loss\_mask\_sum}}{\text{max\_episode\_steps}}

    is passed as an additional argument, typically to reweight episodes by
    their unpadded length.

    **5. Critic warmup**

    If ``critic_warmup=True``, the returned actor loss is forced to zero:

    .. code-block:: python

        if critic_warmup:
            pg_loss = torch.tensor(0.0, device=pg_loss.device)

    This disables policy gradients during a critic warmup phase. In the
    current implementation, the logged ``actor/policy_loss`` metric is
    also zero in this case.

    **6. Reported metrics**

    Besides the scalar loss, this function returns a metrics dictionary
    with the following entries (all ``torch.Tensor`` scalars):

    - ``actor/policy_loss``: final aggregated actor loss (after all
      weighting and warmup handling).
    - ``actor/proximal_ratio``: masked mean of
      :math:`r_{\text{prox}} = \exp(\log \pi_\theta - \log \pi_{\text{prox}})`.
    - ``actor/clipped_proximal_ratio``: masked mean of the clipped ratio
      :math:`\tilde{r}_{\text{prox}}`.
    - ``actor/clip_fraction``: fraction of valid tokens where the PPO clip
      branch is active, i.e. where
      :math:`\ell_2 > \ell_1` (clipped loss chosen).
    - ``actor/dual_clip_fraction``: fraction of valid tokens where the
      dual-clip bound is active, i.e. where
      :math:`\ell_{\text{ppo}} > c_{\text{clip}} \cdot |A|`.
    - ``actor/proximal_approx_kl``: an approximate KL between the proximal
      and current policies, estimated as

      .. math::

          D_{\text{KL}}(\pi_{\text{prox}} \,\|\, \pi_\theta)
          \approx - \mathbb{E}_{(s,a)}[
              \log \pi_\theta(a|s) - \log \pi_{\text{prox}}(a|s)
          ],

      computed as a masked mean over ``(logprobs - proximal_logprobs)``.
    - ``actor/behav_approx_kl``: an approximate KL between the Behavior
      and proximal policies, estimated as

      .. math::

          D_{\text{KL}}(\pi_{\text{behav}} \,\|\, \pi_{\text{prox}})
          \approx - \mathbb{E}_{(s,a)}[
              \log \pi_{\text{prox}}(a|s) - \log \pi_{\text{behav}}(a|s)
          ],

      computed as a masked mean over ``(proximal_logprobs - old_logprobs)``
      using ``behav_mask``.

    Args:
        logprobs (torch.Tensor):
            Log probabilities of the current policy
            :math:`\log \pi_\theta(a|s)`, shape ``[batch_size, seq_len]``.
        proximal_logprobs (torch.Tensor):
            Log probabilities of the proximal policy
            :math:`\log \pi_{\text{prox}}(a|s)`, same shape.
        old_logprobs (torch.Tensor):
            Log probabilities of the Behavior policy
            :math:`\log \pi_{\text{behav}}(a|s)`, same shape.
        clip_ratio_low (float):
            Lower bound :math:`\epsilon_{\text{low}}` for PPO clipping.
            The ratio is clipped to
            :math:`[1 - \epsilon_{\text{low}}, 1 + \text{clip\_ratio\_high}]`.
        clip_ratio_high (float):
            Upper bound :math:`\epsilon_{\text{high}}` for PPO clipping.
        advantages (torch.Tensor):
            Per-token advantages :math:`A`, same shape as ``logprobs``.
        loss_mask (Optional[torch.Tensor]):
            Boolean mask of valid tokens, shape ``[batch_size, seq_len]``.
            Positions with ``False`` are excluded from loss and metrics.
            If ``None``, a full-ones mask is used.
        c_clip (Optional[float]):
            Dual-clip coefficient :math:`c_{\text{clip}} > 1.0`. If
            provided, per-token losses are further bounded by
            :math:`c_{\text{clip}} \cdot |A|`. If ``None``, dual-clip is
            disabled.
        loss_agg_func (Callable):
            Aggregation function taking arguments
            ``(values, mask, loss_mask_ratio)`` and returning a scalar
            loss. Defaults to :func:`masked_mean`. If
            ``max_episode_steps`` and ``loss_mask_sum`` are set, this is
            overridden to :func:`masked_mean_ratio`.
        max_episode_steps (Optional[int]):
            Maximum episode length used to compute ``loss_mask_ratio`` for
            ratio-based aggregation. If ``None``, ``loss_mask_ratio`` is
            left as ``None``.
        loss_mask_sum (Optional[torch.Tensor]):
            Tensor containing the sum of ``loss_mask`` over time for each
            episode, used together with ``max_episode_steps`` to compute
            ``loss_mask_ratio``. Must be broadcastable if provided.
        critic_warmup (Optional[bool]):
            If ``True``, the returned actor loss is set to zero, disabling
            policy gradients (critic warmup phase).
        behave_weight_threshold (Optional[float]):
            Optional upper bound on Behavior importance weights
            :math:`w_{\text{behav}}`. Tokens with
            :math:`w_{\text{behav}} > \text{behave\_weight\_threshold}` are
            filtered out (their mask set to False, weight set to 0).

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            - ``pg_loss``: Scalar actor loss after aggregation and possible
              warmup zeroing. This is the value that should be backpropagated.
            - ``metrics_data``: Dictionary of scalar tensors with logging
              metrics as described above (policy loss, ratios, fractions,
              and approximate KLs).
    """

    loss_mask_ratio = None
    if (
        max_episode_steps is not None
        and loss_mask_sum is not None
        and loss_mask is not None
    ):
        loss_mask_ratio = (loss_mask_sum * 1.0) / max_episode_steps
        loss_agg_func = masked_mean_ratio

    if loss_mask is None:
        loss_mask = torch.ones_like(logprobs).bool()
    loss_mask_count = loss_mask.count_nonzero() or 1

    proximal_ratio = torch.where(
        loss_mask, torch.exp(logprobs - proximal_logprobs), 0.0
    )
    clipped_proximal_ratio = torch.clamp(
        proximal_ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high
    )

    pg_loss1 = -advantages * proximal_ratio
    pg_loss2 = -advantages * clipped_proximal_ratio

    pg_loss = torch.max(pg_loss1, pg_loss2)

    if c_clip is not None:
        assert c_clip > 1.0, f"c_clip should be greater than 1.0, got {c_clip}"
        pg_loss3 = torch.sign(advantages) * c_clip * advantages
        dual_clip_mask = pg_loss3.detach() < pg_loss.detach()
        pg_loss = torch.min(pg_loss, pg_loss3)
    else:
        dual_clip_mask = torch.zeros_like(pg_loss, dtype=torch.bool)

    behav_weight = torch.exp(proximal_logprobs - old_logprobs)
    behav_mask = (
        (behav_weight <= behave_weight_threshold).logical_and_(loss_mask)
        if behave_weight_threshold is not None
        else loss_mask
    )
    behav_mask_count = behav_mask.count_nonzero() or 1
    behav_weight = torch.where(behav_mask, behav_weight, 0.0)
    pg_loss = loss_agg_func(pg_loss * behav_weight, loss_mask, loss_mask_ratio)

    if critic_warmup:
        pg_loss = torch.tensor(0.0, device=pg_loss.device)

    # computing metrics
    with torch.no_grad():
        clip_mask = pg_loss1 < pg_loss2
        clip_fraction = (
            clip_mask.logical_and_(loss_mask).count_nonzero() / loss_mask_count
        )
        dual_clip_fraction = (
            dual_clip_mask.logical_and_(loss_mask).count_nonzero() / loss_mask_count
        )
        proximal_approx_kl = torch.where(loss_mask, logprobs - proximal_logprobs, 0.0)
        proximal_approx_kl = -proximal_approx_kl.sum() / loss_mask_count
        behav_approx_kl = torch.where(behav_mask, proximal_logprobs - old_logprobs, 0.0)
        behav_approx_kl = -behav_approx_kl.sum() / behav_mask_count
        behav_clip_fraction = 1.0 - (behav_mask_count / loss_mask_count)

    metrics_data = {
        "actor/policy_loss": pg_loss.detach(),
        "actor/proximal_ratio": masked_mean(proximal_ratio.detach(), loss_mask),
        "actor/clipped_proximal_ratio": masked_mean(
            clipped_proximal_ratio.detach(), loss_mask
        ),
        "actor/clip_fraction": clip_fraction,
        "actor/dual_clip_fraction": dual_clip_fraction,
        "actor/behav_clip_fraction": behav_clip_fraction,
        "actor/proximal_approx_kl": proximal_approx_kl,
        "actor/behav_approx_kl": behav_approx_kl,
    }
    return pg_loss, metrics_data


def compute_ppo_actor_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    advantages: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    clip_ratio_c: Optional[float] = None,
    loss_agg_func: Callable[..., torch.Tensor] = masked_mean,
    max_episode_steps: Optional[int] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    critic_warmup: Optional[bool] = False,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Compute PPO actor loss function.

    Args:
        logprobs (torch.FloatTensor): Log probabilities of actions.
        old_logprobs (torch.FloatTensor): Old log probabilities of actions.
        clip_ratio_low (float): Lower bound of clipping ratio.
        clip_ratio_high (float): Upper bound of clipping ratio.
        advantages (torch.FloatTensor): GAE (normalized) advantages.
        loss_mask (Optional[torch.BoolTensor], optional): Mask for valid entries. Defaults to None.
        clip_ratio_c (Optional[float], optional): Optional clipping coefficient. Defaults to None.
        loss_agg_func (callable, optional): Aggregation function (e.g., masked_mean). Defaults to None.
        max_episode_steps (Optional[int], optional): Max episode length for normalization. Defaults to None.

    Returns:
        Tuple[torch.Tensor, Dict]: (actor_loss, metrics_dict)
    """

    loss_mask_ratio = None

    if (
        max_episode_steps is not None
        and loss_mask_sum is not None
        and loss_mask is not None
    ):
        loss_mask_ratio = (loss_mask_sum * 1.0) / max_episode_steps
        loss_agg_func = masked_mean_ratio

    if loss_mask is None:
        loss_mask = torch.ones_like(logprobs).bool()

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
    if clip_ratio_c is not None:
        assert clip_ratio_c > 1.0, clip_ratio_c
        policy_loss3 = torch.sign(advantages) * clip_ratio_c * advantages
        dual_clip_mask = policy_loss3.detach() < policy_loss.detach()
        policy_loss = torch.min(policy_loss, policy_loss3)
    else:
        dual_clip_mask = torch.zeros_like(clip_mask)

    policy_loss = loss_agg_func(
        policy_loss, loss_mask, loss_mask_ratio
    )  # default max_episode_steps is None

    clip_mask = policy_loss1.detach() < policy_loss2.detach()
    dual_clip_mask.logical_and_(loss_mask)

    clip_fraction = clip_mask.logical_and_(loss_mask).count_nonzero() / loss_mask_count
    approx_kl = -approx_kl.sum() / loss_mask_count

    dual_cliped_ratio = torch.where(dual_clip_mask, ratio, 0)

    if critic_warmup:
        policy_loss = torch.tensor(0.0, device=policy_loss.device)

    # Compile metrics for logging
    ratio_for_metrics = ratio.detach()
    clipped_ratio_for_metrics = clipped_ratio.detach()
    dual_cliped_ratio_for_metrics = dual_cliped_ratio.detach()
    loss_mask_for_metrics = loss_mask

    # Only broadcast when ratio has action_dim dimension and loss_mask's last dim is 1
    # This handles token_level mode: ratio [bsz, num_chunks, action_dim], loss_mask [bsz, num_chunks, 1]
    if (
        len(ratio.shape) > 2
        and ratio.shape[:-1] == loss_mask.shape[:-1]
        and loss_mask.shape[-1] == 1
        and ratio.shape[-1] > 1
    ):
        # Broadcast loss_mask to match ratio's shape for metrics computation
        loss_mask_for_metrics = loss_mask.expand_as(ratio)

    metrics_data = {
        "actor/policy_loss": policy_loss.detach(),
        "actor/ratio": masked_mean(ratio_for_metrics, loss_mask_for_metrics),
        "actor/clipped_ratio": masked_mean(
            clipped_ratio_for_metrics, loss_mask_for_metrics
        ),
        "actor/dual_cliped_ratio": masked_mean(
            dual_cliped_ratio_for_metrics, loss_mask_for_metrics
        ),
        "actor/approx_kl": approx_kl.detach(),
        "actor/clip_fraction": clip_fraction.detach(),
    }
    return policy_loss, metrics_data


def compute_ppo_critic_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    prev_values: torch.Tensor,
    value_clip: float,
    huber_delta: float,
    loss_mask: Optional[torch.Tensor] = None,
    max_episode_steps: Optional[int] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Compute PPO critic loss function.

    Args:
        values (torch.Tensor): Current value predictions.
        returns (torch.Tensor): Return values.
        prev_values (torch.Tensor): Previous value predictions.
        value_clip (float): Value clipping threshold.
        huber_delta (float): Huber loss delta parameter.

    Returns:
        Tuple[torch.Tensor, Dict]: (critic_loss, metrics_dict)
    """
    loss_mask_ratio = None
    loss_agg_func = masked_mean

    if (
        max_episode_steps is not None
        and loss_mask_sum is not None
        and loss_mask is not None
    ):
        loss_mask_ratio = (loss_mask_sum * 1.0) / max_episode_steps
        loss_agg_func = masked_mean_ratio

    value_pred_clipped = prev_values + (values - prev_values).clamp(
        -value_clip, value_clip
    )  # [bsz, ] | [bsz, chunk-step]

    value_loss_original = huber_loss(
        returns - values, huber_delta
    )  # [bsz, ] | [bsz, chunk-step]
    value_loss_clipped = huber_loss(
        returns - value_pred_clipped, huber_delta
    )  # [bsz, ] | [bsz, chunk-step]
    value_loss = torch.max(value_loss_original, value_loss_clipped)
    value_loss = loss_agg_func(value_loss, loss_mask, loss_mask_ratio)

    value_clip_indicator = (value_pred_clipped - prev_values).abs() > value_clip
    value_clip_ratio = value_clip_indicator.float().mean()

    # explained variance
    if loss_mask is not None:
        masked_returns = returns[loss_mask]
        masked_values = values[loss_mask]
    else:
        masked_returns = returns
        masked_values = values

    var_returns = torch.var(masked_returns)
    if torch.isnan(var_returns) or var_returns == 0:
        explained_variance = torch.tensor(float("nan"), device=returns.device)
    else:
        var_diff = torch.var(masked_returns - masked_values)
        if torch.isnan(var_diff):
            explained_variance = torch.tensor(float("nan"), device=returns.device)
        else:
            explained_variance = 1 - var_diff / var_returns

    # Compile metrics for logging
    metrics_data = {
        "critic/value_loss": value_loss.detach().item(),
        "critic/value_clip_ratio": value_clip_ratio.detach().item(),
        "critic/explained_variance": explained_variance.detach().item(),
    }
    return value_loss, metrics_data


@register_policy_loss("actor_critic")
def compute_ppo_actor_critic_loss(**kwargs) -> tuple[torch.Tensor, dict]:
    """
    Compute PPO actor loss function.

    Args:
        logprobs (torch.Tensor): Log probabilities of actions
        values (torch.Tensor): Current value predictions
        old_log_prob (torch.Tensor): Previous log probabilities
        advantages (torch.Tensor): Advantage values
        returns (torch.Tensor): Return values
        prev_values (torch.Tensor): Previous value predictions
        clip_ratio_low (float): Lower clipping ratio for PPO
        clip_ratio_high (float): Upper clipping ratio for PPO
        value_clip (float): Value clipping threshold
        huber_delta (float): Huber loss delta parameter

    Returns:
        Tuple[torch.Tensor, Dict]: Loss and metrics dictionary
    """
    metrics_data = {}
    actor_loss, actor_metrics_data = compute_decoupled_ppo_actor_loss(**kwargs)
    critic_loss, critic_metrics_data = compute_ppo_critic_loss(**kwargs)

    loss = actor_loss + critic_loss
    metrics_data.update(actor_metrics_data)
    metrics_data.update(critic_metrics_data)

    return loss, metrics_data


@register_policy_loss("actor")
def compute_grpo_actor_loss_fn(**kwargs) -> tuple[torch.Tensor, dict]:
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
    actor_loss, actor_metrics_data = compute_decoupled_ppo_actor_loss(**kwargs)
    metrics_data.update(actor_metrics_data)

    return actor_loss, metrics_data
