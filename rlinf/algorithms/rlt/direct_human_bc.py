"""Direct, observation-aligned human BC and chunk-level execution validity."""

from __future__ import annotations

import torch


def execution_weight(batch: dict, record_weight: torch.Tensor | None) -> torch.Tensor:
    """Require actual actions for every real step; allow terminal padding only.

    Q predicts the return of a whole chunk. One unknown or out-of-range action
    invalidates that Q sample, but does not discard its other human BC steps.
    Missing masks indicate an older replay format; never guess their values.
    """
    fi = batch["forward_inputs"]
    available = fi["exec_action_available"].bool()
    available = available.reshape(available.shape[0], -1)
    padding = fi["transition_padding"].bool().reshape_as(available).to(available.device)
    terminal = (
        batch["dones"]
        .reshape(available.shape[0], -1)
        .bool()
        .any(-1, keepdim=True)
        .to(available.device)
    )
    # Padding must be a suffix of a terminal chunk, never an interior gap.
    suffix = ~(padding[:, :-1] & ~padding[:, 1:]).any(-1, keepdim=True)
    valid_padding = (~padding.any(-1, keepdim=True) | terminal) & suffix
    weight = (
        (available | padding).all(-1, keepdim=True)
        & (~padding).any(-1, keepdim=True)
        & valid_padding
    ).float()
    if record_weight is not None:
        weight = weight * record_weight.to(weight.device)
    return weight


def direct_bc_loss(
    pi: torch.Tensor,
    reference: torch.Tensor,
    forward_inputs: dict,
    record_weight: torch.Tensor | None = None,
    human_weight: float = 1.0,
    skip_auto: torch.Tensor | None = None,
    human_within: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fit bounded human corrections, retaining VLA BC on executed AUTO steps.

    Args:
        pi: Predicted normalized actions [B, C, D].
        reference: Original frozen VLA reference [B, C, D].
        forward_inputs: Replay action targets and per-step availability masks.
        record_weight: Optional per-chunk recording mask [B, 1].
        human_weight: Relative weight of valid human steps.
        skip_auto: Optional [B, 1] chunks whose AUTO steps are no BC target
            (their VLA reference was penalized); human steps still count.
        human_within: Optional [D] residual limits (RL units). A human step is a
            BC target only if its target lies within them of the clamped VLA
            reference on every dimension; larger corrections stay out of the
            RL update (kept in the replay, e.g. for the next SFT).
    """
    b, c, _ = pi.shape
    fi = forward_inputs
    kind = fi["iv_kind"].reshape(b, c).to(pi.device)
    padding = fi["transition_padding"].reshape(b, c).to(pi.device).bool()
    human = (
        (kind == 3)
        & ~padding
        & fi["human_bc_available"].reshape(b, c).to(pi.device).bool()
    )
    auto = (
        (kind == 0)
        & ~padding
        & fi["exec_action_available"].reshape(b, c).to(pi.device).bool()
    )
    if record_weight is not None:
        recorded = record_weight.reshape(b, 1).to(pi.device).bool()
        human, auto = human & recorded, auto & recorded
    if skip_auto is not None:
        auto = auto & ~skip_auto.reshape(b, 1).to(pi.device).bool()
    human_target = fi["human_bc_actions"].reshape_as(pi).to(pi.device, pi.dtype)
    human_all = human
    if human_within is not None:
        dev = (human_target - reference.clamp(-0.999, 0.999)).abs()       # ref_tilde, the residual's zero point
        human = human & (dev <= human_within.to(dev)).all(-1)
    target = torch.where(
        human[..., None], human_target, reference.clamp(-1, 1)
    ).detach()
    weight = auto.to(pi.dtype) + human.to(pi.dtype) * human_weight
    error = (pi - target).square().mean(-1)
    loss = (error * weight).sum() / weight.sum().clamp_min(1.0)

    def masked_error(values, mask):
        return ((values * mask).sum() / mask.sum().clamp_min(1)).detach().item()

    return loss, {
        "bc_loss": loss.detach().item(),
        "bc_human_loss": masked_error(error, human),
        "bc_ref_loss": masked_error(error, auto),
        "bc_human_vla_baseline": masked_error(
            (reference - target).square().mean(-1), human
        ),
        "bc_human_small_ratio": (human.sum() / human_all.sum().clamp_min(1)).detach().item(),
        "human_bc_steps": float(human.sum().item()),
        "human_mask_ratio": human.float().mean().item(),
        "policy_mask_ratio": auto.float().mean().item(),
        "bc_valid_ratio": (human | auto).float().mean().item(),
    }
