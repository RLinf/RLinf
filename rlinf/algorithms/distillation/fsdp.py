# Copyright 2026 The RLinf Authors.
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

"""Differentiable distribution losses on the model output vocabulary."""

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import DISTRIBUTION_MODES, DistillationLossConfig


@dataclass
class DistillationDistributionOutput:
    """Per-token objectives and probability-mass diagnostics."""

    distribution_losses: Tensor
    student_mass: Tensor
    teacher_mass: Tensor
    tail_loss: Tensor


def compute_distribution_loss(
    logits: Tensor,
    teacher_ids: Tensor | None,
    teacher_logprobs: Tensor,
    config: DistillationLossConfig,
    *,
    valid_mask: Tensor | None = None,
) -> DistillationDistributionOutput:
    """Compute sparse or full KL over the complete model output vocabulary.

    Args:
        logits: Student logits with vocabulary in the last dimension.
        teacher_ids: Fixed support IDs, or None for ordered full targets.
        teacher_logprobs: Frozen probabilities on the selected support.
        config: Direction, tail and clamp settings.
        valid_mask: Trainable positions, excluding padding and context.

    Returns:
        Loss and probability masses with the vocabulary dimension reduced.
    """
    mode = config.canonical_loss_mode
    if mode not in DISTRIBUTION_MODES:
        raise ValueError(f"Expected a distribution loss, got {mode}")
    if logits.shape[:-1] != teacher_logprobs.shape[:-1]:
        raise ValueError("Student and Teacher token dimensions must match")
    mask = (
        torch.ones_like(logits[..., 0], dtype=torch.bool)
        if valid_mask is None
        else valid_mask.bool()
    )
    if mask.shape != logits.shape[:-1]:
        raise ValueError("valid_mask must match Student token dimensions")
    z = torch.where(mask.unsqueeze(-1), logits.float(), 0.0)
    t = torch.where(mask.unsqueeze(-1), teacher_logprobs.detach().float(), 0.0)
    if not torch.isfinite(z).all() or not torch.isfinite(t).all():
        raise ValueError(
            "Expected finite Student logits and Teacher logprobs at valid positions"
        )
    log_z = torch.logsumexp(z, dim=-1, keepdim=True)
    full = teacher_ids is None
    if full:
        if config.topk != -1 or t.shape != z.shape:
            raise ValueError(
                "Full targets require topk=-1 and matching vocabulary dimensions"
            )
        qlog = z - log_z
    else:
        if (
            config.topk == -1
            or teacher_ids.shape != t.shape
            or t.shape[-1] != config.topk
        ):
            raise ValueError(
                "Sparse targets must have the configured positive K IDs and probabilities"
            )
        if teacher_ids.dtype != torch.long:
            raise ValueError("Teacher IDs must have dtype int64")
        ids = torch.where(mask.unsqueeze(-1), teacher_ids, 0)
        selected = ids[mask]
        if ((selected < 0) | (selected >= z.shape[-1])).any():
            raise ValueError("Teacher IDs must belong to the Student output vocabulary")
        ordered = selected.sort(dim=-1).values
        if (ordered[..., 1:] == ordered[..., :-1]).any():
            raise ValueError(
                "Teacher support must contain unique IDs at every valid position"
            )
        qlog = z.gather(-1, ids) - log_z
    p, q = t.exp(), qlog.exp()
    pmass, qmass = p.sum(-1), q.sum(-1)
    if (pmass[mask] > 1 + 1e-5).any():
        raise ValueError(
            "Teacher probabilities must retain full-vocabulary normalization"
        )
    if full and not torch.allclose(
        pmass[mask], torch.ones_like(pmass[mask]), atol=1e-5, rtol=1e-5
    ):
        raise ValueError("Full Teacher probabilities must sum to one")
    reverse = mode.startswith("reverse")
    if config.log_prob_min_clamp is not None:
        qlog = qlog.clamp_min(config.log_prob_min_clamp)
        t = t.clamp_min(config.log_prob_min_clamp)
        p = t.exp()
    loss = (q * (qlog - t) if reverse else p * (t - qlog)).sum(-1)
    tail = loss * 0
    if mode.endswith("_tail") and not full and config.topk < z.shape[-1]:
        ptail = (1 - pmass).clamp_min(0)
        outside = torch.ones_like(z, dtype=torch.bool).scatter(-1, ids, False)
        # Every row has at least one outside ID; logsumexp stays finite.
        qtail_log = torch.logsumexp(
            z.masked_fill(~outside, -torch.inf), -1
        ) - log_z.squeeze(-1)
        qtail = qtail_log.exp()
        if reverse:
            tail = qtail * (qtail_log - ptail.clamp_min(config.tail_mass_eps).log())
        else:
            safe_p = torch.where(ptail > 0, ptail, 1.0)
            tail = ptail * (safe_p.log() - qtail.clamp_min(config.tail_mass_eps).log())
        loss = loss + tail
    if mode == "forward_kl_topk":
        loss = loss.clamp_min(0)
    return DistillationDistributionOutput(
        torch.where(mask, loss, 0.0),
        torch.where(mask, qmass, 0.0),
        torch.where(mask, pmass, 0.0),
        torch.where(mask, tail, 0.0),
    )
