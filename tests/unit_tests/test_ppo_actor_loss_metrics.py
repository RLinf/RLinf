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

"""Regression tests for issue #1471: the ``actor/approx_kl`` and
``actor/clip_fraction`` metrics (and their decoupled-PPO siblings) must be
normalized per masked element, not inflated by ``action_dim`` when
``reward_type: chunk_level`` (mask ``[B, C, 1]``) meets ``logprob_type:
token_level`` (metrics ``[B, C, action_dim]``).
"""

import torch

from rlinf.algorithms.losses import (
    compute_decoupled_ppo_actor_loss,
    compute_ppo_actor_loss,
)

B, C, ACTION_DIM = 2, 3, 7


def _inputs():
    torch.manual_seed(0)
    logprobs = (torch.randn(B, C, ACTION_DIM) * 0.1).float()
    old_logprobs = (torch.randn(B, C, ACTION_DIM) * 0.1).float()
    advantages = torch.randn(B, C, 1).float()
    # chunk_level mask: last dim collapsed to 1, with a couple of masked rows.
    loss_mask = torch.ones(B, C, 1, dtype=torch.bool)
    loss_mask[0, 0, 0] = False
    loss_mask[1, 2, 0] = False
    return logprobs, old_logprobs, advantages, loss_mask


def test_ppo_actor_approx_kl_not_inflated_by_action_dim():
    logprobs, old_logprobs, advantages, loss_mask = _inputs()

    _, metrics = compute_ppo_actor_loss(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        advantages=advantages,
        loss_mask=loss_mask,
    )

    # Independently compute the per-element mean over the broadcast mask.
    expanded = loss_mask.expand(B, C, ACTION_DIM)
    log_ratio = logprobs - old_logprobs
    expected_kl = -(log_ratio * expanded).sum() / expanded.sum()

    torch.testing.assert_close(metrics["actor/approx_kl"], expected_kl)

    # The un-broadcast normalization the bug used would be action_dim x larger.
    inflated = -(log_ratio * loss_mask).sum() / loss_mask.sum()
    assert not torch.allclose(metrics["actor/approx_kl"], inflated)
    torch.testing.assert_close(inflated, expected_kl * ACTION_DIM)


def test_ppo_actor_metrics_invariant_to_mask_width():
    """Passing an already-expanded mask must yield identical metrics -- the
    function normalizes to the metric width internally either way."""
    logprobs, old_logprobs, advantages, loss_mask = _inputs()
    kwargs = {
        "logprobs": logprobs,
        "old_logprobs": old_logprobs,
        "clip_ratio_low": 0.2,
        "clip_ratio_high": 0.2,
        "advantages": advantages,
        "clip_ratio_c": 3.0,
    }

    _, narrow = compute_ppo_actor_loss(loss_mask=loss_mask, **kwargs)
    _, wide = compute_ppo_actor_loss(
        loss_mask=loss_mask.expand(B, C, ACTION_DIM).contiguous(), **kwargs
    )

    for key in ("actor/approx_kl", "actor/clip_fraction"):
        torch.testing.assert_close(narrow[key], wide[key], msg=f"mismatch on {key}")


def test_decoupled_ppo_actor_approx_kl_not_inflated_by_action_dim():
    logprobs, old_logprobs, advantages, loss_mask = _inputs()

    _, metrics = compute_decoupled_ppo_actor_loss(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        advantages=advantages,
        loss_mask=loss_mask,
    )

    # proximal_logprobs defaults to old_logprobs, so proximal KL == logprobs - old.
    expanded = loss_mask.expand(B, C, ACTION_DIM)
    diff = logprobs - old_logprobs
    expected_kl = -(diff * expanded).sum() / expanded.sum()

    torch.testing.assert_close(metrics["actor/proximal_approx_kl"], expected_kl)

    inflated = -(diff * loss_mask).sum() / loss_mask.sum()
    torch.testing.assert_close(inflated, expected_kl * ACTION_DIM)


def test_decoupled_ppo_metrics_invariant_to_mask_width():
    logprobs, old_logprobs, advantages, loss_mask = _inputs()
    kwargs = {
        "logprobs": logprobs,
        "old_logprobs": old_logprobs,
        "clip_ratio_low": 0.2,
        "clip_ratio_high": 0.2,
        "advantages": advantages,
        "clip_ratio_c": 3.0,
    }

    _, narrow = compute_decoupled_ppo_actor_loss(loss_mask=loss_mask, **kwargs)
    _, wide = compute_decoupled_ppo_actor_loss(
        loss_mask=loss_mask.expand(B, C, ACTION_DIM).contiguous(), **kwargs
    )

    for key in (
        "actor/clip_fraction",
        "actor/dual_clip_fraction",
        "actor/proximal_approx_kl",
        "actor/proximal_ratio",
    ):
        torch.testing.assert_close(narrow[key], wide[key], msg=f"mismatch on {key}")
