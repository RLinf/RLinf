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

import math

import torch

from rlinf.algorithms.losses import (
    compute_grpo_actor_loss_fn,
    compute_ppo_actor_loss,
)
from rlinf.utils.utils import seq_mean_token_mean


def test_dual_clip_bounds_extreme_ratio_for_negative_advantage():
    loss, metrics = compute_ppo_actor_loss(
        logprobs=torch.tensor([[math.log(1.0e8)]], dtype=torch.float32),
        old_logprobs=torch.zeros(1, 1, dtype=torch.float32),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
        advantages=-torch.ones(1, 1, dtype=torch.float32),
        loss_mask=torch.ones(1, 1, dtype=torch.bool),
    )

    assert torch.allclose(loss, torch.tensor(3.0))
    assert metrics["actor/dual_cliped_ratio"].item() > 1.0e7


def test_log_ratio_clamp_keeps_extreme_ratio_gradient_finite():
    for advantage in (-1.0, 1.0):
        logprobs = torch.tensor([[117.56]], dtype=torch.float32, requires_grad=True)
        loss, metrics = compute_ppo_actor_loss(
            logprobs=logprobs,
            old_logprobs=torch.zeros(1, 1, dtype=torch.float32),
            clip_ratio_low=0.2,
            clip_ratio_high=0.2,
            clip_ratio_c=3.0,
            clip_log_ratio_min=-20.0,
            clip_log_ratio_max=20.0,
            advantages=torch.full((1, 1), advantage, dtype=torch.float32),
            loss_mask=torch.ones(1, 1, dtype=torch.bool),
        )

        loss.backward()

        assert torch.isfinite(loss)
        assert torch.isfinite(metrics["actor/ratio"])
        assert torch.isfinite(logprobs.grad).all()


def test_finite_fraction_is_not_scaled_by_episode_loss_weight():
    _, metrics = compute_ppo_actor_loss(
        logprobs=torch.zeros(1, 4, dtype=torch.float32),
        old_logprobs=torch.zeros(1, 4, dtype=torch.float32),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        advantages=torch.ones(1, 4, dtype=torch.float32),
        loss_mask=torch.ones(1, 4, dtype=torch.bool),
        loss_mask_sum=torch.full((1, 4), 2.0, dtype=torch.float32),
        max_episode_steps=10,
        log_logprob_diagnostics=True,
    )

    assert torch.allclose(metrics["actor/logprob_finite_fraction"], torch.tensor(1.0))


def test_empty_loss_mask_uses_neutral_ratio_metrics():
    logprobs = torch.zeros(1, 4, dtype=torch.float32, requires_grad=True)
    loss, metrics = compute_ppo_actor_loss(
        logprobs=logprobs,
        old_logprobs=torch.zeros(1, 4, dtype=torch.float32),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        advantages=torch.ones(1, 4, dtype=torch.float32),
        loss_mask=torch.zeros(1, 4, dtype=torch.bool),
        log_logprob_diagnostics=True,
    )

    loss.backward()

    assert torch.equal(loss, torch.tensor(0.0))
    assert torch.equal(logprobs.grad, torch.zeros_like(logprobs))
    assert torch.equal(metrics["actor/ratio"], torch.tensor(1.0))
    assert torch.equal(metrics["actor/clipped_ratio"], torch.tensor(1.0))
    assert torch.equal(metrics["actor/ratio_abs"], torch.tensor(0.0))
    assert torch.equal(metrics["actor/approx_kl"], torch.tensor(0.0))
    assert torch.equal(metrics["actor/logprob_finite_fraction"], torch.tensor(1.0))
    assert "actor/nonempty_microbatch_fraction" not in metrics


def test_empty_loss_mask_fast_path_uses_neutral_ratio_metrics():
    _, metrics = compute_ppo_actor_loss(
        logprobs=torch.zeros(1, 4, dtype=torch.float32),
        old_logprobs=torch.zeros(1, 4, dtype=torch.float32),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        advantages=torch.ones(1, 4, dtype=torch.float32),
        loss_mask=torch.zeros(1, 4, dtype=torch.bool),
        fast_path_zero_loss_mask=True,
    )

    assert torch.equal(metrics["actor/ratio"], torch.tensor(1.0))
    assert torch.equal(metrics["actor/clipped_ratio"], torch.tensor(1.0))
    assert "actor/nonempty_microbatch_fraction" not in metrics


def test_masked_nonfinite_logprobs_do_not_pollute_token_loss():
    logprobs = torch.tensor(
        [[0.0, float("nan")], [0.0, float("inf")]],
        dtype=torch.float32,
        requires_grad=True,
    )
    loss, metrics = compute_ppo_actor_loss(
        logprobs=logprobs,
        old_logprobs=torch.zeros(2, 2, dtype=torch.float32),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        advantages=torch.tensor([[1.0], [-1.0]], dtype=torch.float32),
        loss_mask=torch.tensor([[True, False], [True, False]]),
        loss_agg_func=seq_mean_token_mean,
        log_logprob_diagnostics=True,
    )

    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logprobs.grad).all()
    assert torch.equal(metrics["actor/logprob_finite_fraction"], torch.tensor(1.0))


def test_non_pi0_action_level_grpo_is_unchanged():
    loss, _ = compute_grpo_actor_loss_fn(
        logprobs=torch.zeros(2, 2, dtype=torch.float32),
        old_logprobs=torch.zeros(2, 2, dtype=torch.float32),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        advantages=torch.tensor([[1.0, 1.0], [-1.0, -1.0]], dtype=torch.float32),
        loss_mask=torch.ones(2, 2, dtype=torch.bool),
    )

    assert torch.equal(loss, torch.tensor(0.0))
