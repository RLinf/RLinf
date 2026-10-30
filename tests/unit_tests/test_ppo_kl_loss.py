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

"""Unit tests for the PPO KL penalty term and adaptive coefficient (issue #1110)."""

import pytest
import torch

from rlinf.algorithms.losses import adapt_kl_coefficient, compute_ppo_actor_loss

BSZ, NUM_ACTIONS = 8, 4


def _make_inputs(seed: int = 0):
    torch.manual_seed(seed)
    old_logprobs = torch.randn(BSZ, NUM_ACTIONS)
    # Start exactly at the behavior policy; drift is applied by the test.
    return {
        "logprobs": old_logprobs.clone(),
        "old_logprobs": old_logprobs,
        "clip_ratio_low": 0.2,
        "clip_ratio_high": 0.2,
        "advantages": torch.randn(BSZ, NUM_ACTIONS),
    }


def test_kl_penalty_disabled_by_default_matches_clipping_only():
    inputs = _make_inputs()
    loss_default, metrics_default = compute_ppo_actor_loss(**inputs)
    loss_zero, metrics_zero = compute_ppo_actor_loss(**inputs, kl_beta=0.0)
    assert torch.allclose(loss_default, loss_zero)
    assert torch.allclose(metrics_default["actor/kl_loss"], torch.tensor(0.0))


def test_kl_penalty_term_is_zero_when_policy_matches_behavior():
    inputs = _make_inputs()
    _, metrics = compute_ppo_actor_loss(**inputs, kl_beta=0.1)
    assert torch.allclose(metrics["actor/kl_loss"], torch.tensor(0.0), atol=1e-6)


def test_kl_penalty_grows_with_policy_drift():
    """The loss must respond to how far logprobs drift from old_logprobs.

    The k1 estimator used here is ``mean(old - new)`` (same convention as the
    ``actor/approx_kl`` metric): drifting the new logprobs *below* the behavior
    policy must increase both the measured KL and the total loss.
    """
    inputs = _make_inputs()
    kl_beta = 1.0
    losses = []
    for drift in (0.0, 0.5, 1.0, 2.0):
        drifted = dict(inputs)
        drifted["logprobs"] = inputs["old_logprobs"] - drift
        loss, metrics = compute_ppo_actor_loss(**drifted, kl_beta=kl_beta)
        losses.append(loss)
        torch.testing.assert_close(
            metrics["actor/approx_kl"],
            torch.tensor(drift, dtype=torch.float32),
        )
        torch.testing.assert_close(
            metrics["actor/kl_loss"], torch.tensor(drift, dtype=torch.float32)
        )
    for smaller, larger in zip(losses, losses[1:]):
        assert larger > smaller, "loss must increase with KL drift"


def test_kl_penalty_scales_linearly_with_kl_beta():
    inputs = _make_inputs()
    inputs["logprobs"] = inputs["old_logprobs"] - 0.7
    loss_base, _ = compute_ppo_actor_loss(**inputs, kl_beta=0.0)
    losses = {}
    for kl_beta in (0.0, 0.1, 1.0, 10.0):
        losses[kl_beta], metrics = compute_ppo_actor_loss(**inputs, kl_beta=kl_beta)
        torch.testing.assert_close(
            metrics["actor/kl_beta"], torch.tensor(kl_beta, dtype=torch.float32)
        )
    # loss(kl_beta) == loss_base + kl_beta * kl_term, i.e. affine in kl_beta.
    kl_term = (losses[1.0] - losses[0.0]) / 1.0
    for kl_beta in (0.1, 10.0):
        torch.testing.assert_close(
            losses[kl_beta], loss_base + kl_beta * kl_term, rtol=1e-5, atol=1e-6
        )


def test_kl_penalty_gradient_pulls_logprobs_back():
    """Single CPU update step: gradients must push logprobs toward the behavior policy.

    With logprobs drifted *below* old_logprobs (kl > 0), the KL gradient must
    point in the direction that raises logprobs back toward old_logprobs.
    """
    inputs = _make_inputs()
    inputs["advantages"] = torch.zeros_like(inputs["advantages"])  # isolate KL term
    logprobs = (inputs["old_logprobs"] - 1.0).requires_grad_(True)
    loss, _ = compute_ppo_actor_loss(
        logprobs=logprobs,
        **{k: v for k, v in inputs.items() if k != "logprobs"},
        kl_beta=1.0,
    )
    loss.backward()
    assert logprobs.grad is not None
    # d/dlogprobs [-mean(old - logprobs)] = -1/N < 0: gradient descent increases
    # logprobs, i.e. moves the policy back toward the behavior policy.
    torch.testing.assert_close(
        logprobs.grad, -torch.ones_like(logprobs) / logprobs.numel()
    )


def test_kl_penalty_respects_loss_mask():
    inputs = _make_inputs()
    mask = torch.ones_like(inputs["old_logprobs"], dtype=torch.bool)
    mask[:, -2:] = False
    logprobs = inputs["old_logprobs"].clone()
    logprobs[~mask] -= 5.0  # drift only on masked-out entries
    _, metrics = compute_ppo_actor_loss(
        logprobs=logprobs,
        **{k: v for k, v in inputs.items() if k != "logprobs"},
        loss_mask=mask,
        kl_beta=1.0,
    )
    torch.testing.assert_close(
        metrics["actor/kl_loss"], torch.tensor(0.0), atol=1e-6, rtol=1e-6
    )


def test_kl_penalty_disabled_during_critic_warmup():
    inputs = _make_inputs()
    inputs["logprobs"] = inputs["old_logprobs"] - 1.0
    _, metrics = compute_ppo_actor_loss(**inputs, kl_beta=1.0, critic_warmup=True)
    torch.testing.assert_close(
        metrics["actor/kl_loss"], torch.tensor(0.0), atol=1e-6, rtol=1e-6
    )


class TestAdaptKlCoefficient:
    def test_disabled_without_target(self):
        assert adapt_kl_coefficient(0.1, 10.0, None) == 0.1

    def test_increases_when_kl_too_high(self):
        assert adapt_kl_coefficient(0.1, 1.5 * 0.05 + 1e-6, 0.05) == pytest.approx(0.15)

    def test_decreases_when_kl_too_low(self):
        assert adapt_kl_coefficient(0.1, 0.05 / 1.5 - 1e-6, 0.05) == pytest.approx(0.05)

    def test_unchanged_within_band(self):
        assert adapt_kl_coefficient(0.1, 0.05, 0.05) == pytest.approx(0.1)
        assert adapt_kl_coefficient(0.1, 0.06, 0.05) == pytest.approx(0.1)
        assert adapt_kl_coefficient(0.1, 0.04, 0.05) == pytest.approx(0.1)

    def test_recovers_from_persistent_divergence(self):
        """A stuck-too-high KL must keep inflating the coefficient past the band."""
        kl_beta = 0.01
        for _ in range(20):
            kl_beta = adapt_kl_coefficient(kl_beta, 0.5, 0.05)
        assert kl_beta > 1.5 * 0.05  # now large enough to bring KL under control

    def test_calms_down_after_recovery(self):
        """Once KL falls below the band the coefficient shrinks back down."""
        kl_beta = 0.5
        for _ in range(20):
            kl_beta = adapt_kl_coefficient(kl_beta, 0.001, 0.05)
        assert kl_beta < 0.05


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
