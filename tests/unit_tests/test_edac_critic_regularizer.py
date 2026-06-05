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

import pytest
import torch

from rlinf.algorithms.critic_regularizers import (  # noqa: E402
    compute_edac_critic_diversity_loss,
    compute_sac_critic_loss,
    prepare_edac_actions,
)


def test_edac_loss_returns_zero_for_single_critic_head():
    actions = torch.randn(4, 3, requires_grad=True)
    q_values = actions[:, :1]

    loss = compute_edac_critic_diversity_loss(q_values, actions)
    loss.backward()

    torch.testing.assert_close(loss, torch.zeros((), device=actions.device))
    torch.testing.assert_close(actions.grad, torch.zeros_like(actions))


def test_edac_loss_penalizes_aligned_action_gradients_more_than_orthogonal_ones():
    actions = torch.randn(5, 2, requires_grad=True)

    aligned_weights = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    orthogonal_weights = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    aligned_q_values = actions @ aligned_weights.T
    orthogonal_q_values = actions @ orthogonal_weights.T

    aligned_loss = compute_edac_critic_diversity_loss(aligned_q_values, actions)
    orthogonal_loss = compute_edac_critic_diversity_loss(orthogonal_q_values, actions)

    assert aligned_loss > orthogonal_loss


def test_edac_loss_matches_known_cosine_values():
    actions = torch.randn(4, 2, requires_grad=True)

    aligned_q_values = actions @ torch.tensor([[1.0, 0.0], [1.0, 0.0]]).T
    orthogonal_q_values = actions @ torch.tensor([[1.0, 0.0], [0.0, 1.0]]).T
    opposed_q_values = actions @ torch.tensor([[1.0, 0.0], [-1.0, 0.0]]).T

    torch.testing.assert_close(
        compute_edac_critic_diversity_loss(aligned_q_values, actions),
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        compute_edac_critic_diversity_loss(orthogonal_q_values, actions),
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        compute_edac_critic_diversity_loss(opposed_q_values, actions),
        torch.tensor(-1.0),
    )


def test_edac_loss_matches_three_head_off_diagonal_normalization():
    actions = torch.randn(4, 2, requires_grad=True)
    weights = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    q_values = actions @ weights.T

    loss = compute_edac_critic_diversity_loss(q_values, actions)

    torch.testing.assert_close(loss, torch.tensor(-1.0 / 3.0))


def test_edac_loss_flattens_chunked_action_dimensions():
    actions = torch.randn(3, 2, 4, requires_grad=True)
    q_first = actions[..., 0].sum(dim=1)
    q_second = actions[..., 1].sum(dim=1)
    q_values = torch.stack((q_first, q_second), dim=-1)

    loss = compute_edac_critic_diversity_loss(q_values, actions)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_edac_loss_preserves_device_and_handles_float64_chunked_actions():
    actions = torch.randn(3, 2, 4, dtype=torch.float64, requires_grad=True)
    q_first = actions[..., 0].sum(dim=1)
    q_second = actions[..., 1].sum(dim=1)
    q_values = torch.stack((q_first, q_second), dim=-1)

    loss = compute_edac_critic_diversity_loss(q_values, actions)

    assert loss.shape == torch.Size([])
    assert loss.device == actions.device
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


def test_edac_loss_handles_zero_action_gradients_without_nan():
    actions = torch.randn(4, 3, requires_grad=True)
    zero_q = (actions * 0.0).sum(dim=-1, keepdim=True)
    q_values = zero_q.expand(-1, 2)

    loss = compute_edac_critic_diversity_loss(q_values, actions)

    torch.testing.assert_close(loss, torch.zeros_like(loss))


def test_edac_loss_rejects_action_disconnected_q_values():
    actions = torch.randn(4, 2, requires_grad=True)
    q_values = torch.nn.Parameter(torch.ones(actions.shape[0], 2))

    with pytest.raises(ValueError, match="actions|disconnected"):
        compute_edac_critic_diversity_loss(q_values, actions)


def test_edac_loss_rejects_detached_q_values():
    actions = torch.randn(4, 2, requires_grad=True)
    q_values = torch.ones(actions.shape[0], 2)

    with pytest.raises(ValueError, match="q_values|grad"):
        compute_edac_critic_diversity_loss(q_values, actions)


def test_edac_loss_backpropagates_to_critic_parameters():
    actions = torch.randn(6, 3, requires_grad=True)
    weights = torch.nn.Parameter(torch.tensor([[1.0, 0.0], [0.5, 1.0], [0.0, -1.0]]))
    q_values = actions @ weights

    loss = compute_edac_critic_diversity_loss(q_values, actions)
    loss.backward()

    assert weights.grad is not None
    assert torch.isfinite(weights.grad).all()


def test_sac_critic_loss_uses_plain_mse_when_edac_is_disabled():
    q_values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target_q_values = torch.tensor([[1.5], [2.5]])

    loss, metrics = compute_sac_critic_loss(
        q_values,
        target_q_values,
        actions=None,
        edac_eta=0.0,
    )

    expected_loss = torch.nn.functional.mse_loss(
        q_values, target_q_values.expand_as(q_values)
    )
    torch.testing.assert_close(loss, expected_loss)
    assert metrics["mse_loss"] == pytest.approx(expected_loss.item())
    assert "edac_diversity_loss" not in metrics
    assert "edac_eta" not in metrics


def test_sac_critic_loss_aligns_target_dtype_to_q_values():
    q_values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    target_q_values = torch.tensor([[1.5], [2.5]], dtype=torch.float32)

    loss, metrics = compute_sac_critic_loss(
        q_values,
        target_q_values,
        actions=None,
        edac_eta=0.0,
    )

    expected_loss = torch.nn.functional.mse_loss(
        q_values, target_q_values.to(dtype=q_values.dtype).expand_as(q_values)
    )
    torch.testing.assert_close(loss, expected_loss)
    assert loss.dtype == q_values.dtype
    assert metrics["mse_loss"] == pytest.approx(expected_loss.item())


def test_sac_critic_loss_detaches_target_values():
    q_values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    target_q_values = torch.tensor([[1.5], [2.5]], requires_grad=True)

    loss, _ = compute_sac_critic_loss(
        q_values,
        target_q_values,
        actions=None,
        edac_eta=0.0,
    )
    loss.backward()

    assert q_values.grad is not None
    assert target_q_values.grad is None


def test_sac_critic_loss_requires_actions_when_edac_is_enabled():
    q_values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target_q_values = torch.tensor([[1.5], [2.5]])

    with pytest.raises(ValueError, match="actions"):
        compute_sac_critic_loss(
            q_values,
            target_q_values,
            actions=None,
            edac_eta=0.1,
        )


def test_sac_critic_loss_adds_weighted_edac_regularizer_when_enabled():
    actions = torch.randn(5, 2, requires_grad=True)
    weights = torch.tensor([[1.0, 0.0], [1.0, 0.5]])
    q_values = actions @ weights.T
    target_q_values = torch.zeros(actions.shape[0], 1)
    edac_eta = 0.25

    expected_mse = torch.nn.functional.mse_loss(
        q_values, target_q_values.expand_as(q_values)
    )
    expected_edac = compute_edac_critic_diversity_loss(q_values, actions)

    loss, metrics = compute_sac_critic_loss(
        q_values,
        target_q_values,
        actions=actions,
        edac_eta=edac_eta,
    )

    torch.testing.assert_close(loss, expected_mse + edac_eta * expected_edac)
    assert metrics["mse_loss"] == pytest.approx(expected_mse.item())
    assert metrics["edac_diversity_loss"] == pytest.approx(expected_edac.item())
    assert metrics["edac_eta"] == pytest.approx(edac_eta)


def test_sac_critic_loss_preserves_q_value_dtype_when_edac_is_enabled():
    actions = torch.randn(5, 2, requires_grad=True)
    weights = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    q_values = (actions @ weights.T).to(torch.bfloat16)
    target_q_values = torch.zeros(actions.shape[0], 1, dtype=torch.bfloat16)

    loss, _ = compute_sac_critic_loss(
        q_values,
        target_q_values,
        actions=actions,
        edac_eta=0.25,
    )

    assert loss.dtype == q_values.dtype


def test_sac_critic_loss_locks_head_mean_mse_plus_pairwise_edac_scale():
    actions = torch.randn(4, 2, requires_grad=True)
    weights = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    q_values = actions @ weights.T
    target_q_values = torch.zeros(actions.shape[0], 1)
    edac_eta = 0.25

    loss, _ = compute_sac_critic_loss(
        q_values,
        target_q_values,
        actions=actions,
        edac_eta=edac_eta,
    )

    expected_mse = ((q_values - target_q_values.expand_as(q_values)) ** 2).mean()
    expected_pairwise_edac = torch.tensor(-1.0 / 3.0)
    torch.testing.assert_close(loss, expected_mse + edac_eta * expected_pairwise_edac)


def test_prepare_edac_actions_returns_original_action_when_disabled():
    actions = torch.randn(2, 3)

    prepared = prepare_edac_actions(actions, edac_eta=0.0, use_crossq=False)

    assert prepared is actions
    assert not prepared.requires_grad


def test_prepare_edac_actions_allows_crossq_when_edac_is_disabled():
    actions = torch.randn(2, 3)

    prepared = prepare_edac_actions(actions, edac_eta=0.0, use_crossq=True)

    assert prepared is actions
    assert not prepared.requires_grad


def test_prepare_edac_actions_detaches_action_history_when_enabled():
    actions = torch.randn(2, 3, requires_grad=True)

    prepared = prepare_edac_actions(actions, edac_eta=0.1, use_crossq=False)

    assert prepared is not actions
    assert prepared.requires_grad
    assert prepared.is_leaf
    prepared.pow(2).sum().backward()
    assert prepared.grad is not None
    assert actions.grad is None


def test_prepare_edac_actions_rejects_crossq_when_edac_is_enabled():
    actions = torch.randn(2, 3)

    with pytest.raises(NotImplementedError, match="CrossQ"):
        prepare_edac_actions(actions, edac_eta=0.1, use_crossq=True)
