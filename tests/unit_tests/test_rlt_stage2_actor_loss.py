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

import pytest

torch = pytest.importorskip("torch")

from rlinf.models.embodiment.rlt_stage2.components import (
    DirectGaussianActor,
    TwinQCritic,
    actor_loss,
    compute_td_target,
)
from rlinf.models.embodiment.rlt_stage2.rollout import TransitionSource


def test_direct_gaussian_actor_conditions_on_reference_and_uses_fixed_noise():
    actor = DirectGaussianActor(
        state_dim=1,
        action_chunk_dim=2,
        hidden_dim=4,
        num_hidden_layers=0,
        sigma=0.25,
        ref_dropout=1.0,
    )
    linear = actor.mlp.net[0]
    with torch.no_grad():
        linear.weight.copy_(
            torch.tensor(
                [
                    [1.0, 1.0, 0.0],
                    [-1.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            )
        )
        linear.bias.zero_()

    x = torch.tensor([[0.2]], dtype=torch.float32)
    a_tilde = torch.tensor([[0.3, 0.4]], dtype=torch.float32)

    deterministic = actor(x, a_tilde, deterministic=True)
    torch.testing.assert_close(
        deterministic,
        torch.tensor([[0.5, 0.2]], dtype=torch.float32),
    )

    dropped_ref = actor(
        x,
        a_tilde,
        deterministic=True,
        apply_ref_dropout=True,
    )
    torch.testing.assert_close(
        dropped_ref,
        torch.tensor([[0.2, -0.2]], dtype=torch.float32),
    )

    torch.manual_seed(0)
    noisy = actor(x, a_tilde, deterministic=False)
    torch.manual_seed(0)
    expected_noise = deterministic + torch.randn_like(deterministic) * actor.sigma
    torch.testing.assert_close(noisy, expected_noise.clamp(-1.0, 1.0))


def test_compute_td_target_uses_chunk_discount_and_twin_target_minimum():
    actor = DirectGaussianActor(
        state_dim=1,
        action_chunk_dim=2,
        hidden_dim=4,
        num_hidden_layers=0,
        sigma=0.0,
    )
    actor_linear = actor.mlp.net[0]
    with torch.no_grad():
        actor_linear.weight.copy_(
            torch.tensor(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            )
        )
        actor_linear.bias.zero_()

    critic = TwinQCritic(
        state_dim=1,
        action_chunk_dim=2,
        hidden_dim=4,
        num_hidden_layers=0,
    )
    with torch.no_grad():
        q1_linear = critic.q1_target.mlp.net[0]
        q2_linear = critic.q2_target.mlp.net[0]
        q1_linear.weight.copy_(torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32))
        q1_linear.bias.fill_(10.0)
        q2_linear.weight.copy_(torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32))
        q2_linear.bias.fill_(1.0)

    rewards = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    dones = torch.tensor([[0.0]], dtype=torch.float32)
    next_x = torch.tensor([[0.5]], dtype=torch.float32)
    next_a_tilde = torch.tensor([[0.25, 0.75]], dtype=torch.float32)

    target = compute_td_target(
        rewards=rewards,
        dones=dones,
        next_x=next_x,
        next_a_tilde=next_a_tilde,
        target_actor=actor,
        critic=critic,
        gamma=0.5,
        chunk_length=2,
    )

    next_q_min = 1.0 + 0.5 + 2.0 * 0.25 + 3.0 * 0.75
    expected = 1.0 + 0.5 * 2.0 + (0.5**2) * next_q_min
    torch.testing.assert_close(target, torch.tensor([[expected]], dtype=torch.float32))


def test_compute_td_target_drops_bootstrap_on_terminal_chunk():
    actor = DirectGaussianActor(
        state_dim=1,
        action_chunk_dim=2,
        hidden_dim=4,
        num_hidden_layers=0,
        sigma=0.0,
    )
    critic = TwinQCritic(
        state_dim=1,
        action_chunk_dim=2,
        hidden_dim=4,
        num_hidden_layers=0,
    )

    target = compute_td_target(
        rewards=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        dones=torch.tensor([[1.0]], dtype=torch.float32),
        next_x=torch.zeros((1, 1), dtype=torch.float32),
        next_a_tilde=torch.zeros((1, 2), dtype=torch.float32),
        target_actor=actor,
        critic=critic,
        gamma=0.5,
        chunk_length=2,
    )

    torch.testing.assert_close(target, torch.tensor([[2.0]], dtype=torch.float32))


def test_actor_loss_matches_rlt_q_plus_reference_regularizer_formula():
    q_value = torch.tensor([[2.0]], dtype=torch.float32)
    a = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]], dtype=torch.float32)
    a_tilde = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)

    total_loss, metrics = actor_loss(
        q_value,
        a,
        a_tilde,
        bc_weight=3.0,
        q_weight=0.5,
        delta_weight=0.25,
    )

    bc_loss = (1.0 + 1.0) / 4.0
    q_loss = -0.5 * 2.0
    delta_loss = 0.0
    expected = 3.0 * bc_loss + q_loss + 0.25 * delta_loss
    torch.testing.assert_close(total_loss, torch.tensor(expected, dtype=torch.float32))
    torch.testing.assert_close(metrics["bc_loss"], torch.tensor(bc_loss))
    torch.testing.assert_close(metrics["q_mean"], torch.tensor(2.0))


def test_actor_loss_uses_executed_human_actions_as_bc_target():
    q_value = torch.tensor([[0.0]], dtype=torch.float32)
    a = torch.tensor([[[0.0, 0.0], [2.0, 0.0]]], dtype=torch.float32)
    a_tilde = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32)
    action_chunk = torch.tensor([[[0.0, 0.0], [2.0, 0.0]]], dtype=torch.float32)
    source_chunk = torch.tensor(
        [[TransitionSource.RL, TransitionSource.HUMAN]],
        dtype=torch.uint8,
    )

    total_loss, metrics = actor_loss(
        q_value,
        a,
        a_tilde,
        action_chunk=action_chunk,
        source_chunk=source_chunk,
        bc_weight=1.0,
        q_weight=0.0,
        delta_weight=0.0,
    )

    torch.testing.assert_close(total_loss, torch.tensor(0.0))
    torch.testing.assert_close(metrics["bc_human_loss"], torch.tensor(0.0))
    torch.testing.assert_close(metrics["human_mask_ratio"], torch.tensor(0.5))
