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

import torch

from rlinf.models.embodiment.rlt_stage2.components import actor_loss
from rlinf.models.embodiment.rlt_stage2.transition import TransitionSource


def test_actor_loss_human_bc_weight_is_backward_compatible():
    q_value = torch.tensor([[0.25], [0.5]], dtype=torch.float32)
    a = torch.tensor(
        [
            [[0.2, 0.0], [0.4, 0.0]],
            [[0.0, 0.3], [0.0, 0.6]],
        ],
        dtype=torch.float32,
    )
    a_tilde = torch.zeros_like(a)
    action_chunk = torch.ones_like(a)
    source_chunk = torch.tensor(
        [
            [TransitionSource.RL, TransitionSource.RL],
            [TransitionSource.HUMAN, TransitionSource.MIXED],
        ],
        dtype=torch.uint8,
    )

    default_loss, default_metrics = actor_loss(
        q_value,
        a,
        a_tilde,
        action_chunk=action_chunk,
        source_chunk=source_chunk,
        bc_weight=2.0,
        q_weight=0.5,
        delta_weight=0.25,
    )
    explicit_zero_loss, explicit_zero_metrics = actor_loss(
        q_value,
        a,
        a_tilde,
        action_chunk=action_chunk,
        source_chunk=source_chunk,
        bc_weight=2.0,
        q_weight=0.5,
        delta_weight=0.25,
        human_bc_weight=0.0,
    )

    assert torch.allclose(default_loss, explicit_zero_loss)
    assert torch.allclose(
        default_metrics["bc_human_weighted_loss"],
        explicit_zero_metrics["bc_human_weighted_loss"],
    )
    assert default_metrics["bc_human_weighted_loss"].item() == 0.0


def test_actor_loss_human_bc_weight_adds_normalized_human_penalty():
    q_value = torch.tensor([[0.25], [0.5]], dtype=torch.float32)
    a = torch.tensor(
        [
            [[0.2, 0.0], [0.4, 0.0]],
            [[0.0, 0.3], [0.0, 0.6]],
        ],
        dtype=torch.float32,
    )
    a_tilde = torch.zeros_like(a)
    action_chunk = torch.ones_like(a)
    source_chunk = torch.tensor(
        [
            [TransitionSource.RL, TransitionSource.RL],
            [TransitionSource.HUMAN, TransitionSource.MIXED],
        ],
        dtype=torch.uint8,
    )

    base_loss, base_metrics = actor_loss(
        q_value,
        a,
        a_tilde,
        action_chunk=action_chunk,
        source_chunk=source_chunk,
        bc_weight=2.0,
        q_weight=0.5,
        delta_weight=0.25,
        human_bc_weight=0.0,
    )
    weighted_loss, weighted_metrics = actor_loss(
        q_value,
        a,
        a_tilde,
        action_chunk=action_chunk,
        source_chunk=source_chunk,
        bc_weight=2.0,
        q_weight=0.5,
        delta_weight=0.25,
        human_bc_weight=1.5,
    )

    expected_extra = 2.0 * 1.5 * base_metrics["bc_human_loss"]
    assert torch.allclose(weighted_loss - base_loss, expected_extra)
    assert torch.allclose(
        weighted_metrics["bc_human_weighted_loss"],
        1.5 * base_metrics["bc_human_loss"],
    )


def test_actor_loss_human_bc_weight_is_inactive_without_human_sources():
    q_value = torch.tensor([[0.25], [0.5]], dtype=torch.float32)
    a = torch.tensor(
        [
            [[0.2, 0.0], [0.4, 0.0]],
            [[0.0, 0.3], [0.0, 0.6]],
        ],
        dtype=torch.float32,
    )
    a_tilde = torch.zeros_like(a)
    action_chunk = torch.ones_like(a)
    source_chunk = torch.tensor(
        [
            [TransitionSource.BASE, TransitionSource.RL],
            [TransitionSource.RL, TransitionSource.BASE],
        ],
        dtype=torch.uint8,
    )

    base_loss, _ = actor_loss(
        q_value,
        a,
        a_tilde,
        action_chunk=action_chunk,
        source_chunk=source_chunk,
        bc_weight=2.0,
        q_weight=0.5,
        delta_weight=0.25,
        human_bc_weight=0.0,
    )
    weighted_loss, weighted_metrics = actor_loss(
        q_value,
        a,
        a_tilde,
        action_chunk=action_chunk,
        source_chunk=source_chunk,
        bc_weight=2.0,
        q_weight=0.5,
        delta_weight=0.25,
        human_bc_weight=10.0,
    )

    assert torch.allclose(weighted_loss, base_loss)
    assert weighted_metrics["bc_human_loss"].item() == 0.0
    assert weighted_metrics["bc_human_weighted_loss"].item() == 0.0
