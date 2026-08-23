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

"""Unit tests for the dependency-free StreamingVLA SFP primitives."""

import pytest
import torch

from rlinf.models.embodiment.streamingvla.sfp import (
    compute_sfp_flow_targets,
    create_sinusoidal_pos_embedding,
    make_attention_masks_and_position_ids,
    sample_beta,
)
from rlinf.models.embodiment.streamingvla.training import (
    sample_streamingvla_step_inputs,
    streamingvla_step_seed,
)


def test_sfp_flow_targets_match_reference_formula():
    """Action-state interpolation, noise, and velocity stay in fp32."""
    actions = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[-1.0, 1.0], [2.0, -2.0], [4.0, 3.0]],
        ],
        dtype=torch.bfloat16,
    )
    action_states = torch.tensor([[10.0, 20.0], [-3.0, 5.0]])
    time = torch.tensor([0.5, 0.9])
    noise = torch.tensor([[[0.25, -0.5]], [[1.0, 2.0]]])

    scaled_t, index, alpha, x_t, u_t, added_noise = compute_sfp_flow_targets(
        actions,
        action_states,
        time,
        noise,
        action_horizon=3,
        sigma=0.16,
        noise_decay=4.0,
    )

    expected_interpolated = torch.tensor([[[12.5, 24.0]], [[0.8, 6.1]]])
    expected_velocity = torch.tensor([[[9.0, 12.0]], [[12.0, 9.0]]])
    expected_noise = 0.16 * torch.exp(-4.0 * time)[:, None, None] * noise

    torch.testing.assert_close(scaled_t, torch.tensor([1.5, 2.7]))
    torch.testing.assert_close(index, torch.tensor([1, 2]))
    torch.testing.assert_close(alpha, torch.tensor([0.5, 0.7]))
    torch.testing.assert_close(added_noise, expected_noise)
    torch.testing.assert_close(x_t, expected_interpolated + expected_noise)
    torch.testing.assert_close(u_t, expected_velocity - 4.0 * expected_noise)
    for tensor in (scaled_t, alpha, x_t, u_t, added_noise):
        assert tensor.dtype == torch.float32


def test_sfp_flow_targets_validate_shapes():
    """Malformed horizons, action states, timesteps, and noise fail early."""
    actions = torch.zeros(2, 3, 4)
    time = torch.zeros(2)

    with pytest.raises(ValueError, match="action_horizon"):
        compute_sfp_flow_targets(actions, None, time, None, action_horizon=2)
    with pytest.raises(ValueError, match="action_states"):
        compute_sfp_flow_targets(
            actions,
            torch.zeros(2, 3),
            time,
            None,
            action_horizon=3,
        )
    with pytest.raises(ValueError, match="time must have shape"):
        compute_sfp_flow_targets(
            actions, None, torch.zeros(2, 1), None, action_horizon=3
        )
    with pytest.raises(ValueError, match="noise must have shape"):
        compute_sfp_flow_targets(
            actions, None, time, torch.zeros(2, 3, 4), action_horizon=3
        )


def test_prefix_attention_mask_and_position_ids():
    """Prefix tokens stay bidirectional and suffix tokens stay causal."""
    pad_masks = torch.tensor([[True, True, True, True], [True, True, False, True]])
    categories = torch.tensor([[False, False, True, True], [False, True, True, True]])

    attention, positions = make_attention_masks_and_position_ids(pad_masks, categories)

    expected_first = torch.tensor(
        [
            [True, True, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    expected_second = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [False, False, False, False],
            [True, True, False, True],
        ]
    )
    torch.testing.assert_close(attention[0], expected_first)
    torch.testing.assert_close(attention[1], expected_second)
    torch.testing.assert_close(positions, torch.tensor([[0, 1, 2, 3], [0, 1, 1, 2]]))


def test_time_sampling_and_embedding_are_reproducible():
    """Beta sampling respects the Torch seed and embeddings validate shape."""
    torch.manual_seed(42)
    first = sample_beta(1.5, 1.0, 32, torch.device("cpu"))
    torch.manual_seed(42)
    second = sample_beta(1.5, 1.0, 32, torch.device("cpu"))
    torch.testing.assert_close(first, second)
    assert first.dtype == torch.float32
    assert torch.all((first >= 0.0) & (first <= 1.0))

    embedding = create_sinusoidal_pos_embedding(
        torch.tensor([0.0, 0.5]), 4, min_period=0.004, max_period=4.0
    )
    torch.testing.assert_close(embedding[0, :2], torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(embedding[0, 2:], torch.ones(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="divisible by 2"):
        create_sinusoidal_pos_embedding(
            torch.tensor([0.0]), 3, min_period=0.004, max_period=4.0
        )


def test_attention_inputs_must_be_rank_two():
    """Mask construction rejects ambiguous broadcast shapes."""
    with pytest.raises(ValueError, match="rank-2"):
        make_attention_masks_and_position_ids(
            torch.ones(2, dtype=torch.bool),
            torch.ones(1, 2, dtype=torch.bool),
        )


def test_step_random_inputs_are_repeatable_and_partition_invariant():
    """Step RNG is stable, micro-batch independent, and side-effect free."""
    rng_state = torch.random.get_rng_state()
    first_time, first_noise = sample_streamingvla_step_inputs(
        seed=42,
        rank=1,
        global_step=7,
        local_batch_size=8,
        action_dim=32,
        device=torch.device("cpu"),
    )
    second_time, second_noise = sample_streamingvla_step_inputs(
        seed=42,
        rank=1,
        global_step=7,
        local_batch_size=8,
        action_dim=32,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(first_time, second_time, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first_noise, second_noise, rtol=0.0, atol=0.0)
    torch.testing.assert_close(torch.random.get_rng_state(), rng_state)
    assert torch.equal(torch.cat([first_time[:4], first_time[4:]]), first_time)
    assert torch.equal(torch.cat([first_noise[:4], first_noise[4:]]), first_noise)
    assert streamingvla_step_seed(42, 1, 7) != streamingvla_step_seed(42, 1, 8)
