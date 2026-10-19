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

"""DiffusionART tensor-kernel tests."""

import inspect

import pytest
import torch

from rlinf.workers.actor.fsdp_art_policy_worker import (
    _art_loss,
    _art_scalar_covariance,
    _prompt_disjoint_windows,
    _validate_art_rollout,
)


def test_art_velocity_target_validation_and_prompt_windows(monkeypatch):
    """The supported ART path keeps its sign and validates outside the hot path."""
    assert tuple(inspect.signature(_art_loss).parameters) == (
        "current_velocity",
        "old_velocity",
        "sigma",
        "sigma_next",
        "guidance_direction",
        "guidance_magnitude",
        "loss_space",
        "mask_metric",
        "mask_threshold",
        "reference_noise_level",
        "corrective_mask_enabled",
    )
    states = torch.zeros(4, 2, 2)
    sigmas = torch.tensor([[1.0, 0.5]]).expand(4, -1)
    sigmas_next = torch.tensor([[0.5, 0.0]]).expand(4, -1)
    magnitudes = torch.full((4,), 0.25)
    _validate_art_rollout(states, sigmas, sigmas_next, magnitudes)

    old = torch.zeros(4, 2)
    current = old.clone().requires_grad_(True)
    direction = torch.tensor([[1.0, -1.0]]).expand_as(old)

    def fail_hot_path_scan(*args, **kwargs):
        raise AssertionError("art_loss must not scan tensor values in the hot path")

    with monkeypatch.context() as context:
        context.setattr(torch, "isfinite", fail_hot_path_scan)
        terms = _art_loss(
            current,
            old,
            sigmas[:, 1],
            sigmas_next[:, 1],
            direction,
            magnitudes,
            mask_metric="fixed_mse",
            mask_threshold=1.0,
        )

    expected_target = old - magnitudes[:, None] * direction
    torch.testing.assert_close(terms["target"], expected_target)
    torch.testing.assert_close(
        terms["per_item_loss"],
        sigmas[:, 1] * (expected_target - current).square().mean(1),
    )
    terms["per_item_loss"].sum().backward()
    assert torch.all(current.grad * expected_target < 0)

    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    order = [3, 4, 0, 7, 2, 5, 1, 6]
    windows = _prompt_disjoint_windows(groups, updates=2, order=order)
    assert windows == ((3, 4, 0, 7), (2, 5, 1, 6))
    assert set(windows[0]).isdisjoint(windows[1])


def test_art_rollout_rejects_non_decreasing_sigma():
    states = torch.zeros(1, 2, 2)
    with pytest.raises(ValueError, match="sigma_next < sigma"):
        _validate_art_rollout(
            states,
            torch.tensor([[1.0, 0.5]]),
            torch.tensor([[0.5, 0.5]]),
            torch.ones(1),
        )


def test_art_complete_group_covariance_and_constant_rewards():
    """Guidance uses complete prompt groups and freezes constant-reward groups."""
    endpoints = torch.tensor(
        [[0.0, 0.0], [2.0, 4.0], [10.0, 5.0], [14.0, 9.0], [7.0, -3.0], [9.0, 5.0]],
        requires_grad=True,
    )
    rewards = torch.tensor([0.0, 2.0, 1.0, 3.0, 0.1, 0.1], requires_grad=True)
    groups = torch.tensor([0, 0, 1, 1, 2, 2])
    stats = _art_scalar_covariance(
        endpoints,
        rewards,
        groups,
        group_count=3,
        zero_signal_mode="exact",
    )

    covariance = torch.tensor([[2.0, 4.0], [4.0, 4.0], [0.0, 0.0]])
    direction = covariance / covariance.square().mean(1, keepdim=True).sqrt().clamp_min(
        1e-5
    )
    torch.testing.assert_close(stats["group_covariance"], covariance)
    torch.testing.assert_close(stats["group_directions"], direction)
    torch.testing.assert_close(
        stats["group_guidance_magnitudes"], torch.tensor([1.0, 1.0, 0.0])
    )
    torch.testing.assert_close(stats["group_active"], torch.tensor([True, True, False]))
    torch.testing.assert_close(stats["group_counts"], torch.full((3,), 2.0))
    assert all(not value.requires_grad for value in stats.values())
