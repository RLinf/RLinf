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
from omegaconf import OmegaConf

from rlinf.algorithms.diffusion.art import (
    art_loss,
    art_scalar_covariance,
    compute_art_loss,
    prepare_art_training_inputs,
    prompt_disjoint_windows,
    validate_art_rollout,
)


def test_art_velocity_target_validation_and_prompt_windows(monkeypatch):
    """The supported ART path keeps its sign and validates outside the hot path."""
    assert tuple(inspect.signature(art_loss).parameters) == (
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
    validate_art_rollout(states, sigmas, sigmas_next, magnitudes)

    old = torch.zeros(4, 2)
    current = old.clone().requires_grad_(True)
    direction = torch.tensor([[1.0, -1.0]]).expand_as(old)

    def fail_hot_path_scan(*args, **kwargs):
        raise AssertionError("art_loss must not scan tensor values in the hot path")

    with monkeypatch.context() as context:
        context.setattr(torch, "isfinite", fail_hot_path_scan)
        terms = art_loss(
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
    windows = prompt_disjoint_windows(groups, updates=2, order=order)
    assert windows == ((3, 4, 0, 7), (2, 5, 1, 6))
    assert set(windows[0]).isdisjoint(windows[1])


def test_art_rollout_rejects_non_decreasing_sigma():
    states = torch.zeros(1, 2, 2)
    with pytest.raises(ValueError, match="sigma_next < sigma"):
        validate_art_rollout(
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
    stats = art_scalar_covariance(
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


@pytest.mark.parametrize("exclude_final", [False, True])
@pytest.mark.parametrize("reward_reduction", ["mean", "sum"])
def test_art_preparation_builds_guidance_and_preserves_timestep_rng_order(
    exclude_final, reward_reduction
):
    states = torch.arange(8 * 4 * 2).reshape(8, 4, 2).float()
    sigmas = torch.tensor([[1.0, 0.75, 0.5, 0.25]]).expand(8, -1)
    sigmas_next = torch.tensor([[0.75, 0.5, 0.25, 0.0]]).expand(8, -1)
    endpoints = torch.tensor([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]] * 2)
    rewards = torch.tensor([[0.0], [1.0], [2.0], [3.0], [0.1], [0.1], [0.1], [0.1]])
    rewards = torch.cat((rewards, rewards * 2), dim=1)
    prompts = torch.ones(8, 2, 3)
    inputs = {
        "art_endpoint": endpoints,
        "art_states": states,
        "art_sigmas": sigmas,
        "art_sigmas_next": sigmas_next,
        "prompt_embeds": prompts,
    }
    config = OmegaConf.create(
        {
            "group_size": 4,
            "art_timestep_fraction": 0.5,
            "art_exclude_final_step": exclude_final,
            "art_reward_reduction": reward_reduction,
        }
    )
    pool = 3 if exclude_final else 4
    count = max(1, int(pool * 0.5))
    expected_generator = torch.Generator().manual_seed(37)
    expected_steps = torch.stack(
        [torch.randperm(pool, generator=expected_generator)[:count] for _ in range(8)]
    )
    generator = torch.Generator().manual_seed(37)
    metrics, has_guidance = prepare_art_training_inputs(
        inputs, rewards, config=config, generator=generator
    )
    assert has_guidance
    assert "art_endpoint" not in inputs
    assert inputs["prompt_embeds"] is prompts
    rows = torch.arange(8)[:, None]
    torch.testing.assert_close(inputs["art_states"], states[rows, expected_steps])
    torch.testing.assert_close(inputs["art_sigmas"], sigmas[rows, expected_steps])
    torch.testing.assert_close(
        inputs["art_sigmas_next"], sigmas_next[rows, expected_steps]
    )
    torch.testing.assert_close(generator.get_state(), expected_generator.get_state())
    direction = torch.tensor([1.0, 2.0]) / torch.tensor(2.5).sqrt()
    torch.testing.assert_close(
        inputs["art_guidance_direction"][:4], direction.expand(4, -1)
    )
    torch.testing.assert_close(inputs["art_guidance_direction"][4:], torch.zeros(4, 2))
    torch.testing.assert_close(
        inputs["art_guidance_magnitude"], torch.tensor([1.0] * 4 + [0.0] * 4)
    )
    assert metrics["actor/active_group_frac"] == 0.5
    reduced_rewards = rewards.sum(1) if reward_reduction == "sum" else rewards.mean(1)
    assert metrics["actor/art_reward_mean"] == pytest.approx(
        reduced_rewards.mean().item()
    )
    assert metrics["actor/art_reward_std"] == pytest.approx(
        reduced_rewards.std(correction=0).item()
    )


def test_art_preparation_reports_zero_guidance_for_constant_rewards():
    inputs = {
        "art_endpoint": torch.arange(8).reshape(4, 2).float(),
        "art_states": torch.ones(4, 1, 2),
        "art_sigmas": torch.ones(4, 1),
        "art_sigmas_next": torch.zeros(4, 1),
    }
    metrics, active = prepare_art_training_inputs(
        inputs,
        torch.full((4, 1), 0.1),
        config=OmegaConf.create({"group_size": 4, "art_exclude_final_step": True}),
        generator=torch.Generator().manual_seed(0),
    )
    assert not active
    assert metrics["actor/active_group_frac"] == 0
    assert metrics["actor/covariance_rms"] == 0
    assert metrics["actor/guidance_rms"] == 0
    assert inputs["art_states"].shape == (4, 1, 2)


@pytest.mark.parametrize("mask_metric", ["sde_kl", "fixed_mse"])
@pytest.mark.parametrize("loss_space", ["velocity_linear_sigma", "velocity_uniform"])
@pytest.mark.parametrize("mask_enabled", [False, True])
def test_art_objective_freezes_targets_and_masks_only_active_outward_movement(
    mask_metric, loss_space, mask_enabled
):
    current = torch.tensor([[-0.2, -0.2], [0.2, 0.2], [-0.2, -0.2]], requires_grad=True)
    old = torch.zeros_like(current, requires_grad=True)
    direction = torch.ones_like(current, requires_grad=True)
    magnitude = torch.tensor([1.0, 1.0, 0.0], requires_grad=True)
    sigma = torch.full((3,), 0.5)
    config = OmegaConf.create(
        {
            "art_mask_metric": mask_metric,
            "art_loss_space": loss_space,
            "art_corrective_mask_enabled": mask_enabled,
        }
    )
    loss, metrics = compute_art_loss(
        current,
        old,
        sigma,
        torch.zeros_like(sigma),
        direction,
        magnitude,
        config=config,
    )
    weight = 0.5 if loss_space == "velocity_linear_sigma" else 1.0
    unmasked = (current + magnitude.detach()[:, None]).square().mean(1) * weight
    expected = (unmasked[1] if mask_enabled else unmasked[:2].sum()) / 3
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert metrics["actor/art_loss"] == pytest.approx(expected.item())
    assert metrics["actor/active_count"] == 2
    assert metrics["actor/unmasked_loss"] == pytest.approx(unmasked[:2].mean().item())
    assert metrics["actor/mask_frac"] == (0.5 if mask_enabled else 0)
    assert torch.count_nonzero(current.grad[2]) == 0
    assert (torch.count_nonzero(current.grad[0]) == 0) == mask_enabled
    assert torch.count_nonzero(current.grad[1]) == 2
    assert old.grad is None
    assert direction.grad is None
    assert magnitude.grad is None


@pytest.mark.parametrize("updates", [1, 2, 4])
def test_art_prompt_windows_cover_each_row_once_and_balance_each_group(updates):
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    order = [3, 4, 0, 7, 2, 5, 1, 6]
    windows = prompt_disjoint_windows(groups, updates=updates, order=order)
    assert sorted(row for window in windows for row in window) == list(range(8))
    for window in windows:
        torch.testing.assert_close(
            torch.bincount(groups[list(window)]), torch.full((2,), 4 // updates)
        )
