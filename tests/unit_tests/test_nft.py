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

"""NFT schedules, EMA state, and tensor objectives on CPU."""

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from rlinf.algorithms.diffusion.nft import (
    build_nft_schedule,
    compute_nft_loss,
    get_nft_tau,
    nft_requires_ema,
    resample_nft_inputs,
    update_nft_ema,
)


@pytest.mark.parametrize("container", [list, tuple, OmegaConf.create])
@pytest.mark.parametrize(
    ("step", "expected"), [(0, 0.2), (2, 0.2), (4, 0.5), (6, 0.8), (8, 0.8)]
)
def test_nft_tau_schedule(container, step, expected):
    tau = container([0.2, 0.8, 2, 6])
    assert get_nft_tau(tau, step) == pytest.approx(expected)
    assert nft_requires_ema(tau)


@pytest.mark.parametrize("tau", [1.0, [1.0, 1.0, 0, 5]])
def test_nft_on_policy_tau_does_not_require_ema(tau):
    assert get_nft_tau(tau, 3) == 1.0
    assert not nft_requires_ema(tau)


def test_nft_tau_scalar_and_degenerate_schedule():
    assert get_nft_tau(0.4, 17) == 0.4
    assert nft_requires_ema(0.4)
    assert get_nft_tau([0.2, 0.8, 3, 3], 3) == 0.2
    assert get_nft_tau([0.2, 0.8, 3, 3], 4) == 0.8
    assert nft_requires_ema([1.0, 0.5, 2, 6])


@pytest.mark.parametrize("tau", [0.0, 0.25, 1.0])
def test_nft_ema_updates_floating_weights_and_copies_other_state(tau):
    weight = torch.tensor([1.0, 3.0], dtype=torch.float64)
    count = torch.tensor([2], dtype=torch.int64)
    state = {"weight": weight, "count": count, "tag": "old"}
    current = {
        "weight": torch.tensor([5.0, 7.0], requires_grad=True),
        "count": torch.tensor([9], dtype=torch.int64),
        "tag": "new",
    }
    update_nft_ema(state, current, tau)
    torch.testing.assert_close(
        weight, torch.tensor([1 + 4 * tau, 3 + 4 * tau], dtype=weight.dtype)
    )
    assert state["weight"] is weight
    assert state["count"] is count
    assert count.item() == 9
    assert state["tag"] == "new"
    assert not weight.requires_grad


@pytest.mark.parametrize("shape", [(2, 3, 4), (2, 3, 2, 2, 2)])
def test_nft_resampling_preserves_rng_order_and_broadcasts_schedule(shape):
    x0 = torch.arange(torch.tensor(shape).prod()).reshape(shape).float()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(43)
        expected_steps = torch.randint(0, 4, (2,))
        expected_noise = torch.randn_like(x0)
        torch.manual_seed(43)
        xcur, steps = resample_nft_inputs(x0, num_steps=4)
    schedule, timesteps = build_nft_schedule(
        steps, x0.device, x0.dtype, num_steps=4, ndim=x0.ndim
    )
    torch.testing.assert_close(schedule, torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]))
    torch.testing.assert_close(steps, expected_steps)
    assert timesteps.shape == (2,) + (1,) * (x0.ndim - 1)
    torch.testing.assert_close(xcur, (1 - timesteps) * x0 + timesteps * expected_noise)


def _reference_nft_loss(velocity, inputs, advantages, mask, config, schedule):
    """Evaluate the two candidates directly in the requested target space."""
    batch_size = velocity.shape[0]
    t = schedule[inputs["nft_step_index"]].reshape(batch_size, 1, 1)
    dt = (
        schedule[inputs["nft_step_index"]] - schedule[inputs["nft_step_index"] + 1]
    ).reshape(batch_size, 1, 1)
    safe_t = torch.where(t == 1, schedule[1], t)
    sigma2 = (
        t / (1 - safe_t) * inputs["nft_noise_level"].reshape(batch_size, 1, 1).square()
    )
    old = inputs["nft_v"].detach()
    delta = velocity - old
    beta = config.get("nft_beta", 1.0)
    x = inputs["nft_xcur"]
    target_space = config.get("nft_target_space", "xnext")
    target = inputs["nft_" + target_space]
    action_level = config.get("nft_gather_type", "action_level") == "action_level"
    dims = (-1,) if action_level else (-2, -1)
    reduction = config.get("nft_energy_reduction", "mean")
    scales = config.get("nft_weight_scale", 1.0)
    scales = (
        [float(scales), float(scales)] if isinstance(scales, (int, float)) else scales
    )

    def energy(candidate, scale):
        clean = x - candidate * t
        noise = x + candidate * (1 - t)
        if target_space == "x0":
            prediction = clean - sigma2 / 2 * noise
        else:
            prediction = clean * (1 - t + dt) + noise * (t - dt - sigma2 * dt / (2 * t))
        error = (prediction - target).square() * scale
        return error.mean(dims) if reduction == "mean" else error.sum(dims)

    positive = energy(old + beta * delta, scales[0])
    negative = energy(old - beta * delta, scales[1])
    advantage = advantages.reshape(batch_size, -1)
    if not action_level:
        advantage = advantage[:, 0]
        mask = mask.reshape(batch_size, -1)[:, 0]
    else:
        advantage = advantage.expand_as(positive)
        mask = mask.reshape(batch_size, -1).expand_as(positive)
    if config.get("adv_type", "raw") != "raw":
        maximum = config.get("adv_clip_max", 1.0)
        advantage = (advantage.clamp(-maximum, maximum) + maximum) / (2 * maximum)
    if config.get("nft_loss_form", "dpo") == "dpo":
        per_item = F.softplus(
            config.get("dpo_beta", 1.0)
            / 2
            * (2 * advantage - 1)
            * (positive - negative)
        )
    else:
        scale = config.get("adv_clip_max", 1.0) / beta
        per_item = (advantage * positive + (1 - advantage) * negative) * scale
        clip = config.get("nft_clip_ratio")
        if clip is not None:
            delta_norm = delta.square().sum(dims, keepdim=True).sqrt().clamp_min(1e-8)
            old_norm = old.square().sum(dims, keepdim=True).sqrt().clamp_min(1e-8)
            coefficient = (clip * old_norm / delta_norm).clamp(max=1)
            clipped = torch.where(
                coefficient < 1, (delta * coefficient).detach(), delta
            )
            clipped_pos = energy(old + beta * clipped, scales[0])
            clipped_neg = energy(old - beta * clipped, scales[1])
            per_item = torch.maximum(
                per_item,
                (advantage * clipped_pos + (1 - advantage) * clipped_neg) * scale,
            )
    return (per_item * mask).sum() / mask.sum()


@pytest.mark.parametrize("target_space", ["x0", "xnext"])
@pytest.mark.parametrize("gather_type", ["action_level", "chunk_level"])
@pytest.mark.parametrize("energy_reduction", ["mean", "sum"])
@pytest.mark.parametrize("adv_type", ["raw", "grpo"])
@pytest.mark.parametrize("loss_form,clip", [("dpo", None), ("mse", None), ("mse", 0.1)])
@pytest.mark.parametrize("noise_level", [0.0, 0.3])
def test_nft_objective_and_gradient_match_candidate_energies(
    target_space, gather_type, energy_reduction, adv_type, loss_form, clip, noise_level
):
    generator = torch.Generator().manual_seed(71)
    inputs = {
        key: torch.randn(2, 2, 3, generator=generator, dtype=torch.float64)
        for key in ("nft_xcur", "nft_v", "nft_x0", "nft_xnext")
    }
    inputs["nft_v"].requires_grad_(True)
    inputs["nft_step_index"] = torch.tensor([0, 2])
    inputs["nft_noise_level"] = torch.full((2,), noise_level, dtype=torch.float64)
    velocity = torch.randn(
        2, 2, 3, generator=generator, dtype=torch.float64, requires_grad=True
    )
    reference_velocity = velocity.detach().clone().requires_grad_()
    advantages = (
        torch.tensor([[-2.0], [0.7]])
        if adv_type == "grpo"
        else torch.tensor([[0.1], [0.7]])
    )
    mask = torch.tensor([[True], [False]])
    config = OmegaConf.create(
        {
            "nft_target_space": target_space,
            "nft_gather_type": gather_type,
            "nft_energy_reduction": energy_reduction,
            "adv_type": adv_type,
            "adv_clip_max": 1.5,
            "nft_beta": 0.6,
            "dpo_beta": 0.8,
            "nft_loss_form": loss_form,
            "nft_clip_ratio": clip,
            "nft_weight_mode": "constant",
            "nft_weight_scale": [0.7, 1.3],
        }
    )
    schedule, _ = build_nft_schedule(
        inputs["nft_step_index"], velocity.device, velocity.dtype, num_steps=4
    )
    loss, metrics = compute_nft_loss(
        velocity, inputs, advantages, mask, schedule=schedule, config=config
    )
    expected = _reference_nft_loss(
        reference_velocity, inputs, advantages, mask, config, schedule
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    expected.backward()
    torch.testing.assert_close(velocity.grad, reference_velocity.grad)
    assert inputs["nft_v"].grad is None
    assert metrics["actor/nft_loss"] == pytest.approx(expected.item())
    assert ("actor/clip_loss_frac" in metrics) == (clip is not None)


@pytest.mark.parametrize(
    "mode", ["constant", "t", "1-t", "inv_t2", "inv_t3", "sigma", "adaptive", "auto"]
)
@pytest.mark.parametrize("noise_level", [0.0, 0.3])
def test_nft_energy_weights_and_detached_adaptive_gradient(mode, noise_level):
    velocity = torch.tensor([[[0.3, -0.4]]], dtype=torch.float64, requires_grad=True)
    old = torch.tensor([[[0.1, 0.2]]], dtype=torch.float64)
    inputs = {
        "nft_xcur": torch.zeros_like(old),
        "nft_x0": torch.ones_like(old),
        "nft_v": old,
        "nft_step_index": torch.tensor([1]),
        "nft_noise_level": torch.tensor([noise_level], dtype=torch.float64),
    }
    schedule, _ = build_nft_schedule(
        inputs["nft_step_index"], velocity.device, velocity.dtype, num_steps=4
    )
    config = OmegaConf.create(
        {"nft_target_space": "x0", "nft_weight_mode": mode, "nft_loss_form": "mse"}
    )
    loss, metrics = compute_nft_loss(
        velocity,
        inputs,
        torch.ones(1, 1),
        torch.ones(1, 1, dtype=torch.bool),
        schedule=schedule,
        config=config,
    )
    t = 0.75
    sigma2 = t / (1 - t) * noise_level**2
    prediction = -velocity * t - sigma2 / 2 * velocity * (1 - t)
    error = prediction - 1
    effective_mode = (
        ("adaptive" if noise_level == 0 else "sigma") if mode == "auto" else mode
    )
    weight = {
        "constant": 1.0,
        "t": t**2,
        "1-t": (1 - t) ** 2,
        "inv_t2": 1 / (t**2 + 1e-4),
        "inv_t3": 1 / (t**3 + 1e-4),
        "sigma": 1 / (0.25 * sigma2 + 1e-4),
        "adaptive": 1 / error.detach().abs().mean().clamp_min(1e-5),
    }[effective_mode]
    expected = (error.square() * weight).mean()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(
        torch.autograd.grad(loss, velocity)[0],
        torch.autograd.grad(expected, velocity)[0],
    )
    assert metrics["actor/E_pos_mean"] == pytest.approx(expected.item())


@pytest.mark.parametrize("shape", [(2, 3, 4), (2, 3, 2, 2, 2)])
@pytest.mark.parametrize("gather_type", ["action_level", "chunk_level"])
def test_nft_all_masked_loss_has_zero_gradient_for_action_and_video_batches(
    shape, gather_type
):
    velocity = torch.full(shape, 0.2, requires_grad=True)
    inputs = {
        "nft_xcur": torch.zeros(shape),
        "nft_x0": torch.ones(shape),
        "nft_v": torch.zeros(shape),
        "nft_step_index": torch.tensor([0, 1]),
        "nft_noise_level": torch.zeros(2),
    }
    schedule, _ = build_nft_schedule(
        inputs["nft_step_index"], velocity.device, velocity.dtype, num_steps=4
    )
    loss, metrics = compute_nft_loss(
        velocity,
        inputs,
        torch.ones(2, 1),
        torch.zeros(2, 1, dtype=torch.bool),
        schedule=schedule,
        config=OmegaConf.create(
            {"nft_target_space": "x0", "nft_gather_type": gather_type}
        ),
    )
    loss.backward()
    assert loss.item() == 0
    assert torch.count_nonzero(velocity.grad) == 0
    assert metrics["actor/E_pos_mean_pos_only"] == 0
    assert metrics["actor/E_neg_mean_neg_only"] == 0


def test_nft_action_signals_mask_individual_actions_and_slice_rollout_targets():
    velocity = torch.full((2, 2, 3), 0.2, dtype=torch.float64, requires_grad=True)
    inputs = {
        "nft_xcur": torch.zeros(2, 3, 3, dtype=torch.float64),
        "nft_x0": torch.ones(2, 3, 3, dtype=torch.float64),
        "nft_v": torch.zeros(2, 3, 3, dtype=torch.float64),
        "nft_step_index": torch.tensor([2, 2]),
        "nft_noise_level": torch.zeros(2, dtype=torch.float64),
    }
    advantages = torch.tensor([[1.0, 0.0], [0.25, 0.75]], dtype=torch.float64)
    mask = torch.tensor([[True, False], [False, True]])
    config = OmegaConf.create(
        {
            "nft_target_space": "x0",
            "nft_loss_form": "mse",
            "nft_weight_mode": "constant",
        }
    )
    schedule, _ = build_nft_schedule(
        inputs["nft_step_index"], velocity.device, velocity.dtype, num_steps=4
    )
    loss, _ = compute_nft_loss(
        velocity, inputs, advantages, mask, schedule=schedule, config=config
    )
    positive = (-0.5 * velocity - 1).square().mean(-1)
    negative = (0.5 * velocity - 1).square().mean(-1)
    expected = ((advantages * positive + (1 - advantages) * negative) * mask).sum() / 2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.count_nonzero(velocity.grad[~mask]) == 0
    assert torch.count_nonzero(velocity.grad[mask]) == 6


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"nft_loss_form": "unknown"}, "Unsupported nft_loss_form"),
        ({"nft_target_space": "unknown"}, "Unsupported nft_target_space"),
        ({"nft_gather_type": "unknown"}, "Unsupported nft_gather_type"),
        ({"nft_weight_scale": [1.0]}, "nft_weight_scale list"),
    ],
)
def test_nft_objective_rejects_unsupported_settings(settings, message):
    inputs = {
        "nft_xcur": torch.zeros(2, 2, 3),
        "nft_x0": torch.ones(2, 2, 3),
        "nft_v": torch.zeros(2, 2, 3),
        "nft_step_index": torch.tensor([1, 2]),
        "nft_noise_level": torch.zeros(2),
    }
    config = OmegaConf.create({"nft_target_space": "x0", **settings})
    schedule, _ = build_nft_schedule(
        inputs["nft_step_index"], torch.device("cpu"), torch.float32, num_steps=4
    )
    with pytest.raises(ValueError, match=message):
        compute_nft_loss(
            torch.zeros(2, 2, 3),
            inputs,
            torch.ones(2, 1),
            torch.ones(2, 1, dtype=torch.bool),
            schedule=schedule,
            config=config,
        )
