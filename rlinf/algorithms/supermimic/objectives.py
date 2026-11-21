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

"""World-model loss, imagined return and critic targets."""

from __future__ import annotations

import torch

from rlinf.envs.sim.g1_mimic.constants import DT, MAX_TRACKING_REWARD

# Bryson's rule: a channel group's squared error is divided by its nominal
# error squared. Units are m, rad, rad/s and m/s.
NOMINAL_ERRORS = {
    "root_pos": 0.01,
    "root_ori": 0.02,
    "joint_pos": 0.1,
    "joint_vel": 3.14,
    "body_pos": 0.05,
    "body_rot": 0.1,
    "body_vel": 1.0,
    "body_avel": 3.14,
}


def value_ceiling(gamma: float) -> float:
    """Largest discounted return: every step earns the tracking-kernel maximum."""
    return MAX_TRACKING_REWARD / (1.0 - gamma)


def world_model_loss(
    pred: dict[str, torch.Tensor], true: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bryson-weighted state loss and the plain MSE of each channel group.

    Each group's mean squared error is weighted by ``1 / nominal^2`` and by
    its channel count, then divided by the total channel count. The groups
    come out in ``pred``'s key order.
    """
    mse = torch.stack([(pred[name] - true[name]).square().mean() for name in pred])
    # Filled on the device: a CUDA graph cannot record a host-to-device copy.
    widths = torch.stack(
        [mse.new_full((), float(pred[name].shape[-1])) for name in pred]
    )
    weights = torch.stack(
        [mse.new_full((), NOMINAL_ERRORS[name] ** -2) for name in pred]
    )
    return (mse * weights * widths).sum() / widths.sum(), mse


def imagined_return(
    rewards: torch.Tensor,
    term_prob: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Discounted return of each imagined rollout, ``[B]``.

    ``rewards`` and ``term_prob`` are ``[B, H]``. Step ``t`` counts with the
    survival ``prod_{j<t} (1 - p_j)``; the critic value of the last state
    counts with the survival of all ``H`` steps. ``term_prob`` stays soft so
    the gradient reaches the termination test.
    """
    horizon = rewards.shape[1]
    # A running product, not cumprod: cumprod's backward reads on the host
    # whether an input is zero, which a CUDA graph cannot record.
    survival = [torch.ones_like(term_prob[:, 0])]
    for t in range(horizon):
        survival.append(survival[-1] * (1.0 - term_prob[:, t]))
    alive = torch.stack(survival[1:], dim=1)
    before = torch.stack(survival[:-1], dim=1)
    discount = gamma ** torch.arange(
        horizon, device=rewards.device, dtype=rewards.dtype
    )
    return (discount * before * rewards).sum(1) + gamma**horizon * alive[
        :, -1
    ] * last_value


def lambda_returns(
    values: torch.Tensor,
    last_value: torch.Tensor,
    rewards: torch.Tensor,
    term_prob: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """GAE value targets ``A_t + V_t`` of a time-major rollout, ``[H, B]``.

    A soft termination ``p_t`` discounts what follows step ``t`` by
    ``(1 - p_t) gamma``.
    """
    targets = torch.empty_like(values)
    advantage = torch.zeros_like(last_value)
    next_value = last_value
    for t in reversed(range(values.shape[0])):
        carry = (1.0 - term_prob[t]) * gamma
        delta = rewards[t] + carry * next_value - values[t]
        advantage = delta + carry * lam * advantage
        targets[t] = advantage + values[t]
        next_value = values[t]
    return targets


def reward_per_second(mean_return: torch.Tensor, horizon: int) -> torch.Tensor:
    """Mean imagined return per second of rollout, the logged policy reward."""
    return mean_return / horizon / DT
