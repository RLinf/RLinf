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

"""Pure-tensor SuperMimic objectives (paper Eq. 2 and Eq. 3)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rlinf.envs.sim.g1_mimic.constants import (
    EPISODE_CONT,
    EPISODE_FAILURE,
    EPISODE_TIMEOUT,
    JOINT_POS,
    JOINT_VEL,
    ROOT_ANG_VEL,
    ROOT_LIN_VEL,
    ROOT_POS,
    STATE_LOSS_SCALES,
)
from rlinf.envs.sim.g1_mimic.features import so3_relative_rotvec


def survival_weights(d_hat: torch.Tensor) -> torch.Tensor:
    """c_0 = 1, c_{t+1} = c_t (1 - d_{t+1}). ``d_hat`` is [B, H] and stays soft."""
    ones = torch.ones(d_hat.shape[0], 1, device=d_hat.device, dtype=d_hat.dtype)
    c_rest = torch.cumprod(1.0 - d_hat, dim=1)
    return torch.cat((ones, c_rest), dim=1)


def actor_objective(
    rewards: torch.Tensor,
    d_hat: torch.Tensor,
    v_last: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Paper Eq. 2. Mean over the batch. ``d_hat`` must not be detached or bool."""
    horizon = rewards.shape[1]
    weights = survival_weights(d_hat)
    time = torch.arange(horizon, device=rewards.device, dtype=rewards.dtype)
    discount = gamma**time
    stepwise = (discount * weights[:, :-1] * rewards).sum(dim=1)
    bootstrap = (gamma**horizon) * weights[:, -1] * v_last
    return (stepwise + bootstrap).mean()


def gae_targets(
    rewards: torch.Tensor,
    d_hat: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Paper Eq. 3. ``values`` is [B, H+1], ``rewards``/``d_hat`` are [B, H]."""
    horizon = rewards.shape[1]
    returns = torch.zeros_like(values)
    returns[:, -1] = values[:, -1]
    for t in range(horizon - 1, -1, -1):
        survive = 1.0 - d_hat[:, t]
        blend = (1.0 - lam) * values[:, t + 1] + lam * returns[:, t + 1]
        returns[:, t] = rewards[:, t] + gamma * survive * blend
    return returns[:, :-1]


def wm_termination_target(end_kind: torch.Tensor) -> torch.Tensor:
    """BCE label: only failure termination is a positive dynamics event."""
    return (end_kind == EPISODE_FAILURE).to(dtype=torch.float32)


def collect_bootstrap_mask(end_kind: torch.Tensor) -> torch.Tensor:
    """Real-collect return: timeout bootstraps, clip-end and failure do not."""
    return end_kind == EPISODE_TIMEOUT


def is_episode_boundary(end_kind: torch.Tensor) -> torch.Tensor:
    """True where an episode ended for any reason."""
    return end_kind != EPISODE_CONT


def state_prediction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sum of squared scaled errors. Not a mean."""
    pos = (pred[..., ROOT_POS] - target[..., ROOT_POS]) / STATE_LOSS_SCALES["root_pos"]
    # Relative SO(3) error. Subtracting two absolute rotvecs is not a rotation.
    rot = (
        so3_relative_rotvec(pred[..., 3:7], target[..., 3:7])
        / STATE_LOSS_SCALES["root_rotvec"]
    )
    lin = (pred[..., ROOT_LIN_VEL] - target[..., ROOT_LIN_VEL]) / STATE_LOSS_SCALES[
        "root_lin_vel"
    ]
    ang = (pred[..., ROOT_ANG_VEL] - target[..., ROOT_ANG_VEL]) / STATE_LOSS_SCALES[
        "root_ang_vel"
    ]
    q = (pred[..., JOINT_POS] - target[..., JOINT_POS]) / STATE_LOSS_SCALES["joint_pos"]
    qd = (pred[..., JOINT_VEL] - target[..., JOINT_VEL]) / STATE_LOSS_SCALES[
        "joint_vel"
    ]
    stacked = torch.cat((pos, rot, lin, ang, q, qd), dim=-1)
    return (stacked**2).sum(dim=-1).mean()


def termination_bce(logit: torch.Tensor, end_kind: torch.Tensor) -> torch.Tensor:
    """BCE of the termination logit against failure-only labels."""
    target = wm_termination_target(end_kind).to(device=logit.device, dtype=logit.dtype)
    return F.binary_cross_entropy_with_logits(logit, target)
