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

"""Tracking reward and failure tests shared by the simulator and the world model.

The anchor is the pelvis, tracked body 0: the root terms compare it, the
body pose terms re-anchor the reference at its xy and yaw, and failure reads
its height and tilt.
"""

from __future__ import annotations

import torch

from rlinf.envs.sim.g1_mimic.constants import (
    DT,
    FAILURE_HEIGHT_M,
    FAILURE_TILT,
    ROOT_QUAT,
)
from rlinf.envs.sim.g1_mimic.features import (
    heading_yaw,
    projected_gravity,
    rot_apply,
    rot_mul,
    rot_mul_nt,
    yaw_rotation,
)
from rlinf.envs.sim.g1_mimic.g1_fk import PELVIS_BODY, TORSO_BODY, g1_kinematics

TORSO = TORSO_BODY
PELVIS = PELVIS_BODY

# (weight, std) of the six tracking kernels.
ROOT_POS_KERNEL = (0.5, 0.3)
ROOT_ORI_KERNEL = (0.5, 0.4)
BODY_POS_KERNEL = (1.0, 0.3)
BODY_ORI_KERNEL = (1.0, 0.4)
BODY_LIN_VEL_KERNEL = (1.0, 1.0)
BODY_ANG_VEL_KERNEL = (1.0, 3.14)
ACTION_RATE_WEIGHT = -0.1
JOINT_LIMIT_WEIGHT = -2.0
# Width of the soft failure threshold, relative to the hard one.
TERMINATION_SOFTNESS = 0.05


def differentiable_fk(
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """14 tracked bodies of a 71-d state on the G1 kinematic tree."""
    return g1_kinematics().forward(state)


def forward_kinematics_pair(
    state: torch.Tensor, ref_state: torch.Tensor
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Body poses of ``state`` and ``ref_state`` from one tree walk."""
    if state.shape[0] != ref_state.shape[0]:
        return differentiable_fk(state), differentiable_fk(ref_state)
    pos, rot, lin, ang = differentiable_fk(torch.cat((state, ref_state), dim=0))
    n = state.shape[0]
    return (
        (pos[:n], rot[:n], lin[:n], ang[:n]),
        (pos[n:], rot[n:], lin[n:], ang[n:]),
    )


def rotation_angle(rot: torch.Tensor, ref_rot: torch.Tensor) -> torch.Tensor:
    """Geodesic angle between ``[..., 3, 3]`` rotations."""
    cos = ((rot * ref_rot).sum(dim=(-2, -1)) - 1.0) * 0.5
    return torch.acos(cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7))


def _kernel(error: torch.Tensor, kernel: tuple[float, float]) -> torch.Tensor:
    weight, std = kernel
    return weight * torch.exp(-error / (std * std))


def tracking_terms(
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
    body: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ref: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Sum of the six weighted tracking kernels, before ``DT``.

    ``body`` and ``ref`` are world positions ``[..., 14, 3]``, rotations
    ``[..., 14, 3, 3]`` and linear / angular velocities ``[..., 14, 3]``.
    The reference bodies are re-anchored at the xy and yaw of the root pose
    ``root_pos`` / ``root_rot``.
    """
    pos, rot, lin, ang = body
    ref_pos, ref_rot, ref_lin, ref_ang = ref
    anchor = PELVIS
    turn = yaw_rotation(
        heading_yaw_of_matrix(rot_mul_nt(root_rot, ref_rot[..., anchor, :, :]))
    ).unsqueeze(-3)
    shift = torch.cat((root_pos[..., :2], ref_pos[..., anchor, 2:3]), dim=-1).unsqueeze(
        -2
    )
    target_pos = shift + rot_apply(turn, ref_pos - ref_pos[..., anchor : anchor + 1, :])
    target_rot = rot_mul(turn, ref_rot)
    root_pos = (pos[..., anchor, :] - ref_pos[..., anchor, :]).square().sum(-1)
    root_ori = rotation_angle(
        rot[..., anchor, :, :], ref_rot[..., anchor, :, :]
    ).square()
    body_pos = (pos - target_pos).square().sum(-1).mean(-1)
    body_ori = rotation_angle(rot, target_rot).square().mean(-1)
    body_lin = (lin - ref_lin).square().sum(-1).mean(-1)
    body_ang = (ang - ref_ang).square().sum(-1).mean(-1)
    return (
        _kernel(root_pos, ROOT_POS_KERNEL)
        + _kernel(root_ori, ROOT_ORI_KERNEL)
        + _kernel(body_pos, BODY_POS_KERNEL)
        + _kernel(body_ori, BODY_ORI_KERNEL)
        + _kernel(body_lin, BODY_LIN_VEL_KERNEL)
        + _kernel(body_ang, BODY_ANG_VEL_KERNEL)
    )


def tracking_reward(
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
    body: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ref: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    joint_pos: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    soft_limits: torch.Tensor,
) -> torch.Tensor:
    """Per-step reward: tracking kernels, action rate and soft joint limits, times ``DT``.

    ``action`` and ``last_action`` are raw policy outputs, before the PD scale.
    """
    rate = (action - last_action).square().sum(-1)
    over = (soft_limits[:, 0] - joint_pos).clamp_min(0.0) + (
        joint_pos - soft_limits[:, 1]
    ).clamp_min(0.0)
    return DT * (
        tracking_terms(root_pos, root_rot, body, ref)
        + ACTION_RATE_WEIGHT * rate
        + JOINT_LIMIT_WEIGHT * over.sum(-1)
    )


def heading_yaw_of_matrix(rot: torch.Tensor) -> torch.Tensor:
    """Yaw of a rotation matrix's x axis on the ground plane."""
    return torch.atan2(rot[..., 1, 0], rot[..., 0, 0])


def termination_probability(
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
    ref_pos: torch.Tensor,
    ref_rot: torch.Tensor,
) -> torch.Tensor:
    """Soft pelvis failure test: a sigmoid of the larger normalized violation.

    ``ref_pos`` / ``ref_rot`` are the reference pelvis position ``[..., 3]``
    and rotation ``[..., 3, 3]``.
    """
    height = (ref_pos[..., 2] - root_pos[..., 2]).abs() / FAILURE_HEIGHT_M
    tilt = (ref_rot[..., 2, 2] - root_rot[..., 2, 2]).abs() / FAILURE_TILT
    return torch.sigmoid((torch.maximum(height, tilt) - 1.0) / TERMINATION_SOFTNESS)


def failure_signals(
    state: torch.Tensor, ref_state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pelvis height error, gravity-z error, and the hard failure test."""
    height_err = state[..., 2] - ref_state[..., 2]
    tilt = (
        projected_gravity(state[..., ROOT_QUAT])[..., 2]
        - projected_gravity(ref_state[..., ROOT_QUAT])[..., 2]
    )
    failed = (height_err.abs() > FAILURE_HEIGHT_M) | (tilt.abs() > FAILURE_TILT)
    return height_err, tilt, failed


def xy_yaw_drift(
    state: torch.Tensor, ref_state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Planar position distance and absolute yaw difference to the reference."""
    pos_drift = (state[..., :2] - ref_state[..., :2]).norm(dim=-1)
    yaw_drift = heading_yaw(state[..., 3:7]) - heading_yaw(ref_state[..., 3:7])
    yaw_drift = torch.atan2(torch.sin(yaw_drift), torch.cos(yaw_drift)).abs()
    return pos_drift, yaw_drift
