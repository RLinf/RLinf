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

"""Analytic tracking reward and differentiable FK.

Root position/orientation terms compare ``torso_link``. Failure termination
reads the pelvis. Those are not the same body.
"""

from __future__ import annotations

import torch

from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    DT,
    FAILURE_HEIGHT_M,
    FAILURE_TILT,
    JOINT_POS,
    NUM_BODIES,
    REWARD_WEIGHTS,
    ROOT_ANG_VEL,
    ROOT_LIN_VEL,
)
from rlinf.envs.sim.g1_mimic.features import (
    heading_yaw,
    projected_gravity,
    yaw_rotation,
)
from rlinf.envs.sim.g1_mimic.g1_fk import PELVIS_BODY, TORSO_BODY, g1_kinematics

# Reward root is torso_link. Termination reads the pelvis.
TORSO = TORSO_BODY
PELVIS = PELVIS_BODY


def differentiable_fk(
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """14 tracking bodies on the G1 kinematic tree."""
    return g1_kinematics().forward(state)


def reanchor_reference(
    robot_state: torch.Tensor, ref_state: torch.Tensor
) -> torch.Tensor:
    """Snap the reference root's xy and yaw onto the robot.

    Root height, joint angles, and body-frame angular velocity stay on the
    reference. World-frame linear velocity is rotated by the same yaw. Root
    position and orientation rewards are computed before this call, so global
    drift remains in those terms and not in the body terms.
    """
    squeezed = robot_state.dim() == 1
    if squeezed:
        robot_state = robot_state.unsqueeze(0)
        ref_state = ref_state.unsqueeze(0)
    robot_yaw = heading_yaw(robot_state[..., 3:7])
    ref_yaw = heading_yaw(ref_state[..., 3:7])
    dyaw = robot_yaw - ref_yaw
    align = yaw_rotation(dyaw)
    new_ref = ref_state.clone()
    new_ref[..., 0] = robot_state[..., 0]
    new_ref[..., 1] = robot_state[..., 1]
    from rlinf.envs.sim.g1_mimic.features import quat_mul, rotvec_to_quat

    yaw_fix = rotvec_to_quat(
        torch.stack(
            (
                torch.zeros_like(dyaw),
                torch.zeros_like(dyaw),
                dyaw,
            ),
            dim=-1,
        )
    )
    new_ref[..., 3:7] = quat_mul(yaw_fix, ref_state[..., 3:7])
    new_ref[..., ROOT_LIN_VEL] = torch.einsum(
        "...ij,...j->...i", align, ref_state[..., ROOT_LIN_VEL]
    )
    return new_ref.squeeze(0) if squeezed else new_ref


def _exp_sq(squared_error: torch.Tensor, scale: float) -> torch.Tensor:
    """``exp(-mean(||e||^2) / std^2)``. ``squared_error`` is already squared."""
    return torch.exp(-squared_error / (scale**2))


def _rotation_angle_sq(rot: torch.Tensor, ref_rot: torch.Tensor) -> torch.Tensor:
    """Squared geodesic angle. ``rot`` is [..., 3, 3]."""
    rel = torch.matmul(ref_rot.transpose(-1, -2), rot)
    trace = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos).square()


def analytic_reward(
    state: torch.Tensor,
    ref_state: torch.Tensor,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    joint_limit_low: torch.Tensor | None = None,
    joint_limit_high: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Table A.2 terms, no self-collision, joint-limit weight -2, times dt."""
    squeezed = state.dim() == 1
    if squeezed:
        state = state.unsqueeze(0)
        ref_state = ref_state.unsqueeze(0)
        action = action.unsqueeze(0)
        prev_action = prev_action.unsqueeze(0)
    pos, rot, lin, ang = differentiable_fk(state)
    ref_pos, ref_rot, ref_lin, ref_ang = differentiable_fk(ref_state)
    # Root terms stay in the world frame. Body pose terms use xy+yaw reanchor.
    e_root_pos = (pos[:, TORSO] - ref_pos[:, TORSO]).square().sum(dim=-1)
    e_root_ori = _rotation_angle_sq(rot[:, TORSO], ref_rot[:, TORSO])
    anchored = reanchor_reference(state, ref_state)
    apos, arot, _, _ = differentiable_fk(anchored)
    e_body_pos = (pos - apos).square().sum(dim=-1).mean(dim=-1)
    e_body_ori = _rotation_angle_sq(rot, arot).mean(dim=-1)
    e_body_lin = (lin - ref_lin).square().sum(dim=-1).mean(dim=-1)
    e_body_ang = (ang - ref_ang).square().sum(dim=-1).mean(dim=-1)

    body_pos = REWARD_WEIGHTS["body_pos"] * _exp_sq(e_body_pos, 0.3)
    body_ori = REWARD_WEIGHTS["body_ori"] * _exp_sq(e_body_ori, 0.4)
    body_lin = REWARD_WEIGHTS["body_lin_vel"] * _exp_sq(e_body_lin, 1.0)
    body_ang = REWARD_WEIGHTS["body_ang_vel"] * _exp_sq(e_body_ang, 3.14)
    root_pos = REWARD_WEIGHTS["root_pos"] * _exp_sq(e_root_pos, 0.3)
    root_ori = REWARD_WEIGHTS["root_ori"] * _exp_sq(e_root_ori, 0.4)
    tracking = body_pos + body_ori + body_lin + body_ang + root_pos + root_ori

    if joint_limit_low is None:
        joint_limit_low = torch.full(
            (ACTION_DIM,), -2.5, device=state.device, dtype=state.dtype
        )
    if joint_limit_high is None:
        joint_limit_high = torch.full(
            (ACTION_DIM,), 2.5, device=state.device, dtype=state.dtype
        )
    q = state[..., JOINT_POS]
    over = (q - joint_limit_high).clamp_min(0.0) + (joint_limit_low - q).clamp_min(0.0)
    limit = REWARD_WEIGHTS["joint_limit"] * over.sum(dim=-1)
    smooth = REWARD_WEIGHTS["action_rate"] * (action - prev_action).pow(2).sum(dim=-1)
    total = (tracking + limit + smooth) * DT

    out = {
        "reward": total,
        "body_tracking": (body_pos + body_ori + body_lin + body_ang) * DT,
        "root_pos": root_pos * DT,
        "root_ori": root_ori * DT,
        "joint_limit": limit * DT,
        "action_smooth": smooth * DT,
    }
    if squeezed:
        return {k: v.squeeze(0) for k, v in out.items()}
    return out


def failure_signals(
    state: torch.Tensor, ref_state: torch.Tensor, termination_body: str = "pelvis"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Height error, tilt error, and hard failure from the pelvis by default."""
    squeezed = state.dim() == 1
    if squeezed:
        state = state.unsqueeze(0)
        ref_state = ref_state.unsqueeze(0)
    pos, _, _, _ = differentiable_fk(state)
    ref_pos, _, _, _ = differentiable_fk(ref_state)
    body = PELVIS if termination_body == "pelvis" else TORSO
    # Signed errors. The hard threshold uses the absolute value.
    # Tilt is the reference-frame gravity z minus the robot's, not tilt from upright.
    height_err = pos[:, body, 2] - ref_pos[:, body, 2]
    tilt = projected_gravity(state[..., 3:7])[..., 2] - projected_gravity(
        ref_state[..., 3:7]
    )[..., 2]
    failed = (height_err.abs() > FAILURE_HEIGHT_M) | (tilt.abs() > FAILURE_TILT)
    if squeezed:
        return height_err.squeeze(0), tilt.squeeze(0), failed.squeeze(0)
    return height_err, tilt, failed


def termination_head_input(
    height_err: torch.Tensor, tilt: torch.Tensor
) -> torch.Tensor:
    if height_err.dim() == 0:
        height_err = height_err.unsqueeze(0)
        tilt = tilt.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    feat = torch.stack((height_err, tilt, height_err.abs(), tilt.abs()), dim=-1)
    return feat.squeeze(0) if squeeze else feat


def xy_yaw_drift(
    state: torch.Tensor, ref_state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    pos_drift = (state[..., :2] - ref_state[..., :2]).norm(dim=-1)
    yaw_drift = heading_yaw(state[..., 3:7]) - heading_yaw(ref_state[..., 3:7])
    yaw_drift = torch.atan2(torch.sin(yaw_drift), torch.cos(yaw_drift)).abs()
    return pos_drift, yaw_drift


assert NUM_BODIES == 14
