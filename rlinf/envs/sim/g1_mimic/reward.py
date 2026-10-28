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

"""Tracking reward, failure signals, and torso-anchored goal features.

BeyondMimic's anchor body is ``torso_link``: the root position and
orientation terms compare it, the body pose terms re-anchor the reference at
it, and the optional 9-d anchor goal expresses the reference torso in the
robot torso frame. Failure termination and the critic root error read the
pelvis, the floating base, as Appendix A states.
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
)
from rlinf.envs.sim.g1_mimic.features import (
    heading_yaw,
    projected_gravity,
    rotmat_to_6d,
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


def reanchor_bodies(
    pos: torch.Tensor,
    rot: torch.Tensor,
    ref_pos: torch.Tensor,
    ref_rot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference body poses moved onto the robot torso's xy and yaw.

    BeyondMimic's ``update_relative_body_poses``: the shift takes the robot
    torso's xy and the reference torso's height, and the turn is the yaw of
    ``R_robot_torso R_ref_torso^T``. Inputs are ``[B, bodies, 3]`` positions
    and ``[B, bodies, 3, 3]`` rotations from ``differentiable_fk``.
    """
    turn = yaw_rotation(
        heading_yaw_of_matrix(rot[:, TORSO] @ ref_rot[:, TORSO].transpose(-1, -2))
    )
    shift = torch.cat((pos[:, TORSO, :2], ref_pos[:, TORSO, 2:3]), dim=-1)
    offset = ref_pos - ref_pos[:, TORSO : TORSO + 1]
    anchored_pos = shift.unsqueeze(1) + torch.einsum("bij,bkj->bki", turn, offset)
    anchored_rot = turn.unsqueeze(1) @ ref_rot
    return anchored_pos, anchored_rot


def heading_yaw_of_matrix(rot: torch.Tensor) -> torch.Tensor:
    """Yaw of a rotation matrix's x axis on the ground plane."""
    return torch.atan2(rot[..., 1, 0], rot[..., 0, 0])


def torso_anchor_9d(state: torch.Tensor, ref_state: torch.Tensor) -> torch.Tensor:
    """Reference torso pose in the robot torso frame: position (3) and Rot6D (6).

    BeyondMimic's ``motion_anchor_pos_b`` / ``motion_anchor_ori_b``, the 9-d
    anchor CuboiLeo appends to the 680-d goal. It depends on the current,
    possibly imagined, state, so it is recomputed every step.
    """
    pos, rot, _, _ = differentiable_fk(state)
    ref_pos, ref_rot, _, _ = differentiable_fk(ref_state)
    torso = rot[:, TORSO]
    offset = torch.einsum("bji,bj->bi", torso, ref_pos[:, TORSO] - pos[:, TORSO])
    relative = torso.transpose(-1, -2) @ ref_rot[:, TORSO]
    return torch.cat((offset, rotmat_to_6d(relative)), dim=-1)


def compose_goal(
    goal_ref: torch.Tensor,
    state: torch.Tensor | None,
    ref_state: torch.Tensor | None,
    goal_anchor: bool,
) -> torch.Tensor:
    """680-d reference, optionally plus the 9-d torso anchor at the same time."""
    if not goal_anchor:
        return goal_ref
    if state is None or ref_state is None:
        raise ValueError(
            "goal_anchor=True needs state and ref_state, got "
            f"state={'None' if state is None else 'a tensor'}, "
            f"ref_state={'None' if ref_state is None else 'a tensor'}"
        )
    return torch.cat((goal_ref, torso_anchor_9d(state, ref_state)), dim=-1)


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
    # Root terms stay in the world frame, so global drift stays in them.
    # Body pose terms re-anchor the reference at the robot torso; body
    # velocities are compared in the world frame, as in BeyondMimic.
    e_root_pos = (pos[:, TORSO] - ref_pos[:, TORSO]).square().sum(dim=-1)
    e_root_ori = _rotation_angle_sq(rot[:, TORSO], ref_rot[:, TORSO])
    apos, arot = reanchor_bodies(pos, rot, ref_pos, ref_rot)
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
    body = PELVIS if termination_body == "pelvis" else TORSO
    # Signed errors. The hard threshold uses the absolute value.
    # Tilt is the reference-frame gravity z minus the robot's, not tilt from upright.
    if body == g1_kinematics().free_tracked:
        # The tree walk copies the free-joint body's position from the state.
        height_err = state[:, 2] - ref_state[:, 2]
    else:
        pos, _, _, _ = differentiable_fk(state)
        ref_pos, _, _, _ = differentiable_fk(ref_state)
        height_err = pos[:, body, 2] - ref_pos[:, body, 2]
    tilt = (
        projected_gravity(state[..., 3:7])[..., 2]
        - projected_gravity(ref_state[..., 3:7])[..., 2]
    )
    failed = (height_err.abs() > FAILURE_HEIGHT_M) | (tilt.abs() > FAILURE_TILT)
    if squeezed:
        return height_err.squeeze(0), tilt.squeeze(0), failed.squeeze(0)
    return height_err, tilt, failed


def analytic_failure_prob(height_err: torch.Tensor, tilt: torch.Tensor) -> torch.Tensor:
    """Soft version of the hard failure test, with no learned weights.

    Each threshold becomes a sigmoid of width 0.02 m (height) or 0.04 (tilt),
    the same form as the velocity task's soft fall. Selected by
    ``algorithm.termination_head: analytic``; the paper default is learned.
    """
    p_height = torch.sigmoid((height_err.abs() - FAILURE_HEIGHT_M) / 0.02)
    p_tilt = torch.sigmoid((tilt.abs() - FAILURE_TILT) / 0.04)
    return 1.0 - (1.0 - p_height) * (1.0 - p_tilt)


def termination_head_input(
    height_err: torch.Tensor, tilt: torch.Tensor
) -> torch.Tensor:
    """Termination-head features ``[height, tilt, |height|, |tilt|]``."""
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
    """Planar position distance and absolute yaw difference to the reference."""
    pos_drift = (state[..., :2] - ref_state[..., :2]).norm(dim=-1)
    yaw_drift = heading_yaw(state[..., 3:7]) - heading_yaw(ref_state[..., 3:7])
    yaw_drift = torch.atan2(torch.sin(yaw_drift), torch.cos(yaw_drift)).abs()
    return pos_drift, yaw_drift


assert NUM_BODIES == 14
