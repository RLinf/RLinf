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
    ROOT_POS,
    ROOT_QUAT,
)
from rlinf.envs.sim.g1_mimic.features import (
    heading_invariant_features,
    heading_yaw,
    normalize_xi,
    projected_gravity,
    quat_to_rotmat,
    root_error_4d,
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


def forward_kinematics_pair(
    state: torch.Tensor, ref_state: torch.Tensor
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Body poses of ``state`` and ``ref_state`` from one tree walk.

    The two batches stay independent. Reference rows usually have no
    gradient; they still share the walk so the tree is not launched twice.
    """
    if state.shape[0] != ref_state.shape[0]:
        return differentiable_fk(state), differentiable_fk(ref_state)
    pos, rot, lin, ang = differentiable_fk(torch.cat((state, ref_state), dim=0))
    n = state.shape[0]
    return (
        (pos[:n], rot[:n], lin[:n], ang[:n]),
        (pos[n:], rot[n:], lin[n:], ang[n:]),
    )


def anchor_index(reward_anchor: str) -> int:
    """Body index used to re-anchor the tracking reward."""
    if reward_anchor == "torso":
        return TORSO
    if reward_anchor == "pelvis":
        return PELVIS
    raise ValueError(f"reward_anchor must be torso or pelvis, got {reward_anchor!r}")


def reanchor_bodies(
    pos: torch.Tensor,
    rot: torch.Tensor,
    ref_pos: torch.Tensor,
    ref_rot: torch.Tensor,
    reward_anchor: str = "torso",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference body poses moved onto one robot body's xy and yaw.

    BeyondMimic's ``update_relative_body_poses``. ``torso`` uses
    ``torso_link``. ``pelvis`` is zita-ch's anchor: the robot pelvis xy, the
    reference pelvis height, and the yaw of ``R_pelvis R_ref_pelvis^T``.
    Inputs are ``[B, bodies, 3]`` positions and ``[B, bodies, 3, 3]``
    rotations from ``differentiable_fk``.
    """
    anchor = anchor_index(reward_anchor)
    turn = yaw_rotation(
        heading_yaw_of_matrix(rot[:, anchor] @ ref_rot[:, anchor].transpose(-1, -2))
    )
    shift = torch.cat((pos[:, anchor, :2], ref_pos[:, anchor, 2:3]), dim=-1)
    offset = ref_pos - ref_pos[:, anchor : anchor + 1]
    anchored_pos = shift.unsqueeze(1) + torch.einsum("bij,bkj->bki", turn, offset)
    anchored_rot = turn.unsqueeze(1) @ ref_rot
    return anchored_pos, anchored_rot


def heading_yaw_of_matrix(rot: torch.Tensor) -> torch.Tensor:
    """Yaw of a rotation matrix's x axis on the ground plane."""
    return torch.atan2(rot[..., 1, 0], rot[..., 0, 0])


def zita_critic_features(
    state: torch.Tensor,
    ref_state: torch.Tensor,
    xi: torch.Tensor,
    state_fk: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | None = None,
    ref_fk: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """zita-ch critic features except the goal, which is appended separately.

    Layout: heading-invariant state (71), ``q_ref - q`` (29), 14 body-position
    errors, root position error (3), relative root Rot6D (6), then ``xi``.
    Reference bodies are shifted onto the robot pelvis xy and the reference
    pelvis height, and yawed by ``R R_ref^T``. The error is then written in
    the robot heading frame. ``xi`` is not normalized.
    """
    invariant = heading_invariant_features(state)
    joint_err = ref_state[..., JOINT_POS] - state[..., JOINT_POS]
    position = state[..., ROOT_POS]
    ref_position = ref_state[..., ROOT_POS]
    rot = quat_to_rotmat(state[..., ROOT_QUAT])
    ref_rot = quat_to_rotmat(ref_state[..., ROOT_QUAT])
    heading = yaw_rotation(heading_yaw(state[..., ROOT_QUAT]))
    turn = yaw_rotation(heading_yaw_of_matrix(rot @ ref_rot.transpose(-1, -2)))
    if state_fk is None or ref_fk is None:
        state_fk, ref_fk = forward_kinematics_pair(state, ref_state)
    body = state_fk[0]
    ref_body = ref_fk[0]
    anchor = torch.cat((position[..., :2], ref_position[..., 2:3]), dim=-1)
    offset = ref_body - ref_position.unsqueeze(-2)
    anchored = anchor.unsqueeze(-2) + torch.einsum("bij,bkj->bki", turn, offset)
    delta = anchored - body
    body_err = torch.einsum("bij,bkj->bki", heading.transpose(-1, -2), delta).flatten(
        -2
    )
    root_err = torch.einsum(
        "bij,bj->bi", heading.transpose(-1, -2), ref_position - position
    )
    relative = rotmat_to_6d(rot.transpose(-1, -2) @ ref_rot)
    return torch.cat((invariant, joint_err, body_err, root_err, relative, xi), dim=-1)


def motion_critic_input(
    state: torch.Tensor,
    goal: torch.Tensor,
    xi: torch.Tensor,
    ref_state: torch.Tensor,
    critic_features: str,
    state_fk: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | None = None,
    ref_fk: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    scale_xi: bool = False,
) -> torch.Tensor:
    """Feature vector consumed by ``CriticNet``.

    ``root4`` keeps the historical order: invariant state, goal, ``xi``, 4-d
    root error. ``zita`` appends the goal after :func:`zita_critic_features`.
    ``scale_xi`` applies zita-ch's fixed context scale before either layout.
    """
    squeezed = state.dim() == 1
    if squeezed:
        state = state.unsqueeze(0)
        goal = goal.unsqueeze(0)
        xi = xi.unsqueeze(0)
        ref_state = ref_state.unsqueeze(0)
    if scale_xi:
        xi = normalize_xi(xi)
    if critic_features == "root4":
        out = torch.cat(
            (
                heading_invariant_features(state),
                goal,
                xi,
                root_error_4d(state, ref_state),
            ),
            dim=-1,
        )
    elif critic_features == "zita":
        out = torch.cat(
            (zita_critic_features(state, ref_state, xi, state_fk, ref_fk), goal),
            dim=-1,
        )
    else:
        raise ValueError(
            f"critic_features must be root4 or zita, got {critic_features!r}"
        )
    return out.squeeze(0) if squeezed else out


def rollout_critic_feature(
    state: torch.Tensor,
    goal: torch.Tensor,
    xi: torch.Tensor,
    ref_state: torch.Tensor,
    critic_features: str,
    state_fk: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ref_fk: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    scale_xi: bool = False,
    keep_grad: bool = False,
) -> torch.Tensor:
    """Critic features at one imagined state.

    The actor bootstraps from the terminal value only. ``keep_grad`` is for
    that state, so the value stays on the predicted state. Earlier steps are
    the critic's regression inputs and leave the rollout graph. Dynamics and
    the reward keep the original tensors.
    """
    if not keep_grad:
        state = state.detach()
        goal = goal.detach()
        xi = xi.detach()
        ref_state = ref_state.detach()
        state_fk = (
            state_fk[0].detach(),
            state_fk[1].detach(),
            state_fk[2].detach(),
            state_fk[3].detach(),
        )
        ref_fk = (
            ref_fk[0].detach(),
            ref_fk[1].detach(),
            ref_fk[2].detach(),
            ref_fk[3].detach(),
        )
    return motion_critic_input(
        state,
        goal,
        xi,
        ref_state,
        critic_features,
        state_fk,
        ref_fk,
        scale_xi=scale_xi,
    )


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
    state_fk: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | None = None,
    ref_fk: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    reward_anchor: str = "torso",
) -> dict[str, torch.Tensor]:
    """Table A.2 terms, no self-collision, joint-limit weight -2, times dt.

    ``reward_anchor`` selects the body whose xy and yaw snap the reference
    for the four body terms, and whose world pose is the root position and
    orientation error. ``torso`` is this repository's reward. ``pelvis`` is
    zita-ch's.
    """
    squeezed = state.dim() == 1
    if squeezed:
        state = state.unsqueeze(0)
        ref_state = ref_state.unsqueeze(0)
        action = action.unsqueeze(0)
        prev_action = prev_action.unsqueeze(0)
    if state_fk is None or ref_fk is None:
        state_fk, ref_fk = forward_kinematics_pair(state, ref_state)
    pos, rot, lin, ang = state_fk
    ref_pos, ref_rot, ref_lin, ref_ang = ref_fk
    # Root terms stay in the world frame, so global drift stays in them.
    # Body pose terms re-anchor the reference at ``reward_anchor``; body
    # velocities are compared in the world frame, as in BeyondMimic.
    anchor = anchor_index(reward_anchor)
    e_root_pos = (pos[:, anchor] - ref_pos[:, anchor]).square().sum(dim=-1)
    e_root_ori = _rotation_angle_sq(rot[:, anchor], ref_rot[:, anchor])
    apos, arot = reanchor_bodies(pos, rot, ref_pos, ref_rot, reward_anchor)
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
