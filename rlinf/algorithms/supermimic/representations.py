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

"""State layouts the world model predicts in.

The replay stores a layout's full state; the world model reads its
root-relative encoding, predicts velocity increments and integrates them
into the next root pose and encoding.

* ``gencoord``: the 71-d simulator state. Encoding ``[z, gravity_b, v_b,
  w_b, q, qd]`` (68), prediction ``[dv_b, dw_b, dqd]`` (35).
* ``maxcoord``: the 14 tracked bodies' world pose and twist (182). Encoding
  per body: position, 6-D rotation and twist in the pelvis frame, then the
  pelvis height and gravity (214); prediction ``[dv, dw]`` per body (84).
  Joint angles come back through :class:`ChainInverseKinematics`.

Poses integrate ``(D + 1) / (2 D)`` of the increment for ``D = 4`` physics
substeps, velocities the whole increment. The layouts follow
HybridRobotics/supermimic (Apache-2.0).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from rlinf.envs.sim.g1_mimic.constants import (
    DT,
    JOINT_POS,
    JOINT_VEL,
    NUM_BODIES,
    ROOT_ANG_VEL,
    ROOT_LIN_VEL,
    ROOT_QUAT,
    STATE_DIM,
)
from rlinf.envs.sim.g1_mimic.features import (
    quat_mul,
    quat_to_rotmat,
    rot_apply,
    rot_apply_t,
    rot_mul,
    rot_mul_t,
    rotmat_to_6d,
)
from rlinf.envs.sim.g1_mimic.g1_ik import ChainInverseKinematics
from rlinf.envs.sim.g1_mimic.mjlab_env import DECIMATION
from rlinf.envs.sim.g1_mimic.reward import differentiable_fk

POSE_WEIGHT = (DECIMATION + 1) / (2 * DECIMATION)


def rotation_from_6d(six: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt rotation ``[..., 3, 3]`` from two stored columns ``[..., 6]``."""
    cols = six.unflatten(-1, (3, 2))
    first = F.normalize(cols[..., 0], dim=-1)
    second = cols[..., 1]
    second = F.normalize(
        second - (first * second).sum(-1, keepdim=True) * first, dim=-1
    )
    return torch.stack((first, second, torch.linalg.cross(first, second)), dim=-1)


def _rotation_parts(
    rotvec: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unit axis, cosine and sine of a rotation vector, smooth at zero."""
    angle = torch.sqrt(rotvec.square().sum(-1, keepdim=True) + 1e-8)
    return rotvec / angle, torch.cos(angle), torch.sin(angle)


def _rotate(
    vec: torch.Tensor, axis: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rodrigues rotation of ``vec`` by precomputed axis parts."""
    along = (axis * vec).sum(-1, keepdim=True) * axis
    return cos * vec + sin * torch.linalg.cross(axis, vec) + (1.0 - cos) * along


def integrate_quat(
    quat: torch.Tensor, ang_vel_w: torch.Tensor, dt: float
) -> torch.Tensor:
    """Unit wxyz quaternion after turning at world rate ``ang_vel_w`` for ``dt``."""
    rotvec = ang_vel_w * dt
    angle = torch.sqrt(rotvec.square().sum(-1, keepdim=True) + 1e-8)
    turn = torch.cat(
        (torch.cos(0.5 * angle), torch.sin(0.5 * angle) * rotvec / angle), dim=-1
    )
    out = quat_mul(turn, quat)
    return out / out.norm(dim=-1, keepdim=True)


def _down_in(rot: torch.Tensor) -> torch.Tensor:
    """World ``(0, 0, -1)`` in the frame of ``rot``."""
    return -rot[..., 2, :]


def _root_blocks(root: torch.Tensor) -> dict[str, torch.Tensor]:
    """Root position and rotation as squared-error channels.

    The rotation is scaled by ``sqrt(1/2)``: half the squared Frobenius
    distance is ``2 (1 - cos theta)``, which is ``theta^2`` to leading order.
    """
    rot = quat_to_rotmat(root[..., 3:7]).flatten(-2)
    return {"root_pos": root[..., :3], "root_ori": rot * math.sqrt(0.5)}


class GeneralizedCoordinates(nn.Module):
    """The 71-d simulator state; the reward walks the G1 tree for body poses."""

    name = "gencoord"
    state_dim = STATE_DIM
    encoding_dim = 68
    prediction_dim = 35
    block_widths = {"root_pos": 3, "root_ori": 9, "joint_pos": 29, "joint_vel": 29}

    def stored_state(
        self,
        state: torch.Tensor,
        bodies: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Replay row of one simulator readout."""
        del bodies
        return state

    def root(self, full: torch.Tensor) -> torch.Tensor:
        """World root position and wxyz quaternion ``[..., 7]``."""
        return full[..., :7]

    def encode(self, full: torch.Tensor) -> torch.Tensor:
        rot = quat_to_rotmat(full[..., ROOT_QUAT])
        return torch.cat(
            (
                full[..., 2:3],
                _down_in(rot),
                rot_apply_t(rot, full[..., ROOT_LIN_VEL]),
                full[..., ROOT_ANG_VEL],
                full[..., JOINT_POS],
                full[..., JOINT_VEL],
            ),
            dim=-1,
        )

    def _split(self, enc: torch.Tensor):
        return enc[..., 4:7], enc[..., 7:10], enc[..., 10:39], enc[..., 39:68]

    def advance(
        self, root: torch.Tensor, enc: torch.Tensor, pred: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Next root pose and encoding from the increments in the current root frame."""
        lin, ang, q, qd = self._split(enc)
        dv, dw, dqd = pred[..., 0:3], pred[..., 3:6], pred[..., 6:35]
        rot = quat_to_rotmat(root[..., 3:7])
        lin_w = rot_apply(rot, lin + dv)
        ang_w = rot_apply(rot, ang + dw)
        pos = root[..., :3] + rot_apply(rot, lin + POSE_WEIGHT * dv) * DT
        quat = integrate_quat(
            root[..., 3:7], rot_apply(rot, ang + POSE_WEIGHT * dw), DT
        )
        next_rot = quat_to_rotmat(quat)
        enc = torch.cat(
            (
                pos[..., 2:3],
                _down_in(next_rot),
                rot_apply_t(next_rot, lin_w),
                rot_apply_t(next_rot, ang_w),
                q + (qd + POSE_WEIGHT * dqd) * DT,
                qd + dqd,
            ),
            dim=-1,
        )
        return torch.cat((pos, quat), dim=-1), enc

    def loss_blocks(
        self, root: torch.Tensor, enc: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        _, _, q, qd = self._split(enc)
        return {**_root_blocks(root), "joint_pos": q, "joint_vel": qd}

    def joints(self, enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, q, qd = self._split(enc)
        return q, qd

    def bodies(
        self, root: torch.Tensor, enc: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """World position, rotation and twist ``[..., 14, ...]`` of the tracked bodies."""
        lin, ang, q, qd = self._split(enc)
        rot = quat_to_rotmat(root[..., 3:7])
        state = torch.cat((root, q, rot_apply(rot, lin), ang, qd), dim=-1)
        lead = state.shape[:-1]
        out = differentiable_fk(state.reshape(-1, STATE_DIM))
        return tuple(part.reshape(*lead, *part.shape[1:]) for part in out)


class MaximalCoordinates(nn.Module):
    """World pose and twist of every tracked body; joints come back by IK."""

    name = "maxcoord"
    state_dim = NUM_BODIES * 13
    encoding_dim = NUM_BODIES * 15 + 4
    prediction_dim = NUM_BODIES * 6
    block_widths = {
        "root_pos": 3,
        "root_ori": 9,
        "body_pos": (NUM_BODIES - 1) * 3,
        "body_rot": (NUM_BODIES - 1) * 6,
        "body_vel": (NUM_BODIES - 1) * 3,
        "body_avel": (NUM_BODIES - 1) * 3,
    }

    def __init__(self):
        super().__init__()
        self.solver = ChainInverseKinematics()

    def stored_state(
        self,
        state: torch.Tensor,
        bodies: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Replay row: body positions, wxyz quaternions, linear and angular velocities."""
        del state
        return torch.cat([part.flatten(-2) for part in bodies], dim=-1)

    def _bodies_of(self, full: torch.Tensor):
        n = NUM_BODIES
        return (
            full[..., : 3 * n].unflatten(-1, (n, 3)),
            full[..., 3 * n : 7 * n].unflatten(-1, (n, 4)),
            full[..., 7 * n : 10 * n].unflatten(-1, (n, 3)),
            full[..., 10 * n : 13 * n].unflatten(-1, (n, 3)),
        )

    def root(self, full: torch.Tensor) -> torch.Tensor:
        pos, quat, _, _ = self._bodies_of(full)
        return torch.cat((pos[..., 0, :], quat[..., 0, :]), dim=-1)

    def encode(self, full: torch.Tensor) -> torch.Tensor:
        pos, quat, lin, ang = self._bodies_of(full)
        rot = quat_to_rotmat(quat)
        base = rot[..., :1, :, :]
        return torch.cat(
            (
                rot_apply_t(base, pos - pos[..., :1, :]).flatten(-2),
                rotmat_to_6d(rot_mul_t(base, rot)).flatten(-2),
                rot_apply_t(base, lin).flatten(-2),
                rot_apply_t(base, ang).flatten(-2),
                pos[..., 0, 2:3],
                _down_in(rot[..., 0, :, :]),
            ),
            dim=-1,
        )

    def _split(self, enc: torch.Tensor):
        n = NUM_BODIES
        return (
            enc[..., : 3 * n].unflatten(-1, (n, 3)),
            enc[..., 3 * n : 9 * n].unflatten(-1, (n, 6)),
            enc[..., 9 * n : 12 * n].unflatten(-1, (n, 3)),
            enc[..., 12 * n : 15 * n].unflatten(-1, (n, 3)),
        )

    def advance(
        self, root: torch.Tensor, enc: torch.Tensor, pred: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate in the current pelvis frame, then re-express in the next one."""
        n = NUM_BODIES
        pos, six, lin, ang = self._split(enc)
        dv = pred[..., : 3 * n].unflatten(-1, (n, 3))
        dw = pred[..., 3 * n : 6 * n].unflatten(-1, (n, 3))
        lin_next, ang_next = lin + dv, ang + dw
        lin_pose, ang_pose = lin + POSE_WEIGHT * dv, ang + POSE_WEIGHT * dw
        # The pelvis entry should be zero; subtracting it drops its drift.
        pelvis = pos[..., :1, :]
        moved = (pos - pelvis) + (lin_pose - lin_pose[..., :1, :]) * DT
        turn = _rotation_parts(ang_pose * DT)
        cols = six.unflatten(-1, (3, 2))
        col0, col1 = _rotate(cols[..., 0], *turn), _rotate(cols[..., 1], *turn)

        rot = quat_to_rotmat(root[..., 3:7])
        root_pos = root[..., :3] + rot_apply(
            rot, pelvis[..., 0, :] + lin_pose[..., 0, :] * DT
        )
        root_quat = integrate_quat(
            root[..., 3:7], rot_apply(rot, ang_pose[..., 0, :]), DT
        )

        back = _rotation_parts(-ang_pose[..., :1, :] * DT)
        six_next = torch.stack((_rotate(col0, *back), _rotate(col1, *back)), dim=-1)
        enc = torch.cat(
            (
                _rotate(moved, *back).flatten(-2),
                six_next.flatten(-3),
                _rotate(lin_next, *back).flatten(-2),
                _rotate(ang_next, *back).flatten(-2),
                root_pos[..., 2:3],
                _down_in(quat_to_rotmat(root_quat)),
            ),
            dim=-1,
        )
        return torch.cat((root_pos, root_quat), dim=-1), enc

    def loss_blocks(
        self, root: torch.Tensor, enc: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # Body 0 is the pelvis, scored through the root pose.
        pos, six, lin, ang = self._split(enc)
        return {
            **_root_blocks(root),
            "body_pos": pos[..., 1:, :].flatten(-2),
            "body_rot": six[..., 1:, :].flatten(-2),
            "body_vel": lin[..., 1:, :].flatten(-2),
            "body_avel": ang[..., 1:, :].flatten(-2),
        }

    def joints(self, enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, six, _, ang = self._split(enc)
        return self.solver(rotation_from_6d(six), ang)

    def bodies(
        self, root: torch.Tensor, enc: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pos, six, lin, ang = self._split(enc)
        rot = quat_to_rotmat(root[..., 3:7]).unsqueeze(-3)
        return (
            root[..., :3].unsqueeze(-2) + rot_apply(rot, pos),
            rot_mul(rot, rotation_from_6d(six)),
            rot_apply(rot, lin),
            rot_apply(rot, ang),
        )


REPRESENTATIONS = {
    GeneralizedCoordinates.name: GeneralizedCoordinates,
    MaximalCoordinates.name: MaximalCoordinates,
}


def make_representation(name: str) -> GeneralizedCoordinates | MaximalCoordinates:
    """Layout ``gencoord`` or ``maxcoord``."""
    if name not in REPRESENTATIONS:
        raise ValueError(
            f"representation must be one of {sorted(REPRESENTATIONS)}, got {name!r}"
        )
    return REPRESENTATIONS[name]()
