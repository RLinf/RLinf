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

"""Closed-form G1 joint angles and rates from tracked body rotations.

Every tracked body other than the pelvis hangs below a tracked ancestor
through two or three hinges. Pushing the fixed link rotations through the
hinges turns the relative rotation of the pair into one XYZ Euler product in
a per-chain frame, so each chain is solved by ``atan2`` and ``asin``. A
three-hinge chain has a second solution ``(u1 + pi, pi - u2, u3 + pi)``; the
one closer to the joint range, then to the neutral pose, is kept.

The method follows HybridRobotics/supermimic (Apache-2.0).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from rlinf.envs.sim.g1_mimic.features import quat_to_rotmat, rot_apply, rot_mul
from rlinf.envs.sim.g1_mimic.g1_fk import (
    g1_model,
    joint_qpos_addresses,
    tracked_body_ids,
)

_TWO_PI = 2.0 * math.pi
# Keeps the derivative of asin finite where |sin(u2)| reaches 1.
_ASIN_MARGIN = 1e-6


def _bounded_atan2(y: torch.Tensor, x: torch.Tensor, damp: float) -> torch.Tensor:
    """``atan2(y, x)`` whose gradient is at most ``1 / (2 sqrt(damp))``."""
    x0, y0 = x.detach(), y.detach()
    return torch.atan2(y0, x0) + (x0 * y - y0 * x) / (x0 * x0 + y0 * y0 + damp)


def _value_with_grad_of(value: torch.Tensor, surrogate: torch.Tensor) -> torch.Tensor:
    """``value`` forward, ``surrogate``'s gradient backward."""
    return surrogate + (value - surrogate).detach()


def _chain_constants(model, body: int, row_of: dict[int, int], col_of: dict[int, int]):
    """Constants of the chain from ``body`` up to its tracked ancestor.

    Returns ``(parent_row, parent_frame, child_frame, sign, neutral, low,
    high, columns)``, padded to three hinges.
    """
    path = [body]
    while int(model.body_parentid[path[-1]]) not in row_of:
        path.append(int(model.body_parentid[path[-1]]))
    ancestor = int(model.body_parentid[path[-1]])
    fixed = torch.eye(3, dtype=torch.float64)
    axes, joints = [], []
    for link in reversed(path):
        quat = torch.from_numpy(model.body_quat[link].copy()).double()
        fixed = fixed @ quat_to_rotmat(quat)
        joint = int(model.body_jntadr[link])
        if joint >= 0:
            axis = torch.from_numpy(model.jnt_axis[joint].copy()).double()
            axes.append(fixed @ axis)
            joints.append(joint)
    if len(joints) not in (2, 3):
        raise RuntimeError(
            f"G1 body {body} is {len(joints)} hinges below its tracked ancestor; "
            "the closed form needs two or three"
        )
    basis = torch.stack((axes[0], axes[1], torch.linalg.cross(axes[0], axes[1])))
    count = len(joints)
    qpos_adr = [int(model.jnt_qposadr[j]) for j in joints]
    sign = torch.zeros(3, dtype=torch.float64)
    neutral = torch.zeros(3, dtype=torch.float64)
    low = torch.zeros(3, dtype=torch.float64)
    high = torch.zeros(3, dtype=torch.float64)
    sign[:count] = 1.0
    if count == 3:
        sign[2] = round(float(axes[2] @ basis[2]))
    neutral[:count] = torch.from_numpy(model.qpos0[qpos_adr].copy())
    ranges = torch.from_numpy(model.jnt_range[joints].copy())
    low[:count], high[:count] = ranges[:, 0], ranges[:, 1]
    columns = [col_of[adr] for adr in qpos_adr] + [-1] * (3 - count)
    return (
        row_of[ancestor],
        basis.T,
        (basis @ fixed).T,
        sign,
        neutral,
        low,
        high,
        columns,
    )


class ChainInverseKinematics(nn.Module):
    """Batched joint solve for the 14 tracked G1 bodies.

    The chain constants are float32 buffers outside the state dict.
    """

    def __init__(self):
        super().__init__()
        model = g1_model()
        tracked = tracked_body_ids()
        row_of = {body: row for row, body in enumerate(tracked)}
        col_of = {adr: col for col, adr in enumerate(joint_qpos_addresses())}
        chains = [_chain_constants(model, body, row_of, col_of) for body in tracked[1:]]
        parent, a, b, sign, neutral, low, high, columns = zip(*chains, strict=True)
        slot_column = torch.tensor(columns).flatten()
        unused = int(slot_column.lt(0).sum())
        buffers = {
            "parent_rows": torch.tensor(parent),
            "child_rows": torch.arange(1, len(tracked)),
            "parent_frame": torch.stack(a).float(),
            "child_frame": torch.stack(b).float(),
            "sign": torch.stack(sign).float(),
            "neutral": torch.stack(neutral).float(),
            "low": torch.stack(low).float(),
            "high": torch.stack(high).float(),
            "has_third": torch.stack(sign)[:, 2] != 0,
            "column_order": slot_column.argsort()[unused:],
        }
        for name, value in buffers.items():
            self.register_buffer(name, value, persistent=False)

    def forward(
        self, rots: torch.Tensor, ang_vels: torch.Tensor, damp: float = 1e-2
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint angles and rates ``[..., 29]`` in ``G1_JOINT_NAMES`` order.

        ``rots`` ``[..., 14, 3, 3]`` and ``ang_vels`` ``[..., 14, 3]`` are the
        tracked bodies in one common frame. ``damp`` bounds every gradient and,
        within ``cos^2(u2) < damp`` of a three-hinge gimbal lock, the rates of
        its outer pair; elsewhere the values are exact.
        """
        frame = rot_mul(rots[..., self.parent_rows, :, :], self.parent_frame)
        frame = frame.transpose(-1, -2)
        euler = rot_mul(
            frame, rot_mul(rots[..., self.child_rows, :, :], self.child_frame)
        )
        rel = ang_vels[..., self.child_rows, :] - ang_vels[..., self.parent_rows, :]
        w = rot_apply(frame, rel)

        two_first = _bounded_atan2(euler[..., 2, 1], euler[..., 1, 1], damp)
        two_second = _bounded_atan2(euler[..., 0, 2], euler[..., 0, 0], damp)
        first = _bounded_atan2(-euler[..., 1, 2], euler[..., 2, 2], damp)
        second = torch.asin(
            euler[..., 0, 2].clamp(-1.0 + _ASIN_MARGIN, 1.0 - _ASIN_MARGIN)
        )
        third = _bounded_atan2(-euler[..., 0, 1], euler[..., 0, 0], damp)
        u = torch.where(
            self.has_third[:, None],
            torch.stack((first, second, third), dim=-1),
            torch.stack((two_first, two_second, torch.zeros_like(third)), dim=-1),
        )
        mirror = torch.stack(
            (first + math.pi, math.pi - second, third + math.pi), dim=-1
        )
        q, q_mirror = self._angles(u), self._angles(mirror)
        use_mirror = (
            self.has_third & (self._cost(q_mirror) < self._cost(q))
        ).unsqueeze(-1)
        u = torch.where(use_mirror, mirror, u)
        q = torch.where(use_mirror, q_mirror, q)

        num = euler[..., 1, 2] * w[..., 1] + euler[..., 2, 2] * w[..., 2]
        cos_sq = euler[..., 1, 2].square() + euler[..., 2, 2].square()
        rate3 = _value_with_grad_of(num / cos_sq.clamp(min=damp), num / (cos_sq + damp))
        rate3 = torch.where(self.has_third, rate3, 0.0)
        rate2 = torch.cos(u[..., 0]) * w[..., 1] + torch.sin(u[..., 0]) * w[..., 2]
        rate1 = w[..., 0] - euler[..., 0, 2] * rate3
        rates = self.sign * torch.stack((rate1, rate2, rate3), dim=-1)
        order = self.column_order
        return q.flatten(-2)[..., order], rates.flatten(-2)[..., order]

    def _angles(self, u: torch.Tensor) -> torch.Tensor:
        """Hinge angles on the turn of the circle centered on each range."""
        q = self.sign * u + self.neutral
        center = 0.5 * (self.low + self.high)
        return q - _TWO_PI * torch.round((q - center) / _TWO_PI)

    def _cost(self, q: torch.Tensor) -> torch.Tensor:
        """Range violation; the distance to neutral breaks ties."""
        violation = (self.low - q).clamp(min=0) + (q - self.high).clamp(min=0)
        return (violation + 1e-3 * (q - self.neutral).abs()).sum(-1)
