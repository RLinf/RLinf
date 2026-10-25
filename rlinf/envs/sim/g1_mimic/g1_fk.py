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

"""Differentiable forward kinematics of the Unitree G1 tree.

The tree is read from the mjlab G1 MJCF. Positions match MuJoCo ``xpos``.
Linear velocity is the body-frame origin, and angular velocity is in the
world frame, matching ``mj_objectVelocity`` after the COM-to-origin shift.
"""

from __future__ import annotations

import torch

from rlinf.envs.sim.g1_mimic.constants import ACTION_DIM, G1_JOINT_NAMES
from rlinf.envs.sim.g1_mimic.features import quat_mul, quat_normalize, quat_to_rotmat

# mjlab's G1 tracking bodies. Pelvis is the termination body. Torso is the
# reward root. They are not the same link.
TRACKED_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
PELVIS_BODY = TRACKED_BODY_NAMES.index("pelvis")
TORSO_BODY = TRACKED_BODY_NAMES.index("torso_link")


def _rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    if vec.dim() == 1:
        vec = vec.expand(quat.shape[0], 3)
    rot = quat_to_rotmat(quat)
    return torch.einsum("...ij,...j->...i", rot, vec)


def _axis_angle_quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / axis.norm().clamp_min(1e-8)
    half = 0.5 * angle
    sine = torch.sin(half)
    return torch.stack(
        (torch.cos(half), axis[0] * sine, axis[1] * sine, axis[2] * sine),
        dim=-1,
    )


class G1Kinematics:
    """Batched kinematics. Constants follow ``state`` to its device."""

    def __init__(self, model) -> None:
        import mujoco

        hinge = int(mujoco.mjtJoint.mjJNT_HINGE)
        free = int(mujoco.mjtJoint.mjJNT_FREE)
        names = []
        for j in range(model.njnt):
            if int(model.jnt_type[j]) == hinge:
                names.append(
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                )
        if tuple(names) != G1_JOINT_NAMES:
            raise RuntimeError(
                "G1 MJCF joint order does not match G1_JOINT_NAMES. "
                f"Model starts {names[:3]}, expected {G1_JOINT_NAMES[:3]}."
            )
        body_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            for i in range(model.nbody)
        ]
        missing = [name for name in TRACKED_BODY_NAMES if name not in body_names]
        if missing:
            raise RuntimeError(f"G1 MJCF is missing tracking bodies {missing}")
        self.nbody = int(model.nbody)
        self.parents = [int(model.body_parentid[i]) for i in range(self.nbody)]
        self.body_pos = torch.tensor(model.body_pos, dtype=torch.float32)
        self.body_quat = torch.tensor(model.body_quat, dtype=torch.float32)
        self.tracked = torch.tensor(
            [body_names.index(name) for name in TRACKED_BODY_NAMES],
            dtype=torch.long,
        )
        joints: list[list[tuple]] = []
        hinge_i = 0
        lows: list[float] = []
        highs: list[float] = []
        for i in range(self.nbody):
            owned = []
            jadr = int(model.body_jntadr[i])
            jnum = int(model.body_jntnum[i])
            if jadr >= 0:
                for k in range(jnum):
                    j = jadr + k
                    jtype = int(model.jnt_type[j])
                    if jtype == free:
                        owned.append(("free",))
                    elif jtype == hinge:
                        anchor = torch.tensor(model.jnt_pos[j], dtype=torch.float32)
                        owned.append(
                            (
                                "hinge",
                                hinge_i,
                                torch.tensor(model.jnt_axis[j], dtype=torch.float32),
                                anchor,
                                float(anchor.norm()) > 1e-8,
                            )
                        )
                        lows.append(float(model.jnt_range[j, 0]))
                        highs.append(float(model.jnt_range[j, 1]))
                        hinge_i += 1
                    else:
                        raise RuntimeError(f"unsupported G1 joint type {jtype}")
            joints.append(owned)
        if hinge_i != ACTION_DIM:
            raise RuntimeError(f"expected {ACTION_DIM} hinge joints, found {hinge_i}")
        self.joints = joints
        self.joint_low = torch.tensor(lows, dtype=torch.float32)
        self.joint_high = torch.tensor(highs, dtype=torch.float32)
        if self.joint_low.shape != (ACTION_DIM,) or not torch.all(
            self.joint_high > self.joint_low
        ):
            raise RuntimeError("G1 hinge ranges are missing or not ordered")
        self._cache: dict[tuple, tuple] = {}

    def _packed(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._cache:
            packed = []
            for owned in self.joints:
                row = []
                for joint in owned:
                    if joint[0] == "free":
                        row.append(("free",))
                    else:
                        _, index, axis, anchor, has_anchor = joint
                        row.append(
                            (
                                "hinge",
                                index,
                                axis.to(device=device, dtype=dtype),
                                anchor.to(device=device, dtype=dtype),
                                has_anchor,
                            )
                        )
                packed.append(row)
            self._cache[key] = (
                self.body_pos.to(device=device, dtype=dtype),
                self.body_quat.to(device=device, dtype=dtype),
                self.tracked.to(device=device),
                packed,
            )
        return self._cache[key]

    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return tracked body position, rotation, linear and angular velocity.

        Shapes are ``[B, 14, ...]``. ``state`` is the 71-d simulator state.
        Root linear velocity is world-frame. Root angular velocity is body-frame,
        which is how the free joint stores it.
        """
        squeezed = state.dim() == 1
        if squeezed:
            state = state.unsqueeze(0)
        batch = state.shape[0]
        device, dtype = state.device, state.dtype
        body_pos, body_quat, tracked, joints = self._packed(device, dtype)
        # Lists, not slice writes. Imagination backprop reads parent poses
        # after the child is computed, and an inplace write bumps the version.
        positions = [torch.zeros(batch, 3, device=device, dtype=dtype)]
        quats = [
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype).expand(
                batch, 4
            )
        ]
        linear = [torch.zeros(batch, 3, device=device, dtype=dtype)]
        angular = [torch.zeros(batch, 3, device=device, dtype=dtype)]
        for i in range(1, self.nbody):
            pid = self.parents[i]
            pos_i = positions[pid] + _rotate(quats[pid], body_pos[i])
            quat_i = quat_mul(quats[pid], body_quat[i].expand(batch, 4))
            w = angular[pid]
            v = linear[pid] + torch.cross(
                angular[pid], pos_i - positions[pid], dim=-1
            )
            for joint in joints[i]:
                if joint[0] == "free":
                    pos_i = state[:, 0:3]
                    quat_i = state[:, 3:7]
                    v = state[:, 36:39]
                    w = _rotate(quat_i, state[:, 39:42])
                else:
                    _, index, axis, anchor, has_anchor = joint
                    angle = state[:, 7 + index]
                    rate = state[:, 42 + index]
                    qj = _axis_angle_quat(axis, angle)
                    if has_anchor:
                        world_anchor = pos_i + _rotate(quat_i, anchor)
                        quat_i = quat_mul(quat_i, qj)
                        pos_i = world_anchor - _rotate(quat_i, anchor)
                    else:
                        quat_i = quat_mul(quat_i, qj)
                    axis_w = _rotate(quat_i, axis)
                    w = w + axis_w * rate.unsqueeze(-1)
                    anchor_w = pos_i + _rotate(quat_i, anchor)
                    v = v + torch.cross(
                        axis_w * rate.unsqueeze(-1), pos_i - anchor_w, dim=-1
                    )
            positions.append(pos_i)
            quats.append(quat_normalize(quat_i))
            linear.append(v)
            angular.append(w)
        pos = torch.stack(positions, dim=1)
        quat = torch.stack(quats, dim=1)
        lin = torch.stack(linear, dim=1)
        ang = torch.stack(angular, dim=1)
        sel = tracked
        rot = quat_to_rotmat(quat[:, sel])
        out = (pos[:, sel], rot, lin[:, sel], ang[:, sel])
        if squeezed:
            return tuple(item.squeeze(0) for item in out)
        return out


_KINEMATICS: G1Kinematics | None = None


def g1_kinematics() -> G1Kinematics:
    """Compile the mjlab G1 spec once and reuse the tree."""
    global _KINEMATICS
    if _KINEMATICS is None:
        from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

        _KINEMATICS = G1Kinematics(get_spec().compile())
    return _KINEMATICS


def asset_joint_limits() -> tuple[torch.Tensor, torch.Tensor]:
    """Hinge ranges from the G1 MJCF, in ``G1_JOINT_NAMES`` order."""
    kin = g1_kinematics()
    return kin.joint_low.clone(), kin.joint_high.clone()


def asset_default_state() -> torch.Tensor:
    """71-d state of mjlab's G1 knees-bent keyframe. Not a LAFAN1 clip."""
    import re

    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import KNEES_BENT_KEYFRAME

    state = torch.zeros(71)
    state[0:3] = torch.tensor(KNEES_BENT_KEYFRAME.pos, dtype=torch.float32)
    state[3:7] = torch.tensor(KNEES_BENT_KEYFRAME.rot, dtype=torch.float32)
    joints = torch.zeros(ACTION_DIM)
    for pattern, value in KNEES_BENT_KEYFRAME.joint_pos.items():
        matched = False
        for i, name in enumerate(G1_JOINT_NAMES):
            if re.fullmatch(pattern, name):
                joints[i] = float(value)
                matched = True
        if not matched:
            raise RuntimeError(f"keyframe pattern {pattern!r} matched no G1 joint")
    state[7:36] = joints
    return state
