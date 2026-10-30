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

Bodies at the same depth are one batched walk. A CUDA batch of 256 or more
is compiled once per process so the walk is not a kernel launch per body.
``SUPERMIMIC_FK_EAGER=1`` keeps the batched walk but skips that compile.
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING

import torch

from rlinf.envs.sim.g1_mimic.constants import ACTION_DIM, G1_JOINT_NAMES
from rlinf.envs.sim.g1_mimic.features import quat_normalize, quat_to_rotmat

if TYPE_CHECKING:
    import mujoco


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
    """``quat_to_rotmat(quat) @ vec`` without building the matrix.

    The tree walk calls this a few times per body; the rotation-matrix form
    launched about 25 kernels per call and dominated update time.
    """
    if vec.dim() == 1:
        vec = vec.expand(quat.shape[0], 3)
    quat = quat_normalize(quat)
    w = quat[..., :1]
    u = quat[..., 1:]
    t = 2.0 * torch.cross(u, vec, dim=-1)
    return vec + w * t + torch.cross(u, t, dim=-1)


def _axis_angle_quat(unit_axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    half = (0.5 * angle).unsqueeze(-1)
    return torch.cat((torch.cos(half), unit_axis * torch.sin(half)), dim=-1)


def _hamilton_table() -> torch.Tensor:
    """``C`` with ``(q1 * q2)_k = sum_ij q1_i q2_j C[i, j, k]``, wxyz order."""
    table = torch.zeros(4, 4, 4)
    for i, j, k, sign in (
        (0, 0, 0, 1),
        (1, 1, 0, -1),
        (2, 2, 0, -1),
        (3, 3, 0, -1),
        (0, 1, 1, 1),
        (1, 0, 1, 1),
        (2, 3, 1, 1),
        (3, 2, 1, -1),
        (0, 2, 2, 1),
        (1, 3, 2, -1),
        (2, 0, 2, 1),
        (3, 1, 2, 1),
        (0, 3, 3, 1),
        (1, 2, 3, 1),
        (2, 1, 3, -1),
        (3, 0, 3, 1),
    ):
        table[i, j, k] = sign
    return table


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """``features.quat_mul`` in three kernels instead of about thirty.

    Elementwise products and a sum, not a matmul, so TF32 settings elsewhere
    in a process cannot change the precision.
    """
    outer = q1.unsqueeze(-1).unsqueeze(-1) * q2.unsqueeze(-2).unsqueeze(-1)
    return (outer * table).sum(dim=(-3, -2))


def _run_levels(state, table, tracked, nbody, levels):
    """One batched walk per depth. ``levels`` is static for a given MJCF."""
    batch = state.shape[0]
    positions = [None] * nbody
    quats = [None] * nbody
    linear = [None] * nbody
    angular = [None] * nbody
    positions[0] = torch.zeros(batch, 3, device=state.device, dtype=state.dtype)
    identity = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], device=state.device, dtype=state.dtype
    )
    quats[0] = identity.expand(batch, 4)
    linear[0] = torch.zeros(batch, 3, device=state.device, dtype=state.dtype)
    angular[0] = torch.zeros(batch, 3, device=state.device, dtype=state.dtype)
    for (
        kind,
        body_ids,
        parent_ids,
        bpos,
        bquat,
        hinge_idx,
        axis,
        unit,
        anchor,
        has_anchor,
    ) in levels:
        parent_pos = torch.stack([positions[p] for p in parent_ids], 1)
        parent_quat = torch.stack([quats[p] for p in parent_ids], 1)
        parent_v = torch.stack([linear[p] for p in parent_ids], 1)
        parent_w = torch.stack([angular[p] for p in parent_ids], 1)
        if kind == "free":
            pos = state[:, 0:3].unsqueeze(1)
            quat = state[:, 3:7].unsqueeze(1)
            vel = state[:, 36:39].unsqueeze(1)
            ang = _rotate(state[:, 3:7], state[:, 39:42]).unsqueeze(1)
        else:
            bpos_b = bpos.unsqueeze(0)
            bquat_b = bquat.unsqueeze(0)
            axis_b = axis.unsqueeze(0)
            unit_b = unit.unsqueeze(0)
            anchor_b = anchor.unsqueeze(0)
            pos = parent_pos + _rotate(parent_quat, bpos_b)
            quat = _quat_mul(parent_quat, bquat_b, table)
            vel = parent_v + torch.cross(parent_w, pos - parent_pos, dim=-1)
            ang = parent_w
            angle = state[:, 7 + hinge_idx]
            rate = state[:, 42 + hinge_idx]
            joint_quat = _axis_angle_quat(unit_b, angle)
            if has_anchor:
                world_anchor = pos + _rotate(quat, anchor_b)
                quat = _quat_mul(quat, joint_quat, table)
                pos = world_anchor - _rotate(quat, anchor_b)
            else:
                quat = _quat_mul(quat, joint_quat, table)
            axis_w = _rotate(quat, axis_b)
            rate_b = rate.unsqueeze(-1)
            ang = ang + axis_w * rate_b
            anchor_w = pos + _rotate(quat, anchor_b)
            vel = vel + torch.cross(axis_w * rate_b, pos - anchor_w, dim=-1)
        quat = quat_normalize(quat)
        for slot, body_id in enumerate(body_ids):
            positions[body_id] = pos[:, slot]
            quats[body_id] = quat[:, slot]
            linear[body_id] = vel[:, slot]
            angular[body_id] = ang[:, slot]
    pos_all = torch.stack(positions, 1)
    quat_all = torch.stack(quats, 1)
    lin_all = torch.stack(linear, 1)
    ang_all = torch.stack(angular, 1)
    rot = quat_to_rotmat(quat_all[:, tracked])
    return (
        pos_all[:, tracked],
        rot,
        lin_all[:, tracked],
        ang_all[:, tracked],
    )


_COMPILED_LEVELS: dict = {}
_COMPILE_FAILED = False
_COMPILE_ANNOUNCED = False


def _compiled_levels(state, table, tracked, nbody, levels):
    """Compile ``_run_levels`` once per device and dtype. Falls back on failure."""
    global _COMPILE_FAILED, _COMPILE_ANNOUNCED
    if _COMPILE_FAILED:
        return _run_levels(state, table, tracked, nbody, levels)
    key = (str(state.device), state.dtype)
    fn = _COMPILED_LEVELS.get(key)
    if fn is None:
        if not _COMPILE_ANNOUNCED:
            warnings.warn(
                "Compiling the G1 kinematic tree for CUDA. The first large "
                "batch waits for this; later batches reuse the compiled walk.",
                RuntimeWarning,
                stacklevel=3,
            )
            _COMPILE_ANNOUNCED = True

        def eager(
            state,
            table=table,
            tracked=tracked,
            nbody=nbody,
            levels=levels,
        ):
            return _run_levels(state, table, tracked, nbody, levels)

        fn = torch.compile(eager, dynamic=True, fullgraph=True)
        _COMPILED_LEVELS[key] = fn
    try:
        return fn(state)
    except Exception as exc:
        name = type(exc).__name__
        module = type(exc).__module__
        if module.startswith("torch._dynamo") or "Compiler" in name:
            _COMPILE_FAILED = True
            warnings.warn(
                f"G1 kinematics stayed on the batched walk ({name}).",
                RuntimeWarning,
                stacklevel=3,
            )
            return _run_levels(state, table, tracked, nbody, levels)
        raise


class G1Kinematics:
    """Batched kinematics. Constants follow ``state`` to its device."""

    def __init__(self, model: mujoco.MjModel) -> None:
        import mujoco

        hinge = int(mujoco.mjtJoint.mjJNT_HINGE)
        free = int(mujoco.mjtJoint.mjJNT_FREE)
        names = []
        for j in range(model.njnt):
            if int(model.jnt_type[j]) == hinge:
                names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
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
        # Tracked index of the free-joint body. Its position is state[:, 0:3].
        free_bodies = [i for i, owned in enumerate(joints) if ("free",) in owned]
        tracked = self.tracked.tolist()
        self.free_tracked = (
            tracked.index(free_bodies[0])
            if len(free_bodies) == 1 and free_bodies[0] in tracked
            else None
        )
        self.joint_low = torch.tensor(lows, dtype=torch.float32)
        self.joint_high = torch.tensor(highs, dtype=torch.float32)
        if self.joint_low.shape != (ACTION_DIM,) or not torch.all(
            self.joint_high > self.joint_low
        ):
            raise RuntimeError(
                f"expected {ACTION_DIM} G1 hinge ranges with high > low, found "
                f"low={self.joint_low.tolist()}, high={self.joint_high.tolist()}"
            )
        self._cache: dict[tuple, tuple] = {}
        self._level_cache: dict[tuple, tuple] = {}
        self._level_specs = self._build_level_specs()

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
                        axis = axis.to(device=device, dtype=dtype)
                        row.append(
                            (
                                "hinge",
                                index,
                                axis,
                                anchor.to(device=device, dtype=dtype),
                                has_anchor,
                                axis / axis.norm().clamp_min(1e-8),
                            )
                        )
                packed.append(row)
            self._cache[key] = (
                self.body_pos.to(device=device, dtype=dtype),
                self.body_quat.to(device=device, dtype=dtype),
                self.tracked.to(device=device),
                packed,
                _hamilton_table().to(device=device, dtype=dtype),
            )
        return self._cache[key]

    def _build_level_specs(self):
        """Group same-depth bodies that share a joint kind. None if any body is irregular."""
        if any(len(owned) != 1 for i, owned in enumerate(self.joints) if i > 0):
            return None
        depth = [0] * self.nbody
        for i in range(1, self.nbody):
            depth[i] = depth[self.parents[i]] + 1
        specs = []
        for level in range(1, max(depth) + 1):
            ids = [i for i in range(self.nbody) if depth[i] == level]
            groups: dict[tuple, list[int]] = {}
            for i in ids:
                joint = self.joints[i][0]
                if joint[0] == "free":
                    key = ("free", False)
                else:
                    key = ("hinge", bool(joint[4]))
                groups.setdefault(key, []).append(i)
            for (kind, has_anchor), members in groups.items():
                parent_ids = tuple(self.parents[i] for i in members)
                bpos = torch.stack([self.body_pos[i] for i in members])
                bquat = torch.stack([self.body_quat[i] for i in members])
                if kind == "free":
                    empty = torch.zeros(len(members), 3)
                    specs.append(
                        (
                            kind,
                            tuple(members),
                            parent_ids,
                            bpos,
                            bquat,
                            torch.zeros(len(members), dtype=torch.long),
                            empty,
                            empty,
                            empty,
                            False,
                        )
                    )
                    continue
                hinge_idx = torch.tensor(
                    [self.joints[i][0][1] for i in members], dtype=torch.long
                )
                axis = torch.stack([self.joints[i][0][2] for i in members])
                anchor = torch.stack([self.joints[i][0][3] for i in members])
                specs.append(
                    (
                        kind,
                        tuple(members),
                        parent_ids,
                        bpos,
                        bquat,
                        hinge_idx,
                        axis,
                        axis,
                        anchor,
                        has_anchor,
                    )
                )
        return tuple(specs)

    def _level_pack(self, device, dtype):
        key = (str(device), dtype)
        cached = self._level_cache.get(key)
        if cached is None:
            _body_pos, _body_quat, tracked, _joints, table = self._packed(device, dtype)
            moved = []
            for spec in self._level_specs:
                (
                    kind,
                    body_ids,
                    parent_ids,
                    bpos,
                    bquat,
                    hinge_idx,
                    axis,
                    _unit,
                    anchor,
                    has_anchor,
                ) = spec
                axis_d = axis.to(device=device, dtype=dtype)
                unit = axis_d / axis_d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                moved.append(
                    (
                        kind,
                        body_ids,
                        parent_ids,
                        bpos.to(device=device, dtype=dtype),
                        bquat.to(device=device, dtype=dtype),
                        hinge_idx.to(device=device),
                        axis_d,
                        unit,
                        anchor.to(device=device, dtype=dtype),
                        has_anchor,
                    )
                )
            cached = (table, tracked.to(device=device), self.nbody, tuple(moved))
            self._level_cache[key] = cached
        return cached

    def _forward_levels(self, state: torch.Tensor):
        """Batched depth walk, without the CUDA compile."""
        squeezed = state.dim() == 1
        if squeezed:
            state = state.unsqueeze(0)
        table, tracked, nbody, levels = self._level_pack(state.device, state.dtype)
        out = _run_levels(state, table, tracked, nbody, levels)
        if squeezed:
            return tuple(item.squeeze(0) for item in out)
        return out

    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return tracked body position, rotation, linear and angular velocity.

        Shapes are ``[B, 14, ...]``. ``state`` is the 71-d simulator state.
        Root linear velocity is world-frame. Root angular velocity is body-frame,
        which is how the free joint stores it. Large CUDA batches compile the
        depth walk; the body-by-body loop remains for an irregular tree.
        A scan that is already being compiled inlines the eager walk so the
        two compiles do not nest.
        """
        if torch.compiler.is_compiling():
            if self._level_specs is not None:
                return self._forward_levels(state)
            return self._forward_serial(state)
        batch = 1 if state.dim() == 1 else state.shape[0]
        eager_only = os.environ.get("SUPERMIMIC_FK_EAGER") == "1"
        if (
            self._level_specs is not None
            and state.is_cuda
            and batch >= 256
            and not eager_only
        ):
            squeezed = state.dim() == 1
            if squeezed:
                state = state.unsqueeze(0)
            table, tracked, nbody, levels = self._level_pack(state.device, state.dtype)
            out = _compiled_levels(state, table, tracked, nbody, levels)
            if squeezed:
                return tuple(item.squeeze(0) for item in out)
            return out
        if self._level_specs is not None:
            return self._forward_levels(state)
        return self._forward_serial(state)

    def _forward_serial(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Body-by-body walk. Reference for the depth-batched path."""
        squeezed = state.dim() == 1
        if squeezed:
            state = state.unsqueeze(0)
        batch = state.shape[0]
        device, dtype = state.device, state.dtype
        body_pos, body_quat, tracked, joints, table = self._packed(device, dtype)
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
            quat_i = _quat_mul(quats[pid], body_quat[i].expand(batch, 4), table)
            w = angular[pid]
            v = linear[pid] + torch.cross(angular[pid], pos_i - positions[pid], dim=-1)
            for joint in joints[i]:
                if joint[0] == "free":
                    pos_i = state[:, 0:3]
                    quat_i = state[:, 3:7]
                    v = state[:, 36:39]
                    w = _rotate(quat_i, state[:, 39:42])
                else:
                    _, index, axis, anchor, has_anchor, unit_axis = joint
                    angle = state[:, 7 + index]
                    rate = state[:, 42 + index]
                    qj = _axis_angle_quat(unit_axis, angle)
                    if has_anchor:
                        world_anchor = pos_i + _rotate(quat_i, anchor)
                        quat_i = _quat_mul(quat_i, qj, table)
                        pos_i = world_anchor - _rotate(quat_i, anchor)
                    else:
                        quat_i = _quat_mul(quat_i, qj, table)
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
