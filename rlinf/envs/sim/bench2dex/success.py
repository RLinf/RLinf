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

# This file contains code copied from the Bench2Dex project: https://github.com/Bench2Dex/Bench2Dex.
# The original code is licensed under the MIT License.

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def _coerce_pose(pose_or_state: Sequence[float] | dict[str, Any]) -> Sequence[float]:
    if isinstance(pose_or_state, dict):
        return pose_or_state["pose_world"]
    return pose_or_state


def pose_quat_wxyz(
    pose_or_state: Sequence[float] | dict[str, Any],
) -> tuple[float, float, float, float]:
    pose = _coerce_pose(pose_or_state)
    return float(pose[6]), float(pose[3]), float(pose[4]), float(pose[5])


def world_axis_from_pose(
    pose_or_state: Sequence[float] | dict[str, Any], local_axis: str = "z"
) -> tuple[float, float, float]:
    w, x, y, z = pose_quat_wxyz(pose_or_state)
    if local_axis == "x":
        return (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))
    if local_axis == "y":
        return (2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x))
    return (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))


def up_dot_from_pose(pose_or_state: Sequence[float] | dict[str, Any]) -> float:
    return float(world_axis_from_pose(pose_or_state, "z")[2])


def tilt_deg_from_pose(pose_or_state: Sequence[float] | dict[str, Any]) -> float:
    up_dot = max(-1.0, min(1.0, up_dot_from_pose(pose_or_state)))
    return float(math.degrees(math.acos(up_dot)))


def quat_rotate_vector(
    pose_world: Sequence[float] | dict[str, Any], local_offset: Sequence[float]
) -> tuple[float, float, float]:
    pose = _coerce_pose(pose_world)
    px, py, pz = float(local_offset[0]), float(local_offset[1]), float(local_offset[2])
    qx, qy, qz, qw = float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])
    t0 = 2.0 * (qy * pz - qz * py)
    t1 = 2.0 * (qz * px - qx * pz)
    t2 = 2.0 * (qx * py - qy * px)
    wx = px + qw * t0 + qy * t2 - qz * t1
    wy = py + qw * t1 + qz * t0 - qx * t2
    wz = pz + qw * t2 + qx * t1 - qy * t0
    return (float(pose[0]) + wx, float(pose[1]) + wy, float(pose[2]) + wz)


def get_joint_positions(state: dict[str, Any]) -> dict[str, float]:
    qpos = state.get("qpos")
    if qpos is None:
        return {}
    if isinstance(qpos, dict):
        return {str(name): float(value) for name, value in qpos.items()}
    values = list(qpos)
    joint_names = list(state.get("joint_names") or [])
    if joint_names:
        return {
            str(name): float(values[idx])
            for idx, name in enumerate(joint_names)
            if idx < len(values)
        }
    return {f"joint_{idx}": float(value) for idx, value in enumerate(values)}


def get_joint_position(
    state: dict[str, Any], joint_name: str, default: float = 0.0
) -> float:
    joint_positions = get_joint_positions(state)
    return float(joint_positions.get(str(joint_name), default))


def _get_pos(
    states: dict[str, dict[str, Any]], obj_id: str
) -> tuple[float, float, float]:
    pose = states[obj_id]["pose_world"]
    return float(pose[0]), float(pose[1]), float(pose[2])


def _get_offset_point(
    states: dict[str, dict[str, Any]],
    obj_id: str,
    *,
    center_key: str = "center_offset",
    local_key: str = "local_offset",
    node: dict[str, Any] | None = None,
) -> tuple[float, float, float]:
    if node is None:
        return _get_pos(states, obj_id)
    offset = node.get(center_key)
    if offset is None:
        offset = node.get(local_key)
    if offset is not None:
        return quat_rotate_vector(states[obj_id], offset)
    return _get_pos(states, obj_id)


def _world_delta_to_local(
    state: dict[str, Any], delta: tuple[float, float, float]
) -> tuple[float, float, float]:
    pose = state["pose_world"]
    inverse_pose = [
        0.0,
        0.0,
        0.0,
        -float(pose[3]),
        -float(pose[4]),
        -float(pose[5]),
        float(pose[6]),
    ]
    return quat_rotate_vector(inverse_pose, delta)


def _check_object_inside(
    node: dict[str, Any], states: dict[str, dict[str, Any]]
) -> bool:
    obj = node["object"]
    ctr = node["container"]
    tolerance = float(node.get("tolerance", 0.12))
    z_offset = float(node.get("z_offset", 0.05))
    _ox, _oy, oz = _get_offset_point(
        states,
        obj,
        center_key="object_center_offset",
        local_key="object_local_offset",
        node=node,
    )
    cx, cy, cz = _get_offset_point(
        states,
        ctr,
        center_key="container_center_offset",
        local_key="container_local_offset",
        node=node,
    )
    ok = math.hypot(_ox - cx, _oy - cy) < tolerance and oz > cz - z_offset
    max_z = node.get("max_z")
    if ok and max_z is not None:
        ok = oz < float(max_z)
    return ok


def _check_object_on(node: dict[str, Any], states: dict[str, dict[str, Any]]) -> bool:
    obj = node["object"]
    target = node["target"]
    tolerance_xy = float(node.get("tolerance_xy", 0.12))
    tolerance_z = float(node.get("tolerance_z", 0.1))
    ox, oy, oz = _get_offset_point(
        states,
        obj,
        center_key="object_center_offset",
        local_key="object_local_offset",
        node=node,
    )
    tx, ty, tz = _get_offset_point(
        states,
        target,
        center_key="target_center_offset",
        local_key="target_local_offset",
        node=node,
    )
    return math.hypot(ox - tx, oy - ty) < tolerance_xy and 0 < oz - tz < tolerance_z


def _check_object_in_zone(
    node: dict[str, Any], states: dict[str, dict[str, Any]]
) -> bool:
    obj = node["object"]
    ox, oy, oz = _get_pos(states, obj)
    lo, hi = node["aabb"]
    return lo[0] <= ox <= hi[0] and lo[1] <= oy <= hi[1] and lo[2] <= oz <= hi[2]


def _check_object_upright(
    node: dict[str, Any], states: dict[str, dict[str, Any]]
) -> bool:
    obj = node["object"]
    tol_deg = float(node.get("tolerance_deg", 25))
    local_axis = str(node.get("local_axis", "z"))
    sign = -1.0 if local_axis.startswith("-") else 1.0
    local_axis = local_axis[1:] if local_axis.startswith("-") else local_axis
    if local_axis == "z" and sign > 0:
        up_dot = up_dot_from_pose(states[obj])
    else:
        _x, _y, up_dot = world_axis_from_pose(states[obj], local_axis)
        up_dot *= sign
    return up_dot > math.cos(math.radians(tol_deg))


def _check_joint_state(node: dict[str, Any], states: dict[str, dict[str, Any]]) -> bool:
    obj = node["object"]
    joint = str(node.get("joint", "joint_0"))
    target = node["target"]
    tol = float(node.get("tolerance", 0.15))
    current = get_joint_position(states[obj], joint, 0.0)
    if target == "open":
        return abs(current) > tol
    if target == "closed":
        return abs(current) < tol
    return abs(current - float(target)) < tol


def _check_object_static(
    node: dict[str, Any],
    states: dict[str, dict[str, Any]],
    ctx: dict[str, Any] | None = None,
) -> bool:
    threshold = float(node.get("threshold", 0.05))
    ang_threshold = float(node.get("ang_threshold", threshold * 5))
    obj = node["object"]
    state = states[obj]
    lv = state.get("lin_vel_world", [0.0, 0.0, 0.0])
    speed = math.sqrt(sum(float(v) * float(v) for v in lv))
    if bool(node.get("use_position_delta", False)) and ctx is not None:
        pos = _get_pos(states, obj)
        key = f"_static_prev_pos_{obj}"
        dt = float(ctx.get("dt", 1.0 / 60.0))
        prev = ctx.get(key)
        ctx[key] = pos
        if prev is not None and dt > 0:
            speed = (
                math.sqrt(
                    (pos[0] - prev[0]) ** 2
                    + (pos[1] - prev[1]) ** 2
                    + (pos[2] - prev[2]) ** 2
                )
                / dt
            )
    av = state.get("ang_vel_world", [0.0, 0.0, 0.0])
    ang_speed = math.sqrt(sum(float(v) * float(v) for v in av))
    if not bool(node.get("check_angular", True)):
        return speed < threshold
    return speed < threshold and ang_speed < ang_threshold


def _check_object_in_container_zone(
    node: dict[str, Any], states: dict[str, dict[str, Any]]
) -> bool:
    obj = node["object"]
    container = node["container"]
    zone_lo = node["zone_lo"]
    zone_hi = node["zone_hi"]
    ox, oy, oz = _get_offset_point(
        states,
        obj,
        center_key="object_center_offset",
        local_key="object_local_offset",
        node=node,
    )
    cx, cy, cz = _get_offset_point(
        states,
        container,
        center_key="container_center_offset",
        local_key="container_local_offset",
        node=node,
    )
    dx, dy, dz = ox - cx, oy - cy, oz - cz
    if bool(node.get("container_frame", False)):
        dx, dy, dz = _world_delta_to_local(states[container], (dx, dy, dz))
    return (
        float(zone_lo[0]) <= dx <= float(zone_hi[0])
        and float(zone_lo[1]) <= dy <= float(zone_hi[1])
        and float(zone_lo[2]) <= dz <= float(zone_hi[2])
    )


def evaluate_condition_tree(
    node: dict[str, Any], states: dict[str, dict[str, Any]], ctx: dict[str, Any]
) -> bool:
    ctype = node["type"]
    if ctype == "custom":
        from rlinf.envs.sim.bench2dex.tasks import CUSTOM_EVALUATORS

        return CUSTOM_EVALUATORS[node["evaluator"]](states, ctx, node.get("params", {}))
    if ctype == "all":
        return all(
            evaluate_condition_tree(child, states, ctx) for child in node["conditions"]
        )
    if ctype == "object_inside":
        return _check_object_inside(node, states)
    if ctype == "object_on":
        return _check_object_on(node, states)
    if ctype == "object_in_zone":
        return _check_object_in_zone(node, states)
    if ctype == "object_in_container_zone":
        return _check_object_in_container_zone(node, states)
    if ctype == "object_upright":
        return _check_object_upright(node, states)
    if ctype == "joint_state":
        return _check_joint_state(node, states)
    if ctype == "object_static":
        return _check_object_static(node, states, ctx)
    raise ValueError(f"Unknown condition type: {ctype!r}")


def evaluate_conditions(
    conditions: list[dict[str, Any]], states: dict[str, Any], ctx: dict[str, Any]
) -> bool:
    return all(evaluate_condition_tree(node, states, ctx) for node in conditions)


class TerminalSuccess:
    """Latch terminal success after consecutive physics steps satisfy the dwell time."""

    def __init__(self, metrics: dict[str, Any], dt: float = 1 / 60) -> None:
        terminal = metrics.get("terminal") or {}
        condition = terminal.get("raw_condition")
        if condition is None:
            raise ValueError("Bench2Dex requires metrics.terminal.raw_condition.")
        self.condition = dict(condition)
        self.dt = float(dt)
        self.dwell_time_s = float(terminal.get("dwell_time_s", 0.5))
        if self.dwell_time_s < 0:
            raise ValueError("metrics.terminal.dwell_time_s must be non-negative.")
        self.context: dict[str, Any] = {"dt": self.dt}
        self.hold_s = 0.0
        self.instant_success = False
        self.success = False

    def update(self, states: dict[str, Any]) -> None:
        """Evaluate one physics step with poses, velocities and joints."""
        self.instant_success = bool(
            evaluate_condition_tree(self.condition, states, self.context)
        )
        self.hold_s = self.hold_s + self.dt if self.instant_success else 0.0
        if (
            not self.success
            and self.instant_success
            and self.hold_s + 1e-12 >= self.dwell_time_s
        ):
            self.success = True
