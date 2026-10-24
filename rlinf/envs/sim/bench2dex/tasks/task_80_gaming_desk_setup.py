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

from rlinf.envs.sim.bench2dex.success import get_joint_position, world_axis_from_pose

StateMap = dict[str, Any]


def _has(states: StateMap, *object_ids: str) -> bool:
    return all(
        object_id in states and states[object_id].get("pose_world") is not None
        for object_id in object_ids
    )


def _pos(states: StateMap, object_id: str) -> tuple[float, float, float]:
    pose = states[object_id]["pose_world"]
    return float(pose[0]), float(pose[1]), float(pose[2])


def _speed(state: dict[str, Any], *, angular: bool = False) -> float:
    key = "ang_vel_world" if angular else "lin_vel_world"
    values = state.get(key, [0.0, 0.0, 0.0])
    return math.sqrt(sum(float(value) ** 2 for value in values))


def _static(
    states: StateMap, *object_ids: str, linear: float = 0.06, angular: float = 0.5
) -> bool:
    return all(
        _speed(states[object_id]) < linear
        and _speed(states[object_id], angular=True) < angular
        for object_id in object_ids
    )


def _axis_dot(state: dict[str, Any], local_axis: str, target: Sequence[float]) -> float:
    axis = world_axis_from_pose(state, local_axis)
    norm = math.sqrt(sum(float(value) ** 2 for value in target)) or 1.0
    return sum(axis[index] * float(target[index]) for index in range(3)) / norm


def _axis_aligned(
    state: dict[str, Any],
    local_axis: str,
    target: Sequence[float],
    tolerance_deg: float,
    *,
    allow_opposite: bool = False,
) -> bool:
    dot = _axis_dot(state, local_axis, target)
    if allow_opposite:
        dot = abs(dot)
    return dot > math.cos(math.radians(tolerance_deg))


def _upright(
    states: StateMap, object_id: str, local_axis: str = "z", tolerance_deg: float = 25.0
) -> bool:
    return _axis_aligned(states[object_id], local_axis, (0.0, 0.0, 1.0), tolerance_deg)


def _joint(states: StateMap, object_id: str, joint_name: str) -> float:
    state = states.get(object_id)
    if state is None:
        return 0.0
    return get_joint_position(state, joint_name, 0.0)


def _pressed_then_released(
    states: StateMap,
    ctx: dict[str, Any],
    *,
    key: str,
    object_id: str,
    joint_name: str,
    press_threshold: float,
    release_threshold: float,
) -> bool:
    value = abs(_joint(states, object_id, joint_name))
    pressed_key = f"{key}_pressed"
    complete_key = f"{key}_complete"
    if value > press_threshold:
        ctx[pressed_key] = True
    if ctx.get(pressed_key, False) and value < release_threshold:
        ctx[complete_key] = True
    return bool(ctx.get(complete_key, False))


def _monitor_straight(states: StateMap) -> bool:
    monitor = "obj_090_display_1"
    if not _has(states, monitor):
        return False
    x, y, _ = _pos(states, monitor)
    # The display must move toward -y from its authored spawn onto the desk.
    y_ok = y < 0.282
    roll_pitch_yaw_ok = _upright(states, monitor, "y", 5.0) and _axis_aligned(
        states[monitor], "x", (1.0, 0.0, 0.0), 5.0, allow_opposite=True
    )
    return y_ok and roll_pitch_yaw_ok and _static(states, monitor)


def _keyboard_esc_pressed(states: StateMap, ctx: dict[str, Any]) -> bool:
    keyboard = "obj_098_keyboard_2"
    return _pressed_then_released(
        states,
        ctx,
        key="stage_esc",
        object_id=keyboard,
        joint_name="joint_100",
        press_threshold=0.0050,
        release_threshold=0.0010,
    )


def _mouse_left_clicked(states: StateMap, ctx: dict[str, Any]) -> bool:
    mouse = "obj_300_mouse_3"
    keyboard = "obj_098_keyboard_2"
    if not _has(states, mouse, keyboard):
        return False
    mx, my, _ = _pos(states, mouse)
    kx, ky, _ = _pos(states, keyboard)
    # The mouse must be back on the keyboard's left side.
    moved_left = my < ky + 0.015
    clicked = _pressed_then_released(
        states,
        ctx,
        key="stage_mouse",
        object_id=mouse,
        joint_name="joint_0",
        press_threshold=0.0075,
        release_threshold=0.0010,
    )
    return moved_left and clicked


def check_success(
    states: StateMap, ctx: dict[str, Any], params: dict[str, Any]
) -> bool:
    keyboard = "obj_098_keyboard_2"
    mouse = "obj_300_mouse_3"
    if not _has(states, "obj_090_display_1", keyboard, mouse):
        return False
    step = int(ctx.get("step", 0))
    if step == 0 and _monitor_straight(states):
        step = 1
    if step == 1 and _keyboard_esc_pressed(states, ctx):
        step = 2
    if step == 2 and _mouse_left_clicked(states, ctx):
        step = 3
    ctx["step"] = step
    return step >= 3
