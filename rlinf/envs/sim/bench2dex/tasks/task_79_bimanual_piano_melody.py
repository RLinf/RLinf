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

from typing import Any

from rlinf.envs.sim.bench2dex.success import get_joint_position

StateMap = dict[str, Any]

_LEFT_MELODY = (18, 18, 23, 23, 24, 24, 23)
_RIGHT_MELODY = (33, 33, 38, 38, 39, 39, 38)
_PIANO_PRESS_THRESHOLD = 0.004
_PIANO_RELEASE_THRESHOLD = 0.002


def _has(states: StateMap, *object_ids: str) -> bool:
    return all(
        object_id in states and states[object_id].get("pose_world") is not None
        for object_id in object_ids
    )


def _joint(states: StateMap, object_id: str, joint_name: str) -> float:
    state = states.get(object_id)
    if state is None:
        return 0.0
    return get_joint_position(state, joint_name, 0.0)


def _bimanual_melody_side(
    states: StateMap,
    ctx: dict[str, Any],
    *,
    key: str,
    melody: tuple[int, ...],
) -> bool:
    # Only the expected key advances the melody and other keys do not reset it; a key
    # counts when it rises above the press threshold after dropping below the release
    # threshold.
    piano = "obj_325_piano_1"
    index_key = f"{key}_index"
    armed_key = f"{key}_armed"
    index = int(ctx.get(index_key, 0))
    if index >= len(melody):
        return True

    expected_num = melody[index]
    joint_name = f"joint_{expected_num}_to_1"
    value = _joint(states, piano, joint_name)
    armed = bool(ctx.get(armed_key, True))

    if not armed and value < _PIANO_RELEASE_THRESHOLD:
        armed = True
    if armed and value > _PIANO_PRESS_THRESHOLD:
        index += 1
        armed = False

    ctx[armed_key] = armed
    ctx[index_key] = index
    return index >= len(melody)


def check_success(
    states: StateMap, ctx: dict[str, Any], params: dict[str, Any]
) -> bool:
    if not _has(states, "obj_325_piano_1"):
        return False
    left_ok = _bimanual_melody_side(states, ctx, key="left", melody=_LEFT_MELODY)
    right_ok = _bimanual_melody_side(states, ctx, key="right", melody=_RIGHT_MELODY)
    return left_ok and right_ok
