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

"""Where each named part sits in an env's action vector, and what it means.

An env builds its own action space, so it is the only thing that knows what the
numbers in it are: ``FrankaEnv.step`` reads ``action[:6]`` as a twist, while
``GimArmEnv.step`` reads the same six as joint angles. Saying so is what lets a
teleop group ask for "the arm" instead of computing a slice -- and what stops a
spacemouse driving an arm that would read its twist as joint targets.

This used to be inferred from the width of the action space. Widths agree far
more often than meanings do, so a robot could accept a device it would misread.
An env declares :meth:`action_parts` instead, and one that does not cannot be
teleoperated.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym

from rlinf.robotics.teleop import ActionKind, ActionPart


@dataclass(frozen=True)
class ActionSpec:
    """An env's action vector, part by part.

    Attributes:
        layout: Where each named part sits.
        kinds: What each part's numbers mean.
    """

    layout: dict[str, slice]
    kinds: dict[str, ActionKind]


def action_spec(env: gym.Env) -> ActionSpec:
    """Return what ``env`` says its action is made of.

    Raises:
        AttributeError: If the env does not declare ``action_parts``. Guessing
            here would hand an operator's commands to a robot that reads them
            as something else.
        ValueError: If the declared parts do not tile the action space exactly,
            which means the declaration and ``step`` disagree.
    """
    try:
        declare = env.get_wrapper_attr("action_parts")
    except AttributeError:
        raise AttributeError(
            f"{type(env.unwrapped).__name__} does not declare action_parts(), so "
            "there is no way to know what a command for its arm means. Declare "
            "it to use teleoperation."
        ) from None
    # Called outside the guard: an AttributeError raised inside the env's own
    # declaration is a bug in it, not a missing declaration.
    parts = declare()

    layout: dict[str, slice] = {}
    kinds: dict[str, ActionKind] = {}
    start = 0
    for part in parts:
        if part.name in layout:
            raise ValueError(
                f"{type(env.unwrapped).__name__} declares {part.name!r} twice."
            )
        layout[part.name] = slice(start, start + part.width)
        kinds[part.name] = part.kind
        start += part.width

    total = int(env.action_space.shape[0])
    if start != total:
        raise ValueError(
            f"{type(env.unwrapped).__name__} declares parts covering {start} "
            f"numbers, but its action space has {total}. The declaration and "
            "step() disagree about the action."
        )
    return ActionSpec(layout=layout, kinds=kinds)


def action_layout(env: gym.Env) -> dict[str, slice]:
    """Return the action layout for ``env``, by part name."""
    return action_spec(env).layout


def mirrored(
    per_arm: tuple[ActionPart, ...], sides: tuple[str, ...]
) -> tuple[ActionPart, ...]:
    """Repeat one arm's parts for each side, qualified by side name.

    A two-armed robot lays the same parts out twice, so it says what one arm
    takes and names the sides.
    """
    return tuple(
        ActionPart(f"{side}.{part.name}", part.width, part.kind)
        for side in sides
        for part in per_arm
    )
