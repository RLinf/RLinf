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

"""Map named robot parts to environment action-vector spans and semantics."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym

from rlinf.robotics.actions import ActionKind, ActionPart


@dataclass(frozen=True)
class ActionSpec:
    """Named layout and semantics of an environment action vector.

    Attributes:
        layout: Slice occupied by each named part.
        kinds: Semantic action type for each part.
    """

    layout: dict[str, slice]
    kinds: dict[str, ActionKind]


def action_spec(env: gym.Env) -> ActionSpec:
    """Return and validate the environment's declared action parts.

    Raises:
        AttributeError: If the environment does not declare ``action_parts``.
        ValueError: If the declared parts do not cover the action space exactly.
    """
    try:
        declare = env.get_wrapper_attr("action_parts")
    except AttributeError:
        raise AttributeError(
            f"{type(env.unwrapped).__name__} does not declare action_parts(), so "
            "there is no way to know what a command for its arm means. Declare "
            "it to use teleoperation."
        ) from None
    # Do not mask AttributeError raised by the declaration itself.
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
    """Repeat an arm layout for each side and qualify the part names."""
    return tuple(
        ActionPart(f"{side}.{part.name}", part.width, part.kind)
        for side in sides
        for part in per_arm
    )
