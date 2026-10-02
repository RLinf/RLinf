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

"""Action metadata a teleoperation device is built against.

A device's ``from_config`` reads these facts to match its mapping to the
robot it will drive: which parts exist, what each one means, and whether the
environment expects joint targets or deltas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import gymnasium as gym

from rlinf.robotics.actions import ActionKind

#: The env config section a device reads shared options from, e.g. ``env.eval``.
EnvConfig = Mapping[str, Any]

#: One entry's own options, from the mapping form of a ``teleop`` list item.
DeviceOptions = Mapping[str, Any]


def _rows(value: Any) -> Optional[tuple[tuple[float, ...], ...]]:
    """Convert per-arm limits to plain floats, preserving ``None``."""
    if value is None:
        return None
    return tuple(tuple(float(item) for item in row) for row in value)


@dataclass(frozen=True)
class EnvFacts:
    """Action metadata a device reads to build itself for this env.

    Attributes:
        layout: Slice occupied by each named action part.
        kinds: Semantic action type for each part.
        joint_action_scale: Divisor used to normalize joint deltas.
        direct_stream: Whether joint targets bypass ``step`` through a stream.
        joint_limit_min: Per-arm lower joint limits, when the env declares them.
        joint_limit_max: Per-arm upper joint limits, when the env declares them.
        joint_step_limits: Largest joint movement one control tick may command.
    """

    layout: Mapping[str, slice]
    kinds: Mapping[str, ActionKind]
    joint_action_scale: float = 0.1
    direct_stream: bool = False
    joint_limit_min: Optional[tuple[tuple[float, ...], ...]] = None
    joint_limit_max: Optional[tuple[tuple[float, ...], ...]] = None
    joint_step_limits: Optional[tuple[float, ...]] = None

    @classmethod
    def about(
        cls,
        env: gym.Env,
        layout: Mapping[str, slice],
        kinds: Mapping[str, ActionKind],
    ) -> "EnvFacts":
        """Build action metadata from an environment."""
        config = getattr(env.unwrapped, "config", None)
        step_limits = getattr(config, "joint_step_limits", None)
        return cls(
            layout=layout,
            kinds=kinds,
            joint_action_scale=float(getattr(config, "joint_action_scale", 0.1)),
            direct_stream=bool(getattr(config, "teleop_direct_stream", False)),
            joint_limit_min=_rows(getattr(config, "joint_limit_min", None)),
            joint_limit_max=_rows(getattr(config, "joint_limit_max", None)),
            joint_step_limits=(
                None if step_limits is None else tuple(float(x) for x in step_limits)
            ),
        )
