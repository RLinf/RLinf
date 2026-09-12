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
from typing import Any, Mapping

import gymnasium as gym

from rlinf.robotics.actions import ActionKind

#: The env config section a device reads shared options from, e.g. ``env.eval``.
EnvConfig = Mapping[str, Any]

#: One entry's own options, from the mapping form of a ``teleop`` list item.
DeviceOptions = Mapping[str, Any]


@dataclass(frozen=True)
class EnvFacts:
    """Action metadata a device reads to build itself for this env.

    Attributes:
        layout: Slice occupied by each named action part.
        kinds: Semantic action type for each part.
        joint_action_scale: Divisor used to normalize joint deltas.
        direct_stream: Whether joint targets bypass ``step`` through a stream.
    """

    layout: Mapping[str, slice]
    kinds: Mapping[str, ActionKind]
    joint_action_scale: float = 0.1
    direct_stream: bool = False

    @classmethod
    def about(
        cls,
        env: gym.Env,
        layout: Mapping[str, slice],
        kinds: Mapping[str, ActionKind],
        cfg: EnvConfig | None = None,
    ) -> "EnvFacts":
        """Build action metadata from an environment.

        The scale comes from the channel that commands the joints, because
        that channel decides what one action unit is worth. An env that keeps
        its own config is read as before.
        """
        inner = env.unwrapped
        config = getattr(inner, "config", None)
        scale = cls._joint_scale(inner)
        if scale is None:
            scale = float(getattr(config, "joint_action_scale", 0.1))
        stream = (cfg or {}).get("teleop_direct_stream", None)
        if stream is None:
            stream = getattr(config, "teleop_direct_stream", False)
        return cls(
            layout=layout,
            kinds=kinds,
            joint_action_scale=scale,
            direct_stream=bool(stream),
        )

    @staticmethod
    def _joint_scale(inner: gym.Env) -> float | None:
        """Radians per action unit, from the first channel driving joints."""
        action = getattr(inner, "action", None)
        for channel in getattr(action, "channels", ()):
            scale = getattr(channel, "scale", None)
            if channel.kind is ActionKind.JOINT_DELTA and scale is not None:
                return float(scale)
        return None
