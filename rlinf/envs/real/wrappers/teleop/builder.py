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

"""Build a composed teleoperation device from environment configuration."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import gymnasium as gym

from rlinf.robotics.parts.teleop import TeleopDevice, TeleopGroup

from .composed import ComposedTeleop
from .facts import EnvFacts
from .layout import action_spec

__all__ = ["EnvFacts", "TeleopDevice", "build_teleop"]


def build_teleop(
    env: gym.Env,
    cfg: Mapping[str, Any],
    devices: Sequence[Any],
    timeout: Optional[float] = None,
) -> ComposedTeleop:
    """Build and connect the configured teleoperation group.

    Args:
        env: Environment that supplies the action layout.
        cfg: Environment configuration containing shared device options.
        devices: Device names, or single-key mappings carrying options, e.g.
            ``{"gello_joint": {"port": "/dev/left", "drives": "left"}}``.
        timeout: Optional operator-control hold window.
    """
    spec = action_spec(env)
    facts = EnvFacts.about(env, spec.layout, spec.kinds)

    entries: list[Any] = []
    asked: list[type[TeleopDevice]] = []
    for item in devices:
        if isinstance(item, str):
            name, options = item, {}
        else:
            ((name, options),) = dict(item).items()
        device_cls = TeleopDevice.named(name)
        entries.append(device_cls.from_config(cfg, options or {}, facts))
        if device_cls not in asked:
            asked.append(device_cls)

    # Build a streamer after its group-owned devices exist.
    streamer = None
    for device_cls in asked:
        streamer = device_cls.streamer(cfg, facts, entries)
        if streamer is not None:
            break

    group = TeleopGroup(entries, available=facts.kinds)
    group.connect()
    if timeout is None:
        timeout = group.hold_window
    return ComposedTeleop(group, facts.layout, timeout=timeout, streamer=streamer)
