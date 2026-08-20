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

"""Building the teleop a config asks for.

Each entry in :data:`DEVICES` knows how to construct one device and the binding
that says what it means. Adding a device is an entry here plus the two classes,
rather than another branch in the wrapper stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import gymnasium as gym

from rlinf.robotics.parts.teleop import (
    Glove,
    PicoController,
    SpaceMouse,
    TeleopLeaderArm,
)
from rlinf.robotics.teleop import (
    ActionKind,
    GloveBinding,
    LeaderArmBinding,
    LeaderJointBinding,
    PicoBinding,
    PicoTcpBinding,
    SpaceMouseBinding,
    TeleopEntry,
    TeleopGroup,
)

from .adapters import DualGelloJointStream
from .composed import ComposedTeleop
from .layout import action_spec
from .pico_config import split_dual_config


@dataclass(frozen=True)
class EnvFacts:
    """What the env tells a device builder about the robot it will drive.

    Attributes:
        layout: Where each named part sits in the action vector.
        kinds: What each part's numbers mean, which is what decides whether a
            device can drive it at all.
        joint_action_scale: The divisor turning a joint delta into a normalized
            action. Belongs to the robot rather than to the operator.
        direct_stream: Whether this env wants joint targets pushed on their own
            thread rather than dispatched by ``step``.
    """

    layout: Mapping[str, slice]
    kinds: Mapping[str, ActionKind]
    joint_action_scale: float = 0.1
    direct_stream: bool = False

    @classmethod
    def about(cls, env: gym.Env, layout, kinds) -> "EnvFacts":
        """Read the facts a device builder may ask for off an env."""
        config = getattr(env.unwrapped, "config", None)
        return cls(
            layout=layout,
            kinds=kinds,
            joint_action_scale=float(getattr(config, "joint_action_scale", 0.1)),
            direct_stream=bool(getattr(config, "teleop_direct_stream", False)),
        )


#: Builder for one entry, given the env config, this entry's own options, and
#: what the env says about itself.
EntryBuilder = Callable[[Mapping[str, Any], Mapping[str, Any], EnvFacts], TeleopEntry]


def _spacemouse(
    cfg: Mapping[str, Any],
    options: Mapping[str, Any],
    facts: EnvFacts,
) -> TeleopEntry:
    return TeleopEntry(
        SpaceMouse(device_index=int(options.get("device_index", 0))),
        SpaceMouseBinding(),
        drives=options.get("drives"),
    )


def _gello(
    cfg: Mapping[str, Any],
    options: Mapping[str, Any],
    facts: EnvFacts,
) -> TeleopEntry:
    port = options.get("port", cfg.get("gello_port"))
    if port is None:
        raise ValueError(
            "teleop device 'gello' requires 'gello_port' in the env config "
            "(e.g. env.eval.gello_port)."
        )
    return TeleopEntry(
        TeleopLeaderArm(port=port), LeaderArmBinding(), drives=options.get("drives")
    )


def _gello_joint(
    cfg: Mapping[str, Any],
    options: Mapping[str, Any],
    facts: EnvFacts,
) -> TeleopEntry:
    drives = options.get("drives")
    if drives is None:
        raise ValueError(
            "teleop device 'gello_joint' drives one arm, so it says which. List "
            "one entry per arm, e.g. teleop: [{gello_joint: {drives: left}}, "
            "{gello_joint: {drives: right}}]."
        )
    port = options.get("port") or cfg.get(f"{drives}_gello_port")
    if port is None:
        raise ValueError(
            "teleop device 'gello_joint' requires a 'port', or "
            f"'{drives}_gello_port' in the env config."
        )
    side = {"left": 0, "right": 1}.get(str(drives), 0)
    # Whether the env reads a target or a change is what it declared its arm
    # command to be; there is no second place to ask.
    arm = facts.kinds.get(f"{drives}.arm", facts.kinds.get("arm"))
    return TeleopEntry(
        TeleopLeaderArm(port=port, joint_space=True),
        LeaderJointBinding(
            side=side,
            use_delta=bool(options.get("use_delta", arm is ActionKind.JOINT_DELTA)),
            action_scale=float(options.get("action_scale", facts.joint_action_scale)),
        ),
        drives=drives,
    )


def _glove(
    cfg: Mapping[str, Any],
    options: Mapping[str, Any],
    facts: EnvFacts,
) -> TeleopEntry:
    glove_cfg = dict(cfg.get("glove_config", {}))
    glove_cfg.update(options)
    return TeleopEntry(
        Glove(
            left_port=glove_cfg.get("left_port", "/dev/ttyACM0"),
            right_port=glove_cfg.get("right_port"),
            frequency=int(glove_cfg.get("frequency", 60)),
            config_file=glove_cfg.get("config_file"),
        ),
        GloveBinding(),
        drives=options.get("drives"),
    )


def _pico(
    cfg: Mapping[str, Any],
    options: Mapping[str, Any],
    facts: EnvFacts,
) -> TeleopEntry:
    drives = options.get("drives")
    pico_cfg = dict(cfg.get("pico", {}))
    hold = bool(
        options.get(
            "hold_current_when_inactive",
            pico_cfg.pop("hold_current_when_inactive", True),
        )
    )
    gripper = bool(options.get("gripper", not bool(cfg.get("no_gripper", True))))

    # The env says whether its arm takes a pose to reach or a delta to apply.
    arm = facts.kinds.get("arm" if drives is None else f"{drives}.arm")
    absolute = arm is ActionKind.CARTESIAN_POSE

    if drives in ("left", "right"):
        left_cfg, right_cfg = split_dual_config(pico_cfg)
        device_cfg = left_cfg if drives == "left" else right_cfg
        side = 0 if drives == "left" else 1
    else:
        device_cfg = {
            key: value
            for key, value in pico_cfg.items()
            if key not in ("left", "right")
        }
        device_cfg.setdefault("hand", "right")
        side = 0 if str(device_cfg["hand"]).lower() == "left" else 1

    binding = (
        PicoTcpBinding(gripper=gripper, side=side, hold_current_when_inactive=hold)
        if absolute
        else PicoBinding(gripper=gripper, side=side)
    )
    return TeleopEntry(PicoController(**device_cfg), binding, drives=drives)


def _gello_joint_stream(
    cfg: Mapping[str, Any], facts: EnvFacts, entries: Sequence[TeleopEntry]
) -> Optional[Any]:
    """The 1 kHz thread a pair of leader arms uses, when this env asks for it.

    Follower tracking is unstable at the policy's step rate, so the joint
    targets go straight to the controllers on their own thread.

    It streams the arms the group already composed rather than opening its own.
    Two readers on one serial port is at best two pollers competing for it, and
    the second pair would be built from the config's global ports -- so a
    per-entry ``port`` override would drive the action and be ignored by the
    stream.
    """
    if not facts.direct_stream:
        return None
    arms = {entry.drives: entry.device for entry in entries if entry.drives}
    missing = {"left", "right"} - set(arms)
    if missing:
        raise ValueError(
            "Direct-stream GELLO drives both arms from their leader arms, so "
            f"it needs an entry for each. Missing: {sorted(missing)}."
        )
    return DualGelloJointStream(
        left_arm=arms["left"],
        right_arm=arms["right"],
        gripper_enabled=True,
        use_delta=facts.kinds.get("left.arm") is ActionKind.JOINT_DELTA,
        action_scale=facts.joint_action_scale,
        direct_stream=True,
        stream_period=cfg.get("gello_joint_stream_period", 0.001),
    )


#: Device name -> the streamer it also commands the robot through, if any. A
#: device is here as well as in :data:`DEVICES` when composition alone does not
#: describe it: the action is one thing, the rate it is delivered at another.
STREAMERS: dict[
    str, Callable[[Mapping[str, Any], EnvFacts, Sequence[TeleopEntry]], Optional[Any]]
] = {
    "gello_joint": _gello_joint_stream,
}


#: Device name -> how to build its entry.
DEVICES: dict[str, EntryBuilder] = {
    "spacemouse": _spacemouse,
    "gello": _gello,
    "gello_joint": _gello_joint,
    "glove": _glove,
    "pico": _pico,
}


def build_teleop(
    env: gym.Env,
    cfg: Mapping[str, Any],
    devices: Sequence[Any],
    timeout: Optional[float] = None,
) -> ComposedTeleop:
    """Build the teleop this env config asks for.

    Args:
        env: The env being wrapped, which supplies the action layout.
        cfg: The env config section, for options devices share.
        devices: Device names, or single-key mappings carrying options, e.g.
            ``{"gello_joint": {"port": "/dev/left", "drives": "left"}}``.
        timeout: Hold window. Left out, the bindings decide: a device held
            down to take over asks for none.
    """
    spec = action_spec(env)
    facts = EnvFacts.about(env, spec.layout, spec.kinds)

    entries, streams = [], []
    for item in devices:
        if isinstance(item, str):
            name, options = item, {}
        else:
            ((name, options),) = dict(item).items()
        if name not in DEVICES:
            raise ValueError(
                f"Unknown teleop device {name!r}. Known: {sorted(DEVICES)}."
            )
        options = dict(options or {})
        entries.append(DEVICES[name](cfg, options, facts))
        if name in STREAMERS and name not in streams:
            streams.append(name)

    # After the loop: a streamer drives the devices the group composed, so it
    # cannot be built until they all exist.
    streamer = None
    for name in streams[:1]:
        streamer = STREAMERS[name](cfg, facts, entries)

    group = TeleopGroup(entries, available=facts.kinds)
    group.connect()
    if timeout is None:
        timeout = group.hold_window
    return ComposedTeleop(group, facts.layout, timeout=timeout, streamer=streamer)
