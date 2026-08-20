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

"""Choosing which teleop devices an environment listens to.

``teleop`` names them. One device is the common case, so ``teleop: spacemouse``
is a name; when several devices divide the robot between them it is a list, and
an entry may carry its own options::

    teleop:
      - {gello_joint: {port: /dev/left,  drives: left}}
      - {gello_joint: {port: /dev/right, drives: right}}

Both spellings land on the same list of entries, which is what the group builder
consumes. Which devices make up a group is settled here, from the config, rather
than inferred from the robot further down. Naming a device this env cannot drive
is an error rather than a surprise once the robot is moving.

The retired spellings -- ``teleop_device`` and the booleans ``use_spacemouse``,
``use_gello``, ``use_pico``, ``use_gello_joint`` -- still select a single device
and warn. They are read here rather than in the wrapper builders so that every
env stack retires them on the same schedule.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Optional, Sequence

#: Retired flag -> the device it selected.
LEGACY_FLAGS: dict[str, str] = {
    "use_spacemouse": "spacemouse",
    "use_gello": "gello",
    "use_pico": "pico",
    "use_gello_joint": "gello_joint",
}

#: The name meaning "nobody takes over from the policy".
NO_DEVICE = "none"


def _legacy_selection(cfg: Mapping[str, Any]) -> tuple[Optional[str], list[str]]:
    """Return the device the retired booleans select, and which were present."""
    present = [flag for flag in LEGACY_FLAGS if flag in cfg]
    enabled = [flag for flag in present if bool(cfg[flag])]

    if len(enabled) > 1:
        raise ValueError(
            "Only one teleop device can be active at a time, but "
            f"{', '.join(sorted(enabled))} are all enabled. Replace them with a "
            f"single 'teleop_device: {LEGACY_FLAGS[enabled[0]]}'."
        )
    if not present:
        return None, []
    return (LEGACY_FLAGS[enabled[0]] if enabled else NO_DEVICE), present


def resolve_teleop_device(
    cfg: Mapping[str, Any],
    *,
    supported: Sequence[str],
    default: str = NO_DEVICE,
) -> str:
    """Return the teleop device this env config selects.

    Args:
        cfg: The env config section, e.g. ``env.eval``.
        supported: Device names this env stack can drive. A name outside this
            set is an error rather than a silently ignored setting, because a
            missing takeover is only discovered with a robot already moving.
        default: The device to use when the config says nothing.

    Raises:
        ValueError: If the config names an unsupported device, enables more than
            one of the retired booleans, or sets ``teleop_device`` and a retired
            boolean that disagree with each other.
    """
    declared = cfg.get("teleop_device")
    legacy, legacy_present = _legacy_selection(cfg)

    if declared is not None and legacy_present:
        # Env configs layer, so a half-migrated stack can hold both keys. If they
        # agree that is merely redundant; if they disagree, choosing either one
        # silently gives somebody the wrong device on real hardware.
        if legacy != str(declared):
            raise ValueError(
                f"This env config sets 'teleop_device: {declared}' and also "
                f"{', '.join(sorted(legacy_present))}, which select "
                f"'{legacy}'. Remove the retired flags; they are ignored in a "
                "future release and cannot be reconciled here."
            )
        warnings.warn(
            f"'{', '.join(sorted(legacy_present))}' is retired and redundant "
            f"alongside 'teleop_device: {declared}'. Remove the old flags.",
            DeprecationWarning,
            stacklevel=2,
        )

    if declared is not None:
        if not legacy_present:
            warnings.warn(
                "'teleop_device' is retired. Use 'teleop', which also takes a "
                "list when several devices divide the robot between them.",
                DeprecationWarning,
                stacklevel=2,
            )
        device = str(declared)
    elif legacy is not None:
        warnings.warn(
            f"'{', '.join(sorted(legacy_present))}' is retired. Use "
            f"'teleop_device: {legacy}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        device = legacy
    else:
        device = default

    allowed = {*supported, NO_DEVICE}
    if device not in allowed:
        raise ValueError(
            f"Unsupported teleop device {device!r} for this environment. "
            f"Supported: {', '.join(sorted(allowed))}."
        )
    return device


def _entry(item: Any) -> tuple[str, Any]:
    """Split one ``teleop`` entry into its device name and the entry itself.

    An entry is either a bare name or a single-key mapping of name to options,
    which is the shape the group builder already takes.
    """
    if isinstance(item, str):
        return item, item
    try:
        pairs = list(dict(item).items())
    except (TypeError, ValueError):
        pairs = []
    if len(pairs) != 1:
        raise ValueError(
            "Each 'teleop' entry is a device name, or a mapping of one name to "
            f"its options, e.g. {{gello_joint: {{drives: left}}}}. Got {item!r}."
        )
    name, options = pairs[0]
    return str(name), {str(name): dict(options or {})}


def resolve_teleop_devices(
    cfg: Mapping[str, Any],
    *,
    supported: Sequence[str],
    default: str = NO_DEVICE,
) -> list[Any]:
    """Return the teleop entries this env config selects, in order.

    Args:
        cfg: The env config section, e.g. ``env.eval``.
        supported: Device names this env stack can drive.
        default: The device to use when the config says nothing.

    Returns:
        Bare names and single-key option mappings, ready for the group builder.
        Empty when nobody takes over from the policy.

    Raises:
        ValueError: If ``teleop`` is empty, holds a malformed entry, names a
            device this env cannot drive, or is combined with ``teleop_device``
            or a retired boolean, which would leave two answers to one question.
    """
    declared = cfg.get("teleop")
    if declared is None:
        device = resolve_teleop_device(cfg, supported=supported, default=default)
        return [] if device == NO_DEVICE else [device]

    # Env configs layer, so a run config saying 'teleop' often sits on a base
    # that still says 'teleop_device'. The one naming every device wins, and
    # says so rather than dropping the other silently.
    superseded = [key for key in ("teleop_device", *LEGACY_FLAGS) if key in cfg]
    if superseded:
        warnings.warn(
            f"'teleop' supersedes {', '.join(sorted(superseded))} in this env "
            "config. Remove the older keys.",
            DeprecationWarning,
            stacklevel=2,
        )

    items = [declared] if isinstance(declared, str) else list(declared)
    if not items:
        raise ValueError(
            "'teleop' is empty. Remove the key, or set 'teleop_device: none' to "
            "leave the policy in control."
        )

    entries = [_entry(item) for item in items]
    names = [name for name, _ in entries]

    if NO_DEVICE in names:
        if len(names) > 1:
            raise ValueError(
                f"'teleop' lists {NO_DEVICE!r} alongside {sorted(set(names) - {NO_DEVICE})}. "
                f"{NO_DEVICE!r} means nobody takes over, so it cannot share the list."
            )
        return []

    allowed = set(supported)
    unsupported = sorted({name for name in names if name not in allowed})
    if unsupported:
        raise ValueError(
            f"Unsupported teleop device(s) {', '.join(repr(n) for n in unsupported)} "
            f"for this environment. Supported: {', '.join(sorted(allowed))}."
        )

    # A name may repeat: two leader arms differ only by the branch each drives.
    return [entry for _, entry in entries]
