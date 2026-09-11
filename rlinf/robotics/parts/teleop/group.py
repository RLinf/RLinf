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

"""Compose teleoperation devices into named robot actions."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from rlinf.utils.logging import get_logger

from ...actions import ActionKind

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .base import TeleopDevice, TeleopPart


@dataclass
class TeleopEntry:
    """A teleoperation device and the branch of the action it fills.

    Attributes:
        device: The device the operator drives, which also says how its
            reading becomes robot action parts.
        drives: Branch of a multi-arm action this entry fills. ``None`` on a
            robot with one of each part, where the device's own names suffice.
    """

    device: "TeleopDevice"
    drives: Optional[str] = None

    @property
    def parts(self) -> tuple[str, ...]:
        """Action parts this entry fills, qualified by its branch."""
        return tuple(self.produces)

    @property
    def produces(self) -> dict[str, ActionKind]:
        """Return the qualified action parts produced by this entry."""
        if self.drives is None:
            return dict(self.device.PRODUCES)
        return {
            f"{self.drives}.{part}": kind for part, kind in self.device.PRODUCES.items()
        }


class TeleopGroup:
    """Combine what several devices drive into one named action.

    Args:
        entries: Devices to compose.
        available: Action parts the robot actually has. A device offering a
            part outside this set does not fill it.

    Raises:
        ValueError: If entries overlap, fill no available part, or use an
            incompatible action kind.
    """

    def __init__(
        self,
        entries: Iterable[TeleopEntry],
        available: Optional[Mapping[str, ActionKind]] = None,
    ) -> None:
        self.entries = list(entries)
        self.available = None if available is None else dict(available)
        self._filled = self._resolve()

    def _resolve(self) -> dict[str, TeleopEntry]:
        """Resolve each action part to exactly one compatible entry."""
        filled: dict[str, TeleopEntry] = {}
        for entry in self.entries:
            claimed = [
                part
                for part in entry.parts
                if self.available is None or part in self.available
            ]
            if not claimed:
                raise ValueError(
                    f"{type(entry.device).__name__} fills none of this robot's "
                    f"action parts. It offers {list(entry.parts)}; the robot has "
                    f"{sorted(self.available or [])}."
                )
            self._check_kinds(entry, claimed)
            for part in claimed:
                if part in filled:
                    raise ValueError(
                        f"{type(entry.device).__name__} and "
                        f"{type(filled[part].device).__name__} both drive "
                        f"{part!r}. Give one of them a different 'drives'."
                    )
                filled[part] = entry
        return filled

    def _check_kinds(self, entry: TeleopEntry, claimed: list[str]) -> None:
        """Reject devices whose action meaning differs from the environment."""
        if self.available is None:
            return
        produced = entry.produces
        for part in claimed:
            wanted, offered = self.available[part], produced[part]
            if wanted != offered:
                raise ValueError(
                    f"{type(entry.device).__name__} produces "
                    f"{offered.value!r} for {part!r}, but this env's action "
                    f"expects {wanted.value!r} there. The two mean different "
                    "things by the same numbers."
                )

    @property
    def parts(self) -> tuple[str, ...]:
        """Return the action parts filled by this group."""
        return tuple(self._filled)

    @property
    def devices(self) -> tuple[TeleopPart, ...]:
        """Return distinct devices in declaration order."""
        seen: list[TeleopPart] = []
        for entry in self.entries:
            if not any(entry.device is device for device in seen):
                seen.append(entry.device)
        return tuple(seen)

    def connect(self) -> None:
        """Open all devices and roll back partial startup."""
        opened: list[Any] = []
        try:
            for device in self.devices:
                if not device.is_connected:
                    device.connect()
                    opened.append(device)
        except BaseException:
            self._close(opened, "rolling back a failed teleop connect")
            raise

    def disconnect(self) -> None:
        """Close every device, newest first."""
        self._close(
            [device for device in self.devices if device.is_connected],
            "disconnecting teleop",
        )

    @staticmethod
    def _close(devices: "Sequence[Any]", doing: str) -> None:
        """Close devices in reverse order and report all cleanup failures."""
        failures: list[BaseException] = []
        for device in reversed(list(devices)):
            try:
                device.disconnect()
            except BaseException as error:  # noqa: BLE001 - reported below
                failures.append(error)
                get_logger().exception(
                    "%s: %s failed to close; continuing with the rest",
                    doing,
                    type(device).__name__,
                )
        if failures:
            raise failures[-1]

    def reset(self, context: Mapping[str, Any] = MappingProxyType({})) -> None:
        """Re-align every device to the robot after it resets."""
        for entry in self.entries:
            entry.device.on_reset(context)

    def action(
        self, context: Mapping[str, Any]
    ) -> tuple[dict[str, np.ndarray], bool, dict[str, Any]]:
        """Drive every device and merge the parts each one fills.

        Returns:
            The action parts by name, whether any operator is driving, and any
            info the devices want recorded.
        """
        # GELLO readers start in a deliberately invalid zero-pose state. The
        # previous wrappers waited until every configured reader was ready, so
        # keep the policy in control until the complete rig has a first sample.
        if not all(getattr(device, "ready", True) for device in self.devices):
            return {}, False, {}

        parts: dict[str, np.ndarray] = {}
        driving = False
        info: dict[str, Any] = {}

        # A device two entries share is read once, so both map the same
        # instant. Anything driven by one entry reads inside drive(), which
        # costs a placed device one round trip instead of two.
        shared = self._shared_readings()

        # Preserve order because one device may publish context for the next.
        running = dict(context)
        for entry in self.entries:
            self._require_context(entry, running)
            asked = entry.device.drive(running, shared.get(id(entry.device)))
            running.update(asked.publishes)
            parts.update(self._claimed(entry, asked.parts))
            driving |= asked.driving
            info.update(self._reported(entry, asked.info))
        return parts, driving, info

    def _shared_readings(self) -> dict[int, Mapping[str, Any]]:
        """Read once per device that more than one entry drives.

        Entries normally each own their device, and those read inside
        ``drive``. Only a device claimed twice is read here, because the two
        entries must agree on what the operator was doing.
        """
        counts: dict[int, int] = {}
        for entry in self.entries:
            counts[id(entry.device)] = counts.get(id(entry.device), 0) + 1
        return {
            id(device): device.get_observation()
            for device in self.devices
            if counts.get(id(device), 0) > 1
        }

    @staticmethod
    def _require_context(entry: TeleopEntry, context: Mapping[str, Any]) -> None:
        """Validate that the environment supplied all required context."""
        missing = [key for key in entry.device.NEEDS if key not in context]
        if missing:
            raise ValueError(
                f"{type(entry.device).__name__} needs {missing} from the robot "
                f"it drives, which this env does not report. It offers "
                f"{sorted(context)}."
            )

    def _claimed(
        self, entry: TeleopEntry, produced: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Qualify what an entry produced, dropping parts this robot lacks."""
        claimed = {}
        for name, value in produced.items():
            qualified = name if entry.drives is None else f"{entry.drives}.{name}"
            if self.available is not None and qualified not in self.available:
                continue
            claimed[qualified] = value
        return claimed

    @staticmethod
    def _reported(entry: TeleopEntry, info: Mapping[str, Any]) -> dict[str, Any]:
        """Qualify info keys with the target branch when required."""
        if entry.drives is None:
            return dict(info)
        return {f"{entry.drives}_{key}": value for key, value in info.items()}

    @property
    def clipped_parts(self) -> tuple[str, ...]:
        """Parts whose device wants them clipped into the env's action space."""
        return tuple(
            part
            for part, entry in self._filled.items()
            if entry.device.CLIPS_TO_ACTION_SPACE
        )

    @property
    def idle_parts(self) -> tuple[str, ...]:
        """Parts that remain under device control while the rig is idle."""
        return tuple(
            part
            for part, entry in self._filled.items()
            if entry.device.APPLIES_WHILE_IDLE
        )

    @property
    def hold_window(self) -> Optional[float]:
        """Return the shortest hold window configured by the devices."""
        windows = [
            entry.device.HOLD_WINDOW
            for entry in self.entries
            if entry.device.HOLD_WINDOW is not None
        ]
        return min(windows) if windows else None

    def on_action_chunk_begin(self) -> None:
        """Notify devices that a new policy-action chunk has started."""
        for entry in self.entries:
            entry.device.on_action_chunk_begin()

    def hold(self, context: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Return named action parts that hold the current robot state."""
        parts: dict[str, np.ndarray] = {}
        for entry in self.entries:
            parts.update(self._claimed(entry, entry.device.hold(context)))
        return parts
