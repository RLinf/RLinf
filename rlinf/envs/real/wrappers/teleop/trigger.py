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

"""Control-ownership triggers for real-world teleoperation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rlinf.envs.real.wrappers.episode.keyboard import KeyboardListener

TAKEOVER = "takeover"
RELEASE = "release"


class InterventionTrigger:
    """Source of explicit operator control-ownership events."""

    def poll(self) -> list[str]:
        """Return events observed since the previous control step."""
        return []

    def reset(self) -> None:
        """Discard events queued before the next environment episode."""

    def close(self) -> None:
        """Release trigger resources."""


class KeyboardInterventionTrigger(InterventionTrigger):
    """Map two keyboard press edges to takeover and policy release."""

    def __init__(self, takeover_key: str = "space", release_key: str = "r") -> None:
        takeover_key = self._normalize_key(takeover_key)
        release_key = self._normalize_key(release_key)
        if not takeover_key or not release_key:
            raise ValueError("takeover_key and release_key must be non-empty")
        if takeover_key == release_key:
            raise ValueError("takeover_key and release_key must be different")
        self.takeover_key = takeover_key
        self.release_key = release_key
        self.listener = KeyboardListener()

    @staticmethod
    def _normalize_key(key: str) -> str:
        """Use the names returned by :class:`KeyboardListener`."""
        key = str(key)
        return {"space": "Key.space"}.get(key, key)

    def poll(self) -> list[str]:
        """Translate queued key presses into ownership events."""
        events: list[str] = []
        for key in self.listener.pop_pressed_keys():
            if key == self.takeover_key:
                events.append(TAKEOVER)
            elif key == self.release_key:
                events.append(RELEASE)
        return events

    def reset(self) -> None:
        """Discard key presses from the previous episode."""
        self.listener.pop_pressed_keys()

    def close(self) -> None:
        """Stop the keyboard listener owned by this trigger."""
        self.listener.close()


def build_intervention_trigger(
    cfg: Mapping[str, Any] | None,
) -> InterventionTrigger | None:
    """Build the configured trigger, or ``None`` for activity mode."""
    settings = dict(cfg or {})
    mode = str(settings.pop("mode", "activity")).lower()
    if mode == "activity":
        if settings:
            unknown = ", ".join(sorted(settings))
            raise ValueError(
                f"teleop_intervention has options for activity mode: {unknown}"
            )
        return None
    if mode != "explicit":
        raise ValueError(
            f"Unsupported teleop intervention mode {mode!r}; expected "
            "'activity' or 'explicit'."
        )
    takeover_key = str(settings.pop("takeover_key", "space"))
    release_key = str(settings.pop("release_key", "r"))
    if settings:
        unknown = ", ".join(sorted(settings))
        raise ValueError(f"Unknown explicit teleop_intervention options: {unknown}")
    return KeyboardInterventionTrigger(
        takeover_key=takeover_key,
        release_key=release_key,
    )
