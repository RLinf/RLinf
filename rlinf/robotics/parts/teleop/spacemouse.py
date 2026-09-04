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

"""SpaceMouse: a 6-DoF puck and two buttons driving the arm and gripper."""

from __future__ import annotations

import threading
from types import MappingProxyType
from typing import Any, Mapping, Optional

import numpy as np

from ...actions import ActionKind
from ..base import Features, Observation
from .base import TeleopAction, TeleopDevice
from .util import jittered_grip


@TeleopDevice.register("spacemouse")
class SpaceMouse(TeleopDevice):
    """A 6-DoF mouse: a twist from the puck, and two buttons.

    Args:
        device_index: Which HID device to open when several are attached.
        dexterous_hand: Whether the robot wears a hand rather than a gripper,
            which swaps which button the info reports as each side.
    """

    PRODUCES = {
        "arm": ActionKind.CARTESIAN_DELTA,
        "end_effector": ActionKind.GRIPPER,
    }

    #: gripper_open is read when offered, but a default covers its absence.
    NEEDS = ()

    def __init__(self, device_index: int = 0, dexterous_hand: bool = False) -> None:
        self._device_index = device_index
        self.dexterous_hand = dexterous_hand
        self._grip: Optional[np.ndarray] = None
        self.left = False
        self.right = False

    @classmethod
    def from_config(
        cls, cfg: Mapping[str, Any], options: Mapping[str, Any], facts: Any
    ) -> Any:
        """Read the button layout from the end effector the robot wears."""
        from .group import TeleopEntry

        return TeleopEntry(
            cls(
                device_index=int(options.get("device_index", 0)),
                dexterous_hand="hand" in facts.kinds,
            ),
            drives=options.get("drives"),
        )

    # Hardware.

    def _open(self) -> Any:
        return SpaceMouseExpert(device_index=self._device_index)

    @property
    def observation_features(self) -> Features:
        """Return the twist and button-state feature declarations."""
        return {
            "twist": {"shape": (6,), "dtype": "float32"},
            "buttons": {"shape": (2,), "dtype": "bool"},
        }

    def get_observation(self) -> Observation:
        """Read the puck and the buttons."""
        twist, buttons = self._device.get_action()
        return {
            "twist": np.asarray(twist, dtype=np.float32),
            "buttons": np.asarray(buttons, dtype=bool),
        }

    # Driving the robot.

    def on_reset(self, context: Mapping[str, Any] = MappingProxyType({})) -> None:
        """Release the buttons and resync the gripper after a reset."""
        self.left = False
        self.right = False
        self._grip = None

    def action(
        self, reading: Mapping[str, Any], context: Mapping[str, Any]
    ) -> TeleopAction:
        """Map the twist onto the arm, and the buttons onto the gripper."""
        buttons = reading["buttons"]
        self.left, self.right = bool(buttons[0]), bool(buttons[1])

        parts: dict[str, np.ndarray] = {"arm": np.asarray(reading["twist"])}

        if self.left:
            self._grip = jittered_grip(is_open=False)
        elif self.right:
            self._grip = jittered_grip(is_open=True)
        elif self._grip is None:
            self._grip = jittered_grip(is_open=bool(context.get("gripper_open", True)))
        parts["end_effector"] = self._grip.copy()

        moved = float(np.linalg.norm(reading["twist"])) > self.MOVEMENT_EPSILON
        info = (
            {"left": self.right, "right": self.left}
            if self.dexterous_hand
            else {"left": self.left, "right": self.right}
        )
        return TeleopAction(
            parts=parts,
            driving=moved or self.left or self.right,
            info=info,
        )

    def publish(self, reading: Mapping[str, Any]) -> dict[str, Any]:
        """Publish whether the glove-control button is held."""
        return {"hand_driving": bool(reading["buttons"][1])}


# The vendor SDK this device speaks to.


class SpaceMouseExpert:
    """Read SpaceMouse motion and button state in a background thread."""

    def __init__(self, device_index: int = 0) -> None:
        import pyspacemouse

        self._device = pyspacemouse.open(device_index=device_index)

        self.state_lock = threading.Lock()
        self.latest_data: dict = {"action": np.zeros(6), "buttons": [0, 0]}
        self._stop = False
        self.thread = threading.Thread(target=self._read_spacemouse, daemon=True)
        self.thread.start()

    def _read_spacemouse(self) -> None:
        while not self._stop:
            device = self._device
            if device is None:
                return
            try:
                state = device.read()
            except Exception:
                # close() releases the HID handle this loop is reading, so a
                # failure during shutdown is expected. Anything else is not.
                if self._stop:
                    return
                raise
            with self.state_lock:
                self.latest_data["action"] = np.array(
                    [-state.y, state.x, state.z, -state.roll, -state.pitch, -state.yaw]
                )  # Express SpaceMouse axes in the robot base frame.
                self.latest_data["buttons"] = state.buttons

    def close(self) -> None:
        """Stop the read loop and release the HID handle.

        Without this the puck stays open and its thread keeps polling after
        the device disconnects, so the next connect finds the HID handle
        taken.
        """
        self._stop = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        device, self._device = self._device, None
        if device is not None:
            device.close()

    def get_action(self) -> tuple[np.ndarray, list]:
        """Return the latest motion and button state."""
        with self.state_lock:
            return self.latest_data["action"], self.latest_data["buttons"]


if __name__ == "__main__":
    import time

    def test_spacemouse() -> None:
        """Print SpaceMouse readings at 10 Hz until interrupted."""
        spacemouse = SpaceMouseExpert()
        with np.printoptions(precision=3, suppress=True):
            while True:
                action, buttons = spacemouse.get_action()
                print(f"Spacemouse action: {action}, buttons: {buttons}")
                time.sleep(0.1)

    test_spacemouse()
