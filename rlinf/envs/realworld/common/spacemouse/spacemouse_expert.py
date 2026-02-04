# Copyright 2025 The RLinf Authors.
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

import threading

import numpy as np


class SpaceMouseExpert:
    """
    This class provides an interface to the SpaceMouse.
    It continuously reads the SpaceMouse state and provide
    a "get_action" method to get the latest action and button state.
    """

    def __init__(self):
        import pyspacemouse

        pyspacemouse.open()

        self.state_lock = threading.Lock()
        self.latest_data = {"action": np.zeros(6), "buttons": [0, 0]}
        # Start a thread to continuously read the SpaceMouse state
        self.thread = threading.Thread(target=self._read_spacemouse)
        self.thread.daemon = True
        self.thread.start()

    def _read_spacemouse(self):
        import pyspacemouse

        while True:
            state = pyspacemouse.read()
            with self.state_lock:
                self.latest_data["action"] = np.array(
                    [-state.y, state.x, state.z, -state.roll, -state.pitch, -state.yaw]
                )  # spacemouse axis matched with robot base frame
                self.latest_data["buttons"] = state.buttons

    def get_action(self) -> tuple[np.ndarray, list]:
        """Returns the latest action and button state of the SpaceMouse."""
        with self.state_lock:
            return self.latest_data["action"], self.latest_data["buttons"]


if __name__ == "__main__":
    import time

    def test_spacemouse():
        """Test the SpaceMouseExpert class.

        This interactive test prints the action and buttons of the spacemouse at a rate of 10Hz.
        The user is expected to move the spacemouse and press its buttons while the test is running.
        It keeps running until the user stops it.

        """
        spacemouse = SpaceMouseExpert()
        with np.printoptions(precision=3, suppress=True):
            while True:
                action, buttons = spacemouse.get_action()
                print(f"Spacemouse action: {action}, buttons: {buttons}")
                time.sleep(0.1)

    test_spacemouse()
