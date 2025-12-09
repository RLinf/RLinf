import threading
import pyspacemouse
import numpy as np
from typing import Tuple


class SpaceMouseExpert:
    """
    This class provides an interface to the SpaceMouse.
    It continuously reads the SpaceMouse state and provide
    a "get_action" method to get the latest action and button state.
    """

    def __init__(self):
        pyspacemouse.open()

        self.state_lock = threading.Lock()
        self.latest_data = {"action": np.zeros(6), "buttons": [0, 0]}
        # Start a thread to continuously read the SpaceMouse state
        self.thread = threading.Thread(target=self._read_spacemouse)
        self.thread.daemon = True
        self.thread.start()

    def _read_spacemouse(self):
        while True:
            state = pyspacemouse.read()
            with self.state_lock:
                self.latest_data["action"] = np.array(
                    [-state.y, state.x, state.z, -state.roll, -state.pitch, -state.yaw]
                )  # spacemouse axis matched with robot base frame
                self.latest_data["buttons"] = state.buttons

    def get_action(self) -> Tuple[np.ndarray, list]:
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


    def main():
        """Call spacemouse test."""
        test_spacemouse()


    if __name__ == "__main__":
        main()
