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

import threading

import numpy as np


class GelloExpert:
    """Interface to the GELLO teleoperation device.

    Continuously reads GELLO joint positions in a background thread,
    computes the corresponding TCP pose via forward kinematics, and
    exposes the result through :meth:`get_action`.

    Args:
        port: Serial port of the GELLO device, e.g.
            ``"/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0"``.
    """

    def __init__(self, port: str):
        from gello_teleop.franka_fk import FrankaFK
        from gello_teleop.gello_teleop_agent import GelloTeleopAgent

        self.agent = GelloTeleopAgent(port=port)
        self.fk = FrankaFK()

        self.state_lock = threading.Lock()
        self._ready = False
        self.latest_data = {
            "target_pos": np.zeros(3),
            "target_quat": np.zeros(4),
            "gripper": np.zeros(1),
        }
        self.thread = threading.Thread(target=self._read_gello, daemon=True)
        self.thread.start()

    def _read_gello(self):
        import time

        while True:
            gello_joints, gello_gripper = self.agent.get_action()
            gello_gripper = np.array([gello_gripper])
            target_pos, target_quat = self.fk.get_fk(gello_joints)

            with self.state_lock:
                self.latest_data["target_pos"] = target_pos
                self.latest_data["target_quat"] = target_quat
                self.latest_data["gripper"] = gello_gripper
                self._ready = True

            time.sleep(0.001)

    @property
    def ready(self) -> bool:
        """Whether at least one GELLO frame has been received."""
        return self._ready

    def get_action(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(target_pos, target_quat, gripper)`` from the latest GELLO reading."""
        with self.state_lock:
            return (
                self.latest_data["target_pos"],
                self.latest_data["target_quat"],
                self.latest_data["gripper"],
            )


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Test the GELLO expert.")
    parser.add_argument(
        "--port",
        type=str,
        required=True,
        help="Serial port of the GELLO device.",
    )
    args = parser.parse_args()

    gello = GelloExpert(port=args.port)
    with np.printoptions(precision=3, suppress=True):
        while True:
            target_pos, target_quat, gripper = gello.get_action()
            print(
                f"pos={target_pos}  quat={target_quat}  gripper={gripper}",
                end="\r",
            )
            time.sleep(0.1)
