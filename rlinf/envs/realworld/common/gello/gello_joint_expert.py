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



from __future__ import annotations

import argparse
import threading
import time

import numpy as np

from rlinf.envs.realworld.franka.franky_controller import (
    JOINT_LIMITS_LOWER,
    JOINT_LIMITS_UPPER,
)

# Mid-point of each joint's limit window; the first Dynamixel read is wrapped
# into the ±π band around this point and then clamped into the limit window.
_GELLO_UNWRAP_REFERENCE = 0.5 * (
    np.asarray(JOINT_LIMITS_LOWER) + np.asarray(JOINT_LIMITS_UPPER)
)


class GelloJointExpert:
    """Interface to the GELLO teleoperation device (joint-space output).
    Args:
        port: Serial port of the GELLO device.
    """

    def __init__(self, port: str):
        from gello_teleop.gello_teleop_agent import GelloTeleopAgent

        self.agent = GelloTeleopAgent(port=port)

        self.state_lock = threading.Lock()
        self._ready = False
        self._stop = False
        self._prev_joints: np.ndarray | None = None
        self.latest_data = {
            "joint_positions": np.zeros(7),
            "gripper": np.zeros(1),
        }
        self.thread = threading.Thread(target=self._read_gello, daemon=True)
        self.thread.start()

    def _read_gello(self):
        consecutive_errors = 0
        max_consecutive_errors = 50

        while not self._stop:
            try:
                gello_joints, gello_gripper = self.agent.get_action()
                gello_gripper = np.array([gello_gripper])

                joints = np.array(gello_joints)
                if self._prev_joints is None:
                    joints = (
                        _GELLO_UNWRAP_REFERENCE
                        + (joints - _GELLO_UNWRAP_REFERENCE + np.pi) % (2.0 * np.pi)
                        - np.pi
                    )
                    joints = np.clip(joints, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)
                else:
                    ref = self._prev_joints
                    joints = ref + (joints - ref + np.pi) % (2.0 * np.pi) - np.pi
                self._prev_joints = joints

                with self.state_lock:
                    self.latest_data["joint_positions"] = joints.copy()
                    self.latest_data["gripper"] = gello_gripper
                    self._ready = True
                consecutive_errors = 0
            except Exception:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    with self.state_lock:
                        self._ready = False
                backoff = min(0.1, 0.001 * (2 ** min(consecutive_errors, 7)))
                time.sleep(backoff)
                continue

            time.sleep(0.001)

    def close(self) -> None:
        """Stop the background read loop."""
        self._stop = True
        t = getattr(self, "thread", None)
        if t is not None and t.is_alive():
            t.join(timeout=1.0)

    @property
    def ready(self) -> bool:
        """Whether at least one GELLO frame has been received."""
        return self._ready

    def get_action(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(joint_positions, gripper)`` from the latest GELLO reading.

        Returns:
            A tuple of ``(joint_positions[7], gripper[1])``.
        """
        with self.state_lock:
            return (
                self.latest_data["joint_positions"].copy(),
                self.latest_data["gripper"].copy(),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the GELLO joint expert.")
    parser.add_argument(
        "--port",
        type=str,
        required=True,
        help="Serial port of the GELLO device.",
    )
    args = parser.parse_args()

    gello = GelloJointExpert(port=args.port)
    with np.printoptions(precision=3, suppress=True):
        while True:
            joint_positions, gripper = gello.get_action()
            print(
                f"joints={joint_positions}  gripper={gripper}",
                end="\r",
            )
            time.sleep(0.1)
