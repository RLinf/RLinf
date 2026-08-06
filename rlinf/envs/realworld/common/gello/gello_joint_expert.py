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
from collections.abc import Callable

import numpy as np

from rlinf.envs.realworld.franka.franky_controller import (
    JOINT_LIMITS_LOWER,
    JOINT_LIMITS_UPPER,
)


class GelloJointExpert:
    """Interface to the GELLO teleoperation device (joint-space output).
    Args:
        port: Serial port of the GELLO device.
    """

    def __init__(
        self,
        port: str | None = None,
        action_source: Callable[[], tuple[np.ndarray, float]] | None = None,
        close_action_source: Callable[[], None] | None = None,
        joint_limits_lower: np.ndarray = JOINT_LIMITS_LOWER,
        joint_limits_upper: np.ndarray = JOINT_LIMITS_UPPER,
        stale_timeout: float = 0.5,
    ):
        if action_source is None:
            from gello_teleop.gello_teleop_agent import GelloTeleopAgent

            if port is None:
                raise ValueError("port is required when action_source is not set")
            action_source = GelloTeleopAgent(port=port).get_action
        self._action_source = action_source
        self._close_action_source = close_action_source
        self._joint_limits_lower = np.asarray(joint_limits_lower)
        self._joint_limits_upper = np.asarray(joint_limits_upper)
        self._stale_timeout = stale_timeout
        self._unwrap_reference = 0.5 * (
            self._joint_limits_lower + self._joint_limits_upper
        )

        self.state_lock = threading.Lock()
        self._ready = False
        self._last_success_time = 0.0
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
                gello_joints, gello_gripper = self._action_source()
                gello_gripper = np.array([gello_gripper])

                joints = np.array(gello_joints)
                if self._prev_joints is None:
                    joints = (
                        self._unwrap_reference
                        + (joints - self._unwrap_reference + np.pi) % (2.0 * np.pi)
                        - np.pi
                    )
                    joints = np.clip(
                        joints, self._joint_limits_lower, self._joint_limits_upper
                    )
                else:
                    ref = self._prev_joints
                    joints = ref + (joints - ref + np.pi) % (2.0 * np.pi) - np.pi
                    joints = np.clip(
                        joints, self._joint_limits_lower, self._joint_limits_upper
                    )
                self._prev_joints = joints

                with self.state_lock:
                    self.latest_data["joint_positions"] = joints.copy()
                    self.latest_data["gripper"] = gello_gripper
                    self._ready = True
                    self._last_success_time = time.monotonic()
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
        if self._close_action_source is not None:
            self._close_action_source()

    @property
    def ready(self) -> bool:
        """Whether a GELLO frame arrived within the freshness timeout."""
        with self.state_lock:
            return self._ready and (
                time.monotonic() - self._last_success_time <= self._stale_timeout
            )

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
