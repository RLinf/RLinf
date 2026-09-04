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

"""GELLO leader arm, reported either as a Cartesian pose or as joints.

The joint variant is the Cartesian one with a different reading and a
different mapping, so it subclasses rather than repeating the lifecycle.
"""

from __future__ import annotations

import threading
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation as R

from ...actions import ActionKind
from ..base import Features, Observation
from .base import TeleopAction, TeleopDevice


@TeleopDevice.register("gello")
class Gello(TeleopDevice):
    """A leader arm posed by hand, reported as a Cartesian target.

    Args:
        port: Serial port of the leader arm.
        gripper: Whether the follower has a gripper to drive.
    """

    PRODUCES = {
        "arm": ActionKind.CARTESIAN_DELTA,
        "end_effector": ActionKind.GRIPPER,
    }

    NEEDS = ("tcp_pose", "action_scale")

    def __init__(self, port: str, gripper: bool = True) -> None:
        self._port = port
        self.gripper = gripper

    @classmethod
    def from_config(
        cls, cfg: Mapping[str, Any], options: Mapping[str, Any], facts: Any
    ) -> Any:
        """Take the port from the device options or the env config."""
        from .group import TeleopEntry

        port = options.get("port", cfg.get("gello_port"))
        if port is None:
            raise ValueError(
                "teleop device 'gello' requires 'gello_port' in the env config "
                "(e.g. env.eval.gello_port)."
            )
        return TeleopEntry(
            cls(port=port, gripper="end_effector" in facts.kinds),
            drives=options.get("drives"),
        )

    # Hardware.

    def _open(self) -> Any:
        return GelloExpert(port=self._port)

    @property
    def observation_features(self) -> Features:
        """A Cartesian target, plus the grip."""
        return {
            "position": {"shape": (3,), "dtype": "float32"},
            "orientation": {"shape": (4,), "dtype": "float32"},
            "grip": {"shape": (1,), "dtype": "float32"},
        }

    def get_observation(self) -> Observation:
        """Read the arm the operator is holding."""
        position, orientation, grip = self._device.get_action()
        return {
            "position": np.asarray(position, dtype=np.float32),
            "orientation": np.asarray(orientation, dtype=np.float32),
            "grip": np.asarray(grip, dtype=np.float32).reshape(1),
        }

    # Driving the robot.

    def action(
        self, reading: Mapping[str, Any], context: Mapping[str, Any]
    ) -> TeleopAction:
        """Difference the leader's pose against the follower's."""
        tcp_pose = np.asarray(context["tcp_pose"])
        scale = np.asarray(context["action_scale"])

        delta_position = (np.asarray(reading["position"]) - tcp_pose[:3]) / scale[0]
        rotation = (
            R.from_quat(np.asarray(reading["orientation"]).copy())
            * R.from_quat(tcp_pose[3:].copy()).inv()
        )
        delta_rotation = rotation.as_euler("xyz") / scale[1]

        parts = {
            "arm": np.clip(
                np.concatenate((delta_position, delta_rotation), axis=0), -1.0, 1.0
            )
        }
        gripper_active = False
        if self.gripper:
            grip = np.asarray(reading["grip"]) / scale[2]
            grip = np.clip(-(2 * grip - 1.0), -1.0, 1.0)
            parts["end_effector"] = grip
            gripper_active = bool(np.abs(grip).item() > 0.5)
        moved = float(np.linalg.norm(parts["arm"])) > self.MOVEMENT_EPSILON
        return TeleopAction(parts=parts, driving=moved or gripper_active)


# The vendor SDK this device speaks to.


class GelloExpert:
    """Read GELLO input and convert joint positions to a TCP pose.

    Args:
        port: Serial port of the GELLO device, e.g.
            ``"/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0"``.
    """

    def __init__(self, port: str) -> None:
        from gello_teleop.franka_fk import FrankaFK
        from gello_teleop.gello_teleop_agent import GelloTeleopAgent

        self.agent = GelloTeleopAgent(port=port)
        self.fk = FrankaFK()

        self.state_lock = threading.Lock()
        self._ready = False
        self._stop = False
        self.latest_data = {
            "target_pos": np.zeros(3),
            "target_quat": np.zeros(4),
            "gripper": np.zeros(1),
        }
        self.thread = threading.Thread(target=self._read_gello, daemon=True)
        self.thread.start()

    def _read_gello(self) -> None:
        import time

        while not self._stop:
            gello_joints, gello_gripper = self.agent.get_action()
            gello_gripper = np.array([gello_gripper])
            target_pos, target_quat = self.fk.get_fk(gello_joints)

            with self.state_lock:
                self.latest_data["target_pos"] = target_pos
                self.latest_data["target_quat"] = target_quat
                self.latest_data["gripper"] = gello_gripper
                self._ready = True

            time.sleep(0.001)

    def close(self) -> None:
        """Stop the read loop and release the leader's serial port.

        Without this the thread keeps polling the arm after the device
        disconnects, and the port stays open against the next connect.
        """
        self._stop = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        agent, self.agent = self.agent, None
        release = getattr(agent, "close", None) or getattr(agent, "stop", None)
        if callable(release):
            release()

    @property
    def ready(self) -> bool:
        """Return whether at least one GELLO frame has been received."""
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
