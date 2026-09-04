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

"""GELLO leader arm reported in joint space, one leader per follower arm."""

from __future__ import annotations

import argparse
import threading
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from rlinf.robotics.parts.arms.franka import (
    JOINT_LIMITS_LOWER,
    JOINT_LIMITS_UPPER,
)

from ...actions import ActionKind
from ..base import Features, Observation
from .base import TeleopAction, TeleopDevice
from .gello import Gello


@TeleopDevice.register("gello_joint")
class GelloJoint(Gello):
    """A leader arm whose joints become the follower's joint target.

    Args:
        port: Serial port of the leader arm.
        side: Index of the arm this leader drives, for reading the follower's
            joint positions out of the context.
        use_delta: Whether the env takes joint deltas rather than targets.
        action_scale: Divisor turning a joint delta into a normalized action.
    """

    PRODUCES = {
        "arm": ActionKind.JOINT_POSITION,
        "end_effector": ActionKind.GRIPPER,
    }

    NEEDS = ("joint_positions",)

    #: Match the threshold used by the former dual-GELLO wrapper.
    MOVEMENT_EPSILON = 0.01

    def __init__(
        self,
        port: str,
        side: int = 0,
        use_delta: bool = False,
        action_scale: float = 0.1,
    ) -> None:
        super().__init__(port=port)
        self.side = side
        self.use_delta = use_delta
        self.action_scale = action_scale
        # Differencing against the follower makes this a delta, not a target.
        self.PRODUCES = {
            "arm": ActionKind.JOINT_DELTA if use_delta else ActionKind.JOINT_POSITION,
            "end_effector": ActionKind.GRIPPER,
        }

    @classmethod
    def from_config(
        cls, cfg: Mapping[str, Any], options: Mapping[str, Any], facts: Any
    ) -> Any:
        """Each leader drives one named arm, so the config has to say which."""
        from .group import TeleopEntry

        drives = options.get("drives")
        if drives is None:
            raise ValueError(
                "teleop device 'gello_joint' drives one arm, so it says which. "
                "List one entry per arm, e.g. teleop: [{gello_joint: {drives: "
                "left}}, {gello_joint: {drives: right}}]."
            )
        port = options.get("port") or cfg.get(f"{drives}_gello_port")
        if port is None:
            raise ValueError(
                "teleop device 'gello_joint' requires a 'port', or "
                f"'{drives}_gello_port' in the env config."
            )
        # Match the mapping to absolute or delta joint semantics.
        arm = facts.kinds.get(f"{drives}.arm", facts.kinds.get("arm"))
        return TeleopEntry(
            cls(
                port=port,
                side={"left": 0, "right": 1}.get(str(drives), 0),
                use_delta=bool(options.get("use_delta", arm is ActionKind.JOINT_DELTA)),
                action_scale=float(
                    options.get("action_scale", facts.joint_action_scale)
                ),
            ),
            drives=drives,
        )

    @classmethod
    def streamer(
        cls,
        cfg: Mapping[str, Any],
        facts: Any,
        entries: Sequence[Any],
    ) -> Optional[Any]:
        """Build the optional 1 kHz dual-leader-arm streamer.

        The streamer reuses devices from ``entries`` so each serial port is
        opened once rather than twice.
        """
        from rlinf.envs.real.wrappers.teleop.adapters import DualGelloJointStream

        if not facts.direct_stream:
            return None
        arms = {entry.drives: entry.device for entry in entries if entry.drives}
        missing = {"left", "right"} - set(arms)
        if missing:
            raise ValueError(
                "Direct-stream GELLO drives both arms from their leader arms, "
                f"so it needs an entry for each. Missing: {sorted(missing)}."
            )
        return DualGelloJointStream(
            left_arm=arms["left"],
            right_arm=arms["right"],
            gripper_enabled=True,
            use_delta=facts.kinds.get("left.arm") is ActionKind.JOINT_DELTA,
            action_scale=facts.joint_action_scale,
            direct_stream=True,
            stream_period=cfg.get("gello_joint_stream_period", 0.001),
        )

    # Hardware.

    def _open(self) -> Any:
        return GelloJointExpert(port=self._port)

    @property
    def observation_features(self) -> Features:
        """Joint positions, plus the grip."""
        return {
            "joint_position": {"shape": (7,), "dtype": "float32"},
            "grip": {"shape": (1,), "dtype": "float32"},
        }

    def get_observation(self) -> Observation:
        """Read the arm the operator is holding."""
        joints, grip = self._device.get_action()
        return {
            "joint_position": np.asarray(joints, dtype=np.float32),
            "grip": np.asarray(grip, dtype=np.float32).reshape(1),
        }

    # Driving the robot.

    def action(
        self, reading: Mapping[str, Any], context: Mapping[str, Any]
    ) -> TeleopAction:
        """Difference the leader's joints against the follower's."""
        target = np.asarray(reading["joint_position"])
        current = np.asarray(context["joint_positions"])[self.side]
        if self.use_delta:
            arm = np.clip((target - current) / self.action_scale, -1.0, 1.0)
        else:
            arm = target.copy()

        grip = np.asarray(reading["grip"])
        grip = np.clip(-(2 * grip - 1.0), -1.0, 1.0)
        moved = float(np.linalg.norm(target - current)) > self.MOVEMENT_EPSILON
        gripper_active = bool(np.abs(grip).item() > 0.5)
        return TeleopAction(
            parts={"arm": arm, "end_effector": grip},
            driving=moved or gripper_active,
        )


# The vendor SDK this device speaks to.


# Unwrap the first Dynamixel reading around the midpoint of each joint range.
_GELLO_UNWRAP_REFERENCE = 0.5 * (
    np.asarray(JOINT_LIMITS_LOWER) + np.asarray(JOINT_LIMITS_UPPER)
)


class GelloJointExpert:
    """Read joint-space input from a GELLO device.

    Args:
        port: Serial port of the GELLO device.
    """

    def __init__(self, port: str) -> None:
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

    def _read_gello(self) -> None:
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
        """Stop the read loop and release the leader's serial port."""
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
