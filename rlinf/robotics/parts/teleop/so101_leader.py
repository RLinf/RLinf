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

"""SO-101 leader arm driving an SO-101 follower, joint for joint."""

from typing import Any, Mapping, Optional

import numpy as np

from ...actions import ActionKind
from ..base import Features, Observation
from .base import TeleopAction, TeleopDevice

#: Arm joints in bus order, matching the follower's.
MOTORS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)

#: The sixth servo, reported on lerobot's own 0..100 gripper scale.
GRIPPER = "gripper"
GRIPPER_SCALE = 100.0


@TeleopDevice.register("so101_leader")
class SO101Leader(TeleopDevice):
    """SO-101 leader arm, reporting joint angles and grip.

    The leader is the same five joints and gripper as the follower, held
    rather than driven, so the operator's pose is the follower's target
    directly. Readings arrive in the follower's own units -- radians and a
    ``0..1`` grip -- so neither half has to convert.

    Args:
        port: Serial device the leader's servo bus is on.
        calibration_id: lerobot calibration identifier for this leader.
        movement_epsilon: Radians of leader motion, summed over the joints,
            below which the operator counts as not driving.
        calibrate: Whether to run lerobot's calibration when the arm has none.
            It asks the operator to move the arm through its range, so only a
            caller holding a terminal should turn it on.
    """

    SDK = "scservo_sdk"

    PRODUCES = {
        "arm": ActionKind.JOINT_POSITION,
        "end_effector": ActionKind.GRIPPER,
    }

    NEEDS = ("joint_positions",)

    #: Joint limits and the gripper range both come from the action space.
    CLIPS_TO_ACTION_SPACE = True

    MOVEMENT_EPSILON = 0.01

    def __init__(
        self,
        port: str,
        calibration_id: Optional[str] = None,
        movement_epsilon: float = 0.01,
        calibrate: bool = False,
    ) -> None:
        self._port = port
        self._calibration_id = calibration_id
        self.MOVEMENT_EPSILON = movement_epsilon
        self._calibrate = calibrate

    @classmethod
    def from_config(
        cls, cfg: Mapping[str, Any], options: Mapping[str, Any], facts: Any
    ) -> Any:
        """Take the port and calibration id from options or the env config.

        Calibration stays off for a configured device. It waits on stdin, and
        a scheduler worker has no terminal to answer it from; the standalone
        entry point is where an operator calibrates an arm.
        """
        from .group import TeleopEntry

        port = options.get("port") or cfg.get("so101_leader_port")
        if port is None:
            raise ValueError(
                "teleop device 'so101_leader' requires a 'port', or "
                "'so101_leader_port' in the env config."
            )
        return TeleopEntry(
            cls(
                port=port,
                calibration_id=options.get("calibration_id")
                or cfg.get("so101_leader_id"),
                movement_epsilon=float(options.get("movement_epsilon", 0.01)),
            ),
            drives=options.get("drives"),
        )

    # Hardware.

    def _open(self) -> Any:
        """Open the leader's servo bus and return lerobot's handle for it.

        Calibration is not run unless the caller asked for it. lerobot's
        procedure asks the operator to move the arm through its range and
        waits on stdin, which would hang a worker that has no terminal.
        """
        try:
            from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
        except ImportError:  # pragma: no cover - older lerobot
            from lerobot.teleoperators.so101_leader import (
                SO101Leader,
                SO101LeaderConfig,
            )

        leader = SO101Leader(
            SO101LeaderConfig(
                port=self._port, id=self._calibration_id, use_degrees=True
            )
        )
        leader.connect(calibrate=False)
        if not leader.is_calibrated:
            if not self._calibrate:
                where = leader.calibration_fpath
                leader.disconnect()
                raise RuntimeError(
                    f"The SO-101 leader on {self._port!r} has no calibration at "
                    f"{where}. Calibrating asks the operator to move the arm "
                    "through its range, so it does not run on its own here. "
                    "Calibrate it once from a terminal with:\n\n"
                    "    python -m rlinf.robotics.parts.teleop.so101_leader "
                    f"--port {self._port} "
                    f"--id {self._calibration_id or '<name-for-this-arm>'} "
                    "--calibrate"
                )
            leader.calibrate()
        return leader

    def _release(self, device: Any) -> None:
        """lerobot spells this ``disconnect``, which the base does not try."""
        device.disconnect()

    @property
    def observation_features(self) -> Features:
        """The operator's joint angles, and how far the trigger is squeezed."""
        return {
            "joint_position": {"shape": (5,), "dtype": "float32"},
            "grip": {"shape": (1,), "dtype": "float32"},
        }

    def get_observation(self) -> Observation:
        """Read the leader the operator is holding.

        lerobot reports degrees and a ``0..100`` gripper; the follower speaks
        radians and ``0..1``, so the conversion happens here rather than
        leaving both units loose in the action.
        """
        reading = self._device.get_action()
        joints = np.deg2rad([reading[f"{motor}.pos"] for motor in MOTORS])
        grip = np.clip(reading[f"{GRIPPER}.pos"] / GRIPPER_SCALE, 0.0, 1.0)
        return {
            "joint_position": np.asarray(joints, dtype=np.float32),
            "grip": np.asarray([grip], dtype=np.float32),
        }

    # Driving the robot.

    def action(
        self, reading: Mapping[str, Any], context: Mapping[str, Any]
    ) -> TeleopAction:
        """Take the leader's pose as the follower's target.

        The grip stays on the ``0..1`` axis the SO-101 environment opens over,
        rather than the signed axis a GELLO reports, so neither half clips.
        """
        target = np.asarray(reading["joint_position"], dtype=float)
        current = np.asarray(context["joint_positions"])[0]
        grip = np.asarray(reading["grip"], dtype=float).reshape(1)
        # Idle until the operator actually moves, so the policy keeps control
        # while the leader is just resting in its holder.
        moved = float(np.linalg.norm(target - current)) > self.MOVEMENT_EPSILON
        return TeleopAction(parts={"arm": target, "end_effector": grip}, driving=moved)


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Read an SO-101 leader arm.")
    parser.add_argument(
        "--port", type=str, required=True, help="Serial port of the leader arm."
    )
    parser.add_argument(
        "--id", type=str, default=None, help="lerobot calibration id of the leader."
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Calibrate the arm first. It asks you to move it through its range.",
    )
    args = parser.parse_args()

    # This is a terminal, so it can answer the prompts calibration asks.
    leader = SO101Leader(
        port=args.port, calibration_id=args.id, calibrate=args.calibrate
    )
    leader.connect()
    # No follower to read, so the arm is measured against where it last was.
    # That is the same comparison action() makes, and it is what decides
    # whether the operator has taken control.
    previous = leader.get_observation()["joint_position"]
    try:
        with np.printoptions(precision=3, suppress=True):
            while True:
                action = leader.action(
                    leader.get_observation(), {"joint_positions": previous[None, :]}
                )
                arm = action.parts["arm"]
                grip = float(action.parts["end_effector"][0])
                print(
                    f"joints={np.rad2deg(arm)} deg  grip={grip:.2f}  "
                    f"driving={action.driving}   ",
                    end="\r",
                )
                previous = arm
                time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        leader.disconnect()
