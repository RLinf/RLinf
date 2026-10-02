# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lazy i2rt boundary for the RLinf-native YAM environment.

This is intentionally the only module under ``rlinf.envs.real.yam`` that
mentions i2rt imports. Importing RLinf, the scheduler, or the Gym registration
therefore works on training nodes that do not have the robot SDK installed.
"""

from __future__ import annotations

import inspect
import threading
import time
from typing import Any

import numpy as np

from .types import NUM_ARM_JOINTS, YamArmState


def _explicit_vector(device_config: Any, name: str) -> np.ndarray | None:
    value = getattr(device_config, name, None)
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64).reshape(NUM_ARM_JOINTS).copy()


def _build_yam(device_config: Any, *, zero_gravity_mode: bool) -> Any:
    """Construct one robot, importing i2rt only at connection time."""
    try:
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import ArmType, GripperType
    except ImportError as error:
        raise ImportError(
            "YAM hardware requires the pinned i2rt SDK in the RLinf environment; "
            "no yam-abc-reproduce checkout is required"
        ) from error

    parameters = inspect.signature(get_yam_robot).parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs: dict[str, Any] = {
        "channel": device_config.channel,
        "arm_type": ArmType.from_string_name(device_config.arm_type),
        "gripper_type": GripperType.from_string_name(device_config.gripper_type),
        "zero_gravity_mode": zero_gravity_mode,
        "ee_mass": device_config.ee_mass,
        "gripper_limits_override": (
            None
            if device_config.gripper_limits is None
            else np.asarray(device_config.gripper_limits, dtype=np.float64)
        ),
        "enable_auto_recovery": device_config.enable_auto_recovery,
        "use_coulomb_friction": device_config.use_coulomb_friction,
    }
    gravity_comp_factor = _explicit_vector(device_config, "gravity_comp_factor")
    if gravity_comp_factor is not None:
        kwargs["gravity_comp_factor"] = gravity_comp_factor

    # i2rt v1.3.3 exposes these values in get_robot_info(), but does not yet
    # accept per-instance numeric overrides. Never mutate its private arrays:
    # require an SDK that provides explicit constructor parameters instead.
    for field_name in ("grav_comp_kd", "coulomb_friction"):
        value = _explicit_vector(device_config, field_name)
        if value is None:
            continue
        if field_name not in parameters and not accepts_kwargs:
            raise RuntimeError(
                f"This i2rt version cannot apply YAM {field_name!r} per device. "
                "Install the pinned RLinf-compatible i2rt build or leave the "
                "field unset to use the SDK arm-model default."
            )
        kwargs[field_name] = value

    supported_kwargs = {
        name: value
        for name, value in kwargs.items()
        if name in parameters or accepts_kwargs
    }
    missing_required_support = sorted(set(kwargs) - set(supported_kwargs))
    if missing_required_support:
        raise RuntimeError(
            "Installed i2rt is missing required get_yam_robot options: "
            + ", ".join(missing_required_support)
        )
    return get_yam_robot(**supported_kwargs)


class _I2RTYamBackend:
    """Shared connection, health, and cleanup behavior."""

    def __init__(self, device_config: Any, *, zero_gravity_mode: bool) -> None:
        self.device_config = device_config
        self._zero_gravity_mode = zero_gravity_mode
        self._robot: Any | None = None
        self._control_thread: threading.Thread | None = None
        self._closed = False

    def connect(self) -> None:
        if self._closed:
            raise RuntimeError("cannot reconnect a closed i2rt YAM backend")
        if self._robot is None:
            self._robot = _build_yam(
                self.device_config,
                zero_gravity_mode=self._zero_gravity_mode,
            )
            self._control_thread = self._find_control_thread(self._robot)

    def assert_healthy(self, max_feedback_age_s: float) -> None:
        robot = self._require_robot()
        chain = getattr(robot, "motor_chain", None)
        if chain is None or not bool(getattr(chain, "running", False)):
            raise RuntimeError(
                f"YAM CAN chain {self.device_config.channel!r} is not running"
            )
        server_thread = getattr(robot, "_server_thread", None)
        if server_thread is None or not server_thread.is_alive():
            raise RuntimeError(
                f"YAM robot server {self.device_config.channel!r} is not alive"
            )
        timestamp_s = self._feedback_timestamp(robot)
        age_s = time.time() - timestamp_s
        if not np.isfinite(timestamp_s) or age_s > max_feedback_age_s:
            raise RuntimeError(
                f"stale YAM feedback on {self.device_config.channel!r}: "
                f"age={age_s:.3f}s, limit={max_feedback_age_s:.3f}s"
            )

    def close(self) -> None:
        if self._closed:
            return
        robot = self._robot
        if robot is None:
            self._closed = True
            return
        # Preserve the handle if cleanup fails so a later close() can retry.
        self._stop_i2rt_threads(robot)
        robot.close()
        self._robot = None
        self._control_thread = None
        self._closed = True

    def _stop_i2rt_threads(self, robot: Any) -> None:
        """Stop i2rt workers before its unjoined CAN socket close.

        The pinned official i2rt build joins the robot-server thread but closes
        the SocketCAN descriptor immediately after setting ``chain.running`` to
        false. Its CAN control thread can consequently still be inside
        ``bus.recv`` and report ``fd=-1``. Stop and join that thread first while
        retaining compatibility with builds that already expose the handle.
        """
        stop_event = getattr(robot, "_stop_event", None)
        server_thread = getattr(robot, "_server_thread", None)
        chain = getattr(robot, "motor_chain", None)
        if stop_event is None or server_thread is None or chain is None:
            return

        stop_event.set()
        server_thread.join(timeout=2.0)
        if server_thread.is_alive():
            raise RuntimeError("i2rt robot server did not stop; CAN was left open")

        chain.running = False
        control_thread = self._control_thread
        if (
            control_thread is not None
            and control_thread is not threading.current_thread()
        ):
            control_thread.join(timeout=2.0)
            if control_thread.is_alive():
                raise RuntimeError(
                    "i2rt CAN control thread did not stop; socket was left open"
                )

    @staticmethod
    def _find_control_thread(robot: Any) -> threading.Thread | None:
        """Locate the CAN worker hidden by official i2rt builds."""
        chain = getattr(robot, "motor_chain", None)
        if chain is None:
            return None
        exposed = getattr(chain, "_control_thread", None)
        if exposed is not None and all(
            callable(getattr(exposed, name, None)) for name in ("join", "is_alive")
        ):
            return exposed
        for candidate in threading.enumerate():
            target = getattr(candidate, "_target", None)
            if (
                getattr(target, "__self__", None) is chain
                and getattr(target, "__name__", "") == "_set_torques_and_update_state"
            ):
                return candidate
        return None

    def _require_robot(self) -> Any:
        if self._robot is None:
            raise RuntimeError(
                f"YAM backend {self.device_config.channel!r} is not connected"
            )
        return self._robot

    @staticmethod
    def _feedback_timestamp(robot: Any) -> float:
        # i2rt v1.3.3 has no public health snapshot. Keep this compatibility
        # access isolated here so it can be replaced when the SDK adds one.
        joint_state = getattr(robot, "_joint_state", None)
        timestamp_s = getattr(joint_state, "timestamp", None)
        if timestamp_s is None:
            raise RuntimeError("installed i2rt does not expose a feedback timestamp")
        return float(timestamp_s)


class I2RTYamFollower(_I2RTYamBackend):
    """Follower implementation using i2rt's normalized seven-DoF interface."""

    def __init__(self, device_config: Any) -> None:
        # Followers must hold their measured startup pose before connect()
        # returns.  This avoids leaving the first arm freely backdrivable while
        # the second CAN chain is still opening.  Calibration tooling may opt
        # into zero-gravity mode via ``device_config.zero_gravity_mode``.
        super().__init__(
            device_config,
            zero_gravity_mode=bool(getattr(device_config, "zero_gravity_mode", False)),
        )

    def read_state(self) -> YamArmState:
        robot = self._require_robot()
        position = np.asarray(robot.get_joint_pos(), dtype=np.float64).reshape(-1)
        if position.shape != (NUM_ARM_JOINTS + 1,):
            raise RuntimeError(
                "YAM follower must expose six arm joints and one normalized "
                f"gripper, got shape {position.shape}"
            )
        if not np.all(np.isfinite(position)):
            raise RuntimeError("YAM follower returned non-finite joint feedback")
        return YamArmState(
            joint_positions=position[:NUM_ARM_JOINTS],
            gripper_position=float(np.clip(position[-1], 0.0, 1.0)),
            timestamp_s=self._feedback_timestamp(robot),
        )

    def command(self, target: np.ndarray) -> None:
        target = np.asarray(target, dtype=np.float64).reshape(NUM_ARM_JOINTS + 1)
        if not np.all(np.isfinite(target)):
            raise ValueError("refusing to send a non-finite YAM follower command")
        # i2rt may clip the first six values in place, so never give it the
        # runtime's accepted-action array directly.
        self._require_robot().command_joint_pos(target.copy())

    def joint_limits(self) -> np.ndarray:
        limits = np.asarray(
            self._require_robot().get_robot_info()["joint_limits"],
            dtype=np.float64,
        )
        if limits.shape != (NUM_ARM_JOINTS, 2) or not np.all(np.isfinite(limits)):
            raise RuntimeError(
                f"i2rt returned invalid YAM joint limits with shape {limits.shape}"
            )
        return limits.copy()

    def hold(self) -> None:
        robot = self._require_robot()
        robot.command_joint_pos(
            np.asarray(robot.get_joint_pos(), dtype=np.float64).copy()
        )


class I2RTYamBackendFactory:
    """Create disconnected i2rt backend adapters for runtime ownership."""

    def create_follower(self, device_config: Any) -> I2RTYamFollower:
        """Create one follower without opening its CAN chain."""
        return I2RTYamFollower(device_config)
