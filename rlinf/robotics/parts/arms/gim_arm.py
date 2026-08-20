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
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from rlinf.robotics.parts.arms import ARM_STATE_FIELDS
from rlinf.robotics.parts.base import ControllablePart, RobotPart
from rlinf.robotics.parts.views import MethodGripper
from rlinf.utils.logging import get_logger

# End-effector frame name in gim_arm URDF.
_EEF_FRAME = "arm6_tool0"

# Feedforward control loop parameters (matching SDK keyboard_control.py).
_CONTROL_DT = 0.01  # 100 Hz
_VEL_CUTOFF_HZ = 4.0
_ACCEL_CUTOFF_HZ = 6.0


def _smoothstep(t: float) -> float:
    """Quintic smoothstep for smooth trajectory interpolation."""
    t = max(0.0, min(1.0, t))
    return 10 * t**3 - 15 * t**4 + 6 * t**5


@dataclass
class GimArmRobotState:
    """State snapshot for the GimArm 6-DOF robot.

    All Cartesian quantities are expressed in the robot base frame.
    """

    tcp_pose: np.ndarray = field(default_factory=lambda: np.zeros(7))
    """End-effector pose ``[x, y, z, qx, qy, qz, qw]`` (m / quaternion).
    Computed via Pinocchio FK from joint positions."""

    tcp_vel: np.ndarray = field(default_factory=lambda: np.zeros(6))
    """End-effector Cartesian velocity ``[vx, vy, vz, wx, wy, wz]`` (m/s, rad/s).
    Computed as ``J @ dq``."""

    arm_joint_position: np.ndarray = field(default_factory=lambda: np.zeros(6))
    """Joint positions ``[q1, ..., q6]`` in radians."""

    arm_joint_velocity: np.ndarray = field(default_factory=lambda: np.zeros(6))
    """Joint velocities ``[dq1, ..., dq6]`` in rad/s."""

    tcp_force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Estimated Cartesian force at EEF ``[fx, fy, fz]`` in N.
    Mapped from momentum-observer external torque via ``J^{-T}``.
    Zero when momentum observer is not active."""

    tcp_torque: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Estimated Cartesian torque at EEF ``[tx, ty, tz]`` in N-m.
    Mapped from momentum-observer external torque via ``J^{-T}``.
    Zero when momentum observer is not active."""

    arm_jacobian: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))
    """Body Jacobian ``(6, 6)`` in LOCAL_WORLD_ALIGNED frame.
    Computed via Pinocchio at current joint positions."""

    gripper_position: float = 0.0
    """Gripper joint position in radians (hardware units)."""

    gripper_open: bool = False
    """``True`` when the gripper position is closer to open than closed."""

    def to_dict(self):
        """Convert the dataclass to a serializable dictionary."""
        return asdict(self)


class GimArm(ControllablePart):
    """GimArm robot arm controller.

    Wraps the ``gim_arm_control`` SDK (CAN bus) independently of scheduling.

    Runs in **MOMENTUM_OBSERVER** mode by default.  A background feedforward
    control thread at 100 Hz computes Butterworth-filtered velocity and
    acceleration from the target position, then sends them via
    ``set_feedforward_target(q, dq, ddq)`` so the SDK can compute proper
    dynamics-based torques (gravity + inertia + Coriolis + external torque
    compensation).

    All ``gim_arm_control`` and ``pinocchio`` imports are deferred to
    :meth:`connect` so this module can be imported on GPU-only nodes that
    do not have the robot SDK installed.
    """

    def __init__(
        self,
        can_interface: str,
        arm_variant: str,
        enable_gripper: bool,
        gripper_type: str,
        control_mode: str = "momentum_observer",
    ):
        self._logger = get_logger()
        self._can_interface = can_interface
        self._arm_variant = arm_variant
        self._enable_gripper = enable_gripper
        self._gripper_type = gripper_type
        self._control_mode = control_mode
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Whether the CAN controller and feedforward loop are active."""
        return self._connected

    @property
    def observation_features(self) -> dict:
        """Describe canonical GimArm arm state fields.

        Gripper fields belong to the end-effector part from :attr:`exports`.
        """
        return {name: {} for name in ARM_STATE_FIELDS}

    @property
    def action_features(self) -> dict:
        """Describe the absolute joint target."""
        return {"joint_position": {}}

    @property
    def exports(self) -> dict[str, RobotPart]:
        """Expose the arm, plus the gripper when one is fitted."""
        parts: dict[str, RobotPart] = {"arm": self}
        if self._enable_gripper:
            parts["end_effector"] = MethodGripper(self, state_field="gripper_position")
        return parts

    def connect(self) -> None:
        """Connect the CAN SDK and start the feedforward control loop."""
        if self._connected:
            return
        import pinocchio as pin
        from gim_arm_control import (
            ButterworthFilter,
            ControllerConfig,
            ControlMode,
        )
        from gim_arm_control import (
            GimArmController as _SDKController,
        )
        from gim_arm_control.utils.urdf_loader import (
            get_urdf_path,
            load_arm6_model,
        )

        self._ControlMode = ControlMode
        self._ButterworthFilter = ButterworthFilter
        sdk_config = ControllerConfig(
            can_interface=self._can_interface,
            arm_variant=self._arm_variant,
            enable_gripper=self._enable_gripper,
            gripper_type=self._gripper_type,
        )
        self._sdk = _SDKController(sdk_config)
        if not self._sdk.start(return_to_zero=True):
            raise RuntimeError(
                f"Failed to start GimArm on CAN interface {self._can_interface!r}."
            )
        self._sdk.set_mode(ControlMode[self._control_mode.upper()])

        urdf_path = get_urdf_path(self._arm_variant)
        self._pin_model, self._pin_data = load_arm6_model(urdf_path)
        assert self._pin_model.nv >= 6, (
            f"Pinocchio model nv={self._pin_model.nv}, expected >= 6 for GimArm."
        )
        self._pin_ee_frame_id = self._pin_model.getFrameId(_EEF_FRAME)
        if self._pin_ee_frame_id >= self._pin_model.nframes:
            raise RuntimeError(
                f"End-effector frame '{_EEF_FRAME}' not found in URDF '{urdf_path}'. "
                f"Available frames: "
                f"{[self._pin_model.frames[i].name for i in range(self._pin_model.nframes)]}"
            )
        self._pin = pin

        reading = self._sdk.get_reading()
        initial_q = np.array(reading.position) if reading is not None else np.zeros(6)
        self._lock = threading.Lock()
        self._target_q = initial_q.copy()
        self._prev_q = initial_q.copy()
        self._prev_dq = np.zeros(6)
        dof = self._sdk.get_dof()
        self._velocity_filter = ButterworthFilter(_VEL_CUTOFF_HZ, _CONTROL_DT, dof)
        self._accel_filter = ButterworthFilter(_ACCEL_CUTOFF_HZ, _CONTROL_DT, dof)
        self._control_running = True
        self._control_thread = threading.Thread(
            target=self._feedforward_loop, daemon=True
        )
        self._control_thread.start()
        self._connected = True

    def reset(self) -> None:
        """Leave task-specific reset positions to the caller."""

    def get_observation(self) -> dict:
        """Return the canonical arm state, without end-effector fields."""
        state = self.get_state().to_dict()
        return {name: state[name] for name in ARM_STATE_FIELDS}

    def send_action(self, action: dict) -> dict:
        """Apply one absolute joint target."""
        if set(action) != {"joint_position"}:
            raise KeyError("GimArm action must contain only 'joint_position'.")
        self.move_joints(action["joint_position"])
        return action

    def disconnect(self) -> None:
        """Stop the control loop and disconnect the SDK."""
        if self._connected:
            self.stop()
            self._connected = False

    # ── Feedforward control loop ─────────────────────────────────────────

    def _feedforward_loop(self):
        """Background loop: filter target and send feedforward commands at 100 Hz."""
        next_time = time.monotonic()
        while self._control_running:
            with self._lock:
                target = self._target_q.copy()

            raw_dq = (target - self._prev_q) / _CONTROL_DT
            dq = self._velocity_filter.process(raw_dq)

            raw_ddq = (dq - self._prev_dq) / _CONTROL_DT
            ddq = self._accel_filter.process(raw_ddq)

            self._prev_q = target.copy()
            self._prev_dq = dq.copy()

            self._sdk.set_feedforward_target(target, dq, ddq)

            next_time += _CONTROL_DT
            now = time.monotonic()
            sleep_duration = next_time - now
            if sleep_duration > 0:
                time.sleep(sleep_duration)
            else:
                next_time = now

    # ── Public API ───────────────────────────────────────────────────────

    def is_robot_up(self) -> bool:
        """Return ``True`` when the SDK has a valid reading and no active faults."""
        reading = self._sdk.get_reading()
        return reading is not None and not reading.has_fault

    def get_state(self) -> GimArmRobotState:
        """Compute and return the current robot state.

        Performs Pinocchio FK and Jacobian evaluation on the latest hardware
        reading.  External torque (if available from the momentum observer) is
        mapped to a Cartesian wrench via ``J^{-T}``.
        """
        reading = self._sdk.get_reading()
        if reading is None:
            raise RuntimeError(
                "get_state: SDK returned no reading (CAN bus disconnected or not yet initialized)."
            )
        q = np.array(reading.position)
        dq = np.array(reading.velocity)
        pin = self._pin

        # Forward kinematics.
        q_pin = pin.neutral(self._pin_model)
        q_pin[:6] = q
        pin.forwardKinematics(self._pin_model, self._pin_data, q_pin)
        pin.updateFramePlacement(self._pin_model, self._pin_data, self._pin_ee_frame_id)
        T = self._pin_data.oMf[self._pin_ee_frame_id]
        tcp_quat = pin.Quaternion(T.rotation).coeffs()  # [qx, qy, qz, qw]
        tcp_pose = np.concatenate([T.translation, tcp_quat])

        # Jacobian in LOCAL_WORLD_ALIGNED frame.
        J = pin.computeFrameJacobian(
            self._pin_model,
            self._pin_data,
            q_pin,
            self._pin_ee_frame_id,
            pin.LOCAL_WORLD_ALIGNED,
        )

        # Slice to the 6 actuated arm joints. The full Jacobian has shape
        # (6, model.nv) which may be wider than (6, 6) if the URDF includes
        # additional joints (e.g. gripper DOFs).
        J_arm = J[:, :6]

        tcp_vel = J_arm @ dq

        # Map external joint torques to Cartesian wrench via J^{-T}.
        tau_ext = reading.external_torque
        tcp_force = np.zeros(3)
        tcp_torque = np.zeros(3)
        if tau_ext is not None:
            try:
                wrench = np.linalg.pinv(J_arm.T) @ np.array(tau_ext)
                tcp_force = wrench[:3]
                tcp_torque = wrench[3:]
            except Exception as e:
                self._logger.warning(
                    f"Failed to compute Cartesian wrench from external torque: {e}"
                )

        # Gripper state.
        gripper_pos = (
            reading.gripper_position if reading.gripper_position is not None else 0.0
        )
        open_pos = self._sdk.gripper_open_position
        closed_pos = self._sdk.gripper_closed_position
        mid = (open_pos + closed_pos) / 2.0
        gripper_open = gripper_pos <= mid

        return GimArmRobotState(
            tcp_pose=tcp_pose,
            tcp_vel=tcp_vel,
            arm_joint_position=q,
            arm_joint_velocity=dq,
            tcp_force=tcp_force,
            tcp_torque=tcp_torque,
            arm_jacobian=J_arm,
            gripper_position=gripper_pos,
            gripper_open=gripper_open,
        )

    def move_joints(self, q_target: np.ndarray) -> None:
        """Set the target joint position (non-blocking).

        The background feedforward control thread picks up the new target,
        computes Butterworth-filtered velocity and acceleration, and sends
        the feedforward command to the SDK at 100 Hz.

        Args:
            q_target: Desired joint positions ``(6,)`` in radians.
        """
        with self._lock:
            self._target_q = np.array(q_target, dtype=np.float64)

    def reset_joint(self, reset_qpos: list[float], duration: float = 3.0) -> None:
        """Gradually move to a joint reset configuration using smooth interpolation.

        Uses quintic smoothstep to interpolate from the current position to
        ``reset_qpos`` over ``duration`` seconds.  The feedforward control
        thread handles filtering and sending commands.

        Args:
            reset_qpos: Target joint positions ``(6,)`` in radians.
            duration: Time in seconds for the interpolation.
        """
        reading = self._sdk.get_reading()
        if reading is None:
            self._logger.warning("reset_joint: no reading available, skipping.")
            return

        start_q = np.array(reading.position, dtype=np.float64)
        target_q = np.array(reset_qpos, dtype=np.float64)
        num_steps = max(1, int(duration / _CONTROL_DT))

        for step in range(num_steps + 1):
            t = step / num_steps
            blend = _smoothstep(t)
            interp_q = start_q + (target_q - start_q) * blend
            with self._lock:
                self._target_q = interp_q
            time.sleep(_CONTROL_DT)

        # Ensure final target is exact.
        with self._lock:
            self._target_q = target_q.copy()

    def open_gripper(self) -> None:
        """Open the gripper to its hardware open position."""
        self._sdk.set_gripper(self._sdk.gripper_open_position)

    def close_gripper(self) -> None:
        """Close the gripper to its hardware closed position."""
        self._sdk.set_gripper(self._sdk.gripper_closed_position)

    def clear_errors(self) -> None:
        """No-op — the GimArm SDK handles fault recovery internally."""
        pass

    def stop(self) -> None:
        """Stop the feedforward control thread and the SDK."""
        self._control_running = False
        if self._control_thread.is_alive():
            self._control_thread.join(timeout=2.0)
        self._sdk.stop()
