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

"""YAM follower arm, driven through ``i2rt`` over a CAN bus.

Registered as ``i2rt``, after the SDK. The gripper shares the arm's CAN chain
and is exported beneath it. i2rt takes the six joints and the normalized
gripper as one seven-wide target, so the arm keeps the last target of each and
sends them together.
"""

import math
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from rlinf.robotics.parts.base import Action, Features, Observation, RobotPart
from rlinf.robotics.parts.claims import DeviceClaim
from rlinf.robotics.parts.views import MethodEndEffector

from .base import Arm, BaseArm


@dataclass
class YamRobotState:
    """State snapshot for one YAM follower."""

    arm_joint_position: np.ndarray = field(default_factory=lambda: np.zeros(6))
    """Joint positions in radians."""

    gripper_position: np.ndarray = field(default_factory=lambda: np.zeros(1))
    """Opening as a fraction, 0 closed to 1 open."""

    timestamp: float = 0.0
    """Host time of the joint feedback frame, in seconds since the epoch."""

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass to a serializable dictionary."""
        return asdict(self)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"YAM {name!r} must be a number, but got bool.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"YAM {name!r} must be finite, but got {value!r}.")
    return result


@Arm.register("i2rt")
class YamArm(BaseArm):
    """YAM follower arm on a CAN bus, via ``i2rt``.

    It holds its measured pose from the moment it connects, so an arm that has
    opened stays put while the rest of a station opens.

    Args:
        channel: SocketCAN interface, such as ``"can0"``.
        arm_type: i2rt arm model.
        gripper_type: i2rt gripper model fitted to the arm.
        ee_mass: End-effector mass in kilograms. ``None`` keeps the model's.
        gripper_limits: Motor stops as ``(closed, open)`` in radians. The
            order carries the motor direction. ``None`` lets i2rt find them.
        gravity_comp_factor: Per-joint gravity compensation multipliers.
            ``None`` keeps the SDK's.
        use_coulomb_friction: Whether i2rt compensates Coulomb friction.
        enable_auto_recovery: Whether i2rt recovers a faulted motor itself.
            Off, a motor fault stops the arm.
        feedback_timeout_s: Joint feedback older than this fails a read.
    """

    SDK = "i2rt"

    DOF: ClassVar[int] = 6

    #: No tool pose, velocity, or force is reported.
    STATE_FIELDS = ("arm_joint_position",)

    def __init__(
        self,
        channel: str = "can0",
        *,
        arm_type: str = "yam",
        gripper_type: str = "flexible_4310",
        ee_mass: Optional[float] = None,
        gripper_limits: Optional[Sequence[float]] = None,
        gravity_comp_factor: Optional[Sequence[float]] = None,
        use_coulomb_friction: bool = False,
        enable_auto_recovery: bool = False,
        feedback_timeout_s: float = 0.25,
    ) -> None:
        if not isinstance(channel, str) or not channel.strip():
            raise ValueError(f"A YAM arm needs a CAN channel, but got {channel!r}.")
        self._channel = channel.strip()
        self._arm_type = str(arm_type)
        self._gripper_type = str(gripper_type)
        self._ee_mass = None if ee_mass is None else _finite(ee_mass, "ee_mass")
        if self._ee_mass is not None and self._ee_mass < 0.0:
            raise ValueError(f"YAM 'ee_mass' must be non-negative, got {ee_mass}.")
        self._gripper_limits = None
        if gripper_limits is not None:
            stops = [_finite(value, "gripper_limits") for value in gripper_limits]
            if len(stops) != 2 or stops[0] == stops[1]:
                raise ValueError(
                    "YAM 'gripper_limits' must be two different stops ordered "
                    f"(closed, open), but got {list(gripper_limits)}."
                )
            self._gripper_limits = np.asarray(stops, dtype=np.float64)
        self._gravity_comp_factor = None
        if gravity_comp_factor is not None:
            factors = [_finite(v, "gravity_comp_factor") for v in gravity_comp_factor]
            if len(factors) != self.DOF or any(v < 0.0 for v in factors):
                raise ValueError(
                    "YAM 'gravity_comp_factor' must be six non-negative values, "
                    f"but got {list(gravity_comp_factor)}."
                )
            self._gravity_comp_factor = np.asarray(factors, dtype=np.float64)
        for name, flag in (
            ("use_coulomb_friction", use_coulomb_friction),
            ("enable_auto_recovery", enable_auto_recovery),
        ):
            if not isinstance(flag, bool):
                raise TypeError(f"YAM {name!r} must be a bool, but got {flag!r}.")
        self._use_coulomb_friction = use_coulomb_friction
        self._enable_auto_recovery = enable_auto_recovery
        self._feedback_timeout_s = _finite(feedback_timeout_s, "feedback_timeout_s")
        if self._feedback_timeout_s <= 0.0:
            raise ValueError(
                f"YAM 'feedback_timeout_s' must be positive, got {feedback_timeout_s}."
            )
        self._robot: Any = None
        self._control_thread: Optional[threading.Thread] = None
        self._joint_target: Optional[np.ndarray] = None
        self._gripper_target: Optional[float] = None
        # Two parts on one CAN channel would fight over the bus.
        self._claim = DeviceClaim(f"yam-arm:{self._channel}", type(self).__name__)

    @classmethod
    def declare(
        cls,
        address: str,
        *,
        gripper_type: Optional[str] = None,
        gripper_connection: Optional[str] = None,
        end_effector_type: Optional[str] = None,
        end_effector_config: Optional[dict] = None,
        **placement: Any,
    ) -> "YamArm":
        """Declare a YAM follower on the CAN channel named by ``address``.

        The gripper shares that bus, so ``gripper_type`` names the i2rt
        gripper model rather than a separate driver.
        """
        cls.refuse_unused(
            gripper_connection=gripper_connection,
            end_effector_type=end_effector_type,
            end_effector_config=end_effector_config,
        )
        if gripper_type is not None:
            placement["gripper_type"] = gripper_type
        return cls(address, **placement)

    @property
    def action_features(self) -> Features:
        """Describe the absolute joint target, in radians."""
        return {"joint_position": {}}

    @property
    def parts(self) -> dict[str, RobotPart]:
        """Return the gripper this arm carries on its own bus."""
        return {
            "end_effector": MethodEndEffector(
                self,
                state_field="gripper_position",
                command="move_gripper",
                is_gripper=True,
            )
        }

    def _open(self) -> Any:
        """Build the i2rt robot and command the pose it stands at."""
        self._claim.acquire()
        try:
            robot = self._build()
            self._robot = robot
            self._control_thread = self._find_control_thread(robot)
            # i2rt starts in position control at the measured pose; repeating
            # it seeds the targets this part keeps.
            self.hold()
        except BaseException:
            self._release(self._robot)
            raise
        return robot

    def _build(self) -> Any:
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import ArmType, GripperType

        settings: dict[str, Any] = {
            "channel": self._channel,
            "arm_type": ArmType.from_string_name(self._arm_type),
            "gripper_type": GripperType.from_string_name(self._gripper_type),
            # A follower holds position; zero-gravity mode would let it fall.
            "zero_gravity_mode": False,
            "ee_mass": self._ee_mass,
            "gripper_limits_override": self._gripper_limits,
            "enable_auto_recovery": self._enable_auto_recovery,
            "use_coulomb_friction": self._use_coulomb_friction,
        }
        if self._gravity_comp_factor is not None:
            settings["gravity_comp_factor"] = self._gravity_comp_factor.copy()
        return get_yam_robot(**settings)

    def _release(self, device: Any) -> None:
        """Stop i2rt's threads, then close the CAN chain.

        Also used to unwind a failed open, so ``device`` may be ``None``.
        """
        try:
            if device is not None:
                self._stop_i2rt_threads(device)
                device.close()
        finally:
            self._robot = None
            self._control_thread = None
            self._joint_target = None
            self._gripper_target = None
            self._claim.release()

    def _stop_i2rt_threads(self, robot: Any) -> None:
        """Join i2rt's workers before its CAN socket closes.

        The pinned i2rt joins its server thread on close but closes the CAN
        descriptor without joining the control thread, which can still be in
        ``bus.recv``. Stopping and joining both first keeps that thread from
        reading a closed socket.
        """
        stop_event = getattr(robot, "_stop_event", None)
        server_thread = getattr(robot, "_server_thread", None)
        chain = getattr(robot, "motor_chain", None)
        if stop_event is None or server_thread is None or chain is None:
            return

        stop_event.set()
        server_thread.join(timeout=2.0)
        if server_thread.is_alive():
            raise RuntimeError(
                f"The i2rt server on {self._channel!r} did not stop, so its CAN "
                "chain was left open."
            )
        chain.running = False
        control_thread = self._control_thread
        if (
            control_thread is not None
            and control_thread is not threading.current_thread()
        ):
            control_thread.join(timeout=2.0)
            if control_thread.is_alive():
                raise RuntimeError(
                    f"The i2rt CAN thread on {self._channel!r} did not stop, so "
                    "its socket was left open."
                )

    @staticmethod
    def _find_control_thread(robot: Any) -> Optional[threading.Thread]:
        """Locate the CAN thread, which the pinned i2rt does not expose."""
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
            raise RuntimeError(f"The YAM arm on {self._channel!r} is not connected.")
        return self._robot

    def _assert_healthy(self, robot: Any) -> None:
        """Refuse a read once the CAN chain or its feedback has stopped."""
        chain = getattr(robot, "motor_chain", None)
        if chain is None or not bool(getattr(chain, "running", False)):
            raise RuntimeError(f"The YAM CAN chain {self._channel!r} is not running.")
        server_thread = getattr(robot, "_server_thread", None)
        if server_thread is None or not server_thread.is_alive():
            raise RuntimeError(f"The YAM robot server {self._channel!r} is not alive.")
        age_s = time.time() - self._feedback_timestamp(robot)
        if not math.isfinite(age_s) or age_s > self._feedback_timeout_s:
            raise RuntimeError(
                f"YAM feedback on {self._channel!r} is {age_s:.3f}s old, past the "
                f"{self._feedback_timeout_s:.3f}s limit."
            )

    def get_state(self) -> YamRobotState:
        """Read joints and gripper, refusing stale or non-finite feedback."""
        robot = self._require_robot()
        self._assert_healthy(robot)
        position = np.asarray(robot.get_joint_pos(), dtype=np.float64).reshape(-1)
        if position.shape != (self.DOF + 1,):
            raise RuntimeError(
                f"A YAM follower reports six joints and a gripper, but "
                f"{self._channel!r} reported shape {position.shape}."
            )
        if not np.all(np.isfinite(position)):
            raise RuntimeError(f"The YAM arm on {self._channel!r} reported NaN.")
        return YamRobotState(
            arm_joint_position=position[: self.DOF].copy(),
            gripper_position=np.array([np.clip(position[-1], 0.0, 1.0)]),
            timestamp=self._feedback_timestamp(robot),
        )

    @staticmethod
    def _feedback_timestamp(robot: Any) -> float:
        # The pinned i2rt has no public health snapshot; its joint state carries
        # the time of the last feedback frame.
        timestamp = getattr(getattr(robot, "_joint_state", None), "timestamp", None)
        if timestamp is None:
            raise RuntimeError("The installed i2rt reports no feedback timestamp.")
        return float(timestamp)

    def joint_limits(self) -> np.ndarray:
        """Return the travel i2rt clips commands to, as ``(6, 2)`` radians."""
        limits = np.asarray(
            self._require_robot().get_robot_info()["joint_limits"], dtype=np.float64
        )
        if limits.shape != (self.DOF, 2) or not np.all(np.isfinite(limits)):
            raise RuntimeError(
                f"i2rt reported invalid YAM joint limits of shape {limits.shape}."
            )
        return limits.copy()

    def send_action(self, action: Action) -> Observation:
        """Move the arm joints to an absolute target in radians."""
        if set(action) != {"joint_position"}:
            raise KeyError(
                "A YAM arm action holds only 'joint_position'; the gripper is "
                "commanded through its own end-effector part."
            )
        return {"joint_position": self.move_joints(action["joint_position"])}

    def move_joints(self, q_target: "Sequence[float]") -> np.ndarray:
        """Command absolute joint positions in radians.

        Returns:
            The target sent, in radians.
        """
        target = np.asarray(q_target, dtype=np.float64).reshape(-1)
        if target.shape != (self.DOF,):
            raise ValueError(
                f"Expected {self.DOF} joint targets for a YAM arm, got shape "
                f"{target.shape}."
            )
        if not np.all(np.isfinite(target)):
            raise ValueError("Refusing a non-finite YAM joint target.")
        self._require_robot()
        self._joint_target = target.copy()
        self._command()
        return target

    def move_gripper(self, target: "Sequence[float]") -> None:
        """Command the gripper to an opening fraction in ``0..1``."""
        value = float(np.asarray(target, dtype=np.float64).reshape(-1)[0])
        if not math.isfinite(value):
            raise ValueError("Refusing a non-finite YAM gripper target.")
        value = float(np.clip(value, 0.0, 1.0))
        self._require_robot()
        # The joint command already carried an unchanged gripper target.
        if value == self._gripper_target:
            return
        self._gripper_target = value
        self._command()

    def hold(self) -> None:
        """Command the measured pose, gripper included.

        Reads without the feedback-age check, so it still holds an arm whose
        feedback has gone stale.
        """
        robot = self._require_robot()
        measured = np.asarray(robot.get_joint_pos(), dtype=np.float64).copy()
        self._joint_target = measured[: self.DOF].copy()
        self._gripper_target = float(measured[-1])
        robot.command_joint_pos(measured)

    def _command(self) -> None:
        # i2rt clips the arm joints of the array it is given in place.
        self._robot.command_joint_pos(
            np.append(self._joint_target, self._gripper_target)
        )

    def is_robot_up(self) -> bool:
        """Report whether the CAN chain and the i2rt server are running."""
        robot = self._robot
        if robot is None:
            return False
        chain = getattr(robot, "motor_chain", None)
        server_thread = getattr(robot, "_server_thread", None)
        return bool(
            chain is not None
            and getattr(chain, "running", False)
            and server_thread is not None
            and server_thread.is_alive()
        )


@dataclass
class YamIKConfig:
    """Limits for one synchronous i2rt IK solve; timing is checked on return."""

    site_name: str = "grasp_site"
    solver: str = "quadprog"
    max_iters: int = 30
    dt: float = 0.01
    position_tolerance: float = 0.001
    rotation_tolerance: float = 0.01
    max_solve_s: float = 0.03

    def __post_init__(self) -> None:
        if not self.site_name or not self.solver:
            raise ValueError("IK site_name and solver must be nonempty")
        if (
            isinstance(self.max_iters, bool)
            or not isinstance(self.max_iters, int)
            or self.max_iters < 1
        ):
            raise ValueError("IK max_iters must be a positive integer")
        for name in (
            "dt",
            "position_tolerance",
            "rotation_tolerance",
            "max_solve_s",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"IK {name} must be finite and positive")
            setattr(self, name, value)


@dataclass(frozen=True)
class YamIKResult:
    """A checked six-joint solution; failed solutions must never be dispatched."""

    success: bool
    q_target: np.ndarray
    position_error: float
    rotation_error: float
    elapsed_s: float
    reason: str | None = None


class YamKinematicsAdapter:
    """Own one independent i2rt configuration without opening any hardware.

    V1 deliberately supports the verified YAM + flexible_4310 model only.
    Its grasp_site is attached before the two finger slides, whose normalized
    positions map linearly to their XML ranges and are frozen throughout IK.
    """

    def __init__(
        self,
        *,
        arm_type: str = "yam",
        gripper_type: str = "flexible_4310",
        config: YamIKConfig | None = None,
        joint_lower: np.ndarray | None = None,
        joint_upper: np.ndarray | None = None,
    ) -> None:
        if (arm_type, gripper_type) != ("yam", "flexible_4310"):
            raise ValueError(
                "YAM VR v1 requires yam + flexible_4310; other models need explicit TCP/joint validation"
            )
        import mink
        import mujoco
        from i2rt.robots.kinematics import Kinematics
        from i2rt.robots.utils import (
            ArmType,
            GripperType,
            combine_arm_and_gripper_xml,
        )

        self.config = config or YamIKConfig()
        xml_path = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.FLEXIBLE_4310)
        try:
            self._kin = Kinematics(xml_path, self.config.site_name)
        finally:
            # This is the exact temporary XML returned by our SDK invocation.
            Path(xml_path).unlink(missing_ok=True)
        # The pinned i2rt version exposes its MuJoCo model only through this
        # configuration. Keep that dependency confined to this adapter.
        self.model = self._kin._configuration.model
        names = [self.model.joint(i).name for i in range(self.model.njnt)]
        arm_names = [f"joint{i}" for i in range(1, 7)]
        finger_names = ["joint7", "joint8"]
        if (
            set(names) != set(arm_names + finger_names)
            or self.model.nq != 8
            or self.model.nv != 8
        ):
            raise ValueError(f"Unsupported YAM model joint layout: {names}")
        self._arm_ids = np.array([self.model.joint(n).id for n in arm_names])
        self._finger_ids = np.array([self.model.joint(n).id for n in finger_names])
        if np.any(
            self.model.jnt_type[self._arm_ids] != mujoco.mjtJoint.mjJNT_HINGE
        ) or np.any(
            self.model.jnt_type[self._finger_ids] != mujoco.mjtJoint.mjJNT_SLIDE
        ):
            raise ValueError("YAM model must have six hinges and two finger slides")
        self._arm_q = self.model.jnt_qposadr[self._arm_ids]
        self._finger_q = self.model.jnt_qposadr[self._finger_ids]
        self.lower = self.model.jnt_range[self._arm_ids, 0].copy()
        self.upper = self.model.jnt_range[self._arm_ids, 1].copy()
        if joint_lower is not None:
            self.lower = np.maximum(self.lower, self._vector(joint_lower))
        if joint_upper is not None:
            self.upper = np.minimum(self.upper, self._vector(joint_upper))
        if np.any(self.lower >= self.upper):
            raise ValueError("YAM configured and model joint limits do not overlap")
        self.model.jnt_range[self._arm_ids, 0] = self.lower
        self.model.jnt_range[self._arm_ids, 1] = self.upper
        self.model.site(self.config.site_name)  # Validate before any CAN startup.
        self._limits = [
            mink.ConfigurationLimit(self.model),
            mink.VelocityLimit(self.model, dict.fromkeys(finger_names, 0.0)),
        ]

    @property
    def max_solve_s(self) -> float:
        """Seconds a solve may take before its result is refused."""
        return self.config.max_solve_s

    @staticmethod
    def _vector(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError("YAM kinematics requires six finite joint positions")
        return q

    def _qpos(self, q: np.ndarray, gripper_position: float) -> np.ndarray:
        q = self._vector(q)
        if not np.isfinite(gripper_position) or not 0 <= gripper_position <= 1:
            raise ValueError("YAM gripper position must be finite and within [0, 1]")
        full = self.model.qpos0.copy()
        full[self._arm_q] = q
        ranges = self.model.jnt_range[self._finger_ids]
        full[self._finger_q] = ranges[:, 0] + gripper_position * (
            ranges[:, 1] - ranges[:, 0]
        )
        return full

    def fk(self, q: np.ndarray, gripper_position: float) -> np.ndarray:
        """Return a 4x4 TCP pose in the model's fixed robot-base frame."""
        return self._kin.fk(self._qpos(q, gripper_position)).copy()

    def solve(
        self, target: np.ndarray, q_seed: np.ndarray, gripper_position: float
    ) -> YamIKResult:
        """Solve and verify IK, including frozen fingers and joint limits.

        The control runtime limits command slew relative to measured joints.
        max_solve_s is a post-return rejection threshold, not a preemptive
        deadline. The underlying SDK call is bounded by max_iters.
        """
        started = time.monotonic()
        q_seed = self._vector(q_seed).copy()
        target = np.asarray(target, dtype=np.float64)
        reason = None
        pos_error = rot_error = float("inf")
        q_result = q_seed.copy()
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            raise ValueError("IK target must be a finite 4x4 transform")
        if (
            not np.allclose(target[3], [0, 0, 0, 1])
            or not np.allclose(target[:3, :3].T @ target[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(target[:3, :3]), 1.0)
        ):
            raise ValueError("IK target must contain a valid rigid rotation")
        initial = self._qpos(q_seed, gripper_position)
        if np.any(q_seed < self.lower) or np.any(q_seed > self.upper):
            return YamIKResult(
                False,
                q_result,
                pos_error,
                rot_error,
                time.monotonic() - started,
                "seed_out_of_limits",
            )
        try:
            ok, solved = self._kin.ik(
                target,
                self.config.site_name,
                init_q=initial,
                limits=self._limits,
                solver=self.config.solver,
                dt=self.config.dt,
                max_iters=self.config.max_iters,
                pos_threshold=self.config.position_tolerance,
                ori_threshold=self.config.rotation_tolerance,
            )
            solved = np.asarray(solved).copy()
            if solved.shape != initial.shape or not np.all(np.isfinite(solved)):
                reason = "invalid_solution"
            elif not ok:
                reason = "not_converged"
            elif not np.allclose(
                solved[self._finger_q], initial[self._finger_q], atol=1e-8, rtol=0
            ):
                reason = "finger_motion"
            else:
                q_result = solved[self._arm_q].copy()
                actual = self.fk(q_result, gripper_position)
                pos_error = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
                rot_error = float(
                    Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude()
                )
                if np.any(q_result < self.lower) or np.any(q_result > self.upper):
                    reason = "joint_limits"
                elif (
                    pos_error > self.config.position_tolerance
                    or rot_error > self.config.rotation_tolerance
                ):
                    reason = "residual"
        except Exception as exc:
            reason = f"solver_error:{type(exc).__name__}"
        elapsed = time.monotonic() - started
        if elapsed > self.config.max_solve_s:
            reason = "solve_timeout"
        return YamIKResult(
            reason is None, q_result, pos_error, rot_error, elapsed, reason
        )
