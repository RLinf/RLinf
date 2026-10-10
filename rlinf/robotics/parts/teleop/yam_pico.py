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

"""PICO VR controllers driving a dual YAM in joint space.

One handheld controller per arm. Each accumulates the operator's motion from
the pose the arm stood at when the grip engaged, solves IK against the YAM
model, and reports an absolute joint target. Both arms are always commanded:
one the operator is not holding is commanded to the pose it stood at when
control was lost, which keeps the vector complete for collection without
handing the arm back to the policy.

This device fills joint-position slots, so it is not the shared Cartesian
``pico`` device: there the arm slot means a pose or delta, here it means six
absolute joint angles.
"""

from __future__ import annotations

import time
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from rlinf.utils.logging import get_logger

from ...actions import ActionKind
from ..base import Features, Observation
from .base import TeleopAction, TeleopDevice

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rlinf.envs.real.yam.config import YamPicoConfig

    from .group import TeleopEntry

#: The two arms a dual YAM station drives, in action-vector order.
SIDES = ("left", "right")


def _delta_to_tcp_pose(
    action: np.ndarray,
    tcp_pose: np.ndarray,
    action_scale: np.ndarray,
    *,
    clip_motion: bool = True,
) -> np.ndarray:
    """Compose a normalized motion delta with the measured TCP pose.

    Returns xyz in meters followed by an xyzw quaternion. ``clip_motion``
    caps the spatial rotation error; the VR path passes ``False`` so the
    operator's full wrist angle reaches IK, where the joint-step limits
    decide how much of it one control tick may execute.
    """
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    action_scale = np.asarray(action_scale, dtype=np.float64)
    target_pos = np.asarray(tcp_pose[:3], dtype=np.float64) + action[:3] * float(
        action_scale[0]
    )
    rot_action = np.asarray(action[3:6], dtype=np.float64)
    norm = float(np.linalg.norm(rot_action))
    if clip_motion and norm > 1.0:
        rot_action = rot_action / norm
    target_rot = Rotation.from_rotvec(rot_action * float(action_scale[1])) * (
        Rotation.from_quat(np.asarray(tcp_pose[3:7], dtype=np.float64))
    )
    return np.concatenate([target_pos, target_rot.as_quat()])


class _YamPicoArm:
    """One controller composed onto one YAM arm's measured TCP pose.

    The transport in :mod:`rlinf.robotics.parts.transports.pico` owns the
    socket, the base calibration, and the motion accumulated since the grip
    was engaged. This class adds what the YAM VR path needs and the shared
    Cartesian devices do not: a *cumulative* TCP target, anchored where the
    arm stood when the grip closed, whose rotation reaches equal to the
    operator's rotation rather than a per-step slice of it.
    """

    def __init__(self, **expert_kwargs: Any) -> None:
        from ..transports.pico import PicoExpert

        self.hand = str(expert_kwargs["hand"])
        self._expert = PicoExpert(**expert_kwargs)
        self._ref_tcp_pos: Optional[np.ndarray] = None
        self._ref_tcp_rot: Optional[Rotation] = None

    @property
    def ready(self) -> bool:
        """Whether the controller stream is fresh enough to command from."""
        return self._expert.ready

    def read(self) -> dict[str, Any]:
        """Return what the operator is doing with this controller."""
        return self._expert.get_reading()

    def read_buttons(self) -> dict[str, bool]:
        """Return raw controller buttons, for record and discard edges."""
        return self._expert.get_buttons()

    def reset_reference(self) -> None:
        """Forget the TCP pose the last grip was anchored to."""
        self._ref_tcp_pos = None
        self._ref_tcp_rot = None

    def stop(self) -> None:
        """Close this controller's subscription."""
        self._expert.stop()

    def command(
        self,
        reading: Mapping[str, Any],
        tcp_pose: np.ndarray,
        action_scale: np.ndarray,
        *,
        gripper_enabled: bool = True,
        clip_motion: bool = True,
    ) -> tuple[np.ndarray, bool, dict[str, Any]]:
        """Encode the controller target as a scaled spatial pose error.

        With ``clip_motion=False``, motion components may exceed ``[-1, 1]``;
        compose them with :func:`_delta_to_tcp_pose` using the same flag to
        recover the full target and let the robot controller limit motion.
        """
        info = self._describe(reading)
        if not reading.get("held", False):
            # The transport reports a missing calibration, a stale stream, and
            # an invalid pose by releasing the grip, so this covers them all.
            self.reset_reference()
            return np.zeros(7, dtype=np.float32), False, info

        pose = np.asarray(tcp_pose, dtype=np.float64).reshape(-1)
        if self._ref_tcp_pos is None:
            self._ref_tcp_pos = pose[:3].copy()
            self._ref_tcp_rot = Rotation.from_quat(pose[3:7])

        target_pos = self._ref_tcp_pos + np.asarray(
            reading["position_delta"], dtype=np.float64
        )
        target_rot = (
            Rotation.from_rotvec(
                np.asarray(reading["rotation_delta"], dtype=np.float64)
            )
            * self._ref_tcp_rot
        )

        scale = np.asarray(action_scale, dtype=np.float64)
        current_rot = Rotation.from_quat(pose[3:7])
        delta_pos = (target_pos - pose[:3]) / float(scale[0])
        delta_rotvec = (target_rot * current_rot.inv()).as_rotvec()
        max_rot = float(scale[1])
        if max_rot > 1e-9:
            angle = float(np.linalg.norm(delta_rotvec))
            if clip_motion and angle > max_rot:
                delta_rotvec = delta_rotvec * (max_rot / angle)
            delta_rot = delta_rotvec / max_rot
        else:
            delta_rot = np.zeros(3, dtype=np.float64)

        action = np.concatenate((delta_pos, delta_rot))
        if clip_motion:
            action = np.clip(action, -1.0, 1.0)
        if gripper_enabled:
            grip = 0.0
            if reading.get("grip_close", False):
                grip = -1.0
            elif reading.get("grip_open", False):
                grip = 1.0
            action = np.concatenate((action, np.array([grip], dtype=np.float64)))
        return action.astype(np.float32), True, info

    def _describe(self, reading: Mapping[str, Any]) -> dict[str, Any]:
        """Report controller state under the collector's field names."""
        info: dict[str, Any] = {
            "pico_active": bool(reading.get("held", False)),
            "pico_hand": self.hand,
        }
        if reading.get("stale"):
            return {**info, "pico_ready": False, "pico_stale": True}

        info["pico_ready"] = bool(reading.get("ready", False))
        info["pico_calibrated"] = bool(reading.get("calibrated", False))
        info["pico_control_value"] = reading.get("control_value", 0.0)
        if reading.get("invalid_pose"):
            info["pico_invalid_pose"] = True
        if reading.get("held", False):
            close = bool(reading.get("grip_close", False))
            opened = bool(reading.get("grip_open", False))
            info["pico_gripper_close_pressed"] = close
            info["pico_gripper_open_pressed"] = opened
            info["pico_gripper_action"] = -1.0 if close else (1.0 if opened else 0.0)
            info["pico_gripper_close"] = close
        return info


@TeleopDevice.register("yam_pico")
class YamPico(TeleopDevice):
    """A pair of PICO controllers driving both YAM arms in joint space.

    Every reading yields a complete fourteen-value target, so the environment
    never falls back to the policy. An arm the operator is not holding is held
    where it stands; the whole rig holds after a fault until both grips are
    released and re-engaged, which is why ``ready`` tracks the connection
    rather than the controller stream.

    Args:
        config: Controller, IK, and episode-control settings.
        joint_lower: Per-arm lower joint limits that narrow the model's own.
        joint_upper: Per-arm upper joint limits.
        joint_step_limits: Largest movement one control tick may command per
            joint. A target further away is interpolated toward, so the runtime
            accepts it instead of clipping the command away as a rejection.
        experts: Controller bindings to use instead of opening real ones.
        kinematics: IK adapters to use instead of loading the YAM model.
    """

    PRODUCES = {
        "left.arm": ActionKind.JOINT_POSITION,
        "left.end_effector": ActionKind.GRIPPER,
        "right.arm": ActionKind.JOINT_POSITION,
        "right.end_effector": ActionKind.GRIPPER,
    }

    NEEDS = ("joint_positions",)

    #: An arm the operator let go is still commanded, to the pose it holds.
    APPLIES_WHILE_IDLE = True

    #: The controllers report an explicit held state, so no hold window is needed.
    HOLD_WINDOW = 0.0

    def __init__(
        self,
        config: Mapping[str, Any] | "YamPicoConfig" | None = None,
        *,
        joint_step_limits: Sequence[float],
        joint_lower: Optional[Sequence[Sequence[float]]] = None,
        joint_upper: Optional[Sequence[Sequence[float]]] = None,
        experts: Optional[Mapping[str, Any]] = None,
        kinematics: Optional[Mapping[str, Any]] = None,
    ) -> None:
        from rlinf.envs.real.yam.config import YamPicoConfig

        self._config = (
            config
            if isinstance(config, YamPicoConfig)
            else YamPicoConfig(**dict(config or {}))
        )
        self._joint_lower = joint_lower
        self._joint_upper = joint_upper
        self._joint_step_limits = np.asarray(joint_step_limits, dtype=np.float64)
        self._experts = dict(experts or {})
        self._kinematics = dict(kinematics or {})
        for collection in (self._experts, self._kinematics):
            if collection and set(collection) != set(SIDES):
                raise ValueError("YAM PICO requires both left and right instances")
        self._fault: Optional[str] = "release_grips"
        self._preserve_reference = False
        self._grippers: Optional[np.ndarray] = None
        self._idle_joint_targets: dict[str, np.ndarray] = {}
        self._last_ik_warning = -float("inf")
        self._last_step_log = dict.fromkeys(SIDES, -float("inf"))
        self._logger = get_logger()

    # Hardware.

    def _open(self) -> None:
        """Load both IK models and open both controller subscriptions."""
        try:
            self._build_kinematics()
            self._build_experts()
        except BaseException:
            # Half a rig is worse than none: the connections already opened
            # would otherwise outlive the failure that stopped the rest.
            self.stop()
            raise

    def _build_kinematics(self) -> None:
        if self._kinematics:
            return
        from rlinf.envs.real.yam.kinematics import YamKinematicsAdapter

        self._kinematics = {
            side: YamKinematicsAdapter(
                config=self._config.ik,
                joint_lower=None if self._joint_lower is None else self._joint_lower[i],
                joint_upper=None if self._joint_upper is None else self._joint_upper[i],
            )
            for i, side in enumerate(SIDES)
        }

    def _build_experts(self) -> None:
        if self._experts:
            return
        self._experts = {
            side: _YamPicoArm(**self._config.expert_kwargs(side)) for side in SIDES
        }

    def stop(self) -> None:
        """Close both controller subscriptions."""
        for arm in self._experts.values():
            arm.stop()
        self._experts = {}

    @property
    def ready(self) -> bool:
        """Whether both controllers are open and can report an idle state.

        Tied to the connection, not to the stream: a controller that went
        silent must still be driven, so its arms hold rather than hand the
        robot back to the policy.
        """
        return self.is_connected

    # What the operator is doing, and what the robot should do about it.

    @property
    def observation_features(self) -> Features:
        """What each controller reports, per arm."""
        return {
            side: {
                "held": {"dtype": "bool", "shape": ()},
                "position_delta": {"dtype": "float64", "shape": (3,)},
                "rotation_delta": {"dtype": "float64", "shape": (3,)},
                "grip_close": {"dtype": "bool", "shape": ()},
                "grip_open": {"dtype": "bool", "shape": ()},
            }
            for side in SIDES
        }

    def get_observation(self) -> Observation:
        """Poll both controllers."""
        return {side: arm.read() for side, arm in self._experts.items()}

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, Any],
        options: Mapping[str, Any],
        facts: Any,
    ) -> "TeleopEntry":
        """Build the paired device, refusing combinations its runtime cannot honor.

        The guards live here rather than in a task factory because the device
        is built, and refuses, before any hardware is opened.
        """
        from rlinf.envs.real.yam.config import YamPicoConfig

        from .group import TeleopEntry

        if options.get("drives") is not None:
            raise ValueError(
                "yam_pico already names both arms' action parts; remove 'drives'."
            )
        override = cfg.get("override_cfg", {}) or {}
        if not bool(override.get("enforce_runtime_joint_limits", True)):
            raise ValueError("YAM VR requires enforce_runtime_joint_limits=true")

        settings = dict(cfg.get("pico", {}))
        settings.update({k: v for k, v in options.items() if k != "drives"})
        return TeleopEntry(
            cls(
                YamPicoConfig(**settings),
                joint_step_limits=facts.joint_step_limits,
                joint_lower=facts.joint_limit_min,
                joint_upper=facts.joint_limit_max,
            )
        )

    def action(
        self, reading: Mapping[str, Any], context: Mapping[str, Any]
    ) -> TeleopAction:
        """Return an absolute joint target for both arms."""
        measured = np.asarray(context["joint_positions"], dtype=np.float64).reshape(-1)
        if measured.shape != (len(SIDES) * 7,):
            raise ValueError(
                "yam_pico needs the env's 14-value joint_positions vector, got "
                f"{measured.shape}."
            )
        target, info = self._joint_targets(measured, reading)
        return TeleopAction(
            parts={
                "left.arm": target[:6].astype(np.float32),
                "left.end_effector": target[6:7].astype(np.float32),
                "right.arm": target[7:13].astype(np.float32),
                "right.end_effector": target[13:14].astype(np.float32),
            },
            driving=True,
            info=info,
        )

    def on_reset(self, context: Mapping[str, Any] = MappingProxyType({})) -> None:
        """Re-anchor the grips after an episode reset.

        A manual episode boundary that kept the arms where they were asks for
        the anchors to survive, so the operator can continue from the same
        pose without re-gripping.
        """
        if self._preserve_reference:
            self._preserve_reference = False
            return
        self.hold_until_released("release_grips")

    # Episode control hands its own transitions back through these.

    def hold_until_released(self, reason: str) -> None:
        """Hold both arms until the operator releases and re-engages both grips."""
        if self._fault != reason:
            self._logger.warning("YAM VR holding both arms: %s", reason)
        self._fault = reason
        self._preserve_reference = False
        self._grippers = None
        self._idle_joint_targets.clear()
        for arm in self._experts.values():
            arm.reset_reference()

    def preserve_reference(self) -> None:
        """Keep each grip anchor across the next reset."""
        self._preserve_reference = True

    @property
    def fault(self) -> Optional[str]:
        """Why both arms are currently held, or ``None`` while driving."""
        return self._fault

    # Mapping one reading onto joint targets.

    def _pressed_buttons(self) -> tuple[bool, bool]:
        """Return whether either controller reports each episode button."""
        pressed = {side: arm.read_buttons() for side, arm in self._experts.items()}
        return tuple(  # type: ignore[return-value]
            any(bool(pressed[side].get(button, False)) for side in SIDES)
            for button in (self._config.record_button, self._config.discard_button)
        )

    def _joint_targets(
        self, measured: np.ndarray, readings: Mapping[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Solve both arms toward their controller targets.

        Returns the fourteen-value target and per-arm diagnostics. The target
        is the measured vector whenever anything is held, so a fault never
        leaves a half-solved pose behind.
        """
        started = time.monotonic()
        record, discard = self._pressed_buttons()
        infos: dict[str, Any] = {"pico_record": record, "pico_discard": discard}
        if self._grippers is None:
            self._grippers = measured[[6, 13]].copy()
        target = measured.copy()
        scale = np.array(
            [self._config.max_position_delta, self._config.max_rotation_delta, 1.0]
        )
        results = {}
        poses = {}
        reason = None
        for i, side in enumerate(SIDES):
            arm = measured[i * 7 : i * 7 + 7]
            infos[f"{side}_joint_step_fraction"] = 1.0
            infos[f"{side}_limiting_joint"] = 0
            infos[f"{side}_ik_max_joint_delta"] = 0.0
            infos[f"{side}_ik_attempts"] = 0
            infos[f"{side}_ik_target_fraction"] = 0.0
            infos[f"{side}_ik_initial_fault"] = None
            pose = self._kinematics[side].fk(arm[:6], arm[6])
            poses[side] = pose
            tcp = np.concatenate(
                [pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()]
            )
            action, replaced, info = self._experts[side].command(
                readings[side], tcp, scale, gripper_enabled=True, clip_motion=False
            )
            results[side] = (action, replaced, info)
            infos.update({f"{side}_{key}": value for key, value in info.items()})
            if (
                not info.get("pico_ready", False)
                or info.get("pico_invalid_pose", False)
                or not info.get("pico_calibrated", False)
            ):
                reason = f"{side}:unavailable"
        if reason:
            self.hold_until_released(reason)
        elif self._fault is not None:
            # Never re-engage a still-held grip after data loss or reset.
            released = all(
                info.get("pico_control_value", float("inf"))
                < self._config.control_threshold
                for _, _, info in results.values()
            )
            for arm in self._experts.values():
                arm.reset_reference()
            if released:
                self._fault = None
                self._idle_joint_targets = {
                    side: measured[i * 7 : i * 7 + 6].copy()
                    for i, side in enumerate(SIDES)
                }
            infos["yam_pico_fault"] = self._fault
            return measured, infos

        if self._fault is None:
            for i, side in enumerate(SIDES):
                action, replaced, _ = results[side]
                if not replaced:
                    # Latch the release pose once so feedback drift does not
                    # continuously move the position controller's target.
                    if side not in self._idle_joint_targets:
                        self._idle_joint_targets[side] = measured[
                            i * 7 : i * 7 + 6
                        ].copy()
                    target[i * 7 : i * 7 + 6] = self._idle_joint_targets[side]
                    self._grippers[i] = measured[i * 7 + 6]
                    continue
                self._idle_joint_targets.pop(side, None)
                action = np.asarray(action, dtype=np.float64)
                if action.shape != (7,) or not np.all(np.isfinite(action)):
                    self.hold_until_released(f"{side}:invalid_action")
                    break
                current = poses[side]
                tcp = np.r_[
                    current[:3, 3], Rotation.from_matrix(current[:3, :3]).as_quat()
                ]
                target_tcp = _delta_to_tcp_pose(action, tcp, scale, clip_motion=False)
                pose = np.eye(4)
                pose[:3, 3] = target_tcp[:3]
                pose[:3, :3] = Rotation.from_quat(target_tcp[3:]).as_matrix()
                arm = measured[i * 7 : i * 7 + 7]
                result, attempts, fraction, first_reason = (
                    self._solve_with_backtracking(side, pose, current, arm)
                )
                infos[f"{side}_ik_attempts"] = attempts
                infos[f"{side}_ik_target_fraction"] = fraction
                infos[f"{side}_ik_initial_fault"] = first_reason
                infos[f"{side}_ik_elapsed_s"] = result.elapsed_s
                infos[f"{side}_ik_position_error"] = result.position_error
                infos[f"{side}_ik_rotation_error"] = result.rotation_error
                infos[f"{side}_ik_fault"] = (
                    result.reason if not result.success else None
                )
                if not result.success:
                    # Hold only this arm, retaining references for the next
                    # frame. The other arm can still execute a valid solution.
                    missed = f"{side}:ik:{result.reason}"
                    self._grippers[i] = measured[i * 7 + 6]
                    now = time.monotonic()
                    if now - self._last_ik_warning >= 1.0:
                        self._logger.warning(
                            "YAM VR holding %s arm this frame, retrying: %s",
                            side,
                            missed,
                        )
                        self._last_ik_warning = now
                    continue
                # Interpolate toward the full or successfully reduced IK target.
                # A tiny TCP target ahead of lagging feedback can stall motion.
                difference = result.q_target - arm[:6]
                limits = self._joint_step_limits
                demand = np.abs(difference) / limits
                limiting_joint = int(np.argmax(demand))
                fraction = min(1.0, 1.0 / max(float(demand[limiting_joint]), 1e-12))
                target[i * 7 : i * 7 + 6] = arm[:6] + fraction * difference
                infos[f"{side}_joint_step_fraction"] = fraction
                infos[f"{side}_limiting_joint"] = (
                    limiting_joint + 1 if fraction < 1.0 else 0
                )
                infos[f"{side}_ik_max_joint_delta"] = float(np.max(np.abs(difference)))
                now = time.monotonic()
                if fraction < 1.0 and now - self._last_step_log[side] >= 1.0:
                    self._logger.info(
                        "YAM VR %s joint step limited: fraction=%.3f, joint=J%d, "
                        "required=%.3f rad, limit=%.3f rad",
                        side,
                        fraction,
                        limiting_joint + 1,
                        abs(difference[limiting_joint]),
                        limits[limiting_joint],
                    )
                    self._last_step_log[side] = now
                if action[6] < 0:
                    self._grippers[i] = 0.0
                elif action[6] > 0:
                    self._grippers[i] = 1.0
                target[i * 7 + 6] = self._grippers[i]
        # Reject solutions computed from a now-stale stream or a slow solve.
        if self._fault is None and (
            time.monotonic() - started > self._config.max_tick_s
            or not all(arm.ready for arm in self._experts.values())
        ):
            self.hold_until_released("stale_after_ik")
        infos["yam_pico_fault"] = self._fault
        return (measured if self._fault else target), infos

    def _solve_with_backtracking(
        self, side: str, target: np.ndarray, current: np.ndarray, arm: np.ndarray
    ) -> tuple[Any, int, float, Optional[str]]:
        """Retry geometric IK failures toward the same measured TCP reference.

        Attempts share max_solve_s; timing is checked between synchronous SDK
        calls and on return, rather than preempting a call. Each retry halves
        both translation and the shortest spatial rotation from current.
        """
        from dataclasses import replace

        started = time.monotonic()
        result = self._kinematics[side].solve(target, arm[:6].copy(), arm[6])
        first_reason = result.reason
        attempts, fraction = 1, 1.0
        delta_rotation = None
        while True:
            elapsed = time.monotonic() - started
            if elapsed > self._config.ik.max_solve_s:
                result = replace(result, success=False, reason="solve_timeout")
                break
            if (
                result.success
                or result.reason not in {"not_converged", "residual", "joint_limits"}
                or attempts > self._config.ik_backtrack_attempts
            ):
                break
            if delta_rotation is None:
                delta_rotation = Rotation.from_matrix(
                    target[:3, :3] @ current[:3, :3].T
                ).as_rotvec()
            fraction *= 0.5
            intermediate = current.copy()
            intermediate[:3, 3] += fraction * (target[:3, 3] - current[:3, 3])
            intermediate[:3, :3] = (
                Rotation.from_rotvec(fraction * delta_rotation).as_matrix()
                @ current[:3, :3]
            )
            result = self._kinematics[side].solve(intermediate, arm[:6].copy(), arm[6])
            attempts += 1
        return replace(result, elapsed_s=elapsed), attempts, fraction, first_reason
