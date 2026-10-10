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

"""Single-writer control runtime for a dual-arm YAM rig."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from rlinf.utils.logging import get_logger

from .config import DualYamJointEnvConfig
from .types import (
    NUM_ARM_JOINTS,
    DualYamState,
    YamBackendFactory,
    YamCommandResult,
    YamFollowerBackend,
    pack_dual_action,
    split_dual_action,
)


class YamControlRuntime:
    """Own both follower transports and serialize every follower command."""

    def __init__(
        self,
        config: DualYamJointEnvConfig,
        hardware_config: Any,
        backend_factory: YamBackendFactory,
        *,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.hardware_config = hardware_config
        self._backend_factory = backend_factory
        self._clock = clock
        self._sleep = sleeper
        self._logger = get_logger()
        self._followers: tuple[YamFollowerBackend, YamFollowerBackend] | None = None
        self._pending_cleanup: list[Any] = []
        self._close_requested = False
        self._closed = False

    @property
    def followers_connected(self) -> bool:
        """Whether both follower transports are currently owned."""
        return self._followers is not None

    def connect_followers(self) -> None:
        """Create and connect the two follower backends atomically."""
        if self._close_requested:
            raise RuntimeError("cannot reconnect a closed YAM runtime")
        if self._pending_cleanup:
            raise RuntimeError("cannot connect YAM followers while cleanup is pending")
        if self._followers is not None:
            return

        opened: list[YamFollowerBackend] = []
        try:
            for device in (
                self.hardware_config.left_follower,
                self.hardware_config.right_follower,
            ):
                backend = self._backend_factory.create_follower(device)
                opened.append(backend)
                backend.connect()
                # The real backend already starts followers in measured-pose
                # PD hold. Repeat through the backend contract immediately so
                # every implementation is safe before the next arm connects.
                backend.hold()
        except Exception:
            if self._close_backends(opened):
                self._pending_cleanup.extend(opened)
            raise
        self._followers = (opened[0], opened[1])
        # Fail startup when feedback is absent or stale instead of accepting a
        # first command without knowing the current pose.
        try:
            self._validate_follower_joint_limits()
            self.read_state()
        except Exception:
            if self._close_backends(opened):
                self._pending_cleanup.extend(opened)
            self._followers = None
            raise

    def read_state(self) -> DualYamState:
        """Read a fresh, ordered left/right follower state."""
        followers = self._require_followers()
        states = []
        for backend in followers:
            backend.assert_healthy(self.config.feedback_timeout_s)
            state = backend.read_state()
            self._assert_fresh(state.timestamp_s, "follower")
            states.append(state)
        return DualYamState(left=states[0], right=states[1])

    def command(self, action: Any) -> YamCommandResult:
        """Validate and synchronously dispatch one dual-arm target.

        Shape errors are rejected before reading or writing hardware. Non-finite
        targets trigger an explicit measured-pose hold and are reported without
        forwarding the invalid values. Runtime joint limits are optional so YAM
        teleoperation can use the explicit per-step limits and leave final
        joint clipping to i2rt.
        """
        left_requested, right_requested = split_dual_action(action)
        requested = pack_dual_action(left_requested, right_requested)
        followers = self._require_followers()

        if not np.all(np.isfinite(requested)):
            held = self.hold()
            return YamCommandResult(
                requested=requested,
                accepted=held,
                rejection_reason="non_finite_action",
            )

        try:
            measured = self.read_state()
        except Exception:
            self._best_effort_hold()
            raise
        current = (measured.left.as_action(), measured.right.as_action())
        current_vector = measured.as_vector()
        if not np.all(np.isfinite(current_vector)):
            self._best_effort_hold()
            raise RuntimeError("YAM follower returned non-finite measured state")
        requested_arms = (left_requested, right_requested)
        accepted_arms: list[np.ndarray] = []
        for arm_index, (target, current_arm) in enumerate(
            zip(requested_arms, current, strict=True)
        ):
            accepted = target.copy()
            if self.config.enforce_runtime_joint_limits:
                lower = self.config.joint_limit_min[arm_index]
                upper = self.config.joint_limit_max[arm_index]
                if np.any(current_arm[:NUM_ARM_JOINTS] < lower) or np.any(
                    current_arm[:NUM_ARM_JOINTS] > upper
                ):
                    self._best_effort_hold()
                    return YamCommandResult(
                        requested=requested,
                        accepted=current_vector,
                        rejection_reason="measured_joint_out_of_limits",
                    )
                accepted[:NUM_ARM_JOINTS] = np.clip(
                    accepted[:NUM_ARM_JOINTS], lower, upper
                )
                accepted[:NUM_ARM_JOINTS] = np.clip(
                    accepted[:NUM_ARM_JOINTS],
                    current_arm[:NUM_ARM_JOINTS] - self.config.joint_step_limits,
                    current_arm[:NUM_ARM_JOINTS] + self.config.joint_step_limits,
                )
            accepted[-1] = np.clip(accepted[-1], 0.0, 1.0)
            accepted_arms.append(accepted)

        accepted_vector = pack_dual_action(accepted_arms[0], accepted_arms[1])
        try:
            for backend, accepted in zip(followers, accepted_arms, strict=True):
                backend.command(accepted)
        except Exception:
            self._best_effort_hold()
            raise
        return YamCommandResult(
            requested=requested,
            accepted=accepted_vector,
            clipped=not np.array_equal(requested, accepted_vector),
        )

    def move_to(
        self,
        target: Any,
        *,
        duration_s: float,
        max_joint_delta: float,
        tolerance: float,
        timeout_s: float,
    ) -> np.ndarray:
        """Move followers smoothly to an absolute 14-D target and verify it.

        The interpolation bound is independent of normal policy/teleoperation
        limits, so reset remains gradual even when runtime slew limiting is
        disabled. The configured hard joint limits are always enforced here.
        """
        left_target, right_target = split_dual_action(target)
        target_vector = pack_dual_action(left_target, right_target)
        if not np.all(np.isfinite(target_vector)):
            raise ValueError("YAM move target must be finite")
        for arm_index, arm_target in enumerate((left_target, right_target)):
            lower = self.config.joint_limit_min[arm_index]
            upper = self.config.joint_limit_max[arm_index]
            if np.any(arm_target[:NUM_ARM_JOINTS] < lower) or np.any(
                arm_target[:NUM_ARM_JOINTS] > upper
            ):
                raise ValueError(
                    f"YAM move target for arm {arm_index} is outside joint limits"
                )
            if not 0.0 <= arm_target[-1] <= 1.0:
                raise ValueError("YAM move gripper targets must be within [0, 1]")

        duration_s = float(duration_s)
        max_joint_delta = float(max_joint_delta)
        tolerance = float(tolerance)
        timeout_s = float(timeout_s)
        if not all(
            np.isfinite(value)
            for value in (duration_s, max_joint_delta, tolerance, timeout_s)
        ):
            raise ValueError("YAM move timing and tolerance must be finite")
        if duration_s < 0 or max_joint_delta <= 0 or tolerance <= 0:
            raise ValueError("invalid YAM move timing or tolerance")
        if timeout_s < duration_s:
            raise ValueError("YAM move timeout must be at least its duration")

        start = self.read_state().as_vector()
        joint_indices = np.array([*range(6), *range(7, 13)])
        limits = np.full(12, max_joint_delta, dtype=np.float64)
        if self.config.enforce_runtime_joint_limits:
            limits = np.minimum(limits, np.tile(self.config.joint_step_limits, 2))
        demand = np.abs(target_vector[joint_indices] - start[joint_indices]) / limits
        duration_steps = int(np.ceil(duration_s * self.config.step_frequency))
        # smoothstep has a maximum slope of 1.5, so account for it when
        # bounding the difference between consecutive commanded setpoints.
        slew_steps = int(np.ceil(1.5 * np.max(demand)))
        steps = max(1, duration_steps, slew_steps)
        period_s = 1.0 / self.config.step_frequency
        deadline_s = time.monotonic() + timeout_s
        self._logger.info(
            "Moving YAM followers to reset qpos over %d steps (%.2fs nominal)",
            steps,
            steps * period_s,
        )

        try:
            for step in range(1, steps + 1):
                progress = step / steps
                alpha = progress * progress * (3.0 - 2.0 * progress)
                result = self.command(start + alpha * (target_vector - start))
                if result.rejection_reason is not None:
                    raise RuntimeError(
                        "YAM reset command was rejected: " + result.rejection_reason
                    )
                if step < steps:
                    remaining_s = deadline_s - time.monotonic()
                    if remaining_s <= 0:
                        raise TimeoutError("YAM reset timed out during interpolation")
                    self._sleep(min(period_s, remaining_s))

            while True:
                measured = self.read_state().as_vector()
                max_error = float(np.max(np.abs(target_vector - measured)))
                if max_error <= tolerance:
                    held = self.hold()
                    self._logger.info(
                        "YAM reset qpos reached (max error %.4f)", max_error
                    )
                    return held
                remaining_s = deadline_s - time.monotonic()
                if remaining_s <= 0:
                    raise TimeoutError(
                        "YAM reset did not converge within "
                        f"{timeout_s:.2f}s (max error {max_error:.4f}, "
                        f"tolerance {tolerance:.4f})"
                    )
                result = self.command(target_vector)
                if result.rejection_reason is not None:
                    raise RuntimeError(
                        "YAM reset command was rejected: " + result.rejection_reason
                    )
                self._sleep(min(period_s, remaining_s))
        except Exception:
            self._best_effort_hold()
            raise

    def hold(self) -> np.ndarray:
        """Hold measured follower positions and return the held 14-D action."""
        try:
            state = self.read_state()
        except Exception:
            self._best_effort_hold()
            raise
        self._best_effort_hold(raise_first=True)
        return state.as_vector()

    def emergency_hold(self) -> None:
        """Best-effort measured-pose hold for post-command failure paths."""
        self._best_effort_hold()

    def close(self) -> None:
        """Release followers, retaining handles on cleanup failure."""
        if self._closed:
            return
        self._close_requested = True
        errors: list[Exception] = []
        if self._followers is not None:
            close_errors = self._close_backends(reversed(self._followers))
            errors.extend(close_errors)
            if not close_errors:
                self._followers = None
        if self._pending_cleanup:
            close_errors = self._close_backends(reversed(self._pending_cleanup))
            errors.extend(close_errors)
            if not close_errors:
                self._pending_cleanup.clear()
        self._closed = self._followers is None and not self._pending_cleanup
        if errors:
            details = "; ".join(str(error) for error in errors)
            raise RuntimeError(
                f"failed to fully close YAM runtime: {details}"
            ) from errors[0]

    def _require_followers(
        self,
    ) -> tuple[YamFollowerBackend, YamFollowerBackend]:
        if self._followers is None:
            raise RuntimeError("YAM followers are not connected")
        return self._followers

    def _assert_fresh(self, timestamp_s: float, label: str) -> None:
        age_s = self._clock() - timestamp_s
        if not np.isfinite(timestamp_s) or age_s > self.config.feedback_timeout_s:
            raise RuntimeError(
                f"stale YAM {label} feedback: age={age_s:.3f}s, "
                f"limit={self.config.feedback_timeout_s:.3f}s"
            )

    def _validate_follower_joint_limits(self) -> None:
        followers = self._require_followers()
        tolerance = 1e-6
        for arm_index, backend in enumerate(followers):
            sdk_limits = np.asarray(backend.joint_limits(), dtype=np.float64)
            if sdk_limits.shape != (NUM_ARM_JOINTS, 2) or not np.all(
                np.isfinite(sdk_limits)
            ):
                raise RuntimeError(
                    f"YAM follower {arm_index} returned invalid joint limits"
                )
            configured_lower = self.config.joint_limit_min[arm_index]
            configured_upper = self.config.joint_limit_max[arm_index]
            if np.any(configured_lower < sdk_limits[:, 0] - tolerance) or np.any(
                configured_upper > sdk_limits[:, 1] + tolerance
            ):
                raise ValueError(
                    f"configured YAM arm {arm_index} joint limits must be within "
                    f"the i2rt limits {sdk_limits.tolist()}"
                )

    def _best_effort_hold(self, *, raise_first: bool = False) -> None:
        first_error: Exception | None = None
        for backend in self._require_followers():
            try:
                backend.hold()
            except Exception as error:  # pragma: no cover - hardware failure path
                if first_error is None:
                    first_error = error
                self._logger.exception("Failed to hold a YAM follower")
        if raise_first and first_error is not None:
            raise first_error

    def _close_backends(self, backends: Any) -> list[Exception]:
        errors: list[Exception] = []
        for backend in backends:
            try:
                backend.close()
            except Exception as error:  # pragma: no cover - hardware cleanup path
                errors.append(error)
                self._logger.exception("Failed to close a YAM backend")
        return errors
