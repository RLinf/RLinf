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

"""Arm interfaces and the canonical observation schema."""

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, ClassVar, Optional, Protocol

import numpy as np

from rlinf.robotics.parts.base import ControllablePart, Features, Observation
from rlinf.utils.logging import get_logger

#: Canonical arm fields; mounted devices expose their own observations.
ARM_STATE_FIELDS: tuple[str, ...] = (
    "tcp_pose",
    "tcp_vel",
    "arm_joint_position",
    "arm_joint_velocity",
    "tcp_force",
    "tcp_torque",
    "arm_jacobian",
)


@dataclass
class CartesianCompliance:
    """Cartesian impedance settings for a backend that runs the control loop."""

    translational_stiffness: float = 500.0  # N/m
    rotational_stiffness: float = 40.0  # Nm/rad
    nullspace_stiffness: float = 5.0  # Nm/rad, holds the elbow
    translational_clip: float = 0.05  # m, largest error acted on
    rotational_clip: float = 0.3  # rad, largest error acted on
    max_step: float = 0.10  # m per call, 0 disables
    max_step_rad: float = 0.30  # rad per call, 0 disables
    max_delta_tau: float = 0.3  # Nm per control cycle
    gains_time_constant: float = 0.1  # s to blend a gain change
    stiffness_cap: float = 1200.0  # N/m, most a task may ask for
    rotational_stiffness_cap: float = 80.0  # Nm/rad, most a task may ask for
    clip_floor: float = 0.005  # m, least a task may ask for
    rotational_clip_floor: float = 0.02  # rad, least a task may ask for

    @classmethod
    def from_config(
        cls, value: "CartesianCompliance | Mapping[str, float] | None"
    ) -> "CartesianCompliance":
        """Build settings from a YAML mapping, rejecting unknown keys."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        given = dict(value)
        unknown = set(given) - {f.name for f in fields(cls)}
        if unknown:
            raise KeyError(
                f"Unknown compliance settings {sorted(unknown)}. "
                f"Known: {sorted(f.name for f in fields(cls))}."
            )
        return cls(**{key: float(val) for key, val in given.items()})


class ArmState(Protocol):
    """What an arm's own state object has to offer.

    A driver returns whatever dataclass it already keeps -- ``FrankaRobotState``,
    ``GimArmRobotState`` -- and :meth:`BaseArm.get_observation` selects the
    canonical fields out of it. Naming the one method it uses says that without
    naming any one driver's class, and gives ``get_state()`` somewhere to jump
    to instead of ``Any``.
    """

    def to_dict(self) -> dict[str, Any]:
        """This state as a mapping, keyed by the canonical field names."""
        ...


class Arm(ControllablePart):
    """Base category for registered arm backends.

    Drivers register configuration names on this class::

        @Arm.register("franky")
        class FrankyArm(BaseArm): ...
    """

    @classmethod
    def backends(cls) -> dict[str, type]:
        """Return all registered arm backends by configuration name."""
        if cls is Arm:
            from rlinf.robotics.parts.arms import load_drivers

            load_drivers()
        return super().backends()

    @classmethod
    def declare(
        cls,
        address: str,
        *,
        gripper_type: Optional[str] = None,
        gripper_connection: Optional[str] = None,
        end_effector_type: Optional[str] = None,
        end_effector_config: Optional[dict] = None,
        compliance: "Optional[CartesianCompliance]" = None,
        **placement: Any,
    ) -> "Arm":
        """Declare an unconnected arm from standard robot settings.

        Args:
            address: Arm endpoint, such as an IP address, bus, or device path.
            gripper_type: Gripper backend, when the arm builds its own.
            gripper_connection: Where that gripper is attached.
            end_effector_type: End effector fitted, when the arm builds it.
            end_effector_config: Settings for that end effector.
            compliance: Cartesian impedance settings, ignored by backends
                whose controller owns its gains.
            **placement: Placement arguments forwarded to the connection.
        """
        offered = {
            "gripper_type": gripper_type,
            "gripper_connection": gripper_connection,
            "end_effector_type": end_effector_type,
            "end_effector_config": end_effector_config,
        }
        cls.refuse_unused(**offered)
        return cls(address, **placement)

    @classmethod
    def refuse_unused(cls, **offered: Any) -> None:
        """Reject offered settings this backend would otherwise drop."""
        unused = sorted(name for name, value in offered.items() if value is not None)
        if unused:
            raise TypeError(
                f"{cls.__name__} was given {unused}, which it does not take. "
                "Silently dropping them would leave a robot configured one way "
                f"and running another. Override {cls.__name__}.declare() to "
                "map them onto its constructor, or drop them from the config."
            )

    # Operations every arm is asked for, beyond reading and commanding it.

    def is_robot_up(self) -> bool:
        """Whether the arm is ready to be read and commanded."""
        return self.is_connected

    def clear_errors(self) -> None:
        """Clear a latched fault so the arm accepts commands again."""

    def reset_joint(self, positions: "Sequence[float]") -> None:
        """Move the joints to a configuration, outside the action stream.

        Args:
            positions: Target joint positions, one per joint.

        Raises:
            NotImplementedError: If this backend cannot reset its joints.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot reset its joints. Drive it to the "
            "configuration you want through send_action, or use a backend that "
            "implements reset_joint()."
        )

    def reconfigure_compliance_params(self, params: "Mapping[str, float]") -> None:
        """Apply a task's compliance request, as far as the backend can."""
        if params:
            get_logger().warning(
                "%s cannot change compliance while running; %s were not "
                "applied. Set them where this backend configures its "
                "controller instead.",
                type(self).__name__,
                sorted(params),
            )


class BaseArm(Arm, ABC):
    """Arm base class that exposes fields from a state object."""

    #: The fields this arm reports. A driver may narrow it.
    STATE_FIELDS: ClassVar[tuple[str, ...]] = ARM_STATE_FIELDS

    #: Joint movement between two polls below which the arm counts as stopped.
    SETTLE_TOLERANCE: ClassVar[float] = np.deg2rad(0.5)

    #: Seconds between polls while waiting for the arm to stop.
    SETTLE_POLL_INTERVAL: ClassVar[float] = 0.05

    @abstractmethod
    def _open(self) -> Any:
        """Open the arm connection and return its device handle."""

    @abstractmethod
    def _release(self, device: Any) -> None:
        """Release the handle returned by :meth:`_open`."""

    @abstractmethod
    def get_state(self) -> ArmState:
        """The arm's whole state, as something with ``to_dict()``."""

    @property
    def observation_features(self) -> Features:
        """Describe the canonical arm observation fields."""
        return {name: {} for name in self.STATE_FIELDS}

    def get_observation(self) -> Observation:
        """Select the canonical fields out of this arm's state."""
        state = self.read_state().to_dict()
        return {name: state[name] for name in self.STATE_FIELDS}

    def wait_until_still(self, duration: float = 3.0) -> None:
        """Block until the joints stop moving, or ``duration`` expires.

        A backend whose commands return while the arm is still travelling
        calls this from :meth:`reset_joint`, so that a state read taken
        afterwards reports where the arm came to rest rather than the pose it
        was leaving. Settling is the test rather than reaching the target,
        because an arm holds a configuration a little short of it under load.
        """
        deadline = time.monotonic() + duration
        previous = self._joint_reading()
        while time.monotonic() < deadline:
            time.sleep(self.SETTLE_POLL_INTERVAL)
            current = self._joint_reading()
            if np.all(np.abs(current - previous) < self.SETTLE_TOLERANCE):
                return
            previous = current
        get_logger().warning(
            "%s had not stopped moving after %.1fs.", type(self).__name__, duration
        )

    def _joint_reading(self) -> np.ndarray:
        """The joint positions the settle test compares, in radians.

        Read through :meth:`get_state` rather than :meth:`read_state`, whose
        reading is frozen for the duration of a snapshot; a poll needs a
        fresh one every time. Override to read fewer fields than the whole
        state object.
        """
        return np.asarray(self.get_state().arm_joint_position, dtype=float)
