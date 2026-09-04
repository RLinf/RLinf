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

"""Abstract base class for robot end-effectors."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar

import numpy as np

from rlinf.robotics.parts.base import Action, ControllablePart, Features, Observation


class EndEffector(ControllablePart, ABC):
    """Common observation and action interface for end effectors.

    Implementations may own a connection or borrow the connection of the arm
    that carries them.
    """

    @classmethod
    def of(
        cls, end_effector_type: "str | EndEffectorType", **settings: Any
    ) -> "EndEffector":
        """Declare an end effector from a registered backend name.

        Args:
            end_effector_type: A registered name, or an
                :class:`EndEffectorType` carrying one.
            **settings: Offered to that driver's :meth:`declare`.
        """
        name = getattr(end_effector_type, "value", end_effector_type)
        return cls.backend(name).declare(**settings)

    #: Ways an end effector can be reached, offered to every backend.
    #: A backend takes the one it uses by naming it in :meth:`declare`.
    ATTACHMENTS: ClassVar[tuple[str, ...]] = ("ros", "port", "robot_ip")

    @classmethod
    def declare(cls, **settings: Any) -> "EndEffector":
        """Declare this backend from the attachment settings it is offered.

        A robot does not know how a given end effector is wired, so it offers
        every attachment it can supply. This default drops all of them, which
        suits a device reached through none of them. A backend that needs one
        overrides this method and names it.
        """
        return cls(
            **{
                name: value
                for name, value in settings.items()
                if name not in cls.ATTACHMENTS
            }
        )

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimensionality of the end-effector action vector."""

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimensionality of the end-effector state vector."""

    @property
    @abstractmethod
    def control_mode(self) -> str:
        """Control mode: ``"binary"`` (open/close) or ``"continuous"``."""

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Return the current end-effector state as a 1-D array.

        The length of the returned array must equal :pyattr:`state_dim`.
        """

    @abstractmethod
    def command(self, action: np.ndarray) -> bool:
        """Send a command to the end-effector.

        Args:
            action: Action vector whose length equals :pyattr:`action_dim`.

        Returns:
            ``True`` if the command caused a meaningful state change
            (e.g. gripper opened/closed), ``False`` otherwise.
        """

    @property
    def is_gripper(self) -> bool:
        """Whether this end effector opens and closes on one axis.

        An environment asks the part rather than the configuration it was
        built from, so the two cannot drift apart.
        """
        return False

    @property
    def is_hand(self) -> bool:
        """Whether this end effector poses several fingers."""
        return not self.is_gripper

    def open(self, speed: float = 0.3) -> None:
        """Release fully.

        Latching is a separate verb from :meth:`command`, which positions
        without force. An end effector that does not latch has nothing to do
        here.
        """

    def close(self, speed: float = 0.3, force: float = 130.0) -> None:
        """Grasp closed at ``force``, in Newtons.

        This is not ``command`` with a zero target: a grasp holds an object
        against the requested force, where a move only travels to a width.
        """

    @property
    def is_open(self) -> bool:
        """Whether the end effector is holding nothing.

        A gripper reports its own latch state. Anything that does not open and
        close on one axis has no closed state to report, so it reads as open.
        """
        return True

    @property
    def observation_features(self) -> Features:
        """Describe the canonical end-effector state."""
        return {"state": {"shape": (self.state_dim,), "dtype": "float32"}}

    @property
    def action_features(self) -> Features:
        """Describe the canonical end-effector command."""
        return {"target": {"shape": (self.action_dim,), "dtype": "float32"}}

    def get_observation(self) -> Observation:
        """Return the end-effector state under its canonical key."""
        return {"state": self.get_state()}

    def send_action(self, action: Action) -> Observation:
        """Apply the canonical target command."""
        if set(action) != {"target"}:
            raise KeyError("End-effector action must contain only 'target'.")
        self.command(action["target"])
        return {"target": action["target"]}


class EndEffectorType(str, Enum):
    """Supported end-effector types for the Franka robot arm."""

    FRANKA_GRIPPER = "franka_gripper"
    FRANKY_GRIPPER = "franky_gripper"
    ROBOTIQ_GRIPPER = "robotiq_gripper"
    RUIYAN_HAND = "ruiyan_hand"

    @property
    def is_gripper(self) -> bool:
        return self in (
            type(self).FRANKA_GRIPPER,
            type(self).FRANKY_GRIPPER,
            type(self).ROBOTIQ_GRIPPER,
        )

    @property
    def is_hand(self) -> bool:
        return self == type(self).RUIYAN_HAND


def normalize_end_effector_type(
    end_effector_type: str | EndEffectorType,
    gripper_type: str | None = None,
) -> EndEffectorType:
    if isinstance(end_effector_type, str):
        end_effector_type = EndEffectorType(end_effector_type)

    if end_effector_type.is_hand or gripper_type is None:
        return end_effector_type
    # A driver named outright is kept: only the generic 'franka_gripper'
    # default is still open to being narrowed by gripper_type.
    if end_effector_type in (
        EndEffectorType.ROBOTIQ_GRIPPER,
        EndEffectorType.FRANKY_GRIPPER,
    ):
        return end_effector_type

    gt = gripper_type.lower()
    if gt == "franka":
        return EndEffectorType.FRANKA_GRIPPER
    if gt == "robotiq":
        return EndEffectorType.ROBOTIQ_GRIPPER
    raise ValueError(
        f"Unsupported gripper_type={gripper_type!r}. "
        "Supported types: 'franka', 'robotiq'."
    )


class BaseEndEffector(EndEffector, ABC):
    """Base class for end effectors with an independent connection."""

    @abstractmethod
    def _open(self) -> Any:
        """Open the end effector and return its device handle."""

    @abstractmethod
    def _release(self, device: Any) -> None:
        """Release the handle returned by :meth:`_open`."""

    @property
    def finger_names(self) -> list[str]:
        """Human-readable names for each DOF.

        Subclasses may override this to provide meaningful labels.
        The default returns generic names ``["dof_0", "dof_1", ...]``.
        """
        return [f"dof_{i}" for i in range(self.state_dim)]

    def get_detailed_state(self) -> dict[str, Any]:
        """Return diagnostic state using the generic position representation."""
        state = self.get_state()
        return {
            "positions": state.tolist(),
            "finger_names": self.finger_names,
        }

    @abstractmethod
    def reset(self, target_state: np.ndarray | None = None) -> None:
        """Reset the end-effector to a default or specified state.

        Args:
            target_state: Optional target state. If ``None``, reset to the
                implementation-defined default.
        """
