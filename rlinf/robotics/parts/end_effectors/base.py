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
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np

from rlinf.robotics.parts.base import Action, ControllablePart, Features, Observation


class EndEffector(ControllablePart, ABC):
    """Common observation and action interface for end effectors.

    Implementations may own a connection or borrow the connection of the arm
    that carries them.
    """

    @classmethod
    def register(
        cls, *names: str, arm_backend: str | None = None
    ) -> Callable[[type["EndEffector"]], type["EndEffector"]]:
        """Register driver names, optionally scoped to an arm backend.

        Scoped aliases select a transport for a device that can be reached
        through several arm backends. Unscoped names remain available with
        any arm. Each driver registers its own aliases.
        """
        if arm_backend is None:
            return super().register(*names)

        def add(driver: type["EndEffector"]) -> type["EndEffector"]:
            for name in names:
                key = (arm_backend.lower(), name.lower())
                taken = cls._ARM_BACKENDS.get(key)
                if taken is not None and taken is not driver:
                    raise ValueError(
                        f"End-effector alias {key!r} is already registered."
                    )
                cls._ARM_BACKENDS[key] = driver
            return driver

        return add

    _ARM_BACKENDS: ClassVar[dict[tuple[str, str], type["EndEffector"]]] = {}

    @classmethod
    def backend(
        cls, name: str, *, arm_backend: str | None = None
    ) -> type["EndEffector"]:
        """Resolve a driver, preferring an alias for the supplied arm backend."""
        if arm_backend is not None:
            driver = cls._ARM_BACKENDS.get((arm_backend.lower(), name.lower()))
            if driver is not None:
                return driver
        return super().backend(name)

    @classmethod
    def of(cls, end_effector_type: str, **settings: Any) -> "EndEffector":
        """Declare an end effector from a registered backend name.

        Args:
            end_effector_type: A registered driver name.
            **settings: Offered to that driver's :meth:`declare`.
        """
        return cls.backend(end_effector_type).declare(**settings)

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

    #: Vector sizes and control mode, available before a connection is opened.
    #: Registered drivers declare these on the class so dummy tasks can inspect
    #: the same contract without constructing a device. Shared views may expose
    #: instance properties when their host determines the sizes.
    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Width of the command vector; fixed drivers override with an integer."""

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Width of the state vector; fixed drivers override with an integer."""

    @property
    @abstractmethod
    def control_mode(self) -> str:
        """Control mode: ``binary`` or ``continuous``."""

    #: Whether the part opens and closes on one axis.
    is_gripper: bool = False

    #: Whether the part controls a hand with articulated fingers.
    is_hand: bool = False

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

    def get_detailed_state(self) -> dict[str, Any]:
        """Return diagnostic positions; drivers may add device-specific fields."""
        return {"positions": self.get_state().tolist()}

    def reset(self, target_state: np.ndarray | None = None) -> None:
        """Command an optional reset target; otherwise leave the state unchanged.

        Drivers with a device-specific default reset pose override this method.
        """
        if target_state is not None:
            self.command(target_state)
