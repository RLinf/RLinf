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

"""One operator input device: what it reads, and how that drives a robot.

A device answers three questions, and a new one is written by answering them
in a single class::

    @TeleopDevice.register("so101_leader")
    class SO101Leader(TeleopDevice):
        PRODUCES = {"arm": ActionKind.JOINT_POSITION}
        NEEDS = ("joint_positions",)

        def _open(self): ...  # reach the hardware
        def get_observation(self): ...  # what the operator is doing
        def action(self, reading, context): ...  # what the robot should do

Config becomes constructor arguments through :meth:`TeleopDevice.from_config`,
which by default passes the device's options straight through, so a device
whose config names match its arguments writes none of that itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
)

import numpy as np

from ...actions import ActionKind
from ..base import RobotPart

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .group import TeleopEntry


#: Bound where a device class is decorated, so registration returns that
#: class rather than the base it registers into.
RegisteredDevice = TypeVar("RegisteredDevice", bound="TeleopDevice")

#: Context fields a device may request through ``TeleopDevice.NEEDS``.
CONTEXT_KEYS = (
    "tcp_pose",
    "action_scale",
    "joint_positions",
    "gripper_open",
    "hand_reset_pose",
)


@dataclass
class TeleopAction:
    """Action contribution produced from one device reading.

    Attributes:
        parts: The action parts this device fills, by name.
        driving: Whether the operator is actually driving. Devices report small
            residual motion constantly, so each device decides its own
            threshold.
        info: Device state worth recording alongside the step it produced.
        publishes: Context for devices driven later in the same step.
    """

    parts: dict[str, np.ndarray] = field(default_factory=dict)
    driving: bool = False
    info: dict[str, Any] = field(default_factory=dict)
    publishes: dict[str, Any] = field(default_factory=dict)
    """Context this device offers to the devices driven after it."""


class TeleopPart(RobotPart, ABC):
    """Base class for operator input devices."""

    @abstractmethod
    def _open(self) -> Any:
        """Open the device and return its vendor reader."""

    def _release(self, device: Any) -> None:
        """Release the reader by whichever name its vendor gave the method."""
        for method in ("close", "stop"):
            release = getattr(device, method, None)
            if callable(release):
                release()
                return

    @property
    def ready(self) -> bool:
        """Return whether the device has produced a usable reading."""
        return bool(getattr(self._device, "ready", self.is_connected))


class TeleopDevice(TeleopPart):
    """A device the operator drives, and the robot action it produces.

    Subclasses declare what they fill through :pyattr:`PRODUCES`, what robot
    state they need through :pyattr:`NEEDS`, and implement :meth:`_open`,
    :meth:`get_observation` and :meth:`action`.
    """

    #: Registered devices, by the name a config spells them with.
    _REGISTRY: ClassVar[dict[str, type["TeleopDevice"]]] = {}

    #: Action parts this device fills, and what each one means.
    PRODUCES: Mapping[str, ActionKind] = {}

    #: Robot state this device's :meth:`action` reads out of the context.
    NEEDS: tuple[str, ...] = ()

    #: Whether the parts this fills should be clipped into the action space.
    CLIPS_TO_ACTION_SPACE: bool = False

    #: Whether this device's parts still apply while the operator is idle.
    APPLIES_WHILE_IDLE: bool = False

    #: Seconds a held part stays held, or ``None`` to hold indefinitely.
    HOLD_WINDOW: Optional[float] = None

    #: Motion below which a device counts as not being driven. Devices report
    #: small residual movement constantly, so each one sets its own threshold.
    MOVEMENT_EPSILON: float = 0.001

    # Registry.

    @classmethod
    def register(
        cls, *names: str
    ) -> "Callable[[type[RegisteredDevice]], type[RegisteredDevice]]":
        """Register a device under the names a config spells it with."""

        def add(device_cls: "type[RegisteredDevice]") -> "type[RegisteredDevice]":
            for name in names:
                key = name.lower()
                taken = TeleopDevice._REGISTRY.get(key)
                if taken is not None and taken is not device_cls:
                    if (
                        device_cls.__module__ == "__main__"
                        and taken.__qualname__ == device_cls.__qualname__
                    ):
                        # ``python -m`` on a device module re-executes a class
                        # the package already registered. Same device, so keep
                        # the registered one rather than report a collision.
                        continue
                    raise ValueError(
                        f"Teleop device {name!r} is already registered to "
                        f"{taken.__name__}; {device_cls.__name__} cannot take it."
                    )
                TeleopDevice._REGISTRY[key] = device_cls
            return device_cls

        return add

    @staticmethod
    def _load_devices() -> None:
        """Import the shipped devices so they register themselves."""
        from importlib import import_module

        import_module("rlinf.robotics.parts.teleop")

    @classmethod
    def named(cls, name: str) -> "type[TeleopDevice]":
        """Return the device registered under ``name``."""
        if not TeleopDevice._REGISTRY:
            cls._load_devices()
        device_cls = TeleopDevice._REGISTRY.get(str(name).lower())
        if device_cls is None:
            raise ValueError(
                f"Unknown teleop device {name!r}. Available: {cls.names()}."
            )
        return device_cls

    @classmethod
    def names(cls) -> list[str]:
        """Return every registered device name, sorted."""
        if not TeleopDevice._REGISTRY:
            cls._load_devices()
        return sorted(TeleopDevice._REGISTRY)

    # Construction from configuration.

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, Any],
        options: Mapping[str, Any],
        facts: Any,
    ) -> "TeleopEntry":
        """Build this device and say which branch of the action it fills.

        The default passes the device's own options through as keyword
        arguments, which is all a device needs whose config names match its
        constructor. Override to read a key from the wider env config, or to
        choose behaviour from ``facts`` about the robot being driven.

        Args:
            cfg: The environment config, for settings named outside the device.
            options: This device's own options from the ``teleop`` entry.
            facts: Action layout and kinds of the robot being driven.
        """
        from .group import TeleopEntry

        settings = {k: v for k, v in options.items() if k != "drives"}
        return TeleopEntry(cls(**settings), drives=options.get("drives"))

    @classmethod
    def streamer(
        cls,
        cfg: Mapping[str, Any],
        facts: Any,
        entries: "Sequence[TeleopEntry]",
    ) -> Optional[Any]:
        """Build an optional streamer that drives the robot outside ``step``.

        Only a device that bypasses the environment's step loop needs one.

        Args:
            cfg: The environment config.
            facts: Action layout and kinds of the robot being driven.
            entries: Every entry built for this robot, so a streamer can reuse
                devices instead of opening their ports a second time.
        """
        return None

    # What the operator is doing, and what the robot should do about it.

    @abstractmethod
    def action(
        self, reading: Mapping[str, Any], context: Mapping[str, Any]
    ) -> TeleopAction:
        """Convert one reading into the action parts this device fills."""

    def drive(
        self,
        context: Mapping[str, Any],
        reading: Optional[Mapping[str, Any]] = None,
    ) -> TeleopAction:
        """Read this device and map the reading in one call.

        The group drives a device through this rather than reading and mapping
        separately, so a device placed on the operator's machine costs one
        round trip per step instead of shipping the reading back to be mapped.

        Args:
            context: Robot state and anything devices driven earlier published.
            reading: A reading already taken from this device this step. The
                group passes one only when two entries share the device, so
                both map the same instant instead of reading it twice.
        """
        if reading is None:
            reading = self.get_observation()
        published = self.publish(reading)
        # A device's own published context is available to its own mapping,
        # as it was when reading and mapping were separate steps.
        action = self.action(reading, {**context, **published})
        action.publishes = published
        return action

    def publish(self, reading: Mapping[str, Any]) -> dict[str, Any]:
        """Return context this device offers to the devices read after it."""
        return {}

    def hold(self, context: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Return actions holding this device's parts where they are."""
        return {}

    def on_action_chunk_begin(self) -> None:
        """Let go of anything held only until the next chunk of actions."""

    def on_reset(self, context: Mapping[str, Any] = MappingProxyType({})) -> None:
        """Re-align to the robot after it resets.

        Named apart from :meth:`reset`, which every connection already has for
        its own hardware state.
        """
