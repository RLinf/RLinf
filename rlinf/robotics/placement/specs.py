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

"""Declaring a part and where it runs, without building it yet.

A part whose vendor SDK only exists on one machine cannot be constructed here
and moved there. So placement takes a *declaration* -- a class, its constructor
arguments, and a ``node_rank`` -- and builds the part on the target node.

``RobotPart.at(...)`` produces a :class:`PartSpec`. Compose specs exactly where
you would compose parts; :meth:`Robot.connect` resolves them. Nobody writes a
placement call.

    connection = FrankaROSArm.at("10.0.0.1", node_rank=1)
    robot = FrankaRobot(
        arm=connection.export("arm"),
        end_effector=connection.export("end_effector"),
        scene=RealSenseCamera.at(info, node_rank=2),
    )
    robot.connect()

One spec is placed once, however many times it is referenced, so a driver whose
single connection backs several components is not opened twice.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class PartSpec:
    """A part to build, and the node to build it on.

    Attributes:
        part_cls: The :class:`~.parts.base.RobotPart` subclass to construct.
        args: Positional arguments for its constructor.
        kwargs: Keyword arguments for its constructor.
        node_rank: Node to build it on. ``None`` builds it in this process.
        name: Worker-group name when placed remotely. Defaults to something
            derived from the class and node.
    """

    part_cls: type
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    node_rank: Optional[int] = None
    name: Optional[str] = None

    def export(self, name: str) -> "SubpartRef":
        """Refer to one capability this declaration exports, after placement.

        Use it when one connection backs several components::

            connection = Turtle2Connection.at(50, node_rank=0)
            left = Group(
                arm=connection.export("left"),
                gripper=connection.export("left_end_effector"),
            )
        """
        return SubpartRef(self, name)

    def place(self) -> Any:
        """Build the part, here or on its node, and return its handle."""
        return self.part_cls.spawn(
            *self.args,
            node_rank=self.node_rank,
            name=self.name,
            **self.kwargs,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        where = "here" if self.node_rank is None else f"node {self.node_rank}"
        return f"<PartSpec {self.part_cls.__name__} on {where}>"


@dataclass(frozen=True)
class SubpartRef:
    """A reference to one part of a :class:`PartSpec`."""

    spec: PartSpec
    name: str

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<SubpartRef {self.spec.part_cls.__name__}.{self.name}>"


class Placement:
    """Places each distinct spec once and resolves references against it.

    :meth:`Robot.connect` owns one of these. It keeps the handles so the robot
    can release them, and tears down what it already placed if a later part
    fails, so a half-built robot is never handed back.
    """

    def __init__(self) -> None:
        self._handles: dict[int, Any] = {}
        self._order: list[Any] = []

    @property
    def handles(self) -> list[Any]:
        """Handles for every spec placed so far, in placement order."""
        return list(self._order)

    def resolve(self, value: Any) -> Any:
        """Return a live part for a spec, a reference, or a part.

        Anything that is already a part passes through untouched, so composing
        local and placed parts together needs no special case.
        """
        if isinstance(value, SubpartRef):
            return self._handle_for(value.spec).part_named(value.name)
        if isinstance(value, PartSpec):
            handle = self._handle_for(value)
            parts = handle.exports
            # A leaf -- a camera, a gripper on its own port -- is its own part.
            if not parts:
                return handle.part
            # One part means the hardware is that part.
            if len(parts) == 1:
                return next(iter(parts.values()))
            # Several means the hardware is a subtree. Hand back the whole
            # thing: an arm that carries a gripper resolves to both, so a robot
            # composing the arm never has to reach inside for the gripper.
            from ..parts.base import Group

            return Group(parts)
        return value

    def resolve_handle(self, spec: PartSpec) -> Any:
        """Return the handle for a spec, placing it if this is the first use."""
        return self._handle_for(spec)

    def _handle_for(self, spec: PartSpec) -> Any:
        key = id(spec)
        handle = self._handles.get(key)
        if handle is None:
            handle = spec.place()
            self._handles[key] = handle
            self._order.append(handle)
        return handle

    def release(self) -> None:
        """Release every placed handle, newest first."""
        while self._order:
            handle = self._order.pop()
            handle.disconnect()
        self._handles.clear()
