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

"""A robot: the outermost group of parts, and what places them."""

from collections.abc import Mapping
from typing import Any, ClassVar, Optional, TypeVar

from .parts.base import Group, RobotPart

RobotPartType = TypeVar("RobotPartType", bound=RobotPart)


class Robot(Group):
    """A named group of parts that owns their placement.

    A robot is a :class:`~rlinf.robotics.parts.base.Group` like any other, with
    three additions: it knows its registered type, it builds itself from a
    hardware config, and :meth:`connect` places the parts that were declared
    rather than constructed::

        robot = FrankaRobot(
            arm=FrankaROSArm.at(robot_ip, node_rank=1).export("arm"),
            wrist=RealSenseCamera.at(info, node_rank=3),
        )
        robot.connect()

    There are no arm, camera, or base slots. Names carry the meaning, so a robot
    with a lift, a head, or a third arm needs no new concept.
    """

    ROBOT_TYPE: ClassVar[str] = ""

    def __init__(self, parts: Optional[Mapping[str, Any]] = None, **named: Any) -> None:
        super().__init__(parts, **named)
        self.handles: dict[str, Any] = {}
        """Connections this robot placed, keyed by the part that needed them.
        Reach hardware methods outside the part interface through these."""

        self._placement: Any = None
        self._declared: Optional[dict[str, Any]] = None

    @classmethod
    def build(cls, **kwargs: Any) -> "Robot":
        """Compose this robot from its hardware config.

        Only robots reached through the registry need this: it is what
        ``register`` hands to :func:`~rlinf.robotics.discovery.build_robot`, so
        a robot can be composed from its type name alone. A robot you construct
        yourself does not need it -- name the parts and connect::

            class Bench(Robot):
                ROBOT_TYPE = "Bench"


            connection = MyArm.at(port, node_rank=1)
            robot = Bench(
                arm=connection.export("arm"),
                gripper=connection.export("end_effector"),
                eye=MyCamera.at(info),
            )
            robot.connect()

        Composing does not connect.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement build(), which only robots "
            "composed from the registry by type name need. Construct "
            f"{cls.__name__}(part=..., ...) directly instead."
        )

    @classmethod
    def register(cls, config_cls: type, discovery_cls: type) -> type:
        """Register this robot's config, discovery, and builder in one call."""
        from .discovery import register_robot

        return register_robot(config_cls, cls, build=cls.build)(discovery_cls)

    def parts_of_type(self, part_type: type[RobotPartType]) -> dict[str, RobotPartType]:
        """Return every part implementing ``part_type``, by its dotted path."""
        matches: dict[str, RobotPartType] = {}

        def walk(group: Group, prefix: str) -> None:
            for name, part in group.children.items():
                path = f"{prefix}{name}"
                if isinstance(part, part_type):
                    matches[path] = part
                if isinstance(part, Group):
                    walk(part, f"{path}.")

        walk(self, "")
        return matches

    @staticmethod
    def _origin_spec(part: Any) -> Any:
        """The declaration a part came from, when it came from one."""
        from .placement.specs import PartSpec, SubpartRef

        if isinstance(part, SubpartRef):
            return part.spec
        if isinstance(part, PartSpec):
            return part
        return None

    @classmethod
    def _flatten(cls, declared: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
        """The declaration snapshot, keyed the way the tree walk keys parts."""
        flat: dict[str, Any] = {}
        for name, value in declared.items():
            path = f"{prefix}{name}"
            if isinstance(value, dict):
                flat.update(cls._flatten(value, f"{path}."))
            else:
                flat[path] = value
        return flat

    @staticmethod
    def _declared_kind(part_cls: Optional[type]) -> str:
        """What a declared part will be, without building it."""
        from .parts.base import _PART_KINDS

        if part_cls is None:
            return "declared"
        for kind, kind_cls in _PART_KINDS:
            if isinstance(part_cls, type) and issubclass(part_cls, kind_cls):
                return kind
        return "declared"

    def describe(self) -> str:
        """What this robot is made of, where it runs, and what backs each part.

        Readable before anything is opened, which is when it is most useful: a
        declaration already says which node a part will run on and which
        connection it will come from, so a composition can be checked without
        a robot present.

        Parts sharing a ``via`` share one connection, and are therefore opened
        once and commanded in their declared order rather than concurrently.
        """
        from .placement.specs import PartSpec, SubpartRef

        lines = [type(self).__name__]
        rows: list[tuple[str, Any]] = []

        def walk(group: Group, prefix: str) -> None:
            for name, part in group.children.items():
                path = f"{prefix}{name}"
                if isinstance(part, Group):
                    walk(part, f"{path}.")
                else:
                    rows.append((path, part))

        walk(self, "")

        # Placement and ownership are properties of the declaration, and a
        # connected part no longer carries one. The robot kept the snapshot it
        # would restore on failure, so the same answer is available either
        # side of connect rather than only before it.
        declared = self._flatten(self._declared or {})

        # Parts declared from one spec share a connection; number the specs in
        # the order they appear so the grouping is visible at a glance.
        origins: dict[int, str] = {}
        for path, part in rows:
            spec = self._origin_spec(declared.get(path, part))
            if spec is not None and id(spec) not in origins:
                origins[id(spec)] = f"{spec.part_cls.__name__}#{len(origins) + 1}"

        width = max((len(path) for path, _ in rows), default=0)
        for index, (path, part) in enumerate(rows):
            branch = "└──" if index == len(rows) - 1 else "├──"
            spec = self._origin_spec(declared.get(path, part))
            if isinstance(part, SubpartRef):
                # What a connection exports is only known once it is open, so
                # naming the connection's own kind here would say the arm is a
                # connection. It is not.
                kind = "declared"
            elif isinstance(part, PartSpec):
                kind = self._declared_kind(spec.part_cls if spec else None)
            else:
                kind = part.kind
            where = (
                "here"
                if spec is None or spec.node_rank is None
                else str(spec.node_rank)
            )
            via = origins.get(id(spec), "itself") if spec is not None else "itself"
            lines.append(
                f"{branch} {path:<{width}}  {kind:<13} node={where:<5} via {via}"
            )
        return "\n".join(lines)

    @property
    def named_parts(self) -> dict[str, RobotPart]:
        """Every part keyed by its dotted path."""
        return self.parts_of_type(RobotPart)

    def connect(self) -> None:
        """Place any declared parts, then connect everything.

        Each distinct declaration is built once, however many parts refer to it.
        If anything fails, whatever was already placed or connected is torn down
        and the parts go back to their declarations, so you can fix the cause and
        call ``connect`` again.
        """
        from .placement import Placement

        if self._declared is None:
            self._declared = self.declarations()

        placement = self._placement or Placement()
        try:
            for name, handles in self.resolve(placement).items():
                self.handles.setdefault(name, handles[0])
            self._placement = placement
            super().connect()
        except Exception:
            aborted = {id(handle) for handle in placement.handles}
            self.handles = {
                name: handle
                for name, handle in self.handles.items()
                if id(handle) not in aborted
            }
            placement.release()
            self._placement = None
            self.restore(self._declared)
            raise

    def disconnect(self) -> None:
        """Disconnect every part, release the connections, restore declarations.

        A disconnected robot can be connected again: the parts go back to the
        declarations they were composed with, so ``connect`` places fresh ones
        rather than reusing any whose connection is gone.
        """
        super().disconnect()

        placed = set()
        if self._placement is not None:
            placed = {id(handle) for handle in self._placement.handles}
        for handle in reversed(list(self.handles.values())):
            if id(handle) not in placed:
                handle.disconnect()
        if self._placement is not None:
            self._placement.release()
            self._placement = None

        self.handles.clear()
        if self._declared is not None:
            self.restore(self._declared)
