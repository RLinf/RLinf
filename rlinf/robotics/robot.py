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

"""Robot composition and lifecycle management."""

from typing import TYPE_CHECKING, Any, ClassVar, Optional

from .parts.base import PartGroup, RobotPart, RobotPartType

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .discovery import RobotConfig, RobotDiscovery


class Robot(PartGroup):
    """Top-level named part group for a physical robot.

    Each child retains its own connection and placement. Parts mounted below a
    child, such as an arm's end effector, appear at nested paths.
    """

    ROBOT_TYPE: ClassVar[str] = ""

    @classmethod
    def build(cls, **kwargs: Any) -> "Robot":
        """Compose an unconnected robot from hardware settings."""
        raise NotImplementedError(
            f"{cls.__name__} does not implement build(), which only robots "
            "composed from the registry by type name need. Construct "
            f"{cls.__name__}(part=..., ...) directly instead."
        )

    @classmethod
    def register_type(
        cls,
        config_cls: "type[RobotConfig]",
        discovery_cls: "Optional[type[RobotDiscovery]]" = None,
    ) -> "type[RobotDiscovery]":
        """Register the robot class, config, discovery, and builder.

        If ``discovery_cls`` is omitted, a standard discovery class is created
        from :attr:`ROBOT_TYPE`.

        Example::

            FrankaRobot.register_type(FrankaConfig)
        """
        from .discovery import RobotDiscovery, register_robot

        if discovery_cls is None:
            discovery_cls = type(
                f"{cls.__name__}Discovery",
                (RobotDiscovery,),
                {
                    "HW_TYPE": cls.ROBOT_TYPE,
                    "__module__": cls.__module__,
                    "__doc__": f"Find {cls.ROBOT_TYPE} robots on a node.",
                },
            )
        elif not getattr(discovery_cls, "HW_TYPE", ""):
            discovery_cls.HW_TYPE = cls.ROBOT_TYPE

        return register_robot(config_cls, cls, build=cls.build)(discovery_cls)

    @classmethod
    def of_type(cls, robot_type: str, **kwargs: Any) -> "Robot":
        """Compose an unconnected robot from a registered type name."""
        from .discovery import RobotDiscovery

        registration = RobotDiscovery.registry.get(robot_type)
        if registration is None:
            raise KeyError(
                f"Unknown robot type {robot_type!r}. "
                f"Registered: {sorted(RobotDiscovery.registry)}."
            )
        if registration.build is None:
            raise NotImplementedError(
                f"Robot type {robot_type!r} registered no builder."
            )
        return registration.build(**kwargs)

    def parts_of_type(self, part_type: type[RobotPartType]) -> dict[str, RobotPartType]:
        """Return every part implementing ``part_type``, by its dotted path."""
        matches: dict[str, RobotPartType] = {}

        def walk(part: RobotPart, prefix: str) -> None:
            for name, child in part.children.items():
                path = f"{prefix}{name}"
                if isinstance(child, part_type):
                    matches[path] = child
                walk(child, f"{path}.")

        walk(self, "")
        return matches

    def describe(self) -> str:
        """Describe the part tree, placement, and connection ownership.

        The method is safe before connecting hardware. Paths and ownership
        remain stable after connection, but remotely placed parts display their
        synthesized view class.
        """
        rows: list[tuple[str, Optional[RobotPart]]] = []

        def walk(part: RobotPart, prefix: str) -> None:
            children = list(part.children.items())
            for index, (name, child) in enumerate(children):
                last = index == len(children) - 1
                branch = "└── " if last else "├── "
                # Groups have no device metadata; their children provide it.
                rows.append(
                    (
                        prefix + branch + name,
                        None if isinstance(child, PartGroup) else child,
                    )
                )
                walk(child, prefix + ("    " if last else "│   "))

        walk(self, "")

        # Number owners by first appearance to show shared connections.
        origins: dict[int, str] = {}
        for _, part in rows:
            if part is None:
                continue
            owner = part.owner
            if id(owner) not in origins:
                origins[id(owner)] = f"{type(owner).__name__}#{len(origins) + 1}"

        width = max((len(label) for label, _ in rows), default=0)
        lines = [type(self).__name__]
        for label, part in rows:
            if part is None:
                lines.append(label)
                continue
            owner = part.owner
            where = "here" if owner.node_rank is None else str(owner.node_rank)
            lines.append(
                f"{label:<{width}}  {type(part).__name__:<20} "
                f"node={where:<5} via {origins[id(owner)]}"
            )
        return "\n".join(lines)

    @property
    def named_parts(self) -> dict[str, RobotPart]:
        """Return every part keyed by its dotted path."""
        return self.parts_of_type(RobotPart)
