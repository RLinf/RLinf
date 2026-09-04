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

"""Reusable conformance checks for robot parts, connections, and robots.

The checks cover lifecycle behavior, declared observation shapes, composition,
resource ownership, and rollback after a partial connection failure. They can
run under pytest, from a bench script, or directly against a robot::

    from robot_contracts import PartContract, RobotContract


    def test_my_arm_conforms():
        PartContract(lambda: MyArm("10.0.0.1")).assert_kept()


    def test_my_robot_conforms():
        RobotContract(lambda: MyRobot.build(robot_ip="10.0.0.1")).assert_kept()

The module is test infrastructure and does not depend on pytest.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Sequence

__all__ = [
    "ConformanceError",
    "ConnectionContract",
    "Contract",
    "ObservationContract",
    "PartContract",
    "RobotContract",
]


class ConformanceError(AssertionError):
    """Report one or more conformance failures for a subject."""

    def __init__(self, subject: str, failures: Sequence[str]) -> None:
        self.failures = list(failures)
        listed = "\n".join(f"  - {failure}" for failure in self.failures)
        super().__init__(f"{subject} does not keep {len(self.failures)}:\n{listed}")


class ObservationContract:
    """Check a part's observations against its declared features."""

    def __init__(self, part: Any, where: str) -> None:
        self.part = part
        self.where = where

    @staticmethod
    def declared_shape(feature: Any) -> Optional[tuple]:
        """Return the declared shape, or ``None`` for an unshaped feature."""
        if isinstance(feature, dict):
            shape = feature.get("shape")
            if isinstance(shape, (tuple, list)):
                return tuple(shape)
        return None

    def failures(self) -> list[str]:
        """Return all observation key and shape mismatches."""
        try:
            observation = self.part.get_observation()
        except Exception as error:  # noqa: BLE001 - a check reports anything
            return [
                f"{self.where}: get_observation raised {type(error).__name__}: {error}"
            ]

        features = self.part.observation_features
        found: list[str] = []
        extra = sorted(set(observation) - set(features))
        missing = sorted(set(features) - set(observation))
        if extra:
            found.append(f"{self.where} observes {extra}, which it never declared")
        if missing:
            found.append(f"{self.where} declares {missing}, which it does not observe")

        for key in sorted(set(features) & set(observation)):
            wanted = self.declared_shape(features[key])
            actual = getattr(observation[key], "shape", None)
            if wanted is None or actual is None:
                continue
            if tuple(actual) != tuple(wanted):
                found.append(
                    f"{self.where} observes {key} with shape {tuple(actual)}, "
                    f"declares {tuple(wanted)}"
                )
        return found


class Contract(ABC):
    """Base class for conformance checks against a fresh subject.

    Args:
        factory: Callable that returns a fresh, unconnected subject.
    """

    def __init__(self, factory: Callable[[], Any]) -> None:
        self.factory = factory

    @property
    def subject_name(self) -> str:
        """Return the subject name used in failure messages."""
        return type(self.factory()).__name__

    @abstractmethod
    def failures(self) -> list[str]:
        """Return all conformance failures."""

    def assert_kept(self) -> None:
        """Raise :class:`ConformanceError` if any check fails."""
        failures = self.failures()
        if failures:
            raise ConformanceError(self.subject_name, failures)

    # Shared checks

    def lifecycle_failures(self, connection: Any, where: str) -> list[str]:
        """Check connection, reconnection, and idempotent cleanup."""
        found: list[str] = []

        if connection.is_connected:
            found.append(f"{where} reports itself connected before connect()")

        try:
            connection.connect()
        except Exception as error:  # noqa: BLE001
            return found + [f"{where}: connect raised {type(error).__name__}: {error}"]
        if not connection.is_connected:
            found.append(f"{where} does not report itself connected after connect()")

        try:
            connection.connect()
        except Exception as error:  # noqa: BLE001
            found.append(
                f"{where}: connecting an already-connected connection raised "
                f"{type(error).__name__}: {error}"
            )

        try:
            connection.disconnect()
        except Exception as error:  # noqa: BLE001
            return found + [
                f"{where}: disconnect raised {type(error).__name__}: {error}"
            ]
        if connection.is_connected:
            found.append(f"{where} still reports itself connected after disconnect()")

        try:
            connection.disconnect()
        except Exception as error:  # noqa: BLE001
            found.append(
                f"{where}: disconnecting twice raised "
                f"{type(error).__name__}: {error}; cleanup has to be idempotent"
            )

        try:
            connection.connect()
        except Exception as error:  # noqa: BLE001
            found.append(
                f"{where}: reconnecting raised {type(error).__name__}: {error}; "
                "stall recovery closes a connection and opens it again"
            )
        return found

    @staticmethod
    def release(connection: Any) -> None:
        """Disconnect a subject after a check, suppressing reported failures."""
        try:
            connection.disconnect()
        except Exception:  # noqa: BLE001 - anything wrong here is already reported
            pass


class PartContract(Contract):
    """Check the lifecycle and data interface of a robot part.

    Args:
        factory: Callable that returns a fresh, unconnected part.
        action: Valid action for a controllable part. If omitted, action checks
            are skipped.
    """

    def __init__(
        self, factory: Callable[[], Any], action: Optional[dict] = None
    ) -> None:
        super().__init__(factory)
        self.action = action

    def failures(self) -> list[str]:
        """Return lifecycle, observation, and action failures."""
        from rlinf.robotics.parts.base import Connection, ControllablePart, RobotPart

        part = self.factory()
        where = type(part).__name__

        if not isinstance(part, RobotPart):
            if isinstance(part, Connection):
                return [
                    f"{where} is a connection that backs parts without being "
                    "one, so it belongs to ConnectionContract; a part is "
                    "something you can read"
                ]
            return [f"{where} is not a RobotPart"]

        found = self.lifecycle_failures(part, where)
        if part.is_connected:
            found += ObservationContract(part, where).failures()
        if self.action is not None and isinstance(part, ControllablePart):
            found += self._action_failures(part, where)

        self.release(part)
        return found

    def _action_failures(self, part: Any, where: str) -> list[str]:
        """Check valid actions and rejection of unknown part names."""
        found: list[str] = []
        try:
            part.send_action(self.action)
        except Exception as error:  # noqa: BLE001
            found.append(
                f"{where}: send_action({sorted(self.action)}) raised "
                f"{type(error).__name__}: {error}"
            )

        unknown = {"a-part-this-does-not-have": next(iter(self.action.values()))}
        try:
            part.send_action(unknown)
            found.append(
                f"{where} accepted an action naming a part it does not have; "
                "an unknown name should be refused, not silently dropped"
            )
        except Exception:  # noqa: BLE001 - refusing is the point, how is its own
            pass
        return found


class ConnectionContract(Contract):
    """Check a shared hardware connection and the parts it exports."""

    def failures(self) -> list[str]:
        """Return lifecycle and exported-part failures."""
        from rlinf.robotics.parts.base import Connection, RobotPart

        connection = self.factory()
        where = type(connection).__name__

        if not isinstance(connection, Connection):
            return [f"{where} is not a Connection, so it cannot be opened at all"]
        if isinstance(connection, RobotPart):
            return [
                f"{where} is a RobotPart, so a robot can compose it whole. A "
                "link that backs several parts without being one of them should "
                "subclass Connection directly, and leave get_observation off"
            ]

        found = [
            f"{where}.{absent} exists; a connection is not read and composes nothing"
            for absent in ("get_observation", "observation_features", "children")
            if hasattr(connection, absent)
        ]
        found += self.lifecycle_failures(connection, where)
        if connection.is_connected:
            found += self._backed_part_failures(connection, where, RobotPart)

        self.release(connection)
        return found

    @staticmethod
    def _backed_part_failures(connection: Any, where: str, part_cls: type) -> list[str]:
        """Check that every exported value is an observable robot part."""
        parts = connection.parts
        if not parts:
            return [f"{where} backs no parts, so no robot can use it"]

        found: list[str] = []
        for name, backed in parts.items():
            if not isinstance(backed, part_cls):
                found.append(
                    f"{where} backs {name!r}, which is a "
                    f"{type(backed).__name__} and not a readable part"
                )
            else:
                found += ObservationContract(backed, f"{where}.{name}").failures()
        return found


class RobotContract(Contract):
    """Check a robot's part tree, lifecycle, ownership, and rollback."""

    def failures(self) -> list[str]:
        """Return all robot conformance failures."""
        robot = self.factory()
        where = type(robot).__name__

        found = self._describe_failures(robot, where)
        found += self._identity_failures(robot, where)
        found += self.lifecycle_failures(robot, where)
        if robot.is_connected:
            found += self._tree_failures(robot)
        self.release(robot)
        found += self._rollback_failures()
        return found

    @classmethod
    def _identity_failures(cls, robot: Any, where: str) -> list[str]:
        """Check that repeated tree traversal returns the same part objects."""
        first = dict(cls._leaves(robot, ""))
        second = dict(cls._leaves(robot, ""))
        unstable = sorted(path for path in first if first[path] is not second.get(path))
        if unstable:
            return [
                f"{where} hands out a different object for {unstable} each "
                "time the tree is walked"
            ]
        return []

    @staticmethod
    def _describe_failures(robot: Any, where: str) -> list[str]:
        """Check that the robot can be described before connecting."""
        try:
            if where not in robot.describe():
                return [f"{where}.describe() does not name the robot"]
        except Exception as error:  # noqa: BLE001
            return [
                f"{where}.describe() raised {type(error).__name__}: {error} before "
                "connecting, which is when it is most useful"
            ]
        return []

    @classmethod
    def _tree_failures(cls, robot: Any) -> list[str]:
        """Check every leaf's type, observations, and ownership."""
        from rlinf.robotics.parts.base import Connection, RobotPart

        found: list[str] = []
        for path, part in cls._leaves(robot, ""):
            if isinstance(part, Connection) and not isinstance(part, RobotPart):
                found.append(
                    f"{path} is a connection and should not be in the tree; "
                    "compose the parts it backs"
                )
                continue
            found += ObservationContract(part, path).failures()
            found += cls._ownership_failures(robot, path, part)
        return found

    @staticmethod
    def _ownership_failures(robot: Any, path: str, part: Any) -> list[str]:
        """Check that the robot owns and opens the part's connection."""
        found: list[str] = []
        if not part.is_connected:
            found.append(
                f"{path} is in the tree but not connected; the robot opened "
                f"{[type(o).__name__ for o in robot.owners()]} and none of "
                "them was its owner"
            )
        if not any(owner is part.owner for owner in robot.owners()):
            found.append(
                f"{path} rides a {type(part.owner).__name__} the robot never "
                "lists among its owners, so nothing will close it either"
            )
        return found

    @classmethod
    def _leaves(cls, group: Any, prefix: str) -> list[tuple[str, Any]]:
        """Return every readable part and its dotted path."""
        from rlinf.robotics.parts.base import PartGroup

        found: list[tuple[str, Any]] = []
        for name, part in group.children.items():
            path = f"{prefix}{name}"
            if not isinstance(part, PartGroup):
                found.append((path, part))
            found += cls._leaves(part, f"{path}.")
        return found

    def _rollback_failures(self) -> list[str]:
        """Check rollback after the final connection fails to open."""
        robot = self.factory()
        connections = self._declarations(robot)
        if not connections:
            # Nothing to open, so there is no partial state to roll back.
            return []

        # Patch the lifecycle entry point implemented by this concrete class.
        target = type(connections[-1])
        hook = "_open" if "_open" in vars(target) else "connect"
        original = getattr(target, hook)

        def refuse(_self: Any) -> Any:
            raise RuntimeError("conformance check: refusing to open")

        found: list[str] = []
        setattr(target, hook, refuse)
        try:
            try:
                robot.connect()
                found.append(
                    f"{type(robot).__name__}.connect() reported success although "
                    f"{target.__name__}.{hook} refused to open"
                )
            except Exception:  # noqa: BLE001 - the refusal is what we arranged
                pass
            if robot.is_connected:
                found.append(
                    f"{type(robot).__name__} reports itself connected after a "
                    "failed connect; a half-built robot should not be handed back"
                )
        finally:
            setattr(target, hook, original)

        self.release(robot)
        return found

    @staticmethod
    def _declarations(robot: Any) -> list[Any]:
        """Return the distinct connections in tree order."""
        return robot.owners()
