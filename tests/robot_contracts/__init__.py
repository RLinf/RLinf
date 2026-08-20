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

"""What a part, a connection, and a robot each promise, as runnable checks.

Every lifecycle bug this package has had came from a promise nobody was
checking: a reader closed after its handle was cleared, a camera whose thread
could not be started twice, a part observing a value one number wider than it
declared, a robot that raised on the second disconnect. Each was found by hand,
once, on one robot.

So the promises are written down here instead, as three contracts a contributor
points at their own hardware::

    from robot_contracts import PartContract, RobotContract


    def test_my_arm_conforms():
        PartContract(lambda: MyArm("10.0.0.1")).assert_kept()


    def test_my_robot_conforms():
        RobotContract(lambda: MyRobot.build(robot_ip="10.0.0.1")).assert_kept()

They sit beside :mod:`robot_mocks` rather than in the shipped package, because
they test RLinf rather than being part of it, and they need no test framework
of their own -- so the same contract runs under pytest, in a bench script, or
from a REPL with a robot on the desk.
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
    "PlacementParityContract",
    "RobotContract",
]


class ConformanceError(AssertionError):
    """One or more promises a part, connection, or robot did not keep."""

    def __init__(self, subject: str, failures: Sequence[str]) -> None:
        self.failures = list(failures)
        listed = "\n".join(f"  - {failure}" for failure in self.failures)
        super().__init__(f"{subject} does not keep {len(self.failures)}:\n{listed}")


class ObservationContract:
    """What a part observes, against what it said it would.

    Its own class because two callers need it: the contracts below, and the
    bench check that walks a robot on real hardware. One implementation means
    a bench run and a contributor's test cannot disagree about what a part
    promised.
    """

    def __init__(self, part: Any, where: str) -> None:
        self.part = part
        self.where = where

    @staticmethod
    def declared_shape(feature: Any) -> Optional[tuple]:
        """The shape a feature descriptor promises, when it states one.

        Features are plain dictionaries and a part may describe a subtree
        rather than an array, so only a stated shape is worth comparing.
        """
        if isinstance(feature, dict):
            shape = feature.get("shape")
            if isinstance(shape, (tuple, list)):
                return tuple(shape)
        return None

    def failures(self) -> list[str]:
        """Read the part once and report every way the answer disagrees.

        Keys and shapes both. An env builds its observation space from the
        declaration, so a value of the wrong width reaches a policy as data
        rather than as an error -- it corrupts training instead of raising.
        """
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
    """One subject, built by a factory, checked against what it promises.

    Args:
        factory: Builds a fresh, unconnected subject each time it is called.
            Several checks need one that has never been connected, so this is
            a factory rather than an instance.
    """

    def __init__(self, factory: Callable[[], Any]) -> None:
        self.factory = factory

    @property
    def subject_name(self) -> str:
        """What to call the subject when reporting a broken promise."""
        return type(self.factory()).__name__

    @abstractmethod
    def failures(self) -> list[str]:
        """Every promise the subject did not keep, as readable sentences."""

    def assert_kept(self) -> None:
        """Raise unless the subject keeps every promise its kind makes."""
        failures = self.failures()
        if failures:
            raise ConformanceError(self.subject_name, failures)

    # -- shared checks ----------------------------------------------------

    def lifecycle_failures(self, endpoint: Any, where: str) -> list[str]:
        """Connect, disconnect, do it again, and do the last one twice.

        Reconnecting is the one people skip, and it is the one stall recovery
        needs: an env that finds a camera stalled closes it and opens it
        again. Disconnecting twice is what a teardown path does when it is not
        sure the thing ever came up, and raising there replaces whatever error
        the caller was actually handling.
        """
        found: list[str] = []

        if endpoint.is_connected:
            found.append(f"{where} reports itself connected before connect()")

        try:
            endpoint.connect()
        except Exception as error:  # noqa: BLE001
            return found + [f"{where}: connect raised {type(error).__name__}: {error}"]
        if not endpoint.is_connected:
            found.append(f"{where} does not report itself connected after connect()")

        try:
            endpoint.connect()
        except Exception as error:  # noqa: BLE001
            found.append(
                f"{where}: connecting an already-connected endpoint raised "
                f"{type(error).__name__}: {error}"
            )

        try:
            endpoint.disconnect()
        except Exception as error:  # noqa: BLE001
            return found + [
                f"{where}: disconnect raised {type(error).__name__}: {error}"
            ]
        if endpoint.is_connected:
            found.append(f"{where} still reports itself connected after disconnect()")

        try:
            endpoint.disconnect()
        except Exception as error:  # noqa: BLE001
            found.append(
                f"{where}: disconnecting twice raised "
                f"{type(error).__name__}: {error}; cleanup has to be idempotent"
            )

        try:
            endpoint.connect()
        except Exception as error:  # noqa: BLE001
            found.append(
                f"{where}: reconnecting raised {type(error).__name__}: {error}; "
                "stall recovery closes an endpoint and opens it again"
            )
        return found

    @staticmethod
    def release(endpoint: Any) -> None:
        """Let a checked subject go, whatever state the checks left it in."""
        try:
            endpoint.disconnect()
        except Exception:  # noqa: BLE001 - anything wrong here is already reported
            pass


class PartContract(Contract):
    """A part is an endpoint you can read, and optionally command.

    Args:
        factory: Builds a fresh, unconnected part.
        action: An action to send, for a controllable part. Left out, the
            action half of the contract is skipped rather than guessed at.
    """

    def __init__(
        self, factory: Callable[[], Any], action: Optional[dict] = None
    ) -> None:
        super().__init__(factory)
        self.action = action

    def failures(self) -> list[str]:
        """Check the lifecycle, the observation, and the action."""
        from rlinf.robotics.parts.base import Connection, ControllablePart, RobotPart

        part = self.factory()
        where = type(part).__name__

        if isinstance(part, Connection):
            return [
                f"{where} is a Connection, so it belongs to ConnectionContract; "
                "a part is something you can read"
            ]
        if not isinstance(part, RobotPart):
            return [f"{where} is not a RobotPart"]

        found = self.lifecycle_failures(part, where)
        if part.is_connected:
            found += ObservationContract(part, where).failures()
        if self.action is not None and isinstance(part, ControllablePart):
            found += self._action_failures(part, where)

        self.release(part)
        return found

    def _action_failures(self, part: Any, where: str) -> list[str]:
        """Accept what it declares, and refuse what it does not."""
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
    """A connection owns a session and offers what rides on it.

    Its exports borrow that session: they are read through it, and they are
    not opened or closed on their own.
    """

    def failures(self) -> list[str]:
        """Check the lifecycle, then everything the session offers."""
        from rlinf.robotics.parts.base import Connection, RobotPart

        connection = self.factory()
        where = type(connection).__name__

        if not isinstance(connection, Connection):
            return [
                f"{where} is not a Connection. A thing that backs several parts "
                "without being one of them should be, so a robot cannot compose it"
            ]

        found = [
            f"{where}.{absent} exists; a connection is not read and composes nothing"
            for absent in ("get_observation", "observation_features", "children")
            if hasattr(connection, absent)
        ]
        found += self.lifecycle_failures(connection, where)
        if connection.is_connected:
            found += self._export_failures(connection, where, RobotPart, Connection)

        self.release(connection)
        return found

    @staticmethod
    def _export_failures(
        connection: Any, where: str, part_cls: type, connection_cls: type
    ) -> list[str]:
        """Everything the session offers has to be a readable part."""
        exports = connection.exports
        if not exports:
            return [f"{where} exports nothing, so no robot can use it"]

        found: list[str] = []
        for name, export in exports.items():
            if isinstance(export, connection_cls):
                found.append(f"{where} exports {name!r}, which is itself a connection")
            elif not isinstance(export, part_cls):
                found.append(f"{where} exports {name!r}, which is not a part")
            else:
                found += ObservationContract(export, f"{where}.{name}").failures()
        return found


class RobotContract(Contract):
    """A robot is a tree of named parts that comes up whole or not at all."""

    def failures(self) -> list[str]:
        """Check describing, the lifecycle, every part, and the rollback."""
        robot = self.factory()
        where = type(robot).__name__

        found = self._describe_failures(robot, where)
        found += self.lifecycle_failures(robot, where)
        if robot.is_connected:
            found += self._tree_failures(robot)
        self.release(robot)
        found += self._rollback_failures()
        return found

    @staticmethod
    def _describe_failures(robot: Any, where: str) -> list[str]:
        """Describing a robot has to work before one is connected."""
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
        """Every leaf observes what it declared, and no connection is a leaf."""
        from rlinf.robotics.parts.base import Connection

        found: list[str] = []
        for path, part in cls._leaves(robot, ""):
            if isinstance(part, Connection):
                found.append(
                    f"{path} is a connection and should not be in the tree; "
                    "compose the parts it exports"
                )
            else:
                found += ObservationContract(part, path).failures()
        return found

    @classmethod
    def _leaves(cls, group: Any, prefix: str) -> list[tuple[str, Any]]:
        """Every leaf of a composed tree, by its dotted path."""
        from rlinf.robotics.parts.base import Group

        found: list[tuple[str, Any]] = []
        for name, part in group.children.items():
            path = f"{prefix}{name}"
            if isinstance(part, Group):
                found += cls._leaves(part, f"{path}.")
            else:
                found.append((path, part))
        return found

    def _rollback_failures(self) -> list[str]:
        """A robot that fails halfway through connecting leaves nothing open.

        The failure is injected into the last declaration the robot places, so
        whatever came before it has already been built when the error arrives.
        That is the case that leaks.
        """
        robot = self.factory()
        specs = self._declarations(robot)
        if not specs:
            # Nothing is declared, so there is no partial state to roll back.
            return []

        # Whichever entry point this class actually opens through. A part with
        # a lifecycle of its own is allowed to override connect instead of
        # _open, and patching the one it does not use injects nothing and
        # quietly passes.
        target = specs[-1].part_cls
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
        """Every distinct declaration this robot will place, in tree order."""
        from rlinf.robotics.placement.specs import PartSpec, SubpartRef

        specs: list[Any] = []

        def walk(group: Any) -> None:
            for value in group.children.values():
                if hasattr(value, "children"):
                    walk(value)
                    continue
                if isinstance(value, SubpartRef):
                    value = value.spec
                if isinstance(value, PartSpec) and not any(s is value for s in specs):
                    specs.append(value)

        walk(robot)
        return specs


class PlacementParityContract(Contract):
    """A part answers the same hosted on a node as it does in this process.

    Separate from the contracts above because it needs a running cluster,
    where those need nothing but the part.
    """

    def __init__(self, factory: Callable[[], Any], node_rank: int = 0) -> None:
        super().__init__(factory)
        self.node_rank = node_rank

    @property
    def subject_name(self) -> str:
        """Name the subject as the placed thing it is being checked as."""
        return f"{super().subject_name} (placed)"

    def failures(self) -> list[str]:
        """Compare what a local part describes with what a hosted one does."""
        local = self.factory()
        local.connect()
        try:
            here = local.observation_features
        finally:
            self.release(local)

        handle = type(local).spawn(node_rank=self.node_rank)
        try:
            there = handle.part.observation_features
        finally:
            handle.disconnect()

        if here != there:
            return [
                "a hosted part describes different observations from a local "
                f"one: here {sorted(here)}, hosted {sorted(there)}"
            ]
        return []
