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

"""What a task needs from a robot, and where on the robot it finds it.

A task names *roles* -- ``"arm"``, ``"left"`` and ``"right"`` for a two-arm
task, ``"base"`` for a mobile one -- and says what the part filling each must
be, report, and accept. :func:`bind` finds those parts on a composed robot,
with any end effector an arm carries, and checks them against what they
declare, before anything connects, so a robot that cannot run a task is
refused with the reason rather than failing on its first step.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

from rlinf.robotics import Arm, EndEffector, Robot, RobotPart
from rlinf.robotics.fields import mismatch


class RequirementError(ValueError):
    """A robot is missing something a task or control needs."""


@dataclass(frozen=True)
class Needs:
    """What one role needs from the part filling it.

    Attributes:
        kind: The part category that fills the role.
        observes: Fields the part must report, such as ``"tcp_pose"``.
        commands: Fields the part must accept, such as ``"joint_position"``.
        end_effector: ``"gripper"`` or ``"hand"`` when the part must carry one
            of those, ``"any"`` for either, ``None`` when none is needed.
    """

    kind: type[RobotPart] = Arm
    observes: frozenset[str] = field(default_factory=frozenset)
    commands: frozenset[str] = field(default_factory=frozenset)
    end_effector: Optional[str] = None

    def __or__(self, other: "Needs") -> "Needs":
        """Combine two sources' needs for one role, as a task's and a control's."""
        if self.kind is not other.kind:
            raise RequirementError(
                f"One role cannot be both a {self.kind.__name__} and a "
                f"{other.kind.__name__}."
            )
        kinds = {self.end_effector, other.end_effector} - {None}
        if len(kinds) > 1 and "any" in kinds:
            kinds.discard("any")
        if len(kinds) > 1:
            raise RequirementError(
                f"One role cannot need both a {' and a '.join(sorted(kinds))}."
            )
        return Needs(
            kind=self.kind,
            observes=self.observes | other.observes,
            commands=self.commands | other.commands,
            end_effector=next(iter(kinds), None),
        )


def combine(*needs: Mapping[str, Needs]) -> dict[str, Needs]:
    """Merge role-keyed needs from several sources."""
    merged: dict[str, Needs] = {}
    for mapping in needs:
        for role, need in mapping.items():
            merged[role] = merged[role] | need if role in merged else need
    return merged


@dataclass(frozen=True)
class Bound:
    """The parts filling one role, by their dotted paths on the robot."""

    part: str
    end_effector: Optional[str] = None


def at(tree: Mapping[str, Any], path: str) -> Any:
    """Follow a dotted path down a nested reading or feature tree."""
    node: Any = tree
    for key in path.split("."):
        node = node[key]
    return node


def nest(path: str, value: Any) -> dict[str, Any]:
    """Wrap ``value`` in one dict per segment of a dotted path."""
    for key in reversed(path.split(".")):
        value = {key: value}
    return value


def merge(target: dict[str, Any], branch: Mapping[str, Any]) -> dict[str, Any]:
    """Merge a nested action branch into ``target`` in place."""
    for key, value in branch.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            merge(target[key], value)
        else:
            target[key] = value
    return target


class Parts:
    """The arm and end effector filling each role, on one robot."""

    def __init__(self, robot: Robot, bound: Mapping[str, Bound]) -> None:
        self.robot = robot
        self.bound = dict(bound)

    @property
    def roles(self) -> tuple[str, ...]:
        """The roles bound, in the order the task named them."""
        return tuple(self.bound)

    def part(self, role: str) -> RobotPart:
        """The part filling ``role``."""
        return self.robot.parts_of_type(RobotPart)[self.bound[role].part]

    def arm(self, role: str = "arm") -> Arm:
        """The arm filling ``role``."""
        return self.robot.parts_of_type(Arm)[self.bound[role].part]

    def end_effector(self, role: str = "arm") -> Optional[EndEffector]:
        """The end effector filling ``role``, or ``None`` when it has none."""
        path = self.bound[role].end_effector
        if path is None:
            return None
        return self.robot.parts_of_type(EndEffector)[path]

    def read(self, raw: Mapping[str, Any]) -> "Reading":
        """View one whole-robot reading by role."""
        return Reading(raw, self.bound)


class Reading:
    """One whole-robot reading, looked up by role instead of by path."""

    def __init__(self, raw: Mapping[str, Any], bound: Mapping[str, Bound]) -> None:
        self.raw = raw
        self._bound = bound

    def part(self, role: str) -> Mapping[str, Any]:
        """The fields the part filling ``role`` reported."""
        return at(self.raw, self._bound[role].part)

    def arm(self, role: str = "arm") -> Mapping[str, Any]:
        """The fields the arm filling ``role`` reported."""
        return self.part(role)

    def end_effector(self, role: str = "arm") -> Optional[Mapping[str, Any]]:
        """What the end effector filling ``role`` reported, if it has one."""
        path = self._bound[role].end_effector
        return None if path is None else at(self.raw, path)


def bind(robot: Robot, needs: Mapping[str, Needs], *, owner: str) -> Parts:
    """Find the parts filling each role and check them against ``needs``.

    A role binds to the part of its kind at the role's own path (``"arm"``,
    ``"base"``), then at the one path below it (``"left.arm"``), then to the
    robot's only part of that kind. An arm's end effector is the one riding
    it, as on a shared bus, or the one beside it in the same group, as a Franka
    Hand on its own endpoint.

    Args:
        robot: The composed robot, connected or not.
        needs: Role-keyed needs of the task and control together.
        owner: Who is asking, for the error message.

    Raises:
        RequirementError: Naming every role that falls short, with what was
            missing and what the part offers.
    """
    effectors = robot.parts_of_type(EndEffector)
    bound: dict[str, Bound] = {}
    problems: list[str] = []
    for role, need in needs.items():
        candidates = robot.parts_of_type(need.kind)
        path = _part_path(role, candidates)
        if path is None:
            problems.append(
                f"role {role!r} needs a {need.kind.__name__}; the robot has "
                f"{sorted(candidates) or 'none'}"
            )
            continue
        part = candidates[path]
        observed = part.observation_features
        accepted = part.action_features
        missing_obs = sorted(need.observes - set(observed))
        if missing_obs:
            problems.append(
                f"{path} ({type(part).__name__}) does not report {missing_obs}; "
                f"it reports {sorted(observed)}"
            )
        missing_cmd = sorted(need.commands - set(accepted))
        if missing_cmd:
            problems.append(
                f"{path} ({type(part).__name__}) does not accept {missing_cmd}; "
                f"it accepts {sorted(accepted)}"
            )
        # A field's name is not enough: a part whose numbers mean something
        # else is refused here rather than misread every step.
        for name in sorted(need.observes - set(missing_obs)):
            differs = mismatch(name, observed.get(name))
            if differs:
                problems.append(f"{path} ({type(part).__name__}) {differs}")
        for name in sorted(need.commands - set(missing_cmd)):
            differs = mismatch(name, accepted.get(name))
            if differs:
                problems.append(f"{path} ({type(part).__name__}) {differs}")
        effector_path = _effector_path(path, effectors)
        if need.end_effector is not None:
            effector = effectors.get(effector_path) if effector_path else None
            if effector is None:
                problems.append(f"{path} carries no {need.end_effector}")
            elif not _is_kind(effector, need.end_effector):
                problems.append(
                    f"{effector_path} ({type(effector).__name__}) is not a "
                    f"{need.end_effector}"
                )
        bound[role] = Bound(part=path, end_effector=effector_path)
    if problems:
        raise RequirementError(
            f"{owner} cannot run on {type(robot).__name__}: " + "; ".join(problems)
        )
    return Parts(robot, bound)


def _part_path(role: str, candidates: Mapping[str, RobotPart]) -> Optional[str]:
    if role in candidates:
        return role
    below = [path for path in candidates if path.rpartition(".")[0] == role]
    if len(below) == 1:
        return below[0]
    return next(iter(candidates)) if len(candidates) == 1 else None


def _effector_path(
    part_path: str, effectors: Mapping[str, EndEffector]
) -> Optional[str]:
    riding = [path for path in effectors if path.startswith(f"{part_path}.")]
    if riding:
        return riding[0]
    group = part_path.rpartition(".")[0]
    beside = [path for path in effectors if path.rpartition(".")[0] == group]
    return beside[0] if len(beside) == 1 else None


def _is_kind(effector: EndEffector, kind: str) -> bool:
    if kind == "any":
        return True
    return bool(effector.is_gripper if kind == "gripper" else effector.is_hand)
