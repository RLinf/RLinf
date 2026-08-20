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

from abc import ABC, abstractmethod
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

if TYPE_CHECKING:
    from ..placement import PartHandle
    from ..placement.specs import PartSpec

KeyType = TypeVar("KeyType")
ValueType = TypeVar("ValueType")


def run_parallel(
    jobs: Mapping[KeyType, Callable[[], ValueType]],
) -> dict[KeyType, ValueType]:
    """Run independent part operations concurrently, keyed by component name.

    Parts on separate connections do not contend, so reading or commanding
    several of them costs one round trip rather than the sum. A single job runs
    inline to keep the common one-arm case free of thread overhead.
    """
    if len(jobs) <= 1:
        return {key: job() for key, job in jobs.items()}
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {key: executor.submit(job) for key, job in jobs.items()}
        return {key: future.result() for key, future in futures.items()}


class Endpoint(ABC):
    """Something the robot opens on a machine and later closes.

    An endpoint has two properties and nothing else: a location, so it can be
    declared here and built there with :meth:`at`, and a lifecycle, so
    :meth:`_open` reaches the hardware and :meth:`_release` lets it go.
    Connecting and disconnecting are handled here, once, so an endpoint is
    written by saying what its hardware is rather than by re-implementing a
    lifecycle.

    Being observable is a separate matter, and it is what :class:`RobotPart`
    adds. A camera and the ROS link that drives four other components are both
    opened and closed the same way; only one of them means anything when you
    read it.

    An endpoint with a lifecycle of its own -- an arm that must home before it
    is usable -- may override :meth:`connect` and :meth:`disconnect` instead.
    """

    #: The vendor object this part talks to, or ``None`` before it is opened.
    _device: Any = None

    def _open(self) -> Any:
        """Reach the hardware and return whatever speaks to it."""
        raise NotImplementedError(
            f"{type(self).__name__} does not say how to open its hardware. "
            "Implement _open(), or override connect() for a part whose "
            "lifecycle is more than opening a device."
        )

    def _release(self, device: Any) -> None:
        """Let ``device`` go. The default has nothing to release.

        The handle is passed in rather than read back off ``self``, so an
        implementation cannot be defeated by the order ``disconnect`` does
        things in. That is not hypothetical: clearing ``_device`` first left
        every teleop device closing nothing.
        """

    @property
    def is_connected(self) -> bool:
        """Whether the part is ready for observations."""
        return self._device is not None

    def connect(self) -> None:
        """Connect to the physical part, once.

        A part that opens in place rather than returning a handle still counts
        as connected, so ``_open`` may return nothing.
        """
        if self._device is None:
            self._device = self._open() or self

    def disconnect(self) -> None:
        """Release resources owned by the endpoint, once.

        ``_release`` runs while the handle is still there, because that is what
        it releases. Clearing it first left every implementation that reaches
        for ``self._device`` -- which is where the vendor object is documented
        to live -- closing nothing, and the endpoint reported itself
        disconnected while the reader, its thread and its serial port stayed
        open.
        """
        device = self._device
        if device is None:
            return
        try:
            self._release(device)
        finally:
            self._device = None

    def reset(self) -> None:
        """Reset this endpoint when it has resettable state."""

    # -- Composition ------------------------------------------------------

    @property
    def kind(self) -> str:
        """This endpoint's narrowest kind, as a remote proxy mirrors it.

        Named rather than derived at each call site, because the proxy on the
        other side of a placement has to rebuild the same interface from a
        description that crossed a process boundary.
        """
        for kind, endpoint_type in _PART_KINDS:
            if isinstance(self, endpoint_type):
                return kind
        raise TypeError(f"{type(self).__name__} is not an Endpoint.")

    @property
    def exports(self) -> dict[str, "RobotPart"]:
        """The capabilities this endpoint makes available. A leaf has none.

        One ROS link drives two arms and their grippers; one arm carries the
        gripper on its own connection. Either way the endpoint says what rides
        on it, and a robot picks the ones it wants and names them.

        This is not composition. What a robot is *made of* is
        :attr:`Group.children`, and the two were one property for long enough
        to be worth saying: exports belong to a hardware session, children
        belong to a tree of names.
        """
        return {}

    def export(self, name: str) -> "RobotPart":
        """Return one exported capability, or say what is on offer."""
        exports = self.exports
        if name not in exports:
            raise KeyError(
                f"{type(self).__name__} exports no {name!r}. "
                f"Available: {sorted(exports)}."
            )
        return exports[name]

    # -- Subpart-addressed surface ----------------------------------------
    # Public, so a hosted part exposes these as RPCs automatically and one
    # generic proxy can reach any subpart.

    def describe_self(self) -> dict[str, Any]:
        """Describe this part itself: its kind and its feature dictionaries.

        A leaf part -- a camera, a gripper on its own port -- exposes no
        subparts, so this is what a remote handle proxies.
        """
        described: dict[str, Any] = {"kind": self.kind}
        if isinstance(self, RobotPart):
            described["observation"] = self.observation_features
        if isinstance(self, ControllablePart):
            described["action"] = self.action_features
        return described

    def describe_parts(self) -> dict[str, dict[str, Any]]:
        """Describe every part: its kind and its feature dictionaries.

        One call carries everything a remote handle needs to build correctly
        typed proxies, so placement costs a single round trip rather than one
        per subpart per property.
        """
        described: dict[str, dict[str, Any]] = {}
        for name, part in self.exports.items():
            entry: dict[str, Any] = {
                "kind": part.kind,
                "observation": part.observation_features,
            }
            if isinstance(part, ControllablePart):
                entry["action"] = part.action_features
            described[name] = entry
        return described

    def part_observation(self, name: str) -> dict[str, Any]:
        """Read one part's observation."""
        return self.export(name).get_observation()

    def part_action(self, name: str, action: dict[str, Any]) -> dict[str, Any]:
        """Send an action to one controllable part."""
        part = self.export(name)
        if not isinstance(part, ControllablePart):
            raise TypeError(
                f"Part {name!r} of {type(self).__name__} is not controllable."
            )
        return part.send_action(action)

    def part_reset(self, name: str) -> None:
        """Reset one part."""
        self.export(name).reset()

    def part_reopen(self, name: str) -> None:
        """Reopen one part, for a camera that stalled behind a proxy."""
        part = self.export(name)
        reopen = getattr(part, "reopen", None)
        if not callable(reopen):
            raise TypeError(
                f"Part {name!r} of {type(self).__name__} cannot be reopened."
            )
        reopen()

    def shutdown(self) -> None:
        """Disconnect during worker teardown."""
        if self.is_connected:
            self.disconnect()

    # -- Placement --------------------------------------------------------

    @classmethod
    def at(
        cls,
        *args: Any,
        node_rank: Optional[int] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> "PartSpec":
        """Declare this part and the node it runs on, without building it.

        Compose the result exactly where you would compose a part;
        :meth:`Robot.connect` places it. Prefer this over :meth:`spawn`: a part
        whose SDK lives only on the target machine cannot be built here first,
        and declaring lets the robot own teardown and roll back cleanly.
        """
        from ..placement.specs import PartSpec

        return PartSpec(cls, args, kwargs, node_rank=node_rank, name=name)

    @classmethod
    def spawn(
        cls,
        *args: Any,
        node_rank: Optional[int] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> "PartHandle":
        """Construct and connect this part, here or on a chosen node.

        This is the eager, low-level form and it hands you the handle to manage.
        Prefer :meth:`at` inside a robot, which defers placement to
        :meth:`Robot.connect` and lets the robot own the handle.

        With ``node_rank`` unset the part is built in this process. Otherwise it
        is hosted in a scheduler worker on that node and the returned handle
        proxies to it. Both handles expose the same API, so callers never branch
        on placement.

        Any part can be placed, not only arms: a camera can run on the machine
        it is plugged into while the policy runs elsewhere.

        The scheduler is imported here rather than at module scope, so importing
        a part never pulls Ray into the process.
        """
        from ..placement import LocalPartHandle, PartWorkerHost

        if node_rank is None:
            part = cls(*args, **kwargs)
            part.connect()
            return LocalPartHandle(part)

        return PartWorkerHost(cls, args, kwargs, node_rank=node_rank, name=name).spawn()


class RobotPart(Endpoint):
    """An endpoint you can read: a component of the robot itself.

    An arm, a camera, a gripper. What separates a part from a bare
    :class:`Endpoint` is that reading it means something, so a part says what
    it observes and answers with it. Everything else -- declaring it, placing
    it, opening and closing it -- it gets from being an endpoint.
    """

    @property
    @abstractmethod
    def observation_features(self) -> dict[str, Any]:
        """Describe the values returned by :meth:`get_observation`."""

    @abstractmethod
    def get_observation(self) -> dict[str, Any]:
        """Read the current part observation."""


class ControllablePart(RobotPart):
    """Robot part that accepts commands in addition to observations."""

    @property
    @abstractmethod
    def action_features(self) -> dict[str, Any]:
        """Describe the values accepted by :meth:`send_action`."""

    @abstractmethod
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Apply an action and return the action actually sent."""


class EndEffector(ControllablePart):
    """Controllable tool attached to an arm, such as a gripper or hand."""


class Camera(RobotPart):
    """Observation-only camera part."""


class Group(ControllablePart):
    """A part made of named parts.

    Names are the composition: they become the keys of the observation and the
    action, and the path a policy sees. A group holds whatever you give it, so
    an arm, a torso, or a whole robot is the same construct with different
    names.

    Reads fan out across parts that sit on different connections. Parts sharing
    one connection are read and commanded in their declared order, because a
    vendor SDK behind a single link is rarely safe to call concurrently.
    """

    def __init__(self, parts: Optional[Mapping[str, Any]] = None, **named: Any) -> None:
        combined = {**(parts or {}), **named}
        if any(not name or not isinstance(name, str) for name in combined):
            raise ValueError(
                f"{type(self).__name__} part names must be non-empty strings."
            )
        self._parts: dict[str, Any] = combined
        self._handle_of: dict[str, int] = {}
        """Which connection each part came from, so sharing is respected."""

    @property
    def children(self) -> dict[str, Any]:
        """The parts this group is composed of, by the names it gave them."""
        return self._parts

    def child(self, name: str) -> Any:
        """Return one composed part, or say which names exist."""
        if name not in self._parts:
            raise KeyError(
                f"{type(self).__name__} has no part {name!r}. "
                f"Available: {sorted(self._parts)}."
            )
        return self._parts[name]

    @property
    def is_connected(self) -> bool:
        """Whether every part is placed and connected."""
        from ..placement.specs import PartSpec, SubpartRef

        values = list(self._parts.values())
        if any(isinstance(v, (PartSpec, SubpartRef)) for v in values):
            return False
        return all(part.is_connected for part in values)

    @property
    def observation_features(self) -> dict[str, Any]:
        """Describe each part's observation under its name."""
        return {name: part.observation_features for name, part in self._parts.items()}

    @property
    def action_features(self) -> dict[str, Any]:
        """Describe each controllable part's action under its name."""
        return {
            name: part.action_features
            for name, part in self._parts.items()
            if isinstance(part, ControllablePart)
        }

    def _batches(self) -> list[list[str]]:
        """Group part names by connection: distinct ones may run together."""
        order: list[list[str]] = []
        index: dict[int, int] = {}
        for position, name in enumerate(self._parts):
            key = self._handle_of.get(name, -position - 1)
            if key in index:
                order[index[key]].append(name)
            else:
                index[key] = len(order)
                order.append([name])
        return order

    def _fan_out(self, call) -> dict[str, Any]:
        """Run *call* over every part, concurrently where connections differ."""

        def run(names: list[str]) -> dict[str, Any]:
            return {name: call(self._parts[name]) for name in names}

        batches = self._batches()
        results = run_parallel(
            {position: partial(run, names) for position, names in enumerate(batches)}
        )
        merged: dict[str, Any] = {}
        for position in range(len(batches)):
            merged.update(results[position])
        return merged

    def connect(self) -> None:
        """Connect every part, rolling back the ones already connected."""
        connected: list[RobotPart] = []
        try:
            for part in self._parts.values():
                if not part.is_connected:
                    part.connect()
                    connected.append(part)
        except Exception:
            for part in reversed(connected):
                part.disconnect()
            raise

    def disconnect(self) -> None:
        """Disconnect every connected part, in reverse order.

        Disconnecting restores the declarations, so a second call walks a tree
        holding those rather than live parts. A declaration has nothing to
        disconnect -- and asking it raised, which is exactly the wrong thing to
        do in the teardown path a ``finally`` block takes when it is not sure
        whether the robot came up.
        """
        from ..placement.specs import PartSpec, SubpartRef

        for part in reversed(list(self._parts.values())):
            if isinstance(part, (PartSpec, SubpartRef)):
                continue
            if isinstance(part, Group) or part.is_connected:
                part.disconnect()

    def reset(self) -> None:
        """Reset every part."""
        self._fan_out(lambda part: part.reset())

    def get_observation(self) -> dict[str, Any]:
        """Read every part, namespaced by name."""
        return self._fan_out(lambda part: part.get_observation())

    def send_action(self, action: Mapping[str, Any]) -> dict[str, Any]:
        """Dispatch each named action to the part that owns it."""
        unknown = set(action) - set(self._parts)
        if unknown:
            raise KeyError(
                f"{type(self).__name__} has no parts {sorted(unknown)}; "
                f"available: {sorted(self._parts)}."
            )
        not_controllable = [
            name
            for name in action
            if not isinstance(self._parts[name], ControllablePart)
        ]
        if not_controllable:
            raise TypeError(f"Parts {sorted(not_controllable)} are not controllable.")

        requested = dict(action)
        batches = [[n for n in names if n in requested] for names in self._batches()]

        def run(names: list[str]) -> dict[str, Any]:
            return {
                name: self._parts[name].send_action(requested[name]) for name in names
            }

        results = run_parallel(
            {
                position: partial(run, names)
                for position, names in enumerate(batches)
                if names
            }
        )
        applied: dict[str, Any] = {}
        for value in results.values():
            applied.update(value)
        return applied

    def resolve(self, placement: Any) -> dict[str, list[Any]]:
        """Replace declared parts with the parts they resolve to.

        Returns the handles used, keyed by the part name that needed them, so a
        robot can publish them and so sharing is visible to :meth:`_batches`.
        """
        from ..placement.specs import PartSpec, SubpartRef

        used: dict[str, list[Any]] = {}
        for name, value in list(self._parts.items()):
            if isinstance(value, (PartSpec, SubpartRef)):
                spec = value.spec if isinstance(value, SubpartRef) else value
                handle = placement.resolve_handle(spec)
                used[name] = [handle]
                self._handle_of[name] = id(handle)
                self._parts[name] = placement.resolve(value)
            elif isinstance(value, Group):
                nested = value.resolve(placement)
                flat = [h for handles in nested.values() for h in handles]
                if flat:
                    used[name] = flat
                    self._handle_of[name] = id(flat[0])
        return used

    def declarations(self) -> dict[str, Any]:
        """Snapshot the parts as composed, so placement can be undone."""
        return {
            name: value.declarations() if isinstance(value, Group) else value
            for name, value in self._parts.items()
        }

    def restore(self, declared: Mapping[str, Any]) -> None:
        """Put every part back to what it was composed with."""
        for name, value in declared.items():
            current = self._parts.get(name)
            if isinstance(value, dict) and isinstance(current, Group):
                current.restore(value)
            else:
                self._parts[name] = value
        self._handle_of.clear()


class Connection(Endpoint):
    """One link that backs several parts without being one of them.

    A ROS node, a CAN bus or an SDK session often drives more than one
    component: two arms, their grippers, and the wrist cameras on one link.
    Such a connection has a location and a lifecycle, which is what makes it an
    :class:`Endpoint`, but reading it would mean nothing -- :attr:`exports`
    says what it backs, and a robot composes those parts.

    It is deliberately not a :class:`RobotPart`. A connection that had to be
    one could only answer ``get_observation`` by refusing, or by repeating what
    its parts already say; being a sibling instead means a robot cannot compose
    it into the tree by accident, and there is nothing to refuse.

    A connection that *is* one component, like a ROS link to a single arm, is
    an ordinary part that lists itself in :attr:`exports`. The distinction is
    whether an observation of the whole thing means anything.
    """


class MobileBase(ControllablePart):
    """Controllable wheeled or tracked base."""


class LeggedBase(ControllablePart):
    """Controllable legged base."""


#: Ordered most specific first, so a part matches its narrowest kind.
_PART_KINDS: tuple[tuple[str, type], ...] = (
    ("end_effector", EndEffector),
    ("camera", Camera),
    ("controllable", ControllablePart),
    ("part", RobotPart),
    ("connection", Connection),
)
