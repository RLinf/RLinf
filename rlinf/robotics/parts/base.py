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

"""Core abstractions for robot composition and hardware connections.

``Connection`` manages one hardware session and its placement. ``RobotPart``
adds observations, ``ControllablePart`` adds actions, and ``PartGroup`` builds a
named tree from those parts. ``Connection.parts`` describes components backed
by one session; ``RobotPart.children`` describes the public robot tree.
"""

from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from functools import partial
from importlib import import_module
from typing import (
    Any,
    Callable,
    ClassVar,
    Optional,
    TypeVar,
    overload,
)

from rlinf.utils.logging import get_logger

_KeyType = TypeVar("_KeyType")
_ValueType = TypeVar("_ValueType")

#: The category a registry call was made on, so ``Arm.backend("franky")`` is a
#: ``type[Arm]`` and ``Camera.backend("zed")`` a ``type[Camera]`` -- rather than
#: a bare ``type``, which an editor cannot follow anywhere.
DriverType = TypeVar("DriverType", bound="Connection")

#: Bound where a driver class is decorated, so registration returns that
#: class rather than the category it registers into.
RegisteredDriver = TypeVar("RegisteredDriver", bound="Connection")
RobotPartType = TypeVar("RobotPartType", bound="RobotPart")

#: What a part promises about one reading or one command: each name mapped to
#: its ``{"shape": ..., "dtype": ...}``, or to ``{}`` where the driver says only
#: that the name exists. A carrier's features also carry its riders' under
#: their names, so this nests.
Features = dict[str, Any]

#: One reading, by the names :attr:`observation_features` declared. Values are
#: whatever the device measures -- usually a numpy array, and a nested reading
#: for each part riding this one.
Observation = dict[str, Any]

#: One command, by the names :attr:`action_features` declared, shaped like the
#: observation it answers.
Action = Mapping[str, Any]


@dataclass(frozen=True)
class RemoteInfo:
    """Information required to rebuild a connection on another node."""

    connection_cls: type
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    node_rank: Optional[int] = None
    worker_name: Optional[str] = None


class ConnectionMeta(ABCMeta):
    """Remove placement arguments before invoking a driver constructor.

    ``Connection.__new__`` cannot perform this step because Python passes the
    original keyword arguments to ``__init__`` after ``__new__`` returns.
    """

    def __call__(cls, *args: Any, **kwargs: Any) -> "Connection":
        """Construct a connection and retain its remote reconstruction data."""
        node_rank = worker_name = None
        if cls._TAKES_PLACEMENT:
            node_rank = kwargs.pop("node_rank", None)
            worker_name = kwargs.pop("worker_name", None)
        connection = super().__call__(*args, **kwargs)
        connection._remote_info = RemoteInfo(
            cls, args, dict(kwargs), node_rank, worker_name
        )
        return connection


class Connection(ABC, metaclass=ConnectionMeta):
    """A deferred local or remote hardware connection.

    Construction records the driver arguments and optional ``node_rank`` but
    must not open hardware. Implement :meth:`_open` and :meth:`_release` for
    device-specific lifecycle behavior; do not override :meth:`connect` or
    :meth:`disconnect`, which also manage remote placement.

    Subclass ``Connection`` directly when one session backs several logical
    parts but is not itself observable. Expose those parts through
    :attr:`parts` and select them with :meth:`part`. Observable connections
    should subclass :class:`RobotPart`.
    """

    #: Active vendor handle, or ``None`` while disconnected.
    _device: Any = None
    _state_snapshot: "Optional[list[Any]]" = None

    #: Whether the metaclass should consume placement arguments.
    _TAKES_PLACEMENT: ClassVar[bool] = True

    #: Information needed to rebuild this connection on another node.
    _remote_info: Optional[RemoteInfo] = None

    #: Worker group hosting this connection while it is placed remotely.
    _group: Any = None

    #: Original class restored after a remote connection closes.
    _local_cls: Optional[type] = None

    #: Connection that owns this borrowed part's lifecycle.
    _owner: Optional["Connection"] = None

    #: Cached child parts for stable identity across tree traversals.
    _beneath: "Optional[dict[str, RobotPart]]" = None

    def _open(self) -> Any:
        """Open the hardware and return its vendor handle."""
        raise NotImplementedError(
            f"{type(self).__name__} does not say how to open its hardware. "
            "Implement _open(), or override connect() for a part whose "
            "lifecycle is more than opening a device."
        )

    def _release(self, device: Any) -> None:
        """Release the handle returned by :meth:`_open`."""

    def _opened(self) -> None:
        """Run category-level setup after the device opens."""

    def _closing(self) -> None:
        """Undo :meth:`_opened`, while the device is still open."""

    @property
    def is_connected(self) -> bool:
        """Whether this part or its owning connection is open."""
        if self._owner is not None:
            return self._owner.is_connected
        return self._device is not None

    @property
    def node_rank(self) -> Optional[int]:
        """Return the target node, or ``None`` for the current process."""
        return self._remote_info.node_rank if self._remote_info else None

    @property
    def owner(self) -> "Connection":
        """Return the connection responsible for this part's lifecycle.

        Parts with independent hardware own themselves. Borrowed parts resolve
        recursively to the shared connection that opens them.
        """
        return self._owner.owner if self._owner is not None else self

    def connect(self) -> None:
        """Open this connection on its configured node.

        Local connections retain the handle returned by :meth:`_open`. Remote
        connections are rebuilt in a scheduler worker, and this object becomes
        a forwarding view. Repeated calls are ignored.
        """
        if self._owner is not None:
            self._owner.connect()
            return
        if self._device is not None:
            return
        if self._remote_info is None or self._remote_info.node_rank is None:
            device = self._open()
            # Preserve valid false-valued handles such as file descriptor 0.
            self._device = self if device is None else device
            try:
                self._opened()
            except BaseException:
                # Release the device if post-open setup fails.
                try:
                    self._release(self._device)
                finally:
                    self._device = None
                raise
            return

        from ..placement import abandon, host

        group, view = host(self)
        try:
            # A worker is created asynchronously, so one whose device refused
            # to open is still on its way down here. Waiting surfaces that,
            # rather than reporting a robot that was never opened.
            group._is_ready().wait()
        except BaseException:
            abandon(group)
            raise
        self._group = group
        self._local_cls = type(self)
        self._device = group
        self.__class__ = view

    def disconnect(self) -> None:
        """Close this connection and restore its local class.

        Borrowed parts leave cleanup to their owner. Repeated calls are ignored.
        """
        if self._owner is not None:
            return
        device = self._device
        if device is None:
            return
        try:
            if self._group is None:
                # Always release the device, even if category teardown fails.
                try:
                    self._closing()
                finally:
                    self._release(device)
            else:
                from ..placement import shutdown

                self.__class__ = self._local_cls
                try:
                    shutdown(self._group)
                finally:
                    self._group = None
                    self._local_cls = None
        finally:
            self._device = None

    def reset(self) -> None:
        """Reset this connection when it has resettable state."""

    # Driver registry

    @classmethod
    def register(
        cls, *names: str
    ) -> "Callable[[type[RegisteredDriver]], type[RegisteredDriver]]":
        """Register a driver in this device category.

        Names are case-insensitive, and a name cannot refer to two different
        drivers. Give one name unless callers genuinely reach the driver by
        two, as grippers are: by the short backend name and by the
        :class:`EndEffectorType` value.

        Example::

            @EndEffector.register("franka", "franka_gripper")
            class FrankaGripper(BaseGripper): ...
        """

        def add(driver_cls: "type[RegisteredDriver]") -> "type[RegisteredDriver]":
            # Store the registry on the category, not on Connection.
            if "_BACKENDS" not in cls.__dict__:
                cls._BACKENDS = {}
            for name in names:
                key = name.lower()
                taken = cls._BACKENDS.get(key)
                if taken is not None and taken is not driver_cls:
                    raise ValueError(
                        f"{cls.__name__} backend {name!r} is already registered "
                        f"to {taken.__name__}; {driver_cls.__name__} cannot take it."
                    )
                cls._BACKENDS[key] = driver_cls
            return driver_cls

        return add

    @classmethod
    def backends(cls: "type[DriverType]") -> "dict[str, type[DriverType]]":
        """Return registered drivers keyed by backend name."""
        merged: dict[str, type] = {}
        for base in reversed(cls.__mro__):
            merged.update(base.__dict__.get("_BACKENDS", {}))
        return merged

    @classmethod
    def backend(cls: "type[DriverType]", name: str) -> "type[DriverType]":
        """Resolve a backend by name, or return the available names."""
        registered = cls.backends()
        driver_cls = registered.get(str(name).lower())
        if driver_cls is None:
            raise ValueError(
                f"Unsupported {cls.__name__} backend {name!r}. "
                f"Registered: {sorted(registered)}."
            )
        return driver_cls

    # Local device discovery.

    #: Module this driver imports from its vendor SDK.
    SDK: ClassVar[Optional[str]] = None

    @classmethod
    def discover(cls) -> set[str]:
        """Return identifiers for compatible devices attached to this machine.

        The default returns an empty set. Drivers may return serial numbers,
        stable device paths, or another identifier accepted by their config.
        """
        return set()

    def read_state(self) -> Any:
        """This connection's state, shared for the duration of one read.

        A part and the riders it carries all want the same reading, and on a
        serial bus each extra fetch costs a control cycle. While a snapshot is
        open the first call reaches the hardware and the rest are served from
        it. Outside a snapshot every call reaches the hardware, so a caller
        that wants fresh state always gets it.
        """
        if self._state_snapshot is None:
            return self.get_state()
        if not self._state_snapshot:
            # First reader inside the snapshot pays for the hardware read.
            self._state_snapshot.append(self.get_state())
        return self._state_snapshot[0]

    def read_subtree(self, kind: str) -> dict[str, Any]:
        """Read this part and the riders it carries, in one pass.

        Placed remotely this is a single call, so the whole subtree is read on
        the node holding the hardware and the riders share one read of it.

        Args:
            kind: A key of :pyattr:`PartGroup.SUBTREE_READS`.
        """
        return PartGroup._read_part(self, kind)

    @contextmanager
    def _snapshot_state(self) -> "Iterator[None]":
        """Serve one hardware read to everything read inside this block.

        Private, so it never joins the surface a remote view forwards: a
        context manager cannot be held open across a process boundary. A
        remote part snapshots on its own node, inside :meth:`read_subtree`.
        """
        if self._state_snapshot is not None or not hasattr(self, "get_state"):
            # Already inside a snapshot, or nothing to snapshot.
            yield
            return
        # Opened empty and filled on first use, so a caller that only wants
        # metadata never reaches the hardware at all.
        self._state_snapshot = []
        try:
            yield
        finally:
            self._state_snapshot = None

    @classmethod
    def require_sdk(cls, where: str = "this machine") -> None:
        """Raise unless the vendor library this driver needs is installed.

        Args:
            where: Machine label included in dependency errors.
        """
        if cls.SDK is None:
            return
        try:
            import_module(cls.SDK)
        except ModuleNotFoundError as missing:
            raise ModuleNotFoundError(
                f"{cls.__name__} needs the {cls.SDK!r} module, which is not "
                f"installed on {where}. Install this robot's dependencies "
                "with requirements/install.sh."
            ) from missing

    # Composition

    @property
    def parts(self) -> dict[str, "RobotPart"]:
        """Return logical parts backed by this hardware connection.

        These driver-local names do not become public robot paths until the
        parts are composed through :class:`PartGroup`.
        """
        return {}

    def part(self, name: str) -> "RobotPart":
        """Return a named part backed by this connection.

        Use this method when a non-observable connection backs several logical
        parts::

            session = Turtle2Connection(50, node_rank=0)
            left = PartGroup(
                arm=session.part("left"),
                end_effector=session.part("left_end_effector"),
            )

        Parts without their own lifecycle are assigned to this connection.
        Independently connected parts retain their existing owner and placement.
        """
        available = self.parts
        if name not in available:
            raise KeyError(
                f"{type(self).__name__} backs no part {name!r}. "
                f"Available: {sorted(available)}."
            )
        return self._adopt(available[name])

    def _adopt(self, part: "RobotPart") -> "RobotPart":
        """Assign this connection as owner when ``part`` has no lifecycle."""
        if part is not self and part._owner is None and not part._opens_itself():
            part._owner = self
        return part

    @classmethod
    def _opens_itself(cls) -> bool:
        """Whether this class reaches hardware of its own."""
        return cls._open is not Connection._open


class RobotPart(Connection):
    """An observable hardware connection that can enter a robot tree."""

    @property
    @abstractmethod
    def observation_features(self) -> Features:
        """Describe the values returned by :meth:`get_observation`."""

    @property
    def children(self) -> "dict[str, RobotPart]":
        """Return the named parts mounted below this part.

        The mapping is built once to preserve child identity across lifecycle,
        traversal, observation, and action operations.
        """
        if self._beneath is not None:
            return self._beneath
        beneath: "dict[str, RobotPart]" = {}
        for name, part in self.parts.items():
            if part is self:
                raise TypeError(
                    f"{type(self).__name__} lists itself in parts as {name!r}. "
                    "The mapping may contain only parts exported by this "
                    "connection. Remove the self-reference."
                )
            beneath[name] = self._adopt(part)
        self._refuse_shadowed_fields(beneath)
        self._beneath = beneath
        return beneath

    def _refuse_shadowed_fields(self, beneath: "dict[str, RobotPart]") -> None:
        """Reject child names that collide with this part's data fields."""
        try:
            mine = set(self.observation_features) | set(
                getattr(self, "action_features", {}) or {}
            )
        except Exception:  # noqa: BLE001 - a contract that needs hardware
            # Defer hardware-dependent feature checks to the conformance suite.
            return
        shadowed = sorted(set(beneath) & mine)
        if shadowed:
            raise ValueError(
                f"{type(self).__name__} backs parts named {shadowed}, which "
                "are also its own observation or action fields. A carrier's "
                "reading holds both in one mapping, so the field would vanish "
                "and its action would go to the part instead. Rename the part."
            )

    @overload
    def child(self, name: str) -> "RobotPart": ...

    @overload
    def child(self, name: str, part_type: "type[RobotPartType]") -> "RobotPartType": ...

    def child(
        self, name: str, part_type: "Optional[type[RobotPartType]]" = None
    ) -> "RobotPart":
        """Return a child part by name.

        Pass *part_type* to say what the caller expects the part to be. The
        return is typed as that class, so an editor resolves its methods, and a
        part that turns out to be something else is reported here rather than
        as a missing attribute further along.

        Args:
            name: Name the part was composed under.
            part_type: Class the part is expected to implement.

        Raises:
            KeyError: If no child has that name.
            TypeError: If the part does not implement *part_type*.
        """
        available = self.children
        if name not in available:
            raise KeyError(
                f"{type(self).__name__} has no part {name!r}. "
                f"Available: {sorted(available)}."
            )
        part = available[name]
        if part_type is not None and not isinstance(part, part_type):
            raise TypeError(
                f"{type(self).__name__} part {name!r} is a "
                f"{type(part).__name__}, not the {part_type.__name__} the "
                "caller expects. A remotely placed part answers as a "
                "synthesized subclass of its own class, so this names a real "
                "mismatch rather than placement."
            )
        return part

    @abstractmethod
    def get_observation(self) -> Observation:
        """Read the current part observation."""


class ControllablePart(RobotPart):
    """A robot part that accepts actions as well as observations."""

    @property
    @abstractmethod
    def action_features(self) -> Features:
        """Describe the values accepted by :meth:`send_action`."""

    @abstractmethod
    def send_action(self, action: Action) -> Observation:
        """Apply an action and return the action actually sent."""


def _close_all(owners: "Sequence[Connection]", doing: str) -> None:
    """Close connections in reverse order and report all cleanup failures.

    Cleanup continues after an error. The final exception is re-raised after
    all connections have been processed.
    """
    failures: list[BaseException] = []
    for owner in reversed(list(owners)):
        try:
            owner.disconnect()
        except BaseException as error:  # noqa: BLE001 - reported below
            failures.append(error)
            get_logger().exception(
                "%s: %s failed to close; continuing with the rest",
                doing,
                type(owner).__name__,
            )
    if failures:
        raise failures[-1]


class PartGroup(ControllablePart):
    """A named subtree composed from robot parts.

    Child names define public observation and action paths. A group accepts
    ``RobotPart`` or nested ``PartGroup`` instances::

        PartGroup(
            arm=ExampleArm("10.0.0.2"),
            left=PartGroup(camera=...),
        )

    Bare connections must first expose a readable part through
    :meth:`Connection.part`. Operations run concurrently across independent
    connections and sequentially within a shared connection.
    """

    #: Groups are composed rather than placed as a single connection.
    _TAKES_PLACEMENT: ClassVar[bool] = False

    def __init__(
        self,
        parts: "Optional[Mapping[str, RobotPart]]" = None,
        **named: "RobotPart",
    ) -> None:
        combined: "dict[str, RobotPart]" = {**(parts or {}), **named}
        if any(not name or not isinstance(name, str) for name in combined):
            raise ValueError(
                f"{type(self).__name__} part names must be non-empty strings."
            )
        for name, value in combined.items():
            self._check_composable(name, value)
        self._children: "dict[str, RobotPart]" = combined

    def _check_composable(self, name: str, value: Any) -> None:
        """Validate one value before adding it to the part tree."""
        if isinstance(value, RobotPart):
            return
        if isinstance(value, Connection):
            # Include a valid selection in the error when one is available.
            backed = sorted(value.parts)
            example = repr(backed[0]) if backed else '"arm"'
            offers = f" It backs {backed}." if backed else ""
            raise TypeError(
                f"{type(self).__name__} cannot compose {name}="
                f"{type(value).__name__}: it backs parts without being one of "
                f"them, so there is nothing to read.{offers} Pick one instead, "
                f"as in {name}=<{type(value).__name__.lower()}>.part({example})."
            )
        raise TypeError(
            f"{type(self).__name__} cannot compose {name}="
            f"{type(value).__name__}: a robot is made of parts. Pass a "
            "RobotPart, another PartGroup, or one part picked out of a "
            "connection with .part(...)."
        )

    @property
    def children(self) -> "dict[str, RobotPart]":
        """Return the parts supplied when this group was composed."""
        return self._children

    @property
    def owner(self) -> "Connection":
        """Raise because a group may span multiple connections."""
        raise TypeError(
            f"{type(self).__name__} is composed of parts and rides no "
            "connection itself, so it has no single owner. Use "
            "group.child(<name>).owner for one part or owners() for the group."
        )

    @property
    def is_connected(self) -> bool:
        """Whether every connection this group's parts ride on is open."""
        return all(owner.is_connected for owner in self.owners())

    @property
    def observation_features(self) -> Features:
        """Describe each part's observation under its name."""
        return {
            name: PartGroup._read_part(part, "observation_features")
            for name, part in self._children.items()
        }

    @property
    def action_features(self) -> Features:
        """Describe each controllable part's action under its name."""
        return {
            name: PartGroup._read_part(part, "action_features")
            for name, part in self._children.items()
            if isinstance(part, ControllablePart)
        }

    def owners(self) -> list["Connection"]:
        """Return distinct owning connections in declaration order.

        Connections are deduplicated by identity.
        """
        seen: dict[int, Connection] = {}
        for part in self._children.values():
            for owner in self._owners_of(part):
                seen.setdefault(id(owner), owner)
        return list(seen.values())

    #: What each kind of subtree read collects from a part, and which riders
    #: contribute to it. A key travels to another node in place of a callable,
    #: so a remote part can read its own subtree.
    SUBTREE_READS: ClassVar[
        dict[
            str,
            tuple[
                "Callable[[RobotPart], Mapping[str, Any]]",
                "Optional[Callable[[RobotPart], bool]]",
            ],
        ]
    ] = {
        "observation": (lambda part: part.get_observation(), None),
        "observation_features": (lambda part: part.observation_features, None),
        "action_features": (
            lambda part: part.action_features,
            lambda part: isinstance(part, ControllablePart),
        ),
    }

    @staticmethod
    def _read_part(part: "RobotPart", kind: str) -> dict[str, Any]:
        """Read a part and recursively nest its children.

        Args:
            part: The part to read.
            kind: A key of :pyattr:`PartGroup.SUBTREE_READS`, naming what to
                collect and which riders contribute.
        """
        # A remote part reads its own subtree, so one call crosses the process
        # boundary and the riders still share one read of the hardware.
        if part._group is not None and not isinstance(part, PartGroup):
            return part.read_subtree(kind)

        call, include = PartGroup.SUBTREE_READS[kind]
        # A group rides no connection of its own and refuses to name one, so
        # there is nothing to snapshot here; its children snapshot themselves
        # as they are reached.
        owner = None if isinstance(part, PartGroup) else part.owner
        snapshot = (
            owner._snapshot_state() if isinstance(owner, Connection) else nullcontext()
        )
        # The part and its riders share one hardware read.
        with snapshot:
            reading = dict(call(part))
            for name, rider in part.children.items():
                if include is None or include(rider):
                    reading[name] = PartGroup._read_part(rider, kind)
        return reading

    @staticmethod
    def _command_part(part: "RobotPart", action: "Mapping[str, Any]") -> dict[str, Any]:
        """Apply a part's fields and route child branches recursively."""
        riders = part.children
        own = {name: value for name, value in action.items() if name not in riders}
        applied: dict[str, Any] = dict(part.send_action(own)) if own else {}
        for name, value in action.items():
            if name not in riders:
                continue
            rider = riders[name]
            if not isinstance(rider, ControllablePart):
                raise TypeError(f"Part {name!r} is not controllable.")
            applied[name] = PartGroup._command_part(rider, value)
        return applied

    def _batches(self) -> list[list[str]]:
        """Group children whose connection sets overlap.

        Each group runs sequentially, while disjoint groups run concurrently.
        This prevents concurrent access to a shared vendor session, including
        sessions shared across nested groups.
        """
        names = list(self._children)
        touched = [
            frozenset(id(owner) for owner in self._owners_of(self._children[name]))
            for name in names
        ]
        # Union-find over children that share at least one connection.
        parent = list(range(len(names)))

        def root(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for left in range(len(names)):
            for right in range(left + 1, len(names)):
                if touched[left] & touched[right]:
                    parent[root(right)] = root(left)

        order: list[list[str]] = []
        index_of: dict[int, int] = {}
        for position, name in enumerate(names):
            group = root(position)
            if group not in index_of:
                index_of[group] = len(order)
                order.append([])
            order[index_of[group]].append(name)
        return order

    @staticmethod
    def _owners_of(part: "RobotPart") -> list["Connection"]:
        """Return all connections required by a part and its descendants.

        Descending into children includes independently owned mounted devices.
        """
        if isinstance(part, PartGroup):
            return part.owners()
        found = [part.owner]
        for rider in part.children.values():
            found.extend(PartGroup._owners_of(rider))
        return found

    @staticmethod
    def _run_parallel(
        jobs: "Mapping[_KeyType, Callable[[], _ValueType]]",
    ) -> "dict[_KeyType, _ValueType]":
        """Run independent part operations concurrently."""
        if len(jobs) <= 1:
            return {key: job() for key, job in jobs.items()}
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = {key: executor.submit(job) for key, job in jobs.items()}
            return {key: future.result() for key, future in futures.items()}

    def _fan_out(self, call: "Callable[[RobotPart], Any]") -> dict[str, Any]:
        """Run *call* over every part, concurrently where connections differ."""

        def run(names: list[str]) -> dict[str, Any]:
            return {name: call(self._children[name]) for name in names}

        batches = self._batches()
        results = self._run_parallel(
            {position: partial(run, names) for position, names in enumerate(batches)}
        )
        merged: dict[str, Any] = {}
        for position in range(len(batches)):
            merged.update(results[position])
        return merged

    def connect(self) -> None:
        """Open every owning connection and roll back partial startup."""
        opened: list[Connection] = []
        try:
            for owner in self.owners():
                if not owner.is_connected:
                    owner.connect()
                    opened.append(owner)
        except BaseException:
            _close_all(opened, "rolling back a failed connect")
            raise

    def disconnect(self) -> None:
        """Close every open connection in reverse declaration order."""
        _close_all(
            [owner for owner in self.owners() if owner.is_connected], "disconnecting"
        )

    def reset(self) -> None:
        """Reset every part."""
        self._fan_out(lambda part: part.reset())

    def get_observation(self) -> Observation:
        """Read observations into the named part tree."""
        return self._fan_out(lambda part: self._read_part(part, "observation"))

    def send_action(self, action: Action) -> Observation:
        """Dispatch each named action to the part that owns it."""
        unknown = set(action) - set(self._children)
        if unknown:
            raise KeyError(
                f"{type(self).__name__} has no parts {sorted(unknown)}; "
                f"available: {sorted(self._children)}."
            )
        not_controllable = [
            name
            for name in action
            if not isinstance(self._children[name], ControllablePart)
        ]
        if not_controllable:
            raise TypeError(f"Parts {sorted(not_controllable)} are not controllable.")

        requested = dict(action)
        batches = [[n for n in names if n in requested] for names in self._batches()]

        def run(names: list[str]) -> dict[str, Any]:
            return {
                name: self._command_part(self._children[name], requested[name])
                for name in names
            }

        results = self._run_parallel(
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
