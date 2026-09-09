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

"""Scheduler-backed placement for robotics connections.

A connection with ``node_rank`` is rebuilt in a worker on that node. The local
object becomes a synthesized subclass that forwards public methods and
properties while retaining its original type relationships. Lifecycle and
composition methods remain local.
"""

import inspect
from typing import Any, Callable, ClassVar, Optional
from uuid import uuid4

from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.scheduler.worker.worker import WorkerMeta
from rlinf.scheduler.worker.worker_group import WorkerGroup
from rlinf.utils.logging import get_logger

from ..parts.base import Connection, ConnectionMeta

#: Methods and properties that remain on the local connection view.
_STAYS_LOCAL: frozenset[str] = frozenset(
    {
        "connect",
        "disconnect",
        "is_connected",
        "owner",
        "node_rank",
        "part",
        "parts",
        "children",
        "child",
    }
)

#: Public RPC used to read properties from the hosted connection.
_ATTRIBUTE_RPC = "attribute"


def _first(result: Any) -> Any:
    """Unwrap a one-worker result, rejecting groups sized for more."""
    values = result.wait()
    if len(values) != 1:
        raise RuntimeError(
            f"A part must be hosted by exactly one worker, got {len(values)} results."
        )
    return values[0]


def _forwarding_method(name: str) -> Callable[..., Any]:
    """Create a method that forwards calls to the worker."""

    def call(self: Any, *args: Any, **kwargs: Any) -> Any:
        return _first(getattr(self._group, name)(*args, **kwargs))

    call.__name__ = name
    call.__qualname__ = name
    call.__doc__ = f"Run ``{name}`` on the node holding this connection."
    return call


def _forwarding_property(name: str) -> property:
    """Create a property that reads an attribute from the worker."""

    def read(self: Any) -> Any:
        return _first(getattr(self._group, _ATTRIBUTE_RPC)(name))

    read.__name__ = name
    read.__doc__ = f"``{name}`` as the node holding this connection reports it."
    return property(read)


def _public_surface(part_cls: type) -> tuple[list[str], list[str]]:
    """Return public instance methods and properties to forward.

    Local placement methods, class methods, and static methods are excluded.
    """
    methods: list[str] = []
    properties: list[str] = []
    for name in dir(part_cls):
        if name.startswith("_") or name in _STAYS_LOCAL:
            continue
        attribute = inspect.getattr_static(part_cls, name, None)
        if isinstance(attribute, property):
            properties.append(name)
        elif isinstance(attribute, (classmethod, staticmethod)):
            continue
        elif callable(getattr(part_cls, name, None)):
            methods.append(name)
    return methods, properties


class _RemoteViewMeta(WorkerMeta, ConnectionMeta):
    """Reconcile the metaclasses a synthesized view inherits."""


#: Cached synthesized view for each connection class.
_VIEWS: dict[type, type] = {}


def remote_view_of(part_cls: "type[Connection]") -> "type[Connection]":
    """Return the cached remote-view subclass for ``part_cls``.

    The generated subclass forwards public instance methods and properties,
    while preserving ``isinstance`` behavior.
    """
    cached = _VIEWS.get(part_cls)
    if cached is not None:
        return cached

    methods, properties = _public_surface(part_cls)
    namespace: dict[str, Any] = {
        "__module__": part_cls.__module__,
        "__qualname__": f"Remote{part_cls.__name__}",
        "__doc__": f"Remote view of :class:`{part_cls.__name__}`.",
    }
    for name in methods:
        namespace[name] = _forwarding_method(name)
    for name in properties:
        namespace[name] = _forwarding_property(name)

    view = _RemoteViewMeta(f"Remote{part_cls.__name__}", (part_cls,), namespace)
    _VIEWS[part_cls] = view
    return view


def _refuse_collisions(part_cls: type, methods: dict[str, Any]) -> None:
    """Reject connection methods that conflict with the worker API."""
    taken = {name for name in dir(Worker) if not name.startswith("_")}
    taken.add(_ATTRIBUTE_RPC)
    clashing = sorted(set(methods) & taken)
    if clashing:
        raise TypeError(
            f"{part_cls.__name__} cannot be placed on a node: its methods "
            f"{clashing} share a name with the worker that would host it. "
            "Rename them, or keep them private."
        )


class PartWorkerHost:
    """Create and launch a scheduler worker for one connection.

    Args:
        part_cls: The class to construct on the target node.
        args: Positional arguments for its constructor.
        kwargs: Keyword arguments for its constructor.
        node_rank: Cluster node rank that is physically wired to the device.
        worker_name: Worker-group name. Defaults to :meth:`default_name`.
    """

    #: Cached worker subclass for each hosted connection class.
    _worker_classes: ClassVar[dict[type, type]] = {}

    def __init__(
        self,
        part_cls: type,
        args: tuple[Any, ...] = (),
        kwargs: Optional[dict[str, Any]] = None,
        *,
        node_rank: int,
        worker_name: Optional[str] = None,
    ) -> None:
        self.part_cls = part_cls
        self.args = args
        self.kwargs = kwargs or {}
        self.node_rank = node_rank
        self.worker_name = worker_name or self.default_name(part_cls, node_rank)

    @staticmethod
    def default_name(part_cls: type, node_rank: int) -> str:
        """Return a unique worker name for a hosted connection."""
        return f"{part_cls.__name__}-node{node_rank}-{uuid4().hex[:8]}"

    @classmethod
    def worker_class(cls, part_cls: "type[Connection]") -> "type[Worker]":
        """Return (and cache) the ``Worker`` subclass that hosts ``part_cls``."""
        cached = cls._worker_classes.get(part_cls)
        if cached is not None:
            return cached

        def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
            Worker.__init__(self)
            part_cls.__init__(self, *args, **kwargs)
            self.connect()

        def attribute(self: Any, name: str) -> Any:
            """Read a property from the hosted connection."""
            return getattr(self, name)

        namespace: dict[str, Any] = {
            "__init__": __init__,
            attribute.__name__: attribute,
            "__module__": part_cls.__module__,
            "__qualname__": f"{part_cls.__name__}Worker",
            "__doc__": f"Scheduler host for :class:`{part_cls.__name__}`.",
        }

        # WorkerMeta only wraps methods declared directly on the worker class.
        methods = {
            name: func
            for name, func in inspect.getmembers(part_cls, inspect.isfunction)
            if not name.startswith("_")
        }
        _refuse_collisions(part_cls, methods)
        namespace.update(methods)

        worker_cls = _RemoteViewMeta(
            f"{part_cls.__name__}Worker", (Worker, part_cls), namespace
        )
        cls._worker_classes[part_cls] = worker_cls
        return worker_cls

    def launch(self) -> "WorkerGroup":
        """Start the worker and return the group that reaches it."""
        return (
            self.worker_class(self.part_cls)
            .create_group(*self.args, **self.kwargs)
            .launch(
                cluster=Cluster(),
                placement_strategy=NodePlacementStrategy(node_ranks=[self.node_rank]),
                name=self.worker_name,
            )
        )


def host(connection: Connection) -> "tuple[WorkerGroup, type[Connection]]":
    """Host a connection remotely and return its worker group and view class."""
    remote_info = connection._remote_info
    if remote_info is None or remote_info.node_rank is None:
        raise ValueError("Remote hosting requires a connection with a node rank.")
    group = PartWorkerHost(
        remote_info.connection_cls,
        remote_info.args,
        remote_info.kwargs,
        node_rank=remote_info.node_rank,
        worker_name=remote_info.worker_name,
    ).launch()
    return group, remote_view_of(remote_info.connection_cls)


def abandon(group: "WorkerGroup") -> None:
    """Tear down a group whose worker never became usable.

    Used while rolling back a failed connect, so a teardown problem must not
    replace the failure being reported.
    """
    close = getattr(group, "_close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as teardown:
        get_logger().warning(
            "Could not release the worker for a connection that failed to open: %s",
            teardown,
        )


def shutdown(group: "WorkerGroup") -> None:
    """Disconnect the hosted device before terminating its worker."""
    try:
        group.disconnect().wait()
    finally:
        close = getattr(group, "_close", None)
        if callable(close):
            close()
