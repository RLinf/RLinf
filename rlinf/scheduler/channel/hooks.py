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

"""Customization hooks for :class:`~rlinf.scheduler.channel.Channel`.

A channel makes exactly two decisions about the data flowing through it:

1. **What lands in a queue.** By default a put enqueues its item verbatim under
   the key it arrived with. A :class:`Collector` may transform it instead,
   emitting any number of items across any number of keys.
2. **Which consumer receives it.** By default every consumer of a key competes
   for one shared queue. A :class:`Dispatcher` may assign each item to a
   specific consumer, giving each one a private queue.

Both hooks are plain synchronous functions over ordinary Python data -- no Ray,
no asyncio, no worker addresses -- and both ship with defaults that reproduce
the channel's behavior exactly. Implement one, register it by name, and select
it when the channel is created.

Example:
    >>> from rlinf.scheduler.channel import Collector, register_collector
    >>>
    >>> @register_collector("drop_empty")
    ... class DropEmptyCollector(Collector):
    ...     def collect(self, item, key):
    ...         if item:
    ...             yield key, item
"""

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator

__all__ = [
    "ChannelContext",
    "Collector",
    "Dispatcher",
    "LeastLoadedDispatcher",
    "LeastLoadedStealingDispatcher",
    "RoundRobinDispatcher",
    "SharedDispatcher",
    "register_collector",
    "register_dispatcher",
    "resolve_collector",
    "resolve_dispatcher",
]


@dataclass(frozen=True)
class ChannelContext:
    """Read-only description of the channel a hook is attached to.

    Passed to :meth:`Collector.setup` and :meth:`Dispatcher.setup` once, on the
    channel worker, before any traffic flows.

    Attributes:
        name: The channel's name.
        cfg: The run configuration, if one was supplied at channel creation.
        producers: Worker group names expected to put into this channel.
        consumers: Consumer ids expected to get from it, each ``"<group>:<rank>"``.
        options: The hook's own free-form options, supplied at channel creation.
    """

    name: str
    cfg: Any = None
    producers: tuple[str, ...] = ()
    consumers: tuple[str, ...] = ()
    options: dict[str, Any] = field(default_factory=dict)


class Collector:
    """Transform items on the way into a channel.

    Subclass and override :meth:`collect` to reshape, split, merge, filter, or
    re-key items before they are queued. The base implementation passes items
    through unchanged, which is the channel's default behavior.
    """

    def setup(self, ctx: ChannelContext) -> None:
        """Prepare the collector. Called once before any item is collected.

        Args:
            ctx: Description of the channel this collector is attached to.
        """

    def collect(self, item: Any, key: str) -> Iterable[tuple[str, Any]]:
        """Turn one incoming item into the items that should be queued.

        Args:
            item: The item as it was put into the channel.
            key: The key it was put under.

        Yields:
            ``(key, item)`` pairs to enqueue. May yield nothing to drop the
            input, or several pairs to fan it out.
        """
        yield key, item


class Dispatcher:
    """Decide which consumer receives each item.

    The base implementation returns ``None`` for everything, meaning all
    consumers of a key share one queue -- the channel's default behavior.
    Override :meth:`route` to hand each item to a specific consumer instead,
    which gives every consumer a private queue and makes non-blocking polling
    fair by construction.
    """

    def setup(self, ctx: ChannelContext) -> None:
        """Prepare the dispatcher. Called once before any item is routed.

        Args:
            ctx: Description of the channel this dispatcher is attached to.
        """

    @property
    def deals(self) -> bool:
        """Whether this dispatcher gives each consumer a private queue.

        True as soon as :meth:`route` is overridden. Override this property to
        force it either way.
        """
        return type(self).route is not Dispatcher.route

    def route(self, item: Any, key: str) -> str | None:
        """Choose the consumer that should receive an item.

        Args:
            item: The item about to be queued.
            key: The key it will be queued under.

        Returns:
            A consumer id (``"<group>:<rank>"``), or ``None`` to leave the item
            in the key's shared queue for whichever consumer asks first.
        """
        return None

    def rebalance(self, key: str, starved: str, depths: dict[str, int]) -> str | None:
        """Choose a consumer to take work from when one has run dry.

        Only consulted for a blocking get on an empty private queue.

        Args:
            key: The key being read.
            starved: The consumer id with nothing left to read.
            depths: Current queue depth of every consumer on this key.

        Returns:
            The consumer id to take one item from, or ``None`` to wait instead.
        """
        return None


class SharedDispatcher(Dispatcher):
    """Leave every item in the key's shared queue. The channel default."""


class _DealingDispatcher(Dispatcher):
    """Base for dispatchers that assign each item to one consumer.

    Routing state is kept per queue key. Two streams sharing a channel then
    balance independently, so a key whose item count is not a multiple of the
    consumer count cannot skew how a different key is split -- which matters
    because a consumer waiting on a fixed number of items would otherwise be
    left short.
    """

    def __init__(self) -> None:
        """Initialize empty routing state."""
        self._consumers: list[str] = []
        self._assigned: dict[Any, dict[str, int]] = {}

    def setup(self, ctx: ChannelContext) -> None:
        """Record the consumers items may be dealt to."""
        self._consumers = sorted(ctx.consumers)
        self._assigned = {}

    def counts(self, key: Any) -> dict[str, int]:
        """Return this key's per-consumer counts, seeding any newcomer at zero."""
        counts = self._assigned.setdefault(key, {})
        for consumer in self._consumers:
            counts.setdefault(consumer, 0)
        return counts

    def observe(self, consumer: str) -> None:
        """Register a consumer discovered after setup."""
        if consumer not in self._consumers:
            self._consumers = sorted([*self._consumers, consumer])


class RoundRobinDispatcher(_DealingDispatcher):
    """Deal items to consumers in a fixed rotation."""

    def __init__(self) -> None:
        """Initialize the per-key rotation cursors."""
        super().__init__()
        self._cursor: dict[Any, int] = {}

    def route(self, item: Any, key: str) -> str | None:
        """Return the next consumer in this key's rotation."""
        if not self._consumers:
            return None
        counts = self.counts(key)
        cursor = self._cursor.get(key, 0)
        consumer = self._consumers[cursor % len(self._consumers)]
        self._cursor[key] = cursor + 1
        counts[consumer] += 1
        return consumer


class LeastLoadedDispatcher(_DealingDispatcher):
    """Deal each item to the consumer that has been given the fewest so far.

    Ties break on consumer id, so an even producer stream splits exactly evenly
    and reproducibly across consumers.
    """

    def route(self, item: Any, key: str) -> str | None:
        """Return the consumer least loaded on this key."""
        if not self._consumers:
            return None
        counts = self.counts(key)
        consumer = min(self._consumers, key=lambda name: (counts[name], name))
        counts[consumer] += 1
        return consumer


class LeastLoadedStealingDispatcher(LeastLoadedDispatcher):
    """Least-loaded dealing that also lets an idle consumer take from a peer.

    Use when consumers legitimately progress at different speeds and a stalled
    consumer must not hold back the others.
    """

    def rebalance(self, key: str, starved: str, depths: dict[str, int]) -> str | None:
        """Return the deepest peer queue, if any peer has work to spare."""
        peers = {
            consumer: depth
            for consumer, depth in depths.items()
            if consumer != starved and depth > 0
        }
        if not peers:
            return None
        return max(peers, key=lambda name: (peers[name], name))


COLLECTOR_REGISTRY: dict[str, type[Collector]] = {}
DISPATCHER_REGISTRY: dict[str, type[Dispatcher]] = {
    "shared": SharedDispatcher,
    "round_robin": RoundRobinDispatcher,
    "least_loaded": LeastLoadedDispatcher,
    "least_loaded_stealing": LeastLoadedStealingDispatcher,
}

HookSpec = "str | Collector | Dispatcher | type | None"


def register_collector(name: str) -> Callable[[type], type]:
    """Register a collector class under a name.

    Args:
        name: The name channels may select this collector by.

    Returns:
        A class decorator that registers and returns the class unchanged.
    """

    def decorator(cls: type) -> type:
        key = name.lower()
        if key in COLLECTOR_REGISTRY:
            raise ValueError(f"Collector '{name}' is already registered.")
        COLLECTOR_REGISTRY[key] = cls
        return cls

    return decorator


def register_dispatcher(name: str) -> Callable[[type], type]:
    """Register a dispatcher class under a name.

    Args:
        name: The name channels may select this dispatcher by.

    Returns:
        A class decorator that registers and returns the class unchanged.
    """

    def decorator(cls: type) -> type:
        key = name.lower()
        if key in DISPATCHER_REGISTRY:
            raise ValueError(f"Dispatcher '{name}' is already registered.")
        DISPATCHER_REGISTRY[key] = cls
        return cls

    return decorator


def _import_hook(path: str) -> type:
    """Import a hook class from a ``module:ClassName`` path."""
    module_name, _, class_name = path.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ValueError(f"Cannot import module '{module_name}' for hook.") from error
    hook_cls = getattr(module, class_name, None)
    if hook_cls is None:
        raise ValueError(f"Module '{module_name}' has no attribute '{class_name}'.")
    return hook_cls


def _resolve(
    spec: Any, registry: dict[str, type], base: type, default: type, kind: str
) -> Any:
    """Turn a hook spec into an instance.

    Accepts an instance, a class, a registered name, or a ``module:Class`` path.
    ``None`` yields the default hook.
    """
    if spec is None:
        return default()
    if isinstance(spec, base):
        return spec
    if isinstance(spec, str):
        if ":" in spec:
            hook_cls = _import_hook(spec)
        elif spec.lower() in registry:
            hook_cls = registry[spec.lower()]
        else:
            raise ValueError(
                f"{kind} '{spec}' is not registered. Available: "
                f"{sorted(registry)}. Pass 'module:ClassName' to use one that "
                f"lives outside RLinf."
            )
    elif isinstance(spec, type):
        hook_cls = spec
    else:
        raise TypeError(f"Cannot resolve {kind} from {type(spec)}.")

    if not issubclass(hook_cls, base):
        raise TypeError(f"{hook_cls.__name__} is not a {base.__name__} subclass.")
    return hook_cls()


def resolve_collector(spec: Any) -> Collector:
    """Turn a collector spec into a :class:`Collector` instance.

    Args:
        spec: A ``Collector`` instance, a ``Collector`` subclass, a registered
            name, a ``"module:ClassName"`` import path, or ``None`` for the
            pass-through default.

    Returns:
        The resolved collector.
    """
    return _resolve(spec, COLLECTOR_REGISTRY, Collector, Collector, "Collector")


def resolve_dispatcher(spec: Any) -> Dispatcher:
    """Turn a dispatcher spec into a :class:`Dispatcher` instance.

    Args:
        spec: A ``Dispatcher`` instance, a ``Dispatcher`` subclass, a registered
            name, a ``"module:ClassName"`` import path, or ``None`` for the
            shared-queue default.

    Returns:
        The resolved dispatcher.
    """
    return _resolve(
        spec, DISPATCHER_REGISTRY, Dispatcher, SharedDispatcher, "Dispatcher"
    )


def iter_collected(
    collector: Collector, item: Any, key: str
) -> Iterator[tuple[str, Any]]:
    """Run a collector and validate the pairs it produces.

    Args:
        collector: The collector to run.
        item: The item that was put into the channel.
        key: The key it was put under.

    Yields:
        Validated ``(key, item)`` pairs.
    """
    for produced in collector.collect(item, key):
        if not isinstance(produced, tuple) or len(produced) != 2:
            raise TypeError(
                f"{type(collector).__name__}.collect must yield (key, item) "
                f"pairs, got {produced!r}."
            )
        yield produced
