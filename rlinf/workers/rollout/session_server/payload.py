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

"""Payload data, collector, and channel placement for the session server.

A session server writes each generate response into a payload channel as a
:class:`PayloadPart` under the group's payload key. The channel's collector
(:class:`PayloadCollector`) accumulates the parts and emits one
:class:`GroupPayload` per finished group; assembly from parts into training
sequences stays on the read side. A voided group is erased from the collector
by a :class:`PayloadDiscard` on the same key. :class:`PayloadWriter` holds
the writing side: the puts, their settlements, and the pending-byte budget.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterator, Optional

from ray.exceptions import RayActorError

from rlinf.scheduler.channel import Channel, Collector, register_collector
from rlinf.scheduler.collective import PeerCommunicationError
from rlinf.utils.logging import get_logger

logger = get_logger()

# Rough per-element byte cost of a serialized part's token and logprob lists;
# used for the pending-bytes backpressure, which needs an estimate, not an
# exact size.
_BYTES_PER_LIST_ELEMENT = 8
_PART_OVERHEAD_BYTES = 256

__all__ = [
    "PayloadPart",
    "PayloadDiscard",
    "GroupPayload",
    "PayloadCollector",
    "PayloadWriter",
    "routing_key_for",
    "payload_key_for",
    "create_payload_channels",
    "payload_locator_for",
    "payload_channel_for",
]


@dataclass
class PayloadPart:
    """One generate call's engine view, or the finalize record of a group.

    The engine-view fields (``input_ids``, ``output_ids``, ``logprobs``,
    ``version_start``, ``weight_version``, ``finish_reason``) record what the
    engine saw and produced; trimming them to what the agent actually used
    happens on the read side, from the declared verdicts. The finalize part of
    a group carries ``last=True`` together with the group's ``rewards`` and
    ``verdicts``; every other part has ``last=False`` and leaves the three
    verdict fields empty.

    Attributes:
        group_id: Group this part belongs to.
        traj_index: Trajectory within the group.
        turn_index: Generate call within the trajectory, 0-based.
        input_ids: Token ids sent to the engine, as the engine saw them.
        output_ids: Token ids the engine produced.
        logprobs: Per-token logprobs of ``output_ids``, engine view, or None
            when the response carried none.
        version_start: Lower bound on the weight versions that may have
            produced this output.
        weight_version: Weight version the engine reported for this response
            ("default" is the pre-write sentinel; the session server rejects
            it before building a part).
        finish_reason: The engine's finish reason, e.g. ``{"type": "stop"}``.
        prev_kept_len: Kept length the agent declared for the previous turn
            when it declares per turn, else None.
        last: True only on the finalize part.
        rewards: Group rewards, on the finalize part only.
        verdicts: Verdicts indexed by traj_index, on the finalize part only:
            kept lengths, continuity flags, the agent's per-turn is_end flags,
            and each
            trajectory's slice of the training extra fields.
        verify: Assembly-verification record, on the finalize part only and
            only in verify mode: the reference result fields the agent
            reported at close, both sides' per-turn sampling params.
    """

    group_id: int
    traj_index: int
    turn_index: int
    input_ids: list
    output_ids: list
    logprobs: Optional[list]
    version_start: Any
    weight_version: Any
    finish_reason: dict
    prev_kept_len: Optional[int] = None
    last: bool = False
    rewards: Optional[list] = None
    verdicts: Optional[dict] = None
    verify: Optional[dict] = None


@dataclass
class PayloadDiscard:
    """Marker that voids a group's accumulated payload parts.

    A session server puts it on the group's payload key when the group is
    discarded. It carries no data: reaching the collector, it drops that key's
    accumulated parts without emitting anything, so a half-finished group
    leaves no payload behind.
    """

    group_id: int


@dataclass
class GroupPayload:
    """All parts of one group, emitted once its finalize part lands.

    A light container: it holds the parts in the order they were put and does
    no merging, counting, or task-level interpretation -- assembly is the read
    side's job.
    """

    group_id: int
    parts: list[PayloadPart]


@register_collector("payload")
class PayloadCollector(Collector):
    """Accumulate a group's parts on the payload channel.

    Parts are accumulated per key and held until the finalize part (``last``)
    arrives, at which point the group's parts are emitted as one
    :class:`GroupPayload` and the key's state is dropped. A
    :class:`PayloadDiscard` drops the key's state and emits nothing. The
    channel serializes same-key puts, so the accumulation needs no locking.
    """

    def __init__(self) -> None:
        """Initialize with no pending groups."""
        self._pending: dict[Any, list[PayloadPart]] = {}

    def collect(self, item: Any, key: Any) -> Iterator[tuple[Any, Any]]:
        """Fold one part into its key's state.

        Args:
            item: A :class:`PayloadPart`, or a :class:`PayloadDiscard` to
                void the key.
            key: The key the item was put under.

        Yields:
            ``(key, GroupPayload)`` when the item finalizes the group;
            nothing otherwise.

        Raises:
            TypeError: If the item is neither a PayloadPart nor a
                PayloadDiscard.
        """
        if isinstance(item, PayloadDiscard):
            self._pending.pop(key, None)
            return
        if not isinstance(item, PayloadPart):
            raise TypeError(
                f"PayloadCollector accepts PayloadPart or PayloadDiscard, "
                f"got {type(item).__name__} under key {key!r}."
            )
        parts = self._pending.setdefault(key, [])
        parts.append(item)
        if item.last:
            del self._pending[key]
            yield key, GroupPayload(group_id=item.group_id, parts=parts)


class PayloadWriter:
    """Writes the parts of every group to the payload channels.

    Owns the writes, their settlements, and the pending-byte budget. The
    caller keeps the group state: a failed settlement is reported through
    the ``on_write_failure`` callback, which voids the group and reports
    the discard. The caller also keeps the per-group settlement tasks it
    waits for at finalize; this class tracks them only to cancel them at
    shutdown.
    """

    def __init__(
        self,
        payload_channels: list[Channel],
        max_pending_bytes: int,
        put_timeout_s: float,
        on_write_failure: Callable[[int], Awaitable[None]],
        on_peer_failure: Callable[[BaseException], None],
    ):
        """
        Args:
            payload_channels: The same-node payload channels, one per shard.
                Every write names the shard it goes to, so a group's parts
                stay in one channel process.
            max_pending_bytes: Budget for bytes still being written. Past it,
                each response waits for its own write instead of queueing
                more, which slows the writers but bounds memory.
            put_timeout_s: Limit on one write's settlement. A write that
                times out may still land in the channel later, so the failure
                it reports is never cleared again.
            on_write_failure: Called with a group id when one of its writes
                fails or times out.
            on_peer_failure: Called with the error when a write failed because
                the payload channel worker died or its collective direction
                failed. That channel carries nothing more, so the caller stops
                serving instead of failing group after group.
        """
        self._channels = list(payload_channels)
        self._max_pending_bytes = max_pending_bytes
        self._put_timeout_s = put_timeout_s
        self._on_write_failure = on_write_failure
        self._on_peer_failure = on_peer_failure
        self._pending_bytes = 0
        self._tasks: set[asyncio.Task] = set()

    @property
    def num_shards(self) -> int:
        """Number of payload channels this writer writes to."""
        return len(self._channels)

    @property
    def pending_bytes(self) -> int:
        """Bytes put but not yet known to have landed."""
        return self._pending_bytes

    def channel_for(self, shard: int) -> Channel:
        """The payload channel of one shard."""
        return self._channels[shard]

    async def emit_part(
        self, shard: int, part: PayloadPart
    ) -> tuple[int, Optional[asyncio.Task]]:
        """Put one part in the background, with backpressure.

        The put is issued synchronously here (issuance order is the queue
        order); its settlement runs as a task, or is awaited inline once the
        pending byte budget is exceeded. Every-turn parts carry no weight:
        the byte estimate rides on the finalize part's put, the queue element
        the runner's payload_queued_bytes gauge sums.

        Returns:
            The part's estimated size, and its settlement task when the
            settlement runs in the background. The task is None when the
            settlement was awaited inline, in which case the caller's next
            step already sees its outcome.
        """
        nbytes = _estimate_part_bytes(part)
        work = self.channel_for(shard).put(
            part, key=payload_key_for(part.group_id), async_op=True
        )
        self._pending_bytes += nbytes
        if self._pending_bytes > self._max_pending_bytes:
            await self._settle_put(part.group_id, nbytes, work)
            return nbytes, None
        task = asyncio.create_task(self._settle_put(part.group_id, nbytes, work))
        self._track(task)
        return nbytes, task

    async def put(
        self,
        shard: int,
        group_id: int,
        item: Any,
        await_completion: bool,
        weight: int = 0,
    ) -> bool:
        """Put a payload-channel item under the group's key.

        Args:
            shard: The payload channel the item goes to.
            group_id: Group whose key the item is put under.
            item: A :class:`PayloadPart` (the finalize part, awaited) or a
                :class:`PayloadDiscard` marker (background).
            await_completion: True to await the put's settlement and return
                its success; False to monitor it in the background and
                return True.
            weight: Weight handed to the put; the finalize part carries the
                group's byte estimate so the runner's payload_queued_bytes
                gauge sums it.

        Returns:
            The settlement result when awaited, True when backgrounded.
        """
        work = self.channel_for(shard).put(
            item, weight=weight, key=payload_key_for(group_id), async_op=True
        )
        if await_completion:
            return await self._settle_put(group_id, 0, work)
        task = asyncio.create_task(self._settle_put(group_id, 0, work))
        self._track(task)
        return True

    async def remove_key(self, shard: int, group_id: int) -> None:
        """Drop the group's payload queue, after everything already issued.

        Chains on the same key as the puts (AsyncChannelWork order), so this
        runs after the finalize part and discard marker already queued. It
        matters when a finalize part landed although its settlement timed
        out: the collector emitted a container no one will fetch.
        """
        try:
            work = self.channel_for(shard).remove_key(
                key=payload_key_for(group_id), async_op=True
            )
            await asyncio.wait_for(work.async_wait(), timeout=self._put_timeout_s)
        except BaseException as exc:
            logger.warning(
                f"remove_key for group {group_id} failed ({exc!r}); its "
                f"payload queue may hold an unfetched container."
            )

    async def aclose(self) -> None:
        """Cancel the settlements still running."""
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    def _track(self, task: asyncio.Task) -> None:
        """Track a settlement task so shutdown can cancel it."""
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _settle_put(self, group_id: int, nbytes: int, work: Any) -> bool:
        """Wait for one payload put to settle.

        Returns:
            True if the put completed successfully, False if it failed or
            timed out. A timed-out put may still land in the channel later,
            so the failure this reports is never cleared again.
        """
        success = True
        peer_failure = None
        try:
            await asyncio.wait_for(work.async_wait(), timeout=self._put_timeout_s)
        except asyncio.CancelledError:
            raise
        except (PeerCommunicationError, RayActorError) as exc:
            # The channel worker died or its collective direction failed, so
            # this is not one group's bad luck: every later put fails the
            # same way.
            success = False
            peer_failure = exc
            logger.error(
                f"Payload put reached a failed channel peer for group "
                f"{group_id} ({exc!r})."
            )
        except BaseException as exc:
            success = False
            logger.error(f"Payload put failed for group {group_id} ({exc!r}).")
        finally:
            self._pending_bytes -= nbytes
        if peer_failure is not None:
            self._on_peer_failure(peer_failure)
        elif not success:
            await self._on_write_failure(group_id)
        return success


def _estimate_part_bytes(part: PayloadPart) -> int:
    """Rough byte cost of a part's serialized token and logprob lists."""
    nbytes = (len(part.input_ids) + len(part.output_ids)) * _BYTES_PER_LIST_ELEMENT
    if part.logprobs is not None:
        nbytes += len(part.logprobs) * _BYTES_PER_LIST_ELEMENT
    return nbytes + _PART_OVERHEAD_BYTES


def routing_key_for(group_id: int) -> int:
    """Return the key that pins a group's requests to one engine instance.

    The granularity is the group, so all generates of a group stay on one
    server and share the first-turn prefix. Changing the routing granularity
    means changing this function only.
    """
    return group_id


def payload_key_for(group_id: int) -> int:
    """Return the key a group's payload parts are put under.

    The granularity is the group, so one keyed queue holds the whole group
    and one keyed get delivers it. Changing the payload granularity means
    changing this function only.
    """
    return group_id


def create_payload_channels(
    name: str,
    num_shards: int = 1,
    producer: Any = None,
    node_rank: Optional[int] = None,
) -> list[Channel]:
    """Create the payload channels that sit beside one session server.

    Every channel is non-distributed and placed on the session server's node:
    the node is derived from ``producer`` (the session server's worker group
    or group name) unless ``node_rank`` pins it explicitly. With
    ``num_shards > 1`` the node runs one channel process per shard and each
    group maps to one shard, splitting the collector's CPU work across
    processes without splitting a group.

    Args:
        name: Base channel name. A single shard keeps it; shard ``i`` of a
            sharded set is named ``<name>_<i>``.
        num_shards: Number of channel processes to create on the node.
        producer: The session server's worker group or group name, used to
            resolve the node when ``node_rank`` is None.
        node_rank: Explicit node for the channels; takes precedence over
            ``producer``.

    Returns:
        The channels indexed by shard.
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}.")
    channels = []
    for shard in range(num_shards):
        shard_name = name if num_shards == 1 else f"{name}_{shard}"
        channels.append(
            Channel.create(
                name=shard_name,
                collector=PayloadCollector,
                producers=None if producer is None else [producer],
                node_rank=node_rank,
            )
        )
    return channels


def payload_locator_for(channel: Channel) -> str:
    """Return the locator string naming ``channel`` for a PayloadRef.

    The format is ``channel://<name>``; resolving it back is
    :func:`payload_channel_for`.
    """
    return f"channel://{channel.name}"


def payload_channel_for(locator: str, channels: Any) -> Channel:
    """Return the channel of ``channels`` that a payload locator names.

    Args:
        locator: A ``channel://<name>`` string from a PayloadRef.
        channels: Payload channels to search, in any iterable order.

    Returns:
        The channel whose name matches the locator.

    Raises:
        ValueError: If the locator is malformed or no channel matches it.
    """
    prefix = "channel://"
    if not isinstance(locator, str) or not locator.startswith(prefix):
        raise ValueError(
            f"payload locator must be a '{prefix}<name>' string, got {locator!r}."
        )
    name = locator[len(prefix) :]
    for channel in channels:
        if channel.name == name:
            return channel
    raise ValueError(f"no payload channel named {name!r} for locator {locator!r}.")
