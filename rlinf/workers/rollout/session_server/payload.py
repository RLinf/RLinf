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
by a :class:`PayloadDiscard` on the same key.
"""

from dataclasses import dataclass
from typing import Any, Iterator, Optional

from rlinf.scheduler.channel import Channel, Collector, register_collector

__all__ = [
    "PayloadPart",
    "PayloadDiscard",
    "GroupPayload",
    "PayloadCollector",
    "routing_key_for",
    "payload_key_for",
    "payload_shard_for",
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
            kept lengths, continuity flags, and each trajectory's slice of
            the training extra fields.
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


def payload_shard_for(group_id: int, num_shards: int) -> int:
    """Return the payload shard that serves a group.

    Args:
        group_id: Group to place.
        num_shards: Number of payload channels on the session server's node.

    Returns:
        The shard index in ``[0, num_shards)``. A group maps to exactly one
        shard, so its parts never split across channel processes.
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}.")
    return group_id % num_shards


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
        The channels indexed by shard; select a group's channel with
        ``channels[payload_shard_for(group_id, num_shards)]``.
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
