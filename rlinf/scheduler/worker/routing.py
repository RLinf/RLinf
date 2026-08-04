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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch

from rlinf.scheduler.manager import WorkerAddress, WorkerInfo

ROUTING_KEY_PREFIX = "scheduler_route"
ROUTING_DEFAULT_TAG = "default"


def _build_channel_message(
    send_rank: int,
    batch_idx: int,
    mode: str,
    tag: str,
) -> str:
    """Construct a channel message key that matches the expected communication schema.

    Schema:
        {send_rank}_{batch_idx}_{mode}_{tag}

    Args:
        send_rank: Send worker rank.
        batch_idx: Batch index within the worker.
        mode: The message mode.
        tag: Type of message (e.g., "train_obs", "rollout_results").

    Returns:
        A formatted channel message string.
    """
    return f"{send_rank}_{batch_idx}_{mode}_{tag}"


def split_channel_message(channel_message: str) -> tuple[int, int, str, str]:
    """Split a channel message into its components.

    This is used to route rollout worker and env worker communication without a
    rankmap, recovering the send_rank, batch_idx, mode and tag from the message.

    Args:
        channel_message: The batch index string ``{send_rank}_{batch_idx}_{mode}_{tag}``.

    Returns:
        A tuple of send_rank, batch_idx, mode, tag.
    """
    send_rank, batch_idx, mode, tag = channel_message.split("_", 3)
    send_rank = int(send_rank)
    batch_idx = int(batch_idx)
    return send_rank, batch_idx, mode, tag


class CommMapper:
    """Communication mapping helpers with batch sharding among two worker groups that require fixed rank pairing in communications.

    For example, env and rollout should always use the same rank pair for communications.
    """

    @staticmethod
    def build_channel_key(src_rank: int, dst_rank: int, extra: str) -> str:
        """Build a canonical point-to-point channel key."""
        return f"{src_rank}_{dst_rank}_{extra}"

    @staticmethod
    def decoupled_get_batch_size(
        batch_size: int,
        src_world_size: int,
        dst_world_size: int,
        queue_size: int = 0,
    ) -> list[int]:
        """Compute destination ranks and transfer sizes for one source rank."""
        # in decoupled mode, the src_world_size and dst_world_size are the world size of the source and destination workers.
        # the batch_size is the total batch size of the source workers.
        # the return value is a list of batch sizes for this rank destination to send batch split size.
        # the queue_size is the size of the queue to be split.
        # split_size will be the larger of src_world_size and dst_world_size.
        # The queue length for workers with a larger quantity is 1,
        # while workers with a smaller quantity will aggregate the sent data for batch computation.
        # The queue length can be set by queue_size
        split_size = max(src_world_size, dst_world_size)

        assert batch_size % split_size == 0, (
            f"batch_size ({batch_size}) must be divisible by split_size ({split_size})."
        )

        if src_world_size >= dst_world_size:
            # the src_world_size >= dst_world_size
            # for example, src_world_size = 4, dst_world_size = 2, split_size = 4, batch_size = 32
            # the rank 0 [8] rank1 [8] rank2 [8] rank3 [8]
            return [batch_size // split_size]
        else:
            assert dst_world_size > src_world_size, (
                f"dst_world_size ({dst_world_size}) must more than src_world_size ({src_world_size})."
            )
            # the src_world_size < dst_world_size
            # for example, src_world_size = 2, dst_world_size = 8, split_size = 8, batch_size = 32
            # the rank 0 [4, 4, 4, 4] rank1 [4, 4, 4, 4]
            if queue_size <= 0:
                return [
                    batch_size // split_size
                    for _ in range(dst_world_size // src_world_size)
                ]
            else:
                # special case for queue_size == 1
                # for example, rollout_worker_nums = 2, env_worker_nums = 4, batch_size = 32
                # The rollout worker should have been [8, 8], but after setting queue_size, the length will be limited to queue_size.
                # The rollout worker will be [8] if the queue_size is 1.
                # the env worker rank 0 [8] rank1 [8] rank2 [8] rank3 [8]
                assert queue_size <= (dst_world_size // src_world_size), (
                    f"queue_size {queue_size} should be less than (dst_world_size {dst_world_size} // src_world_size {src_world_size})"
                )
                return [batch_size // split_size for _ in range(queue_size)]

    @staticmethod
    def get_dst_ranks(
        batch_size: int, src_world_size: int, dst_world_size: int, src_rank: int
    ) -> list[tuple[int, int]]:
        """Compute destination ranks and transfer sizes for one source rank."""
        assert batch_size % src_world_size == 0, (
            f"batch_size ({batch_size}) must be divisible by src_world_size ({src_world_size})."
        )
        assert batch_size % dst_world_size == 0, (
            f"batch_size ({batch_size}) must be divisible by dst_world_size ({dst_world_size})."
        )
        assert 0 <= src_rank < src_world_size, (
            f"src_rank ({src_rank}) must be in [0, {src_world_size})."
        )

        batch_size_per_src_rank = batch_size // src_world_size
        batch_size_per_dst_rank = batch_size // dst_world_size

        dst_ranks_and_sizes: list[tuple[int, int]] = []
        batch_begin = src_rank * batch_size_per_src_rank
        batch_end = (src_rank + 1) * batch_size_per_src_rank
        while batch_begin < batch_end:
            dst_rank = batch_begin // batch_size_per_dst_rank
            dst_batch_begin = dst_rank * batch_size_per_dst_rank
            dst_remaining = batch_size_per_dst_rank - (batch_begin - dst_batch_begin)
            src_remaining = batch_end - batch_begin
            dst_size = min(dst_remaining, src_remaining)
            dst_ranks_and_sizes.append((dst_rank, dst_size))
            batch_begin += dst_size
        return dst_ranks_and_sizes

    @staticmethod
    def get_src_ranks(
        batch_size: int, src_world_size: int, dst_world_size: int, dst_rank: int
    ) -> list[tuple[int, int]]:
        """Compute source ranks/sizes for one destination rank."""
        assert batch_size % src_world_size == 0, (
            f"batch_size ({batch_size}) must be divisible by src_world_size ({src_world_size})."
        )
        assert batch_size % dst_world_size == 0, (
            f"batch_size ({batch_size}) must be divisible by dst_world_size ({dst_world_size})."
        )
        assert 0 <= dst_rank < dst_world_size, (
            f"dst_rank ({dst_rank}) must be in [0, {dst_world_size})."
        )

        src_ranks_and_sizes: list[tuple[int, int]] = []
        for src_rank in range(src_world_size):
            dst_ranks_and_sizes = CommMapper.get_dst_ranks(
                batch_size=batch_size,
                src_world_size=src_world_size,
                dst_world_size=dst_world_size,
                src_rank=src_rank,
            )
            for mapped_dst_rank, size in dst_ranks_and_sizes:
                if mapped_dst_rank == dst_rank:
                    src_ranks_and_sizes.append((src_rank, size))

        expected_size = batch_size // dst_world_size
        actual_size = sum(size for _, size in src_ranks_and_sizes)
        assert actual_size == expected_size, (
            f"Expected receive size {expected_size} for destination rank {dst_rank}, "
            f"got {actual_size} from mappings {src_ranks_and_sizes}."
        )
        return src_ranks_and_sizes


@dataclass(frozen=True)
class RouteEntry:
    """One routed shard between a source rank and a destination rank."""

    peer_rank: int
    batch_size: int | None
    key: Any


@dataclass(frozen=True)
class DecoupledRouteEntry:
    """One routed shard in decoupled mode."""

    batch_size: int | None
    key: Any
    batch_index: str | None


@dataclass(frozen=True)
class RoutePlan:
    """Deterministic route plan for one sender or receiver rank."""

    src_group_name: str
    dst_group_name: str
    src_rank: int | None
    dst_rank: int | None
    src_world_size: int
    dst_world_size: int
    tag: str
    route_key: Any
    entries: list[RouteEntry]


def normalize_route_tag(tag: str | None) -> str:
    """Normalize an optional route tag to the default routing tag."""
    return tag or ROUTING_DEFAULT_TAG


def normalize_route_key(route_key: Any) -> str:
    """Normalize a route key into a stable string representation."""
    if route_key is None:
        return ""
    if isinstance(route_key, str):
        return route_key
    return repr(route_key)


def build_route_channel_key(
    src_group_name: str,
    dst_group_name: str,
    src_rank: int,
    dst_rank: int,
    tag: str | None,
    route_key: Any = None,
) -> tuple[str, str, str, str, str, int, int]:
    """Build a canonical key for routed channel messages."""
    return (
        ROUTING_KEY_PREFIX,
        src_group_name,
        dst_group_name,
        normalize_route_tag(tag),
        normalize_route_key(route_key),
        src_rank,
        dst_rank,
    )


def get_group_info(
    manager_proxy: Any,
    group_name: str,
) -> WorkerInfo:
    """Return worker group metadata from the manager proxy.

    Args:
        manager_proxy: Worker manager proxy used to query worker metadata.
        group_name: Name of the worker group to query.

    Returns:
        WorkerInfo for the requested group.

    Raises:
        ValueError: If the worker group is not registered.
    """
    info = manager_proxy.get_worker_info(WorkerAddress(group_name, ranks=0))
    if info is None:
        raise ValueError(f"Worker group '{group_name}' is not registered.")
    return info


def get_group_world_size(manager_proxy: Any, group_name: str) -> int:
    """Return the world size of a worker group."""
    return get_group_info(manager_proxy, group_name).group_world_size


def build_send_plan(
    *,
    src_group_name: str,
    dst_group_name: str,
    src_rank: int,
    src_world_size: int,
    dst_world_size: int,
    tag: str | None,
    route_key: Any = None,
    batch_size: int,
) -> RoutePlan:
    """Build the route plan for one sender rank."""
    entries: list[RouteEntry] = []
    stage_batch_size = batch_size
    for dst_rank, batch_size in CommMapper.get_dst_ranks(
        batch_size=stage_batch_size,
        src_world_size=src_world_size,
        dst_world_size=dst_world_size,
        src_rank=src_rank,
    ):
        entries.append(
            RouteEntry(
                peer_rank=dst_rank,
                batch_size=batch_size,
                key=build_route_channel_key(
                    src_group_name=src_group_name,
                    dst_group_name=dst_group_name,
                    src_rank=src_rank,
                    dst_rank=dst_rank,
                    tag=tag,
                    route_key=route_key,
                ),
            )
        )

    return RoutePlan(
        src_group_name=src_group_name,
        dst_group_name=dst_group_name,
        src_rank=src_rank,
        dst_rank=None,
        src_world_size=src_world_size,
        dst_world_size=dst_world_size,
        tag=normalize_route_tag(tag),
        route_key=route_key,
        entries=entries,
    )


def build_send_key(
    *,
    src_group_name: str,
    dst_group_name: str,
    src_rank: int,
    dst_rank: int,
    tag: str | None,
    route_key: Any = None,
) -> str:
    """Build the channel key used to send a routed payload shard.

    The key identifies one logical route from a source worker group/rank to a
    destination worker group/rank. ``tag`` and ``route_key`` can be used to separate
    different message streams that share the same source and destination ranks.

    Args:
        src_group_name: Name of the source worker group.
        dst_group_name: Name of the destination worker group.
        src_rank: Rank of the source worker within its group.
        dst_rank: Rank of the destination worker within its group.
        tag: Optional routing tag used to distinguish message types.
        route_key: Optional extra key used to separate independent streams.

    Returns:
        The channel key string for sending the routed payload.
    """
    return build_route_channel_key(
        src_group_name=src_group_name,
        dst_group_name=dst_group_name,
        src_rank=src_rank,
        dst_rank=dst_rank,
        tag=tag,
        route_key=route_key,
    )


def decoupled_build_send_plan(
    *,
    src_group_name: str,
    dst_group_name: str,
    src_rank: int | None,
    src_world_size: int,
    dst_world_size: int,
    tag: str | None,
    route_key: Any = None,
    batch_size: int,
    mode: str | None = None,
    tag_batch_router: list[str] | None = None,
    send_queue_size: int = 0,
) -> RoutePlan:
    """Build the route plan for one sender rank."""
    entries: list[DecoupledRouteEntry] = []
    stage_batch_size = batch_size

    for index, batch_size in enumerate(
        CommMapper.decoupled_get_batch_size(
            batch_size=stage_batch_size,
            src_world_size=src_world_size,
            dst_world_size=dst_world_size,
            queue_size=send_queue_size,
        )
    ):
        if tag_batch_router is not None:
            # if the tag_batch_router is provided, use the tag_batch_router to get the batch_index
            batch_index = tag_batch_router[index]
            # get the send_rank from the batch_index
            send_rank, _, _, tag = split_channel_message(batch_index)
        else:
            # Otherwise, use src_rank, index and tag to construct batch_index.
            batch_index = _build_channel_message(
                send_rank=src_rank,
                batch_idx=index,
                mode=mode,
                tag=tag,
            )
            # set the send_rank to None
            send_rank = None

        entries.append(
            DecoupledRouteEntry(
                batch_size=batch_size,
                key=build_route_channel_key(
                    src_group_name=src_group_name,
                    dst_group_name=dst_group_name,
                    src_rank=None,
                    dst_rank=send_rank,
                    tag=tag,
                    route_key=route_key,
                ),
                batch_index=batch_index,
            )
        )

    return RoutePlan(
        src_group_name=src_group_name,
        dst_group_name=dst_group_name,
        src_rank=None,
        dst_rank=send_rank,
        src_world_size=src_world_size,
        dst_world_size=dst_world_size,
        tag=normalize_route_tag(tag),
        route_key=route_key,
        entries=entries,
    )


def build_recv_plan(
    *,
    src_group_name: str,
    dst_group_name: str,
    dst_rank: int,
    src_world_size: int,
    dst_world_size: int,
    tag: str | None,
    route_key: Any = None,
    batch_size: int,
) -> RoutePlan:
    """Build the route plan for one receiver rank."""
    entries: list[RouteEntry] = []
    stage_batch_size = batch_size
    for src_rank, batch_size in CommMapper.get_src_ranks(
        batch_size=stage_batch_size,
        src_world_size=src_world_size,
        dst_world_size=dst_world_size,
        dst_rank=dst_rank,
    ):
        entries.append(
            RouteEntry(
                peer_rank=src_rank,
                batch_size=batch_size,
                key=build_route_channel_key(
                    src_group_name=src_group_name,
                    dst_group_name=dst_group_name,
                    src_rank=src_rank,
                    dst_rank=dst_rank,
                    tag=tag,
                    route_key=route_key,
                ),
            )
        )

    return RoutePlan(
        src_group_name=src_group_name,
        dst_group_name=dst_group_name,
        src_rank=None,
        dst_rank=dst_rank,
        src_world_size=src_world_size,
        dst_world_size=dst_world_size,
        tag=normalize_route_tag(tag),
        route_key=route_key,
        entries=entries,
    )


def decoupled_build_recv_plan(
    *,
    src_group_name: str,
    dst_group_name: str,
    recv_rank: int | None,
    src_world_size: int,
    dst_world_size: int,
    tag: str | None,
    route_key: Any = None,
    batch_size: int,
    recv_queue_size: int = 0,
) -> RoutePlan:
    """Build the route plan for one receiver rank."""
    entries: list[DecoupledRouteEntry] = []
    stage_batch_size = batch_size
    for batch_size in CommMapper.decoupled_get_batch_size(
        batch_size=stage_batch_size,
        src_world_size=src_world_size,
        dst_world_size=dst_world_size,
        queue_size=recv_queue_size,
    ):
        entries.append(
            DecoupledRouteEntry(
                batch_size=batch_size,
                key=build_route_channel_key(
                    src_group_name=src_group_name,
                    dst_group_name=dst_group_name,
                    src_rank=None,
                    dst_rank=recv_rank,
                    tag=tag,
                    route_key=route_key,
                ),
                # In decoupled mode, the batch_index is not needed to be provided.
                # None of the recv_from calls will set batch_index,
                # Because that data will be overwritten by the received data.
                batch_index=None,
            )
        )

    return RoutePlan(
        src_group_name=src_group_name,
        dst_group_name=dst_group_name,
        src_rank=None,
        dst_rank=None,
        src_world_size=src_world_size,
        dst_world_size=dst_world_size,
        tag=normalize_route_tag(tag),
        route_key=route_key,
        entries=entries,
    )


def infer_batch_size(data: Any) -> int:
    """Infer a batch size from common routed payloads."""
    if isinstance(data, torch.Tensor) or isinstance(data, np.ndarray):
        return int(data.shape[0])
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        first_non_none = next(
            (value for value in data.values() if value is not None), None
        )
        if first_non_none is None:
            raise ValueError("Cannot infer batch size from an all-None dict payload.")
        return infer_batch_size(first_non_none)
    raise ValueError(f"Unsupported payload type for batch-size inference: {type(data)}")


def split_batch(data: Any, split_sizes: Sequence[int]) -> list[Any]:
    """Split a common batch payload on batch dimension."""
    if isinstance(data, (bool, int, float, str)):
        return [data for _ in split_sizes]
    if isinstance(data, torch.Tensor):
        return [
            chunk.contiguous() for chunk in torch.split(data, list(split_sizes), dim=0)
        ]
    if isinstance(data, np.ndarray):
        split_indices = np.cumsum(split_sizes[:-1]).tolist()
        return list(np.split(data, split_indices, axis=0))
    if isinstance(data, list):
        begin = 0
        shards = []
        for size in split_sizes:
            shards.append(data[begin : begin + size])
            begin += size
        return shards
    if isinstance(data, dict):
        shard_count = len(split_sizes)
        shards = [{} for _ in range(shard_count)]
        for key, value in data.items():
            if value is None:
                for shard in shards:
                    shard[key] = None
                continue
            split_values = split_batch(value, split_sizes)
            for idx, split_value in enumerate(split_values):
                shards[idx][key] = split_value
        return shards
    raise ValueError(f"Unsupported payload type for batch split: {type(data)}")


def merge_batches(batches: Sequence[Any]) -> Any:
    """Merge common routed payloads on batch dimension."""
    if len(batches) == 0:
        raise ValueError("Cannot merge an empty batch list.")
    if len(batches) == 1:
        return batches[0]

    first_non_none = next((batch for batch in batches if batch is not None), None)
    if first_non_none is None:
        return None
    if any(batch is None for batch in batches):
        raise ValueError(
            "Cannot merge a mix of None and non-None payloads generically."
        )

    if isinstance(first_non_none, torch.Tensor):
        return torch.cat(list(batches), dim=0)
    if isinstance(first_non_none, np.ndarray):
        return np.concatenate(list(batches), axis=0)
    if isinstance(first_non_none, list):
        merged: list[Any] = []
        for batch in batches:
            merged.extend(batch)
        return merged
    if isinstance(first_non_none, dict):
        merged: dict[str, Any] = {}
        for key in first_non_none.keys():
            merged[key] = merge_batches([batch[key] for batch in batches])
        return merged
    if all(batch == first_non_none for batch in batches):
        return first_non_none
    raise ValueError(
        f"Unsupported payload type for batch merge: {type(first_non_none)}"
    )


def validate_batch_size(
    data: Any,
    expected_batch_size: int | None,
    infer_batch_size_fn: Callable[[Any], int] | None = None,
) -> None:
    """Validate that a routed payload matches the expected batch size.

    Args:
        data: Routed payload to validate.
        expected_batch_size: Expected leading-dimension batch size. If ``None``,
            validation is skipped.
        infer_batch_size_fn: Optional custom function for inferring batch size.

    Raises:
        ValueError: If the inferred batch size does not match the expected size.
    """
    if expected_batch_size is None:
        return
    infer_fn = infer_batch_size_fn or infer_batch_size
    actual_batch_size = infer_fn(data)
    if actual_batch_size != expected_batch_size:
        raise ValueError(
            f"Expected batch size {expected_batch_size}, but received {actual_batch_size}."
        )


def get_batch_size(
    data: Any,
    infer_batch_size_fn: Callable[[Any], int] | None = None,
) -> int:
    """Get the batch size of a data."""
    if infer_batch_size_fn is None:
        return infer_batch_size(data)
    return infer_batch_size_fn(data)
