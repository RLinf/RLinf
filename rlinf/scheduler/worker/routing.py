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

from ..manager import WorkerAddress, WorkerInfo
from rlinf.utils.comm_mapping import CommMapper


ROUTING_KEY_PREFIX = "scheduler_route"
ROUTING_DEFAULT_TAG = "default"


@dataclass(frozen=True)
class RouteEntry:
    """One routed shard between a source rank and a destination rank."""

    peer_rank: int
    batch_size: int | None
    key: Any


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
    return tag or ROUTING_DEFAULT_TAG


def normalize_route_key(route_key: Any) -> str:
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
    info = manager_proxy.get_worker_info(WorkerAddress(group_name, ranks=0))
    if info is None:
        raise ValueError(f"Worker group '{group_name}' is not registered.")
    return info


def get_group_world_size(manager_proxy: Any, group_name: str) -> int:
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
    local_batch_size: int,
) -> RoutePlan:
    """Build the route plan for one sender rank."""

    entries: list[RouteEntry] = []
    stage_batch_size = local_batch_size * src_world_size
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


def build_recv_plan(
    *,
    src_group_name: str,
    dst_group_name: str,
    dst_rank: int,
    src_world_size: int,
    dst_world_size: int,
    tag: str | None,
    route_key: Any = None,
    local_batch_size: int,
) -> RoutePlan:
    """Build the route plan for one receiver rank."""

    entries: list[RouteEntry] = []
    stage_batch_size = local_batch_size * dst_world_size
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


def infer_batch_size(data: Any) -> int:
    """Infer a batch size from common routed payloads."""

    if isinstance(data, torch.Tensor) or isinstance(data, np.ndarray):
        return int(data.shape[0])
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        first_non_none = next((value for value in data.values() if value is not None), None)
        if first_non_none is None:
            raise ValueError("Cannot infer batch size from an all-None dict payload.")
        return infer_batch_size(first_non_none)
    raise ValueError(f"Unsupported payload type for batch-size inference: {type(data)}")


def split_batch(data: Any, split_sizes: Sequence[int]) -> list[Any]:
    """Split a common batch payload on batch dimension."""

    if isinstance(data, torch.Tensor):
        return [chunk.contiguous() for chunk in torch.split(data, list(split_sizes), dim=0)]
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
        raise ValueError("Cannot merge a mix of None and non-None payloads generically.")

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
    raise ValueError(f"Unsupported payload type for batch merge: {type(first_non_none)}")


def validate_batch_size(
    data: Any,
    expected_batch_size: int | None,
    infer_batch_size_fn: Callable[[Any], int] | None = None,
) -> None:
    if expected_batch_size is None:
        return
    infer_fn = infer_batch_size_fn or infer_batch_size
    actual_batch_size = infer_fn(data)
    if actual_batch_size != expected_batch_size:
        raise ValueError(
            f"Expected batch size {expected_batch_size}, but received {actual_batch_size}."
        )
