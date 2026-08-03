# Copyright 2025 The RLinf Authors.
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

import asyncio
from typing import Any

from rlinf.utils.backbone_cache import ROLLOUT_BACKBONE_SAMPLE_ID_STRIDE
from rlinf.utils.pinned_rollout_cache import PinnedRolloutBackboneCache


async def receive_pinned_rollout_backbone_stream(
    worker: Any,
    *,
    source_rank: int,
    consumer_rank: int,
    model_version: int,
    expected_blocks: int,
    expected_samples: int,
    blocks_per_batch: int,
    rollout_group_name: str,
    timeout_seconds: float,
) -> tuple[PinnedRolloutBackboneCache, dict[str, Any], dict[str, float]]:
    if expected_blocks <= 0 or expected_samples <= 0:
        raise ValueError("pinned backbone stream requires positive expected counts")
    if blocks_per_batch <= 0:
        raise ValueError("pinned backbone batch size must be positive")
    if timeout_seconds <= 0:
        raise ValueError("pinned backbone stream timeout must be positive")

    cache: PinnedRolloutBackboneCache | None = None
    stream_metadata: dict[str, Any] | None = None
    received_bytes = 0
    completed_blocks = 0
    completed_samples = 0
    expected_batches = (expected_blocks + blocks_per_batch - 1) // blocks_per_batch
    received = None
    metadata = None
    tensors = []
    blocks = []
    try:
        for expected_batch_index in range(expected_batches):
            received = None
            metadata = None
            tensors = []
            blocks = []
            recv_work = worker.recv(
                src_group_name=rollout_group_name,
                src_rank=source_rank,
                async_op=True,
                borrowed_ipc=True,
            )
            try:
                received = await asyncio.wait_for(
                    recv_work.async_wait(), timeout=timeout_seconds
                )
            except asyncio.TimeoutError as error:
                raise TimeoutError(
                    "timed out waiting for pinned backbone batch "
                    f"{expected_batch_index}/{expected_batches} from Rollout rank "
                    f"{source_rank}"
                ) from error
            if not isinstance(received, tuple) or len(received) != 2:
                raise RuntimeError("pinned backbone IPC requires tensors and metadata")
            tensors, metadata = received
            if not isinstance(metadata, dict) or metadata.get("schema") != 3:
                raise RuntimeError(f"unsupported pinned backbone metadata: {metadata}")
            if not isinstance(tensors, list) or not tensors or len(tensors) % 2:
                raise RuntimeError("pinned backbone batch requires feature/mask pairs")
            if int(metadata.get("batch_index", -1)) != expected_batch_index:
                raise RuntimeError("pinned backbone batches arrived out of order")
            if int(metadata.get("start_block_index", -1)) != completed_blocks:
                raise RuntimeError("pinned backbone batch block range is invalid")
            if int(metadata.get("total_blocks", -1)) != expected_blocks:
                raise RuntimeError("pinned backbone total block count changed")
            if int(metadata.get("total_samples", -1)) != expected_samples:
                raise RuntimeError("pinned backbone total sample count changed")
            if int(metadata.get("producer_rank", -1)) != source_rank:
                raise RuntimeError("pinned backbone producer rank does not match route")
            if int(metadata.get("consumer_rank", -1)) != consumer_rank:
                raise RuntimeError("pinned backbone consumer rank does not match Actor")
            if int(metadata.get("model_version", -1)) != model_version:
                raise RuntimeError("pinned backbone model version does not match Actor")
            expected_base = source_rank * ROLLOUT_BACKBONE_SAMPLE_ID_STRIDE
            if int(metadata.get("sample_id_base", -1)) != expected_base:
                raise RuntimeError("pinned backbone sample-ID namespace is invalid")
            if int(metadata.get("offset", -1)) != completed_samples:
                raise RuntimeError("pinned backbone batch sample offset is invalid")

            block_sizes = [int(size) for size in metadata.get("block_sizes", [])]
            expected_batch_blocks = min(
                blocks_per_batch, expected_blocks - completed_blocks
            )
            if len(block_sizes) != expected_batch_blocks:
                raise RuntimeError("pinned backbone batch block count mismatch")
            if len(tensors) != 2 * len(block_sizes) or any(
                size <= 0 for size in block_sizes
            ):
                raise RuntimeError("pinned backbone batch tensor schema is invalid")

            batch_bytes = 0
            for block_offset, block_size in enumerate(block_sizes):
                feature = tensors[2 * block_offset]
                mask = tensors[2 * block_offset + 1]
                if feature.shape[0] != block_size or mask.shape[0] != block_size:
                    raise RuntimeError(
                        "pinned backbone feature/mask block size mismatch"
                    )
                blocks.append((feature, mask))
                batch_bytes += sum(
                    tensor.numel() * tensor.element_size() for tensor in (feature, mask)
                )
            if batch_bytes != int(metadata.get("bytes", -1)):
                raise RuntimeError("pinned backbone batch byte count mismatch")

            if cache is None:
                first_feature, first_mask = blocks[0]
                cache = PinnedRolloutBackboneCache(
                    total_samples=expected_samples,
                    feature_example=first_feature,
                    mask_example=first_mask,
                    device=worker.torch_platform.current_device(),
                )
                stream_metadata = dict(metadata)
            elif stream_metadata is None or metadata.get(
                "lease_id"
            ) != stream_metadata.get("lease_id"):
                raise RuntimeError("pinned backbone lease changed within one stream")

            cache.store_blocks(offset=completed_samples, blocks=blocks)
            completed_blocks += len(block_sizes)
            completed_samples += sum(block_sizes)
            received_bytes += batch_bytes
            if expected_batch_index + 1 == expected_batches and (
                completed_blocks != expected_blocks
                or completed_samples != expected_samples
            ):
                raise RuntimeError(
                    "pinned backbone stream ended with incomplete counts"
                )

            lease_id = str(metadata.get("lease_id", ""))
            blocks.clear()
            tensors.clear()
            received = None
            worker.torch_platform.ipc_collect()
            worker.send(
                {
                    "schema": 3,
                    "status": "ok",
                    "lease_id": lease_id,
                    "batch_index": expected_batch_index,
                    "completed_blocks": completed_blocks,
                },
                dst_group_name=rollout_group_name,
                dst_rank=source_rank,
            )

            log_stride = max(1, expected_blocks // 4)
            if expected_batch_index == 0 or completed_blocks % log_stride == 0:
                worker.log_info(
                    "PINNED_FEATURE_BATCH_STORED "
                    f"rank={consumer_rank} lease={lease_id} "
                    f"batches={expected_batch_index + 1}/{expected_batches} "
                    f"blocks={completed_blocks}/{expected_blocks} "
                    f"bytes={received_bytes}"
                )
    except Exception as error:
        if cache is not None:
            cache.clear()
        blocks.clear()
        if isinstance(tensors, list):
            tensors.clear()
        received = None
        worker.torch_platform.ipc_collect()
        if isinstance(metadata, dict) and metadata.get("lease_id"):
            worker.send(
                {
                    "schema": 3,
                    "status": "error",
                    "lease_id": str(metadata["lease_id"]),
                    "batch_index": int(metadata.get("batch_index", -1)),
                    "completed_blocks": completed_blocks,
                    "error": f"{type(error).__name__}: {error}",
                },
                dst_group_name=rollout_group_name,
                dst_rank=source_rank,
            )
        raise

    if cache is None or stream_metadata is None:
        raise RuntimeError("pinned backbone stream produced no cache")
    if completed_blocks != expected_blocks or completed_samples != expected_samples:
        raise RuntimeError("pinned backbone stream ended with incomplete counts")
    try:
        cache.finalize()
    except Exception:
        cache.clear()
        worker.torch_platform.ipc_collect()
        raise
    stats = cache.stats()
    worker.log_info(
        "PINNED_FEATURE_CACHE_READY "
        f"rank={consumer_rank} lease={stream_metadata['lease_id']} "
        f"batches={expected_batches} samples={stats.samples} bytes={stats.bytes} "
        f"alloc_s={stats.allocation_seconds:.6f} d2h_s={stats.d2h_seconds:.6f}"
    )
    return (
        cache,
        stream_metadata,
        {
            "actor/pinned_feature_bytes": float(stats.bytes),
            "actor/pinned_feature_samples": float(stats.samples),
            "actor/pinned_feature_batches": float(expected_batches),
            "actor/pinned_feature_allocation_seconds": stats.allocation_seconds,
            "actor/pinned_feature_d2h_seconds": stats.d2h_seconds,
        },
    )
