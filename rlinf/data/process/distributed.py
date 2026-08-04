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

"""Distributed helpers shared by the offline dataset-annotation scripts.

Used by the ``compute_advantages*`` scripts (``examples/offline_rl/advantage_labeling/recap/process`` and
``examples/offline_rl/advantage_labeling/steam/process``) to shard a dataset across ``torchrun`` ranks, run
single-pass inference, and gather the per-frame results back to rank 0.

Kept deliberately dependency-light — only ``torch``, ``pandas`` and the stdlib
— so it imports cleanly in every annotation environment (e.g. the openpi venv,
whose top-level ``lerobot.common.datasets`` import is broken). Do not add heavy
imports (``rlinf.scheduler``, megatron, lerobot, …) here.
"""

import gc
import logging
import os
from datetime import timedelta
from typing import Any, Optional

import pandas as pd
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def setup_distributed(cfg: Any) -> tuple[int, int, str]:
    """Initialise ``torch.distributed`` for torchrun-launched processes.

    Args:
        cfg: Config exposing an optional ``distributed`` section
            (``backend``, ``timeout`` seconds).

    Returns:
        Tuple of ``(rank, world_size, device_string)``. The device is
        ``cuda[:local_rank]`` when CUDA is available, else ``"cpu"``. Falls
        back to ``(0, 1, <device>)`` when not launched under torchrun.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist_cfg = cfg.get("distributed", {})
        backend = dist_cfg.get("backend", "nccl")
        timeout_seconds = dist_cfg.get("timeout", 1800)

        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                timeout=timedelta(seconds=timeout_seconds),
            )

        # Only bind a CUDA device when CUDA is actually present; CPU-only
        # torchrun jobs (e.g. a gloo backend) must not hard-fail here.
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        else:
            device = "cpu"

        if rank == 0:
            logger.info("Distributed mode enabled: %d ranks", world_size)
            logger.info("  Backend: %s, Timeout: %ss", backend, timeout_seconds)

        return rank, world_size, device

    # Single-process fallback
    return 0, 1, "cuda" if torch.cuda.is_available() else "cpu"


def _release_cuda_memory_for_exit() -> None:
    """Best-effort CUDA allocator cleanup before NCCL teardown."""
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception as exc:
        logger.warning("CUDA synchronize failed during cleanup: %s", exc)
    try:
        torch.cuda.empty_cache()
    except Exception as exc:
        logger.warning("CUDA empty_cache failed during cleanup: %s", exc)


def cleanup_distributed() -> None:
    """Tear down the process group, releasing CUDA memory first.

    NCCL may still need a little device memory to shut down, so the large CUDA
    modules should be dropped by the caller before this runs; we additionally
    free the allocator cache and barrier so no rank destroys the group while a
    peer is mid-collective.
    """
    if not dist.is_initialized():
        return

    _release_cuda_memory_for_exit()
    try:
        if dist.get_world_size() > 1:
            dist.barrier()
    except Exception as exc:
        logger.warning("Final distributed barrier failed during cleanup: %s", exc)

    _release_cuda_memory_for_exit()
    try:
        dist.destroy_process_group()
    except Exception as exc:
        logger.warning(
            "Ignoring distributed process-group cleanup failure during process "
            "exit: %s",
            exc,
        )


def get_shard_indices(
    total_samples: int, rank: int, world_size: int
) -> tuple[int, int]:
    """Even shard with earlier ranks taking the remainder.

    Args:
        total_samples: Total number of samples.
        rank: Current process rank.
        world_size: Total number of processes.

    Returns:
        Tuple of ``(start_index, end_index)`` where ``end`` is exclusive.
    """
    base = total_samples // world_size
    rem = total_samples % world_size
    if rank < rem:
        start = rank * (base + 1)
        end = start + base + 1
    else:
        start = rem * (base + 1) + (rank - rem) * base
        end = start + base
    return start, end


def gather_dataframes_to_rank0(
    local_df: pd.DataFrame, rank: int, world_size: int
) -> pd.DataFrame:
    """Gather per-rank result frames to a single, sorted DataFrame.

    Args:
        local_df: This rank's shard of per-frame results.
        rank: Current process rank.
        world_size: Total number of processes.

    Returns:
        Merged DataFrame sorted by ``(episode_index, frame_index)``. In
        single-process mode ``local_df`` is returned unchanged.
    """
    if world_size == 1:
        return local_df
    all_dfs: list[Optional[list[dict[str, Any]]]] = [None] * world_size
    dist.all_gather_object(all_dfs, local_df.to_dict("records"))
    rows: list[dict[str, Any]] = []
    for shard in all_dfs:
        if shard:
            rows.extend(shard)
    merged = pd.DataFrame(rows)
    if len(merged) > 0:
        merged = merged.sort_values(["episode_index", "frame_index"]).reset_index(
            drop=True
        )
    return merged
