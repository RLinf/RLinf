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

from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from torch.distributed.tensor import DTensor

from rlinf.scheduler import CollectiveGroupOptions, Worker

SendFn = Callable[[Any], Awaitable[None]]
RecvFn = Callable[[], Awaitable[Any]]


class WeightSyncer(ABC):
    def __init__(self):
        self._sender_initialized: bool = False
        self._receiver_initialized: bool = False
        self._comm_options: Optional[CollectiveGroupOptions] = None

    @property
    def comm_options(self) -> Optional[CollectiveGroupOptions]:
        """``CollectiveGroupOptions`` to pass to broadcast/send/recv calls
        performed during weight sync. Populated by :meth:`create` from the
        ``use_ring_sync``, ``nccl_max_ctas`` and ``nccl_min_ctas`` keys on
        the weight syncer config; ``None`` if every option is at its default
        (matching the legacy behavior where no options were supplied)."""
        return self._comm_options

    @abstractmethod
    async def sync(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        send: SendFn,
        version: int | torch.Tensor,
    ) -> None: ...

    @abstractmethod
    async def apply(self, model: torch.nn.Module, recv: RecvFn) -> int: ...

    async def init_sender(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        param_names_need_sync: list[str],
        send: SendFn,
        recv: RecvFn | None = None,
    ) -> None:
        del state_dict, send, recv, param_names_need_sync
        self._sender_initialized = True

    async def init_receiver(
        self,
        state_dict: dict[str, torch.Tensor | DTensor] | None,
        recv: RecvFn,
        send: SendFn | None = None,
    ) -> None:
        del state_dict, recv, send
        self._receiver_initialized = True

    @classmethod
    def create(cls, config: DictConfig) -> "WeightSyncer":
        assert config is not None, "Weight syncer config must be provided"
        syncer_type = OmegaConf.select(config, "type")
        if syncer_type == "bucket":
            from .bucket_syncer import BucketWeightSyncer

            bucket_config = OmegaConf.select(config, "bucket")
            assert bucket_config is not None, (
                "Bucket config must be provided for bucket weight syncer"
            )
            syncer: "WeightSyncer" = BucketWeightSyncer(
                bucket_size=OmegaConf.select(bucket_config, "bucket_size"),
                bucket_dtype=OmegaConf.select(bucket_config, "bucket_dtype"),
                bucket_device=OmegaConf.select(
                    bucket_config, "bucket_device", default=Worker.torch_device_type
                ),
                is_agent=OmegaConf.select(bucket_config, "is_agent", default=False),
                load_instant=OmegaConf.select(
                    bucket_config, "load_instant", default=True
                ),
            )
        elif syncer_type == "patch":
            from .patch_syncer import PatchWeightSyncer

            patch_config = OmegaConf.select(config, "patch")
            assert patch_config is not None, (
                "Patch config must be provided for patch weight syncer"
            )
            syncer = PatchWeightSyncer(
                snapshot_device=OmegaConf.select(
                    patch_config, "snapshot_device", default="cpu"
                ),
                delta_encoding=OmegaConf.select(
                    patch_config, "delta_encoding", default=True
                ),
                compression_algorithm=OmegaConf.select(
                    patch_config,
                    "compression_algorithm",
                    default=OmegaConf.select(
                        patch_config, "compression", default="none"
                    ),
                ),
                transport_device=OmegaConf.select(
                    patch_config, "transport_device", default=Worker.torch_device_type
                ),
                init_sync_enabled=OmegaConf.select(
                    patch_config, "init_sync.enabled", default=False
                ),
                init_sync_prefixes=OmegaConf.select(
                    patch_config, "init_sync.prefixes", default=None
                ),
                init_sync_bucket_size=OmegaConf.select(
                    patch_config,
                    "init_sync.bucket_size",
                    default=OmegaConf.select(
                        patch_config,
                        "init_sync.buckets_size",
                        default=128 * 1024 * 1024,
                    ),
                ),
            )
        else:
            raise ValueError(f"Unsupported weight syncer type: {syncer_type}")

        syncer._comm_options = cls._build_comm_options(config)
        return syncer

    @staticmethod
    def _build_comm_options(
        config: DictConfig,
    ) -> Optional[CollectiveGroupOptions]:
        """Build ``CollectiveGroupOptions`` from the weight syncer config.

        Reads three top-level keys (all optional, all default to the equivalent
        of the underlying ``CollectiveGroupOptions`` default):

        - ``use_ring_sync`` (bool): route the broadcast through the ring
          algorithm (one cross-group hop + parallel fan-out from the first
          receiver) by setting ``CollectiveGroupOptions.use_ring_broadcast``.
        - ``nccl_max_ctas`` / ``nccl_min_ctas`` (int): forwarded to
          ``CollectiveGroupOptions.accel_max_ctas`` / ``accel_min_ctas`` to
          cap how much GPU SM resource NCCL consumes during weight sync.
        """
        use_ring = OmegaConf.select(config, "use_ring_sync", default=False)
        max_ctas = OmegaConf.select(config, "nccl_max_ctas", default=None)
        min_ctas = OmegaConf.select(config, "nccl_min_ctas", default=None)
        if not use_ring and max_ctas is None and min_ctas is None:
            return None
        return CollectiveGroupOptions(
            use_ring_broadcast=bool(use_ring),
            accel_max_ctas=max_ctas,
            accel_min_ctas=min_ctas,
        )

    def sender_initialized(self) -> bool:
        return self._sender_initialized

    def receiver_initialized(self) -> bool:
        return self._receiver_initialized
