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
from typing import Any, Awaitable, Callable

import torch
from omegaconf import DictConfig, OmegaConf
from torch.distributed.tensor import DTensor

SendFn = Callable[[Any], Awaitable[None]]
RecvFn = Callable[[], Awaitable[Any]]


def materialize_tensor(tensor: torch.Tensor | DTensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    assert isinstance(tensor, torch.Tensor), "Expected a torch.Tensor or DTensor"
    return tensor


def normalize_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        key = dtype.lower()
        if key in mapping:
            return mapping[key]
    raise TypeError(f"Unsupported dtype: {dtype}")


def normalize_device(device: torch.device | str) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


class WeightSyncer(ABC):
    def __init__(self):
        self._sender_initialized: bool = False
        self._receiver_initialized: bool = False

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
        send: SendFn,
        recv: RecvFn | None = None,
    ) -> None:
        del state_dict, send, recv
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
            return BucketWeightSyncer(
                bucket_size=OmegaConf.select(bucket_config, "bucket_size"),
                bucket_dtype=OmegaConf.select(bucket_config, "bucket_dtype"),
                bucket_device=OmegaConf.select(bucket_config, "bucket_device"),
                is_agent=OmegaConf.select(bucket_config, "is_agent", default=False),
                load_instant=OmegaConf.select(
                    bucket_config, "load_instant", default=True
                ),
            )
        if syncer_type == "patch":
            from .patch_syncer import PatchWeightSyncer

            patch_config = OmegaConf.select(config, "patch")
            assert patch_config is not None, (
                "Patch config must be provided for patch weight syncer"
            )
            return PatchWeightSyncer(
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
                    patch_config, "transport_device", default="cuda"
                ),
            )
        raise ValueError(f"Unsupported weight syncer type: {syncer_type}")

    def sender_initialized(self) -> bool:
        return self._sender_initialized

    def receiver_initialized(self) -> bool:
        return self._receiver_initialized
