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

from collections.abc import Callable, Iterable, Iterator

import torch
from torch.distributed.tensor import DTensor

from rlinf.scheduler import Worker
from rlinf.utils.utils import synchronize_pending_accel_copies

from .base import (
    RecvFn,
    SendFn,
    WeightSyncer,
    materialize_tensor,
    normalize_device,
    normalize_dtype,
)


def iter_named_tensor_buckets(
    items: Iterable[tuple[str, torch.Tensor | DTensor]],
    version: int | torch.Tensor,
    *,
    bucket_size: int,
    bucket_device: str | torch.device,
    dtype_resolver: Callable[[str, torch.dtype], torch.dtype] | None = None,
) -> Iterator[dict[str, torch.Tensor]]:
    """Yield transport-ready buckets from already-selected named tensors."""
    metadata_keys = {
        BucketWeightSyncer._TOTAL_BUCKETS_KEY,
        BucketWeightSyncer._SYNCER_VERSION_KEY,
    }
    bucket_device = normalize_device(bucket_device)
    prepared_items: list[tuple[str, torch.Tensor | DTensor, torch.dtype]] = []
    currently_hold = 0
    total_buckets = 0

    for key, value in items:
        if key in metadata_keys:
            raise ValueError(f"Bucket payload key conflicts with metadata key: {key}")

        transport_dtype = (
            dtype_resolver(key, value.dtype)
            if dtype_resolver is not None
            else value.dtype
        )
        prepared_items.append((key, value, transport_dtype))
        currently_hold += (
            value.numel() * torch.empty((), dtype=transport_dtype).element_size()
        )
        if currently_hold >= bucket_size:
            total_buckets += 1
            currently_hold = 0

    if currently_hold > 0:
        total_buckets += 1
    assert total_buckets > 0, "No parameters to sync"

    metadata = {
        BucketWeightSyncer._TOTAL_BUCKETS_KEY: torch.tensor(
            total_buckets, dtype=torch.int32, device=bucket_device
        ),
        BucketWeightSyncer._SYNCER_VERSION_KEY: torch.as_tensor(
            version, dtype=torch.int64, device=bucket_device
        ),
    }

    bucket_idx = 0
    currently_hold = 0
    bucket: dict[str, torch.Tensor] = {}
    pending_copy_devices: set[torch.device] = set()
    for key, value, transport_dtype in prepared_items:
        tensor = materialize_tensor(value)
        async_accel_to_cpu = (
            bucket_device.type == "cpu"
            and tensor.device.type == Worker.torch_device_type
        )
        bucket[key] = tensor.to(
            device=bucket_device,
            dtype=transport_dtype,
            non_blocking=async_accel_to_cpu or bucket_device.type != "cpu",
        )
        if async_accel_to_cpu:
            pending_copy_devices.add(tensor.device)
        currently_hold += bucket[key].numel() * bucket[key].element_size()

        if currently_hold >= bucket_size:
            if bucket_idx == 0:
                bucket.update(metadata)
            synchronize_pending_accel_copies(pending_copy_devices)
            yield bucket
            bucket_idx += 1
            bucket = {}
            currently_hold = 0
            pending_copy_devices = set()

    if bucket:
        if bucket_idx == 0:
            bucket.update(metadata)
        synchronize_pending_accel_copies(pending_copy_devices)
        yield bucket


class BucketWeightSyncer(WeightSyncer):
    _TOTAL_BUCKETS_KEY = "total_buckets"
    _SYNCER_VERSION_KEY = "syncer_version"

    def __init__(
        self,
        bucket_size: int,
        bucket_dtype: torch.dtype | str | None,
        bucket_device: str | torch.device,
        is_agent: bool = False,
        load_instant: bool = True,
    ):
        super().__init__()
        self.bucket_size = bucket_size
        self.bucket_dtype = (
            normalize_dtype(bucket_dtype) if bucket_dtype is not None else None
        )
        self.bucket_device = normalize_device(bucket_device)
        self.is_agent = is_agent
        self.load_instant = load_instant

    def _bucket_key(self, key: str, has_visual: bool) -> str | None:
        if "_extra_state" in key:
            return None
        if has_visual and self.is_agent and key.startswith("model.language_model."):
            return "model." + key[len("model.language_model.") :]
        return key

    def _transport_dtype(self, dtype: torch.dtype) -> torch.dtype:
        if self.bucket_dtype is not None and dtype.is_floating_point:
            return self.bucket_dtype
        return dtype

    def iter_buckets(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        version: int | torch.Tensor,
    ):
        has_visual = any("visual." in key for key in state_dict.keys())
        named_items: list[tuple[str, torch.Tensor | DTensor]] = []
        assert hasattr(self, "param_names_need_sync"), (
            "init_sender must be called before iter_buckets"
        )
        assert self.param_names_need_sync, (
            "param_names_need_sync must be set and not empty"
        )
        for key, value in state_dict.items():
            if key not in self.param_names_need_sync_set:
                continue
            name = self._bucket_key(key, has_visual)
            if name is None:
                continue
            named_items.append((name, value))

        yield from iter_named_tensor_buckets(
            named_items,
            version,
            bucket_size=self.bucket_size,
            bucket_device=self.bucket_device,
            dtype_resolver=lambda _key, dtype: self._transport_dtype(dtype),
        )

    def divide_into_buckets(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        version: int | torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        return list(self.iter_buckets(state_dict, version))

    async def init_sender(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        param_names_need_sync: list[str],
        send: SendFn,
        recv: RecvFn | None = None,
    ) -> None:
        del state_dict, send, recv
        self.param_names_need_sync = param_names_need_sync
        self.param_names_need_sync_set = set(param_names_need_sync)
        self._sender_initialized = True

    async def sync(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        send: SendFn,
        version: int | torch.Tensor,
    ) -> None:
        for bucket in self.iter_buckets(state_dict, version):
            await send(bucket)
            del bucket

    async def apply(self, model: torch.nn.Module, recv: RecvFn) -> int:
        bucket: dict[str, torch.Tensor] = await recv()
        total_buckets = int(bucket.pop(self._TOTAL_BUCKETS_KEY).item())
        applied_version = int(bucket.pop(self._SYNCER_VERSION_KEY).item())

        if self.load_instant:
            model.load_state_dict(bucket, strict=False)
        else:
            cpu_buffer: dict[str, torch.Tensor] = {}
            pending_copy_devices: set[torch.device] = set()
            for key, value in bucket.items():
                if value.device.type == "cpu":
                    cpu_buffer[key] = value
                else:
                    cpu_buffer[key] = value.to("cpu", non_blocking=True)
                    pending_copy_devices.add(value.device)
        del bucket

        for _ in range(total_buckets - 1):
            bucket = await recv()
            if self.load_instant:
                model.load_state_dict(bucket, strict=False)
            else:
                for key, value in bucket.items():
                    if value.device.type == "cpu":
                        cpu_buffer[key] = value
                    else:
                        cpu_buffer[key] = value.to("cpu", non_blocking=True)
                        pending_copy_devices.add(value.device)
            del bucket

        if not self.load_instant:
            synchronize_pending_accel_copies(pending_copy_devices)
            model.load_state_dict(cpu_buffer, strict=False)
            del cpu_buffer

        return applied_version
