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

import torch
from torch.distributed.tensor import DTensor

from .base import (
    RecvFn,
    SendFn,
    WeightSyncer,
    materialize_tensor,
    normalize_device,
    normalize_dtype,
)


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
        metadata_keys = {self._TOTAL_BUCKETS_KEY, self._SYNCER_VERSION_KEY}
        bucket_idx = 0
        total_buckets = 0
        currently_hold = 0
        bucket: dict[str, torch.Tensor] = {}
        has_visual = any("visual." in key for key in state_dict.keys())

        for key, value in state_dict.items():
            name = self._bucket_key(key, has_visual)
            if name is None:
                continue
            if name in metadata_keys:
                raise ValueError(
                    f"Bucket payload key conflicts with metadata key: {name}"
                )

            dtype = self._transport_dtype(value.dtype)
            currently_hold += (
                value.numel() * torch.empty((), dtype=dtype).element_size()
            )
            if currently_hold >= self.bucket_size:
                total_buckets += 1
                currently_hold = 0

        if currently_hold > 0:
            total_buckets += 1
        assert total_buckets > 0, "No parameters to sync"

        metadata = {
            self._TOTAL_BUCKETS_KEY: torch.tensor(
                total_buckets, dtype=torch.int32, device=self.bucket_device
            ),
            self._SYNCER_VERSION_KEY: torch.as_tensor(
                version, dtype=torch.int64, device=self.bucket_device
            ),
        }
        currently_hold = 0
        for key, value in state_dict.items():
            name = self._bucket_key(key, has_visual)
            if name is None:
                continue

            tensor = materialize_tensor(value)
            bucket[name] = tensor.to(
                device=self.bucket_device,
                dtype=self._transport_dtype(tensor.dtype),
                non_blocking=True,
            )
            currently_hold += bucket[name].numel() * bucket[name].element_size()

            if currently_hold >= self.bucket_size:
                if bucket_idx == 0:
                    bucket.update(metadata)
                yield bucket
                bucket_idx += 1
                bucket = {}
                currently_hold = 0

        if bucket:
            if bucket_idx == 0:
                bucket.update(metadata)
            yield bucket

    def divide_into_buckets(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        version: int | torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        return list(self.iter_buckets(state_dict, version))

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
            for key, value in bucket.items():
                cpu_buffer[key] = value.to("cpu", non_blocking=True)
        del bucket

        for _ in range(total_buckets - 1):
            bucket = await recv()
            if self.load_instant:
                model.load_state_dict(bucket, strict=False)
            else:
                for key, value in bucket.items():
                    cpu_buffer[key] = value.to("cpu", non_blocking=True)
            del bucket

        if not self.load_instant:
            model.load_state_dict(cpu_buffer, strict=False)
            del cpu_buffer

        return applied_version
