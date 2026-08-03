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

import time
from dataclasses import dataclass

import torch

from rlinf.utils.backbone_cache import BackboneOutput


@dataclass(frozen=True)
class PinnedRolloutBackboneCacheStats:
    bytes: int
    samples: int
    stored_samples: int
    loads: int
    loaded_samples: int
    allocation_seconds: float
    d2h_seconds: float
    cpu_gather_seconds: float
    h2d_seconds: float
    staging_bytes: int


class PinnedRolloutBackboneCache:
    """Actor-owned pinned cache populated from short-lived CUDA IPC views."""

    def __init__(
        self,
        *,
        total_samples: int,
        feature_example: torch.Tensor,
        mask_example: torch.Tensor,
        device: torch.device | str | int,
    ) -> None:
        if total_samples <= 0:
            raise ValueError("pinned rollout backbone cache requires positive samples")
        if feature_example.device.type != "cuda" or mask_example.device.type != "cuda":
            raise ValueError(
                "pinned rollout backbone source tensors must be CUDA tensors"
            )
        if feature_example.shape[0] != mask_example.shape[0]:
            raise ValueError("pinned rollout backbone feature/mask batch mismatch")

        self._device = (
            torch.device("cuda", device)
            if isinstance(device, int)
            else torch.device(device)
        )
        if self._device.type != "cuda":
            raise ValueError(
                "pinned rollout backbone destination must be a CUDA device"
            )
        self._samples = int(total_samples)
        allocation_start = time.perf_counter()
        self._features: torch.Tensor | None = torch.empty(
            (self._samples, *feature_example.shape[1:]),
            dtype=feature_example.dtype,
            device="cpu",
            pin_memory=True,
        )
        self._masks: torch.Tensor | None = torch.empty(
            (self._samples, *mask_example.shape[1:]),
            dtype=mask_example.dtype,
            device="cpu",
            pin_memory=True,
        )
        self._allocation_seconds = time.perf_counter() - allocation_start
        self._bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self._features, self._masks)
        )
        self._copy_stream = torch.cuda.Stream(device=self._device)
        self._stored_samples = 0
        self._loads = 0
        self._loaded_samples = 0
        self._d2h_seconds = 0.0
        self._cpu_gather_seconds = 0.0
        self._h2d_seconds = 0.0
        self._staging: list[dict[str, object]] = [{}, {}]

    @property
    def samples(self) -> int:
        return self._samples

    def store_block(
        self,
        *,
        offset: int,
        feature: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        self.store_blocks(offset=offset, blocks=[(feature, mask)])

    def store_blocks(
        self,
        *,
        offset: int,
        blocks: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        if self._features is None or self._masks is None:
            raise RuntimeError("pinned rollout backbone cache was cleared")
        if offset != self._stored_samples:
            raise ValueError(
                "pinned rollout backbone blocks must arrive in order: "
                f"expected offset {self._stored_samples}, got {offset}"
            )
        if not blocks:
            raise ValueError("pinned rollout backbone batch must not be empty")

        validated_blocks: list[tuple[torch.Tensor, torch.Tensor, int, int]] = []
        end = offset
        for feature, mask in blocks:
            if feature.device != self._device or mask.device != self._device:
                raise ValueError(
                    "pinned rollout backbone block is on the wrong CUDA device"
                )
            if feature.shape[0] != mask.shape[0]:
                raise ValueError("pinned rollout backbone feature/mask block mismatch")
            block_size = int(feature.shape[0])
            block_end = end + block_size
            if block_end > self._samples:
                raise ValueError("pinned rollout backbone block exceeds cache capacity")
            if tuple(feature.shape[1:]) != tuple(self._features.shape[1:]):
                raise ValueError("pinned rollout backbone feature shape changed")
            if tuple(mask.shape[1:]) != tuple(self._masks.shape[1:]):
                raise ValueError("pinned rollout backbone mask shape changed")
            if feature.dtype != self._features.dtype or mask.dtype != self._masks.dtype:
                raise ValueError("pinned rollout backbone dtype changed")
            validated_blocks.append((feature, mask, end, block_end))
            end = block_end

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self._copy_stream):
            start_event.record(self._copy_stream)
            for feature, mask, block_offset, block_end in validated_blocks:
                self._features[block_offset:block_end].copy_(
                    feature.detach(), non_blocking=True
                )
                self._masks[block_offset:block_end].copy_(
                    mask.detach(), non_blocking=True
                )
            end_event.record(self._copy_stream)
        end_event.synchronize()
        self._d2h_seconds += start_event.elapsed_time(end_event) / 1000.0
        self._stored_samples = end

    def finalize(self) -> None:
        if self._stored_samples != self._samples:
            raise RuntimeError(
                "pinned rollout backbone stream ended early: "
                f"stored {self._stored_samples} of {self._samples} samples"
            )

    def _complete_staging_slot(self, slot: dict[str, object]) -> None:
        end_event = slot.pop("end_event", None)
        start_event = slot.pop("start_event", None)
        if end_event is not None:
            assert isinstance(end_event, torch.cuda.Event)
            assert isinstance(start_event, torch.cuda.Event)
            end_event.synchronize()
            self._h2d_seconds += start_event.elapsed_time(end_event) / 1000.0

    def _get_staging_slot(self, sample_count: int) -> dict[str, object]:
        if self._features is None or self._masks is None:
            raise RuntimeError("pinned rollout backbone cache was cleared")
        slot = self._staging[self._loads % len(self._staging)]
        self._complete_staging_slot(slot)
        feature_shape = (sample_count, *self._features.shape[1:])
        mask_shape = (sample_count, *self._masks.shape[1:])
        feature_staging = slot.get("features")
        mask_staging = slot.get("masks")
        if not isinstance(feature_staging, torch.Tensor) or tuple(
            feature_staging.shape
        ) != tuple(feature_shape):
            feature_staging = torch.empty(
                feature_shape,
                dtype=self._features.dtype,
                device="cpu",
                pin_memory=True,
            )
            slot["features"] = feature_staging
        if not isinstance(mask_staging, torch.Tensor) or tuple(
            mask_staging.shape
        ) != tuple(mask_shape):
            mask_staging = torch.empty(
                mask_shape,
                dtype=self._masks.dtype,
                device="cpu",
                pin_memory=True,
            )
            slot["masks"] = mask_staging
        return slot

    def load(self, sample_ids: torch.Tensor) -> BackboneOutput:
        if self._features is None or self._masks is None:
            raise RuntimeError("pinned rollout backbone cache was cleared")
        if self._stored_samples != self._samples:
            raise RuntimeError("pinned rollout backbone cache is incomplete")
        sample_ids = sample_ids.reshape(-1).to(dtype=torch.int64, device="cpu")
        if sample_ids.numel() == 0:
            raise ValueError("pinned rollout backbone sample IDs must be non-empty")
        min_id = int(sample_ids.min().item())
        max_id = int(sample_ids.max().item())
        if min_id < 0 or max_id >= self._samples:
            raise IndexError(
                f"pinned rollout backbone sample IDs [{min_id}, {max_id}] exceed "
                f"cache size {self._samples}"
            )

        slot = self._get_staging_slot(sample_ids.numel())
        feature_staging = slot["features"]
        mask_staging = slot["masks"]
        assert isinstance(feature_staging, torch.Tensor)
        assert isinstance(mask_staging, torch.Tensor)
        gather_start = time.perf_counter()
        torch.index_select(self._features, 0, sample_ids, out=feature_staging)
        torch.index_select(self._masks, 0, sample_ids, out=mask_staging)
        self._cpu_gather_seconds += time.perf_counter() - gather_start

        feature_out = torch.empty_like(feature_staging, device=self._device)
        mask_out = torch.empty_like(mask_staging, device=self._device)
        stream = torch.cuda.current_stream(self._device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream)
        feature_out.copy_(feature_staging, non_blocking=True)
        mask_out.copy_(mask_staging, non_blocking=True)
        end_event.record(stream)
        slot["start_event"] = start_event
        slot["end_event"] = end_event
        self._loads += 1
        self._loaded_samples += sample_ids.numel()
        return {
            "backbone_features": feature_out,
            "backbone_attention_mask": mask_out,
        }

    def stats(self) -> PinnedRolloutBackboneCacheStats:
        for slot in self._staging:
            self._complete_staging_slot(slot)
        staging_bytes = sum(
            value.numel() * value.element_size()
            for slot in self._staging
            for value in slot.values()
            if isinstance(value, torch.Tensor)
        )
        return PinnedRolloutBackboneCacheStats(
            bytes=self._bytes,
            samples=self._samples,
            stored_samples=self._stored_samples,
            loads=self._loads,
            loaded_samples=self._loaded_samples,
            allocation_seconds=self._allocation_seconds,
            d2h_seconds=self._d2h_seconds,
            cpu_gather_seconds=self._cpu_gather_seconds,
            h2d_seconds=self._h2d_seconds,
            staging_bytes=staging_bytes,
        )

    def clear(self) -> None:
        for slot in self._staging:
            self._complete_staging_slot(slot)
            slot.clear()
        self._features = None
        self._masks = None
