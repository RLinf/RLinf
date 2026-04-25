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
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import torch
from torch.distributed.tensor import DTensor

from .base import (
    RecvFn,
    SendFn,
    WeightSyncer,
    materialize_tensor,
    normalize_device,
)
from .compressor import PatchCompressor


def downscale_nonnegative_indices(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.numel() == 0:
        return tensor.to(torch.uint8)
    max_value = int(tensor.max().item())
    if max_value <= torch.iinfo(torch.uint8).max:
        return tensor.to(torch.uint8)
    if max_value <= torch.iinfo(torch.int32).max:
        return tensor.to(torch.int32)
    return tensor.to(torch.int64)


def as_coo_2d_view(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    original_shape = tensor.shape
    if tensor.ndim == 0:
        view = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 1:
        view = tensor.unsqueeze(0)
    elif tensor.ndim == 2:
        view = tensor
    else:
        try:
            view = tensor.view(tensor.shape[0], -1)
        except RuntimeError as error:
            raise ValueError(
                "PatchWeightSyncer only supports ndim>=3 tensors whose trailing "
                "dimensions can be flattened as a view. "
                f"Got shape={tuple(tensor.shape)}, stride={tuple(tensor.stride())}."
            ) from error
    return view, original_shape


@dataclass
class EmptyWeightPatch:
    version: torch.Tensor

    def to(
        self, device: torch.device | str, non_blocking: bool = False
    ) -> "EmptyWeightPatch":
        device = normalize_device(device)
        return EmptyWeightPatch(
            version=self.version.to(device=device, non_blocking=non_blocking),
        )


@dataclass
class WeightPatch:
    version: torch.Tensor
    ordinals: torch.Tensor
    nnz_per_tensor: torch.Tensor
    rows: torch.Tensor
    cols: torch.Tensor
    values: torch.Tensor

    def to(
        self, device: torch.device | str, non_blocking: bool = False
    ) -> "WeightPatch":
        device = normalize_device(device)
        return WeightPatch(
            version=self.version.to(device=device, non_blocking=non_blocking),
            ordinals=self.ordinals.to(device=device, non_blocking=non_blocking),
            nnz_per_tensor=self.nnz_per_tensor.to(
                device=device, non_blocking=non_blocking
            ),
            rows=self.rows.to(device=device, non_blocking=non_blocking),
            cols=self.cols.to(device=device, non_blocking=non_blocking),
            values=self.values.to(device=device, non_blocking=non_blocking),
        )


@dataclass
class CompressedWeightPatch:
    version: torch.Tensor
    ordinals: torch.Tensor
    nnz_per_tensor: torch.Tensor
    rows_compressed: torch.Tensor
    cols_compressed: torch.Tensor
    values_compressed: torch.Tensor
    rows_dtype_code: torch.Tensor
    cols_dtype_code: torch.Tensor
    values_dtype_code: torch.Tensor


WeightPatchTransport = EmptyWeightPatch | WeightPatch | CompressedWeightPatch


class PatchBuilder(ABC):
    def __init__(
        self,
        snapshot: dict[str, torch.Tensor],
        ordered_keys: list[str],
        original_shapes: dict[str, torch.Size],
        delta_encoding: bool,
    ):
        self.snapshot = snapshot
        self.ordered_keys = ordered_keys
        self.original_shapes = original_shapes
        self.delta_encoding = delta_encoding

    @staticmethod
    def delta_encode(
        rows: torch.Tensor, cols: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert rows.numel() > 0, "No indices to encode"
        assert rows.numel() == cols.numel(), (
            "Rows and columns must have the same number of elements"
        )
        if rows.numel() == 1:
            return rows, cols

        row_deltas = torch.empty_like(rows)
        col_deltas = torch.empty_like(cols)
        row_deltas[0] = rows[0]
        col_deltas[0] = cols[0]
        row_deltas[1:] = rows[1:] - rows[:-1]

        same_row = rows[1:] == rows[:-1]
        col_deltas[1:] = torch.where(same_row, cols[1:] - cols[:-1], cols[1:])
        return row_deltas, col_deltas

    @staticmethod
    def delta_decode(
        rows_delta: torch.Tensor, cols_delta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert rows_delta.numel() > 0
        assert rows_delta.numel() == cols_delta.numel()

        rows = torch.cumsum(rows_delta, dim=0, dtype=torch.int64)

        start_mask = torch.zeros_like(rows_delta, dtype=torch.bool)
        start_mask[0] = True
        start_mask[1:] = rows_delta[1:] != 0

        idx = torch.arange(
            rows_delta.numel(), device=rows_delta.device, dtype=torch.int64
        )
        start_idx = torch.where(start_mask, idx, torch.zeros_like(idx))
        start_idx = torch.cummax(start_idx, dim=0).values

        cum_cols = torch.cumsum(cols_delta, dim=0, dtype=torch.int64)
        base = (cum_cols - cols_delta)[start_idx]
        cols = cum_cols - base
        return rows, cols

    @classmethod
    def create(
        cls,
        snapshot: dict[str, torch.Tensor],
        ordered_keys: list[str],
        original_shapes: dict[str, torch.Size],
        snapshot_device: torch.device,
        delta_encoding: bool,
    ) -> PatchBuilder:
        if snapshot_device.type == "cpu":
            return CPUSnapshotPatchBuilder(
                snapshot,
                ordered_keys,
                original_shapes,
                delta_encoding,
            )
        elif snapshot_device.type == "cuda":
            return GPUSnapshotPatchBuilder(
                snapshot,
                ordered_keys,
                original_shapes,
                delta_encoding,
            )
        else:
            raise ValueError(f"Unsupported snapshot device: {snapshot_device}")

    @abstractmethod
    def create_patch(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        version: torch.Tensor | int,
    ) -> EmptyWeightPatch | WeightPatch: ...

    def _validate_state_dict_keys(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
    ) -> None:
        if set(state_dict.keys()) != set(self.ordered_keys):
            raise ValueError("State dict keys do not match snapshot keys")


@dataclass
class _PrefetchedCPUSnapshot:
    ordinal: int
    key: str
    state_2dview: torch.Tensor
    snapshot_value: torch.Tensor
    snapshot_on_state_device: torch.Tensor
    copy_done: torch.cuda.Event


@dataclass
class _PendingSnapshotUpdate:
    snapshot_value: torch.Tensor
    rows: torch.Tensor
    cols: torch.Tensor
    values: torch.Tensor
    copy_done: torch.cuda.Event


class CPUSnapshotPatchBuilder(PatchBuilder):
    def __init__(
        self,
        snapshot: dict[str, torch.Tensor],
        ordered_keys: list[str],
        original_shapes: dict[str, torch.Size],
        delta_encoding: bool,
    ):
        super().__init__(
            snapshot=snapshot,
            ordered_keys=ordered_keys,
            original_shapes=original_shapes,
            delta_encoding=delta_encoding,
        )
        self._copy_streams: dict[torch.device, torch.cuda.Stream] = {}
        self._snapshot_flush_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="patch-snapshot-flush",
        )
        self._pending_snapshot_flush: Future[None] | None = None

    def create_patch(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        version: torch.Tensor | int,
    ) -> EmptyWeightPatch | WeightPatch:
        self._wait_pending_snapshot_flush()
        self._validate_state_dict_keys(state_dict)

        ordinals: list[torch.Tensor] = []
        nnz_per_tensor: list[torch.Tensor] = []
        row_chunks: list[torch.Tensor] = []
        col_chunks: list[torch.Tensor] = []
        value_byte_chunks: list[torch.Tensor] = []
        pending_snapshot_updates: list[_PendingSnapshotUpdate] = []
        patch_device: torch.device | None = None

        if not self.ordered_keys:
            raise RuntimeError("Snapshot contains no tensors")

        prefetched = self._prefetch_snapshot(state_dict, 0)
        for ordinal in range(len(self.ordered_keys)):
            current = prefetched
            prefetched = (
                self._prefetch_snapshot(state_dict, ordinal + 1)
                if ordinal + 1 < len(self.ordered_keys)
                else None
            )

            if patch_device is None:
                patch_device = current.state_2dview.device
            elif patch_device != current.state_2dview.device:
                raise ValueError(
                    "CPUSnapshotPatchBuilder requires all sender state_dict tensors "
                    "to be on the same CUDA device. "
                    f"Expected {patch_device}, got {current.state_2dview.device} "
                    f"for key={current.key}."
                )

            compute_stream = torch.cuda.current_stream(current.state_2dview.device)
            compute_stream.wait_event(current.copy_done)
            current.snapshot_on_state_device.record_stream(compute_stream)

            compare_value = current.state_2dview.to(
                device=current.state_2dview.device,
                dtype=current.snapshot_value.dtype,
                non_blocking=True,
                copy=False,
            )
            changed = compare_value.ne(current.snapshot_on_state_device)
            rows, cols = changed.nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue

            rows = rows.to(torch.int64)
            cols = cols.to(torch.int64)
            values = compare_value[rows, cols]

            pending_snapshot_updates.append(
                self._stage_snapshot_update(
                    snapshot_value=current.snapshot_value,
                    rows=rows,
                    cols=cols,
                    values=values,
                )
            )

            if self.delta_encoding:
                patch_rows, patch_cols = self.delta_encode(rows, cols)
            else:
                patch_rows, patch_cols = rows, cols

            ordinals.append(
                torch.tensor(current.ordinal, dtype=torch.int32, device=rows.device)
            )
            nnz_per_tensor.append(
                torch.tensor(values.numel(), dtype=torch.int32, device=rows.device)
            )
            row_chunks.append(patch_rows.contiguous())
            col_chunks.append(patch_cols.contiguous())
            value_byte_chunks.append(values.contiguous().view(torch.uint8))

        if row_chunks:
            rows_tensor = downscale_nonnegative_indices(torch.cat(row_chunks, dim=0))
            cols_tensor = downscale_nonnegative_indices(torch.cat(col_chunks, dim=0))
            patch = WeightPatch(
                version=torch.tensor(
                    version,
                    dtype=torch.int64,
                    device=row_chunks[0].device,
                ),
                ordinals=torch.stack(ordinals),
                nnz_per_tensor=torch.stack(nnz_per_tensor),
                rows=rows_tensor,
                cols=cols_tensor,
                values=torch.cat(value_byte_chunks, dim=0),
            )
            self._submit_snapshot_updates(pending_snapshot_updates)
            return patch

        if patch_device is None:
            raise RuntimeError("Snapshot contains no tensors")
        patch = EmptyWeightPatch(
            version=torch.tensor(version, dtype=torch.int64, device=patch_device)
        )
        self._submit_snapshot_updates(pending_snapshot_updates)
        return patch

    def _wait_pending_snapshot_flush(self) -> None:
        if self._pending_snapshot_flush is None:
            return
        self._pending_snapshot_flush.result()
        self._pending_snapshot_flush = None

    def _submit_snapshot_updates(
        self,
        pending_snapshot_updates: list[_PendingSnapshotUpdate],
    ) -> None:
        if not pending_snapshot_updates:
            return
        self._pending_snapshot_flush = self._snapshot_flush_executor.submit(
            self._flush_snapshot,
            pending_snapshot_updates,
        )

    def _get_copy_stream(self, device: torch.device) -> torch.cuda.Stream:
        if device.index is None:
            device = torch.device(device.type, torch.cuda.current_device())
        copy_stream = self._copy_streams.get(device)
        if copy_stream is None:
            copy_stream = torch.cuda.Stream(device=device)
            self._copy_streams[device] = copy_stream
        return copy_stream

    def _prefetch_snapshot(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        ordinal: int,
    ) -> _PrefetchedCPUSnapshot:
        key = self.ordered_keys[ordinal]
        value = materialize_tensor(state_dict[key])
        expected_shape = self.original_shapes[key]
        if value.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for key {key}: "
                f"expected {expected_shape}, got {value.shape}"
            )
        state_2dview, _ = as_coo_2d_view(value)
        if state_2dview.device.type != "cuda":
            raise ValueError(
                "CPUSnapshotPatchBuilder requires sender state_dict tensors "
                f"to be on CUDA. Got key={key}, device={state_2dview.device}."
            )

        snapshot_value = self.snapshot[key]
        if snapshot_value.device.type != "cpu":
            raise ValueError(
                "CPUSnapshotPatchBuilder requires snapshots to be on CPU. "
                f"Got key={key}, device={snapshot_value.device}."
            )

        copy_stream = self._get_copy_stream(state_2dview.device)
        with torch.cuda.stream(copy_stream):
            snapshot_on_state_device = snapshot_value.to(
                device=state_2dview.device,
                non_blocking=True,
                copy=True,
            )
            copy_done = torch.cuda.Event()
            copy_done.record(copy_stream)

        return _PrefetchedCPUSnapshot(
            ordinal=ordinal,
            key=key,
            state_2dview=state_2dview,
            snapshot_value=snapshot_value,
            snapshot_on_state_device=snapshot_on_state_device,
            copy_done=copy_done,
        )

    def _stage_snapshot_update(
        self,
        snapshot_value: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        values: torch.Tensor,
    ) -> _PendingSnapshotUpdate:
        rows_cpu = torch.empty_like(rows, device="cpu", pin_memory=True)
        cols_cpu = torch.empty_like(cols, device="cpu", pin_memory=True)
        values_cpu = torch.empty_like(values, device="cpu", pin_memory=True)

        with torch.cuda.device(values.device):
            stream = torch.cuda.current_stream(values.device)
            rows_cpu.copy_(rows, non_blocking=True)
            cols_cpu.copy_(cols, non_blocking=True)
            values_cpu.copy_(values, non_blocking=True)
            copy_done = torch.cuda.Event()
            copy_done.record(stream)

        return _PendingSnapshotUpdate(
            snapshot_value=snapshot_value,
            rows=rows_cpu,
            cols=cols_cpu,
            values=values_cpu,
            copy_done=copy_done,
        )

    def _flush_snapshot(
        self,
        pending_snapshot_updates: list[_PendingSnapshotUpdate],
    ) -> None:
        for update in pending_snapshot_updates:
            update.copy_done.synchronize()
            update.snapshot_value[update.rows, update.cols] = update.values


class GPUSnapshotPatchBuilder(PatchBuilder):
    def create_patch(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        version: torch.Tensor | int,
    ) -> EmptyWeightPatch | WeightPatch:
        self._validate_state_dict_keys(state_dict)

        ordinals: list[torch.Tensor] = []
        nnz_per_tensor: list[torch.Tensor] = []
        row_chunks: list[torch.Tensor] = []
        col_chunks: list[torch.Tensor] = []
        value_byte_chunks: list[torch.Tensor] = []
        patch_device: torch.device | None = None

        for ordinal, key in enumerate(self.ordered_keys):
            value = materialize_tensor(state_dict[key])
            expected_shape = self.original_shapes[key]
            if value.shape != expected_shape:
                raise ValueError(
                    f"Shape mismatch for key {key}: "
                    f"expected {expected_shape}, got {value.shape}"
                )
            value_2dview, _ = as_coo_2d_view(value)
            if value_2dview.device.type != "cuda":
                raise ValueError(
                    "GPUSnapshotPatchBuilder requires sender state_dict tensors "
                    f"to be on CUDA. Got key={key}, device={value_2dview.device}."
                )

            snapshot_value = self.snapshot[key]
            if snapshot_value.device.type != "cuda":
                raise ValueError(
                    "GPUSnapshotPatchBuilder requires snapshots to be on CUDA. "
                    f"Got key={key}, device={snapshot_value.device}."
                )
            if snapshot_value.device != value_2dview.device:
                raise ValueError(
                    "GPU snapshot and state tensor must be on the same CUDA device. "
                    f"Got key={key}, snapshot={snapshot_value.device}, "
                    f"state={value_2dview.device}."
                )
            if patch_device is None:
                patch_device = snapshot_value.device

            compare_value = value_2dview.to(
                device=snapshot_value.device,
                dtype=snapshot_value.dtype,
                non_blocking=True,
                copy=False,
            )
            changed = compare_value.ne(snapshot_value)
            rows, cols = changed.nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue

            rows = rows.to(torch.int64)
            cols = cols.to(torch.int64)
            values = compare_value[rows, cols]

            snapshot_value[rows, cols] = values

            if self.delta_encoding:
                rows, cols = self.delta_encode(rows, cols)

            ordinals.append(
                torch.tensor(ordinal, dtype=torch.int32, device=rows.device)
            )
            nnz_per_tensor.append(
                torch.tensor(values.numel(), dtype=torch.int32, device=rows.device)
            )
            row_chunks.append(rows.contiguous())
            col_chunks.append(cols.contiguous())
            value_byte_chunks.append(values.contiguous().view(torch.uint8))

        if row_chunks:
            rows_tensor = downscale_nonnegative_indices(torch.cat(row_chunks, dim=0))
            cols_tensor = downscale_nonnegative_indices(torch.cat(col_chunks, dim=0))
            return WeightPatch(
                version=torch.tensor(
                    version,
                    dtype=torch.int64,
                    device=row_chunks[0].device,
                ),
                ordinals=torch.stack(ordinals),
                nnz_per_tensor=torch.stack(nnz_per_tensor),
                rows=rows_tensor,
                cols=cols_tensor,
                values=torch.cat(value_byte_chunks, dim=0),
            )

        if patch_device is None:
            raise RuntimeError("Snapshot contains no tensors")
        return EmptyWeightPatch(
            version=torch.tensor(version, dtype=torch.int64, device=patch_device)
        )


class PatchWeightSyncer(WeightSyncer):
    def __init__(
        self,
        snapshot_device: torch.device | str = "cpu",
        transport_device: torch.device | str = "cuda",
        delta_encoding: bool = True,
        compression_algorithm: str = "none",
    ):
        super().__init__()
        self.snapshot: dict[str, torch.Tensor] | None = None
        self.original_shapes: dict[str, torch.Size] | None = None
        self.ordered_keys: list[str] | None = None
        self.patch_builder: PatchBuilder | None = None
        self.delta_encoding = delta_encoding
        self.transport_device = normalize_device(transport_device)
        self.snapshot_device = normalize_device(snapshot_device)
        self.compressor = PatchCompressor.create(
            compression_algorithm=compression_algorithm,
            transport_device=self.transport_device,
        )

    async def init_sender(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        send: SendFn,
        recv: RecvFn | None = None,
    ) -> None:
        assert not self.sender_initialized(), "Sender already initialized"
        del send
        if recv is None:
            raise ValueError("PatchWeightSyncer sender init requires a recv function")

        metadata = await recv()
        self.ordered_keys = metadata["ordered_keys"]
        self.original_shapes = metadata["original_shapes"]
        receiver_dtypes = metadata["receiver_dtypes"]
        if set(state_dict.keys()) != set(self.ordered_keys):
            raise ValueError("Sender state dict keys do not match receiver keys")

        with torch.no_grad():
            snapshot: dict[str, torch.Tensor] = {}
            for key in self.ordered_keys:
                value_2dview, original_shape = as_coo_2d_view(
                    materialize_tensor(state_dict[key])
                )
                if original_shape != self.original_shapes[key]:
                    raise ValueError(
                        f"Shape mismatch for key {key}: "
                        f"expected {self.original_shapes[key]}, got {original_shape}"
                    )
                if (
                    self.snapshot_device.type == "cpu"
                    and value_2dview.device.type != "cuda"
                ):
                    raise ValueError(
                        "CPU snapshot patch sync requires sender state_dict tensors "
                        f"to be on CUDA. Got key={key}, device={value_2dview.device}."
                    )
                snapshot_device = (
                    value_2dview.device
                    if self.snapshot_device.type == "cuda"
                    and self.snapshot_device.index is None
                    else self.snapshot_device
                )
                snapshot_value = value_2dview.detach().to(
                    device=snapshot_device,
                    dtype=receiver_dtypes[key],
                    non_blocking=self.snapshot_device.type != "cpu",
                    copy=True,
                )
                snapshot[key] = (
                    snapshot_value.pin_memory()
                    if self.snapshot_device.type == "cpu"
                    else snapshot_value
                )

        self.snapshot = snapshot
        self.patch_builder = PatchBuilder.create(
            self.snapshot,
            self.ordered_keys,
            self.original_shapes,
            self.snapshot_device,
            self.delta_encoding,
        )
        self._sender_initialized = True

    async def init_receiver(
        self,
        state_dict: dict[str, torch.Tensor | DTensor] | None,
        recv: RecvFn,
        send: SendFn | None = None,
    ) -> None:
        assert not self.receiver_initialized(), "Receiver already initialized"
        del recv
        if state_dict is None:
            raise ValueError("PatchWeightSyncer receiver init requires a state_dict")
        if send is None:
            raise ValueError("PatchWeightSyncer receiver init requires a send function")

        self.ordered_keys = []
        self.original_shapes = {}
        receiver_dtypes: dict[str, torch.dtype] = {}
        for key, tensor in state_dict.items():
            value_2dview, original_shape = as_coo_2d_view(materialize_tensor(tensor))
            self.ordered_keys.append(key)
            self.original_shapes[key] = original_shape
            receiver_dtypes[key] = value_2dview.dtype

        await send(
            {
                "ordered_keys": self.ordered_keys,
                "original_shapes": self.original_shapes,
                "receiver_dtypes": receiver_dtypes,
            }
        )
        self._receiver_initialized = True

    def delta_encode(
        self, rows: torch.Tensor, cols: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return PatchBuilder.delta_encode(rows, cols)

    def delta_decode(
        self, rows_delta: torch.Tensor, cols_delta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return PatchBuilder.delta_decode(rows_delta, cols_delta)

    @torch.no_grad()
    def create_patch(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        version: torch.Tensor | int,
    ) -> EmptyWeightPatch | WeightPatch:
        if self.patch_builder is None:
            raise RuntimeError("Snapshot not initialized")
        return self.patch_builder.create_patch(state_dict, version)

    async def sync(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        send: SendFn,
        version: int | torch.Tensor,
    ) -> None:
        patch = self.create_patch(state_dict, version)
        transport_patch = patch.to(
            device=self.transport_device,
            non_blocking=self.transport_device.type != "cpu",
        )
        if isinstance(transport_patch, EmptyWeightPatch):
            await send(transport_patch)
        else:
            await send(self.compressor.compress(transport_patch))

    @torch.no_grad()
    async def apply(self, model: torch.nn.Module, recv: RecvFn) -> int:
        assert self.ordered_keys is not None and self.original_shapes is not None, (
            "Snapshot info not initialized"
        )

        payload = await recv()

        if isinstance(payload, EmptyWeightPatch):
            return int(payload.version.item())
        patch = self.compressor.decompress(payload)
        applied_version = int(patch.version.item())
        total_nnz = int(patch.nnz_per_tensor.to(torch.int64).sum().item())
        assert patch.rows.numel() == patch.cols.numel(), (
            "Patch rows/cols must have the same number of elements: "
            f"rows={patch.rows.numel()}, cols={patch.cols.numel()}"
        )
        assert patch.rows.numel() == total_nnz, (
            "Patch payload size does not match nnz_per_tensor: "
            f"payload_nnz={patch.rows.numel()}, nnz_sum={total_nnz}"
        )

        state_dict = model.state_dict()

        offset = 0
        value_byte_offset = 0
        for patch_idx in range(patch.ordinals.numel()):
            key = self.ordered_keys[patch.ordinals[patch_idx].item()]
            original_shape = self.original_shapes[key]
            value = state_dict[key]
            value_2dview, _ = as_coo_2d_view(value)
            assert value.shape == original_shape, (
                f"Shape mismatch for key {key}: expected {original_shape}, got {value.shape}"
            )

            nnz = int(patch.nnz_per_tensor[patch_idx].item())
            start_offset = offset
            next_offset = start_offset + nnz

            row_slice = patch.rows[start_offset:next_offset].clone()
            col_slice = patch.cols[start_offset:next_offset].clone()
            offset = next_offset

            value_nbytes = nnz * value_2dview.element_size()
            value_byte_slice = patch.values[
                value_byte_offset : value_byte_offset + value_nbytes
            ]
            value_byte_offset += value_nbytes

            if self.delta_encoding:
                row_delta = row_slice
                col_delta = col_slice
                if row_delta.device != value_2dview.device:
                    row_delta = row_delta.to(
                        device=value_2dview.device,
                        non_blocking=False,
                    )
                if col_delta.device != value_2dview.device:
                    col_delta = col_delta.to(
                        device=value_2dview.device,
                        non_blocking=False,
                    )
                rows, cols = PatchBuilder.delta_decode(row_delta, col_delta)
            else:
                rows = row_slice.to(
                    device=value_2dview.device,
                    dtype=torch.int64,
                    non_blocking=False,
                )
                cols = col_slice.to(
                    device=value_2dview.device,
                    dtype=torch.int64,
                    non_blocking=False,
                )

            value_slice = value_byte_slice.clone().view(value_2dview.dtype)
            value_2dview[rows, cols] = value_slice.to(
                device=value_2dview.device,
                non_blocking=False,
            )

        assert offset == patch.rows.numel(), "Patch offsets do not match payload size"
        return applied_version
