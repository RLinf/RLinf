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
from dataclasses import dataclass

import torch
try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from rlinf.scheduler import Worker
from rlinf.utils.utils import (
    materialize_tensor,
    normalize_device,
    tensors_record_stream,
)

from .base import RecvFn, SendFn, WeightSyncer
from .bucket_syncer import BucketWeightSyncer, iter_named_tensor_buckets
from .compressor import PatchCompressor


def downscale_nonnegative_indices(tensor: torch.Tensor) -> torch.Tensor:
    """Cast nonnegative index tensors to the smallest supported integer dtype.

    Empty tensors are encoded as ``torch.uint8``. Non-empty tensors are cast to
    ``torch.uint8``, ``torch.int32``, or ``torch.int64`` according to the maximum
    value they contain. Callers must only pass nonnegative indices.

    Args:
        tensor: Index tensor whose values are expected to be nonnegative.

    Returns:
        A tensor with the same values stored in the smallest supported integer
        dtype.
    """
    if tensor.numel() == 0:
        return tensor.to(torch.uint8)
    max_value = int(tensor.max().item())
    if max_value <= torch.iinfo(torch.uint8).max:
        return tensor.to(torch.uint8)
    if max_value <= torch.iinfo(torch.int32).max:
        return tensor.to(torch.int32)
    return tensor.to(torch.int64)


def as_coo_2d_view(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    """View a tensor as a 2-D COO indexing target.

    Scalars are viewed as shape ``(1, 1)``, vectors as ``(1, N)``, matrices are
    returned as-is, and higher-rank tensors are viewed as
    ``(shape[0], prod(shape[1:]))``. Higher-rank tensors must be flattenable as a
    view, so non-contiguous layouts that require a copy are rejected.

    Args:
        tensor: Tensor to expose as a 2-D view for COO row/column indexing.

    Returns:
        A tuple ``(view, original_shape)`` where ``view`` is the 2-D tensor view
        and ``original_shape`` is the input tensor shape.

    Raises:
        ValueError: If a tensor with rank three or higher cannot flatten its
            trailing dimensions as a view.
    """
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
    """
    Patch payload used when no synchronized tensor values changed.

    Attributes:
        version: Weight version represented by this empty patch.
    """

    version: torch.Tensor

    def to(
        self, device: torch.device | str, non_blocking: bool = False
    ) -> "EmptyWeightPatch":
        """
        Move the empty patch metadata to a device.

        Args:
            device: Target device for the version tensor.
            non_blocking: Whether to request non-blocking tensor movement.

        Returns:
            A new ``EmptyWeightPatch`` whose tensor fields are on ``device``.
        """
        device = normalize_device(device)
        return EmptyWeightPatch(
            version=self.version.to(device=device, non_blocking=non_blocking),
        )

    def tensors(self) -> list[torch.Tensor]:
        """
        Return a list of all tensor fields in the empty patch.

        Returns:
            A list of all tensor fields in the order they are defined in the
            dataclass.
        """
        return [self.version]


@dataclass
class WeightPatch:
    """Sparse patch payload for changed state dict entries.

    The patch stores changed tensors in COO-like form. ``ordinals`` identifies
    which state dict entries changed, ``nnz_per_tensor`` stores how many changed
    values belong to each ordinal, ``rows`` and ``cols`` store the changed
    positions in each entry's 2-D view, and ``values`` stores the raw bytes of
    the changed values.

    Attributes:
        version: Weight version represented by this patch.
        ordinals: State dict key ordinals for tensors with at least one changed
            value.
        nnz_per_tensor: Number of changed values for each ordinal.
        rows: Row indices for changed values, concatenated across tensors.
        cols: Column indices for changed values, concatenated across tensors.
        values: Raw byte representation of changed values, concatenated across
            tensors.
    """

    version: torch.Tensor
    ordinals: torch.Tensor
    nnz_per_tensor: torch.Tensor
    rows: torch.Tensor
    cols: torch.Tensor
    values: torch.Tensor

    def to(
        self, device: torch.device | str, non_blocking: bool = False
    ) -> "WeightPatch":
        """Move all patch tensors to a device.

        Args:
            device: Target device for every tensor field.
            non_blocking: Whether to request non-blocking tensor movement.

        Returns:
            A new ``WeightPatch`` whose tensor fields are on ``device``.
        """
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

    def tensors(self) -> list[torch.Tensor]:
        """Return a list of all tensor fields in the patch.

        Returns:
            A list of all tensor fields in the order they are defined in the
            dataclass.
        """
        return [
            self.version,
            self.ordinals,
            self.nnz_per_tensor,
            self.rows,
            self.cols,
            self.values,
        ]


@dataclass
class CompressedWeightPatch:
    """Compressed sparse patch payload.

    This payload mirrors ``WeightPatch`` but stores the row indices, column
    indices, and raw value bytes in compressed byte tensors. The dtype-code
    tensors record the original dtypes needed by the compressor to reconstruct
    each compressed field.

    Attributes:
        version: Weight version represented by this patch.
        ordinals: State dict key ordinals for tensors with changed values.
        nnz_per_tensor: Number of changed values for each ordinal.
        rows_compressed: Compressed representation of the row-index tensor.
        cols_compressed: Compressed representation of the column-index tensor.
        values_compressed: Compressed representation of the raw value bytes.
        rows_dtype_code: Encoded dtype metadata for decompressed rows.
        cols_dtype_code: Encoded dtype metadata for decompressed columns.
        values_dtype_code: Encoded dtype metadata for decompressed values.
    """

    version: torch.Tensor
    ordinals: torch.Tensor
    nnz_per_tensor: torch.Tensor
    rows_compressed: torch.Tensor
    cols_compressed: torch.Tensor
    values_compressed: torch.Tensor
    rows_dtype_code: torch.Tensor
    cols_dtype_code: torch.Tensor
    values_dtype_code: torch.Tensor

    def tensors(self) -> list[torch.Tensor]:
        """Return a list of all tensor fields in the compressed patch.

        Returns:
            A list of all tensor fields in the order they are defined in the
            dataclass.
        """
        return [
            self.version,
            self.ordinals,
            self.nnz_per_tensor,
            self.rows_compressed,
            self.cols_compressed,
            self.values_compressed,
            self.rows_dtype_code,
            self.cols_dtype_code,
            self.values_dtype_code,
        ]


WeightPatchTransport = EmptyWeightPatch | WeightPatch | CompressedWeightPatch


class PatchBuilder(ABC):
    def __init__(
        self,
        snapshot: dict[str, torch.Tensor],
        ordered_keys: list[str],
        param_names_need_sync: list[str],
        original_shapes: dict[str, torch.Size],
        delta_encoding: bool,
    ):
        self.snapshot = snapshot
        self.ordered_keys = ordered_keys
        self.param_names_need_sync = param_names_need_sync
        self.param_names_need_sync_set = set(param_names_need_sync)
        self.original_shapes = original_shapes
        self.delta_encoding = delta_encoding
        self.param_names_need_sync_ordinals: dict[str, int] = {
            name: ordinal
            for ordinal, name in enumerate(self.ordered_keys)
            if name in self.param_names_need_sync_set
        }

        if not self.param_names_need_sync:
            raise ValueError("param_names_need_sync must not be empty")

        if not self.ordered_keys:
            raise ValueError("ordered_keys must not be empty")

    @staticmethod
    def delta_encode(
        rows: torch.Tensor, cols: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Delta-encode COO row and column indices for compact transport.

        The input indices are expected to be ordered as produced by
        ``Tensor.nonzero(as_tuple=True)`` on a 2-D view: rows are nondecreasing,
        and columns are ordered within each row. Row indices are encoded as
        first-order deltas. Column indices are encoded as deltas within the same
        row and reset to the absolute column value when a new row starts.

        Args:
            rows: One-dimensional row indices for changed values.
            cols: One-dimensional column indices for changed values. Must have
                the same number of elements as ``rows``.

        Returns:
            A tuple ``(row_deltas, col_deltas)`` with the same shapes and dtypes
            as the input tensors.
        """
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
        """Decode row and column deltas produced by ``delta_encode``.

        Row indices are reconstructed with a cumulative sum. Column deltas are
        accumulated within each row segment and reset whenever the decoded row
        changes. The returned tensors use ``torch.int64`` so they can be used
        directly as PyTorch advanced-indexing inputs after transport dtypes have
        been downscaled.

        Args:
            rows_delta: One-dimensional row deltas produced by ``delta_encode``.
            cols_delta: One-dimensional column deltas produced by
                ``delta_encode``. Must have the same number of elements as
                ``rows_delta``.

        Returns:
            A tuple ``(rows, cols)`` containing decoded ``torch.int64`` indices.
        """
        if rows_delta.numel() == 0:
            raise ValueError("No indices to decode")
        if rows_delta.numel() != cols_delta.numel():
            raise ValueError("Rows and columns must have the same number of elements")

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
        param_names_need_sync: list[str],
        original_shapes: dict[str, torch.Size],
        snapshot_device: torch.device,
        delta_encoding: bool,
    ) -> PatchBuilder:
        if snapshot_device.type == "cpu":
            return CPUSnapshotPatchBuilder(
                snapshot,
                ordered_keys,
                param_names_need_sync,
                original_shapes,
                delta_encoding,
            )
        elif snapshot_device.type == Worker.torch_device_type:
            return GPUSnapshotPatchBuilder(
                snapshot,
                ordered_keys,
                param_names_need_sync,
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


@dataclass
class _PrefetchedCPUSnapshot:
    ordinal: int
    global_ordinal: int
    key: str
    state_2dview: torch.Tensor
    snapshot_value: torch.Tensor
    snapshot_on_state_device: torch.Tensor
    copy_done: torch.Event


class CPUSnapshotPatchBuilder(PatchBuilder):
    def __init__(
        self,
        snapshot: dict[str, torch.Tensor],
        ordered_keys: list[str],
        param_names_need_sync: list[str],
        original_shapes: dict[str, torch.Size],
        delta_encoding: bool,
    ):
        super().__init__(
            snapshot=snapshot,
            ordered_keys=ordered_keys,
            param_names_need_sync=param_names_need_sync,
            original_shapes=original_shapes,
            delta_encoding=delta_encoding,
        )
        self._copy_streams: dict[torch.device, torch.Stream] = {}

    def create_patch(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        version: torch.Tensor | int,
    ) -> EmptyWeightPatch | WeightPatch:
        ordinals: list[torch.Tensor] = []
        nnz_per_tensor: list[torch.Tensor] = []
        row_chunks: list[torch.Tensor] = []
        col_chunks: list[torch.Tensor] = []
        value_byte_chunks: list[torch.Tensor] = []
        patch_device: torch.device | None = None

        prefetched = self._prefetch_snapshot(state_dict, 0)
        for ordinal in range(len(self.param_names_need_sync)):
            current = prefetched
            prefetched = (
                self._prefetch_snapshot(state_dict, ordinal + 1)
                if ordinal + 1 < len(self.param_names_need_sync)
                else None
            )

            if patch_device is None:
                patch_device = current.state_2dview.device
            elif patch_device != current.state_2dview.device:
                raise ValueError(
                    "CPUSnapshotPatchBuilder requires all sender state_dict tensors "
                    "to be on the same accelerator. "
                    f"Expected {patch_device}, got {current.state_2dview.device} "
                    f"for key={current.key}."
                )

            compute_stream = Worker.torch_platform.current_stream(
                current.state_2dview.device
            )
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

            self._update_snapshot(
                snapshot_value=current.snapshot_value,
                rows=rows,
                cols=cols,
                values=values,
            )

            if self.delta_encoding:
                patch_rows, patch_cols = self.delta_encode(rows, cols)
            else:
                patch_rows, patch_cols = rows, cols

            ordinals.append(
                torch.tensor(
                    current.global_ordinal, dtype=torch.int32, device=rows.device
                )
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
            return patch

        if patch_device is None:
            raise RuntimeError("Snapshot contains no tensors")
        patch = EmptyWeightPatch(
            version=torch.tensor(version, dtype=torch.int64, device=patch_device)
        )
        return patch

    def _get_copy_stream(self, device: torch.device | None) -> torch.Stream:
        if device.index is None:
            device = torch.device(device.type, Worker.torch_platform.current_device())
        copy_stream = self._copy_streams.get(device)
        if copy_stream is None:
            copy_stream = Worker.torch_platform.Stream(device=device)
            self._copy_streams[device] = copy_stream
        return copy_stream

    def _prefetch_snapshot(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        ordinal: int,
    ) -> _PrefetchedCPUSnapshot:
        key = self.param_names_need_sync[ordinal]
        value = materialize_tensor(state_dict[key])
        expected_shape = self.original_shapes[key]
        if value.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for key {key}: "
                f"expected {expected_shape}, got {value.shape}"
            )
        state_2dview, _ = as_coo_2d_view(value)
        if state_2dview.device.type != Worker.torch_device_type:
            raise ValueError(
                "CPUSnapshotPatchBuilder requires sender state_dict tensors "
                f"to be on accelerator. Got key={key}, device={state_2dview.device}."
            )

        snapshot_value = self.snapshot[key]
        if snapshot_value.device.type != "cpu":
            raise ValueError(
                "CPUSnapshotPatchBuilder requires snapshots to be on CPU. "
                f"Got key={key}, device={snapshot_value.device}."
            )

        copy_stream = self._get_copy_stream(state_2dview.device)
        with Worker.torch_platform.stream(copy_stream):
            snapshot_on_state_device = snapshot_value.to(
                device=state_2dview.device,
                non_blocking=True,
                copy=True,
            )
            copy_done = Worker.torch_platform.Event()
            copy_done.record(copy_stream)

        return _PrefetchedCPUSnapshot(
            ordinal=ordinal,
            global_ordinal=self.param_names_need_sync_ordinals[key],
            key=key,
            state_2dview=state_2dview,
            snapshot_value=snapshot_value,
            snapshot_on_state_device=snapshot_on_state_device,
            copy_done=copy_done,
        )

    def _update_snapshot(
        self,
        snapshot_value: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        rows_cpu = rows.to(device="cpu", non_blocking=False)
        cols_cpu = cols.to(device="cpu", non_blocking=False)
        values_cpu = values.to(device="cpu", non_blocking=False)
        snapshot_value[rows_cpu, cols_cpu] = values_cpu


class GPUSnapshotPatchBuilder(PatchBuilder):
    def create_patch(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        version: torch.Tensor | int,
    ) -> EmptyWeightPatch | WeightPatch:
        ordinals: list[torch.Tensor] = []
        nnz_per_tensor: list[torch.Tensor] = []
        row_chunks: list[torch.Tensor] = []
        col_chunks: list[torch.Tensor] = []
        value_byte_chunks: list[torch.Tensor] = []
        patch_device: torch.device | None = None

        for param_name in self.param_names_need_sync:
            ordinal = self.param_names_need_sync_ordinals[param_name]
            value = materialize_tensor(state_dict[param_name])
            expected_shape = self.original_shapes[param_name]
            if value.shape != expected_shape:
                raise ValueError(
                    f"Shape mismatch for key {param_name}: "
                    f"expected {expected_shape}, got {value.shape}"
                )
            value_2dview, _ = as_coo_2d_view(value)
            if value_2dview.device.type != Worker.torch_device_type:
                raise ValueError(
                    "GPUSnapshotPatchBuilder requires sender state_dict tensors "
                    f"to be on accelerator. Got key={param_name}, device={value_2dview.device}."
                )

            snapshot_value = self.snapshot[param_name]
            if snapshot_value.device.type != Worker.torch_device_type:
                raise ValueError(
                    "GPUSnapshotPatchBuilder requires snapshots to be on accelerator. "
                    f"Got key={param_name}, device={snapshot_value.device}."
                )
            if snapshot_value.device != value_2dview.device:
                raise ValueError(
                    "GPU snapshot and state tensor must be on the same accelerator. "
                    f"Got key={param_name}, snapshot={snapshot_value.device}, "
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
        transport_device: torch.device | str = None,
        delta_encoding: bool = True,
        compression_algorithm: str = "none",
        init_sync_enabled: bool = False,
        init_sync_prefixes: list[str] | None = None,
        init_sync_bucket_size: int = 128 * 1024 * 1024,
    ):
        super().__init__()
        self.snapshot: dict[str, torch.Tensor] | None = None
        self.original_shapes: dict[str, torch.Size] | None = None
        self.ordered_keys: list[str] | None = None
        self.patch_builder: PatchBuilder | None = None
        self.delta_encoding = delta_encoding
        self.transport_device = normalize_device(transport_device)
        self.snapshot_device = normalize_device(snapshot_device)
        self.init_sync_enabled = init_sync_enabled
        self.init_sync_prefixes = (
            None
            if init_sync_prefixes is None
            else [str(prefix) for prefix in init_sync_prefixes]
        )
        if self.init_sync_enabled and self.init_sync_prefixes == []:
            raise ValueError("Patch init sync prefixes must not be empty")
        self.init_sync_bucket_size = init_sync_bucket_size
        self.compressor = PatchCompressor.create(
            compression_algorithm=compression_algorithm,
            transport_device=self.transport_device,
        )

    def _select_init_sync_weights(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
    ) -> list[tuple[str, torch.Tensor | DTensor]]:
        if self.init_sync_prefixes is None:
            return list(state_dict.items())

        matched_prefixes = dict.fromkeys(self.init_sync_prefixes, False)
        selected_weights: list[tuple[str, torch.Tensor | DTensor]] = []
        for key, value in state_dict.items():
            for prefix in self.init_sync_prefixes:
                if key == prefix or key.startswith(f"{prefix}."):
                    matched_prefixes[prefix] = True
                    selected_weights.append((key, value))
                    break

        unmatched_prefixes = [
            prefix for prefix, matched in matched_prefixes.items() if not matched
        ]
        if unmatched_prefixes:
            raise ValueError(
                "Patch init sync prefixes did not match any state_dict keys: "
                f"{unmatched_prefixes}"
            )

        return selected_weights

    async def _sync_init_weights(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        receiver_dtypes: dict[str, torch.dtype],
        send: SendFn,
    ) -> None:
        selected_weights = self._select_init_sync_weights(state_dict)
        for key, _ in selected_weights:
            if key not in receiver_dtypes:
                raise ValueError(
                    f"Patch init sync sender key {key} does not exist on receiver"
                )

        for bucket in iter_named_tensor_buckets(
            selected_weights,
            0,
            bucket_size=self.init_sync_bucket_size,
            bucket_device=self.transport_device,
            dtype_resolver=lambda key, _dtype: receiver_dtypes[key],
        ):
            await send(bucket)

    @torch.no_grad()
    def _apply_init_weight_bucket(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        bucket: dict[str, torch.Tensor],
    ) -> list[torch.Tensor]:
        fallback_keepalive: list[torch.Tensor] = []
        for key, value in bucket.items():
            if key not in state_dict:
                raise ValueError(
                    f"Patch init sync receiver key {key} does not exist in state_dict"
                )
            target = state_dict[key]
            if isinstance(target, DTensor):
                raise TypeError(
                    "Patch init sync receiver does not support DTensor state_dict values"
                )
            target.copy_(value, non_blocking=True)
        if self.transport_device.type == Worker.torch_device_type:
            fallback_keepalive.extend(tensors_record_stream(bucket.values()))
        return fallback_keepalive

    async def _apply_init_weights(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        recv: RecvFn,
    ) -> None:
        bucket = await recv()
        if not isinstance(bucket, dict):
            raise TypeError(
                "Patch init sync receiver expected a bucket payload dictionary"
            )

        total_buckets = int(bucket.pop(BucketWeightSyncer._TOTAL_BUCKETS_KEY).item())
        bucket.pop(BucketWeightSyncer._SYNCER_VERSION_KEY)
        fallback_keepalive: list[torch.Tensor] = []
        fallback_keepalive.extend(self._apply_init_weight_bucket(state_dict, bucket))

        for _ in range(total_buckets - 1):
            bucket = await recv()
            if not isinstance(bucket, dict):
                raise TypeError(
                    "Patch init sync receiver expected a bucket payload dictionary"
                )
            fallback_keepalive.extend(
                self._apply_init_weight_bucket(state_dict, bucket)
            )
        Worker.torch_platform.current_stream().synchronize()
        fallback_keepalive.clear()

    async def init_sender(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        param_names_need_sync: list[str],
        send: SendFn,
        recv: RecvFn | None = None,
    ) -> None:
        assert not self.sender_initialized(), "Sender already initialized"
        if recv is None:
            raise ValueError("PatchWeightSyncer sender init requires a recv function")

        metadata = await recv()
        self.ordered_keys = metadata["ordered_keys"]
        self.original_shapes = metadata["original_shapes"]
        self.param_names_need_sync = param_names_need_sync
        receiver_dtypes = metadata["receiver_dtypes"]

        if set(state_dict.keys()) != set(self.ordered_keys):
            raise ValueError("Sender state dict keys do not match receiver keys")

        if self.init_sync_enabled:
            await self._sync_init_weights(state_dict, receiver_dtypes, send)

        with torch.no_grad():
            snapshot: dict[str, torch.Tensor] = {}
            for key in self.param_names_need_sync:
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
                    and value_2dview.device.type != Worker.torch_device_type
                ):
                    raise ValueError(
                        "CPU snapshot patch sync requires sender state_dict tensors "
                        f"to be on accelerator. Got key={key}, device={value_2dview.device}."
                    )
                snapshot_device = (
                    value_2dview.device
                    if self.snapshot_device.type == Worker.torch_device_type
                    and self.snapshot_device.index is None
                    else self.snapshot_device
                )
                snapshot_value = value_2dview.detach().to(
                    device=snapshot_device,
                    dtype=receiver_dtypes[key],
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
            self.param_names_need_sync,
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
        if self.init_sync_enabled:
            await self._apply_init_weights(state_dict, recv)
        self._receiver_initialized = True

    @torch.no_grad()
    def create_patch(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        version: torch.Tensor | int,
    ) -> EmptyWeightPatch | WeightPatch:
        if self.patch_builder is None:
            raise RuntimeError("Snapshot not initialized")
        if self.ordered_keys is None:
            raise RuntimeError("Snapshot metadata not initialized")
        if set(state_dict.keys()) != set(self.ordered_keys):
            raise ValueError("State dict keys do not match snapshot keys")
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

        payload: WeightPatchTransport = await recv()

        fallback_keepalive: list[torch.Tensor] = []
        if self.transport_device.type == Worker.torch_device_type:
            fallback_keepalive = tensors_record_stream(payload.tensors())

        if isinstance(payload, EmptyWeightPatch):
            if fallback_keepalive:
                Worker.torch_platform.current_stream().synchronize()
                fallback_keepalive.clear()
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

        if fallback_keepalive:
            Worker.torch_platform.current_stream().synchronize()
            fallback_keepalive.clear()

        return applied_version
