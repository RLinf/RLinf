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

"""Fixed tensor-frame codecs for trajectory-channel payloads.

The first payload on an endpoint establishes a layout.  Later payloads with
the same tensor structure can be sent as a small integer header plus tensor
lanes, without serializing a dataclass skeleton or tensor metadata.
"""

from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import torch

from .data import TrajectoryData


@dataclass(frozen=True)
class TensorFrame:
    """One encoded trajectory payload."""

    header: torch.Tensor
    tensors: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class _TensorLeaf:
    index: int


@dataclass(frozen=True)
class _MetadataLeaf:
    name: str


@dataclass(frozen=True)
class _DataclassNode:
    type_: type
    values: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class _DictNode:
    values: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class _ListNode:
    values: tuple[Any, ...]


@dataclass(frozen=True)
class _TupleNode:
    values: tuple[Any, ...]


_HEADER_FIELDS = (
    "global_step",
    "rank",
    "current_epoch",
    "current_step",
    "stage_id",
    "slot_ids",
)


class TensorStorageAllocator:
    """Allocate receive storage for one immutable tensor-frame layout."""

    def __init__(self, tensors: tuple[torch.Tensor, ...]) -> None:
        self._templates = tensors

    def allocate(self) -> tuple[torch.Tensor, ...]:
        """Allocate independent buffers safe to hand to asynchronous consumers."""
        return tuple(torch.empty_like(tensor) for tensor in self._templates)


class FixedLayoutProtocol:
    """Encode one stable :class:`TrajectoryData` layout as tensor lanes.

    Non-tensor fields remain part of the layout and therefore must be stable.
    Per-message trajectory metadata is carried by the ``int64`` header.
    Callers should use the generic transport for messages that do not match the
    established layout.
    """

    def __init__(self, prototype: TrajectoryData) -> None:
        self._payload_type = type(prototype)
        tensors: list[torch.Tensor] = []
        self._layout = self._build_layout(prototype, tensors, is_root=True)
        self._tensor_specs = tuple(
            (tensor.shape, tensor.dtype, tensor.device) for tensor in tensors
        )
        lane_sizes: dict[tuple[torch.dtype, torch.device], int] = {}
        for tensor in tensors:
            lane_sizes[(tensor.dtype, tensor.device)] = (
                lane_sizes.get((tensor.dtype, tensor.device), 0) + tensor.numel()
            )
        self._lane_specs = tuple(
            (dtype, device, size) for (dtype, device), size in lane_sizes.items()
        )
        self._slot_count = len(prototype.slot_ids or ())

    @classmethod
    def from_payload(cls, payload: TrajectoryData) -> "FixedLayoutProtocol":
        """Build the fixed layout negotiated from an endpoint's first payload."""
        return cls(payload)

    @property
    def header_size(self) -> int:
        """Return the fixed number of int64 header values."""
        return len(_HEADER_FIELDS) + self._slot_count

    def allocate_receive_buffers(self) -> TensorStorageAllocator:
        """Allocate reusable receive buffers matching this protocol's lanes."""
        tensors = tuple(
            torch.empty(size, dtype=dtype, device=device)
            for dtype, device, size in self._lane_specs
        )
        return TensorStorageAllocator(tensors)

    def encode(self, payload: TrajectoryData) -> TensorFrame:
        """Encode a payload that matches the negotiated layout."""
        tensors: list[torch.Tensor] = []
        layout = self._build_layout(payload, tensors, is_root=True)
        if layout != self._layout:
            raise ValueError("Payload does not match the fixed trajectory layout.")
        if len(tensors) != len(self._tensor_specs):
            raise ValueError("Payload tensor count does not match the fixed layout.")
        for tensor, (shape, dtype, device) in zip(
            tensors, self._tensor_specs, strict=True
        ):
            if (
                tensor.shape != shape
                or tensor.dtype != dtype
                or tensor.device != device
            ):
                raise ValueError("Payload tensor spec does not match the fixed layout.")
            if not tensor.is_contiguous():
                raise ValueError("Fixed-layout tensor lanes must be contiguous.")
        return TensorFrame(
            header=self._encode_header(payload),
            tensors=self._pack_tensor_lanes(tensors),
        )

    def decode(
        self,
        header: torch.Tensor,
        tensors: tuple[torch.Tensor, ...],
    ) -> TrajectoryData:
        """Reconstruct one payload from received tensor lanes."""
        tensors = self._unpack_tensor_lanes(tensors)
        metadata = self._decode_header(header)
        payload = self._restore(self._layout, tensors, metadata)
        if not isinstance(payload, self._payload_type):
            raise TypeError("Fixed layout reconstructed an unexpected payload type.")
        return payload

    def _pack_tensor_lanes(
        self, tensors: list[torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        groups: dict[tuple[torch.dtype, torch.device], list[torch.Tensor]] = {}
        for tensor in tensors:
            groups.setdefault((tensor.dtype, tensor.device), []).append(tensor)
        return tuple(
            torch.cat([tensor.reshape(-1) for tensor in group]).contiguous()
            for group in groups.values()
        )

    def _unpack_tensor_lanes(
        self, lanes: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        groups: dict[
            tuple[torch.dtype, torch.device], list[tuple[int, torch.Size, int]]
        ] = {}
        for index, (shape, dtype, device) in enumerate(self._tensor_specs):
            numel = 1
            for dim in shape:
                numel *= dim
            groups.setdefault((dtype, device), []).append((index, shape, numel))
        if len(lanes) != len(groups):
            raise ValueError(
                "Received tensor lane count does not match the fixed layout."
            )
        tensors: list[torch.Tensor | None] = [None] * len(self._tensor_specs)
        for lane, ((dtype, device), specs) in zip(lanes, groups.items(), strict=True):
            if lane.dtype != dtype or lane.device != device:
                raise ValueError(
                    "Received tensor lane spec does not match the fixed layout."
                )
            offset = 0
            for index, shape, numel in specs:
                tensors[index] = lane.narrow(0, offset, numel).reshape(shape)
                offset += numel
            if offset != lane.numel():
                raise ValueError("Received tensor lane has an unexpected size.")
        if any(tensor is None for tensor in tensors):
            raise RuntimeError("Fixed layout did not restore every tensor leaf.")
        return tuple(tensor for tensor in tensors if tensor is not None)

    def _build_layout(
        self,
        value: Any,
        tensors: list[torch.Tensor],
        *,
        is_root: bool = False,
    ) -> Any:
        if isinstance(value, torch.Tensor):
            index = len(tensors)
            tensors.append(value)
            return _TensorLeaf(index)
        if is_dataclass(value) and not isinstance(value, type):
            values = []
            for field in fields(value):
                if is_root and field.name in _HEADER_FIELDS:
                    values.append((field.name, _MetadataLeaf(field.name)))
                else:
                    values.append(
                        (
                            field.name,
                            self._build_layout(getattr(value, field.name), tensors),
                        )
                    )
            return _DataclassNode(type(value), tuple(values))
        if isinstance(value, dict):
            return _DictNode(
                tuple(
                    (key, self._build_layout(nested_value, tensors))
                    for key, nested_value in value.items()
                )
            )
        if isinstance(value, list):
            return _ListNode(
                tuple(
                    self._build_layout(nested_value, tensors) for nested_value in value
                )
            )
        if isinstance(value, tuple):
            return _TupleNode(
                tuple(
                    self._build_layout(nested_value, tensors) for nested_value in value
                )
            )
        return value

    def _encode_header(self, payload: TrajectoryData) -> torch.Tensor:
        slot_ids = payload.slot_ids or ()
        if len(slot_ids) != self._slot_count:
            raise ValueError("Payload slot count does not match the fixed layout.")
        return torch.tensor(
            (
                payload.global_step,
                payload.rank,
                payload.current_epoch,
                payload.current_step,
                payload.stage_id,
                len(slot_ids),
                *slot_ids,
            ),
            dtype=torch.int64,
            device="cpu",
        )

    def _decode_header(self, header: torch.Tensor) -> dict[str, Any]:
        if header.dtype != torch.int64 or header.device.type != "cpu":
            raise ValueError("Trajectory frame headers must be CPU int64 tensors.")
        if not header.is_contiguous():
            raise ValueError("Trajectory frame headers must be contiguous.")
        expected_size = len(_HEADER_FIELDS) + self._slot_count
        if header.numel() != expected_size:
            raise ValueError("Trajectory frame header has an unexpected size.")
        values = header.tolist()
        slot_count = values[5]
        if slot_count != self._slot_count:
            raise ValueError("Trajectory frame header has an unexpected slot count.")
        return {
            "global_step": values[0],
            "rank": values[1],
            "current_epoch": values[2],
            "current_step": values[3],
            "stage_id": values[4],
            "slot_ids": tuple(values[6:]),
        }

    def _restore(
        self,
        layout: Any,
        tensors: tuple[torch.Tensor, ...],
        metadata: dict[str, Any],
    ) -> Any:
        if isinstance(layout, _TensorLeaf):
            return tensors[layout.index]
        if isinstance(layout, _MetadataLeaf):
            return metadata[layout.name]
        if isinstance(layout, _DataclassNode):
            return layout.type_(
                **{
                    name: self._restore(value, tensors, metadata)
                    for name, value in layout.values
                }
            )
        if isinstance(layout, _DictNode):
            return {
                name: self._restore(value, tensors, metadata)
                for name, value in layout.values
            }
        if isinstance(layout, _ListNode):
            return [self._restore(value, tensors, metadata) for value in layout.values]
        if isinstance(layout, _TupleNode):
            return tuple(
                self._restore(value, tensors, metadata) for value in layout.values
            )
        return layout


class TransportEndpoint:
    """Stateful endpoint that upgrades from bootstrap to fixed tensor frames."""

    def __init__(self) -> None:
        self._protocol: FixedLayoutProtocol | None = None
        self._allocator: TensorStorageAllocator | None = None

    @property
    def ready(self) -> bool:
        """Whether the endpoint has negotiated a fixed payload layout."""
        return self._protocol is not None

    def bootstrap(self, payload: TrajectoryData) -> None:
        """Learn the layout from a generic first payload."""
        if self._protocol is None:
            self._protocol = FixedLayoutProtocol.from_payload(payload)
            self._allocator = self._protocol.allocate_receive_buffers()

    def encode(self, payload: TrajectoryData) -> TensorFrame | None:
        """Encode a stable payload, or request generic fallback on mismatch."""
        if self._protocol is None:
            return None
        try:
            return self._protocol.encode(payload)
        except ValueError:
            return None

    def receive_buffers(self) -> tuple[torch.Tensor, ...]:
        """Return the buffers that the collective transport should receive into."""
        if self._allocator is None:
            raise RuntimeError("Transport endpoint has not completed bootstrap.")
        return self._allocator.allocate()

    def receive_frame_buffers(self) -> tuple[torch.Tensor, ...]:
        """Return the header and payload buffers in wire order."""
        if self._protocol is None:
            raise RuntimeError("Transport endpoint has not completed bootstrap.")
        return (
            torch.empty(
                self._protocol.header_size,
                dtype=torch.int64,
                device="cpu",
            ),
            *self.receive_buffers(),
        )

    def decode(
        self, header: torch.Tensor, tensors: tuple[torch.Tensor, ...]
    ) -> TrajectoryData:
        """Decode the endpoint's receive buffers using the negotiated layout."""
        if self._protocol is None or self._allocator is None:
            raise RuntimeError("Transport endpoint has not completed bootstrap.")
        return self._protocol.decode(header, tensors)
