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

from collections.abc import Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate

from rlinf.scheduler import Worker
from rlinf.utils.utils import (
    dtype_size,
    materialize_tensor,
    normalize_device,
    normalize_dtype,
    tensors_record_stream,
)

from .base import RecvFn, SendFn, WeightSyncer

RecvIntoFn = Callable[[torch.Tensor], Awaitable[None]]

# 16-byte alignment satisfies all scalar/vector dtypes (up to 128-bit) and GPU
# coalesced loads; bucket 0 reserves the first 16 bytes for the int64 version.
_ALIGNMENT = 16
_HEADER_BYTES = 16

_STR_TO_DTYPE: dict[str, torch.dtype] = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}
_DTYPE_TO_STR: dict[torch.dtype, str] = {v: k for k, v in _STR_TO_DTYPE.items()}


def _align_up(offset: int) -> int:
    return (offset + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT


def _dtype_to_str(dtype: torch.dtype) -> str:
    name = _DTYPE_TO_STR.get(dtype)
    if name is None:
        raise ValueError(f"Unsupported dtype for flat bucket sync: {dtype}")
    return name


def _str_to_dtype(name: str) -> torch.dtype:
    dtype = _STR_TO_DTYPE.get(name)
    if dtype is None:
        raise ValueError(f"Unsupported serialized dtype name: {name}")
    return dtype


def _materialize_local_or_full(tensor: torch.Tensor | DTensor) -> torch.Tensor:
    """Return a dense local tensor, avoiding full_tensor() for replicated DTensors."""
    if isinstance(tensor, DTensor):
        if all(isinstance(p, Replicate) for p in tensor.placements):
            return tensor.to_local()
        return tensor.full_tensor()
    return materialize_tensor(tensor)


def _to_local(tensor: torch.Tensor | DTensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _resolve_module_refs(
    module: torch.nn.Module,
) -> dict[str, tuple[torch.nn.Module, str]]:
    """Map parameter and buffer names to ``(parent_submodule, attr_name)``.

    Resolving attributes through the owning submodule on each sync keeps
    references valid across FSDP2 parameter swaps between sharded and unsharded
    views.
    """
    refs: dict[str, tuple[torch.nn.Module, str]] = {}
    for mod_name, submod in module.named_modules(remove_duplicate=False):
        prefix = f"{mod_name}." if mod_name else ""
        for name, _ in submod.named_parameters(recurse=False, remove_duplicate=False):
            refs[f"{prefix}{name}"] = (submod, name)
        for name, _ in submod.named_buffers(recurse=False, remove_duplicate=False):
            refs[f"{prefix}{name}"] = (submod, name)
    return refs


def _bucket_key(key: str, has_visual: bool, is_agent: bool) -> str | None:
    if "_extra_state" in key:
        return None
    if has_visual and is_agent and key.startswith("model.language_model."):
        return "model." + key[len("model.language_model.") :]
    return key


@dataclass(frozen=True)
class _TensorSlot:
    """Layout of one tensor inside a flat byte bucket."""

    src_key: str
    dst_key: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    offset: int
    nbytes: int


@dataclass(frozen=True)
class _FlatBucketSpec:
    """Layout of one contiguous byte bucket."""

    total_bytes: int
    has_header: bool
    slots: tuple[_TensorSlot, ...]


class FlatBucketWeightSyncer(WeightSyncer):
    """Synchronize weights by packing each bucket into one reusable flat buffer.

    During one-time initialization (``init_sender`` / ``init_receiver``), the
    sender partitions parameters into 16-byte-aligned buckets capped at
    ``bucket_size`` and sends the compact layout plan to receivers. Subsequent
    syncs transfer one contiguous ``uint8`` tensor per bucket without per-step
    dict metadata or ``load_state_dict`` overhead.

    Peak extra memory is bounded by the largest single bucket (at most
    ``bucket_size`` unless an individual tensor exceeds it): small models
    collapse into a single bucket while large models reuse one fixed buffer
    across buckets. For unsharded 1:1 actor/rollout topologies, callers can
    pair ``sync_module`` with ``send_tensor`` and ``apply(..., recv_into=...)``
    for zero-metadata P2P (CUDA IPC on colocated GPUs).

    Attributes:
        bucket_size: Target maximum bucket payload size in bytes. A single
            tensor is never split, so one bucket may exceed this limit.
        bucket_dtype: Optional dtype used to transport floating-point tensors.
            Non-floating tensors keep their original dtype.
        bucket_device: Device where the reusable flat staging buffer lives.
        is_agent: Whether to rewrite ``model.language_model.`` keys for VLM
            agents, matching :class:`BucketWeightSyncer`.
        load_instant: Whether to unpack each bucket into the receiver model as
            it arrives rather than staging all buckets on CPU first.
        enable_p2p: Whether workers may use the 1:1 P2P ``send_tensor`` /
            ``recv_tensor`` fast path when weights are unsharded.
    """

    def __init__(
        self,
        bucket_size: int,
        bucket_dtype: torch.dtype | str | None = None,
        bucket_device: str | torch.device | None = None,
        is_agent: bool = False,
        load_instant: bool = True,
        enable_p2p: bool = True,
    ):
        super().__init__()
        if bucket_size <= 0:
            raise ValueError(f"bucket_size must be positive, got {bucket_size}")
        self.bucket_size = int(bucket_size)
        self.bucket_dtype = normalize_dtype(bucket_dtype)
        resolved_device = (
            Worker.torch_device_type if bucket_device is None else bucket_device
        )
        self.bucket_device = normalize_device(resolved_device)
        self.is_agent = bool(is_agent)
        self.load_instant = bool(load_instant)
        self.enable_p2p = bool(enable_p2p)

        self._is_sender: bool = False
        self._sender_buckets: list[_FlatBucketSpec] = []
        self._sender_buffer: torch.Tensor | None = None
        self._sender_views: list[list[torch.Tensor]] = []
        self._sender_module_id: int | None = None
        self._sender_module_refs: list[list[tuple[torch.nn.Module, str]]] = []

        self._receiver_buckets: list[_FlatBucketSpec] = []
        self._receiver_max_bytes: int = 0
        self._receiver_buffer: torch.Tensor | None = None
        self._receiver_model_id: int | None = None
        self._receiver_model_refs: list[list[tuple[torch.nn.Module, str] | None]] = []

    def _transport_dtype(self, dtype: torch.dtype) -> torch.dtype:
        if self.bucket_dtype is not None and dtype.is_floating_point:
            return self.bucket_dtype
        return dtype

    def _build_plan(
        self,
        named_items: Iterable[tuple[str, torch.Size, torch.dtype]],
        has_visual: bool,
    ) -> list[_FlatBucketSpec]:
        buckets: list[_FlatBucketSpec] = []
        slots: list[_TensorSlot] = []
        has_header = True
        offset = _HEADER_BYTES

        for src_key, shape, orig_dtype in named_items:
            dst_key = _bucket_key(src_key, has_visual, self.is_agent)
            if dst_key is None:
                continue
            transport_dtype = self._transport_dtype(orig_dtype)
            nbytes = int(shape.numel()) * dtype_size(transport_dtype)
            aligned = _align_up(offset)
            if slots and aligned + nbytes > self.bucket_size:
                buckets.append(
                    _FlatBucketSpec(
                        total_bytes=offset,
                        has_header=has_header,
                        slots=tuple(slots),
                    )
                )
                slots = []
                has_header = False
                offset = 0
                aligned = 0
            slots.append(
                _TensorSlot(
                    src_key=src_key,
                    dst_key=dst_key,
                    shape=tuple(int(d) for d in shape),
                    dtype=transport_dtype,
                    offset=aligned,
                    nbytes=nbytes,
                )
            )
            offset = aligned + nbytes

        if slots:
            buckets.append(
                _FlatBucketSpec(
                    total_bytes=offset,
                    has_header=has_header,
                    slots=tuple(slots),
                )
            )
        if not buckets:
            raise ValueError("No parameters to sync")
        return buckets

    @staticmethod
    def _serialize_plan(buckets: list[_FlatBucketSpec]) -> list[dict[str, Any]]:
        return [
            {
                "total_bytes": bucket.total_bytes,
                "has_header": bucket.has_header,
                "slots": [
                    (
                        slot.dst_key,
                        slot.shape,
                        _dtype_to_str(slot.dtype),
                        slot.offset,
                        slot.nbytes,
                    )
                    for slot in bucket.slots
                ],
            }
            for bucket in buckets
        ]

    @staticmethod
    def _deserialize_plan(wire_plan: list[dict[str, Any]]) -> list[_FlatBucketSpec]:
        if not isinstance(wire_plan, list) or not wire_plan:
            raise ValueError(
                f"Expected non-empty flat bucket wire plan list, got {type(wire_plan).__name__}"
            )
        buckets: list[_FlatBucketSpec] = []
        for entry in wire_plan:
            slots = tuple(
                _TensorSlot(
                    src_key=dst_key,
                    dst_key=dst_key,
                    shape=tuple(shape),
                    dtype=_str_to_dtype(dtype_name),
                    offset=int(offset),
                    nbytes=int(nbytes),
                )
                for dst_key, shape, dtype_name, offset, nbytes in entry["slots"]
            )
            buckets.append(
                _FlatBucketSpec(
                    total_bytes=int(entry["total_bytes"]),
                    has_header=bool(entry["has_header"]),
                    slots=slots,
                )
            )
        return buckets

    @staticmethod
    def _build_views(
        buffer: torch.Tensor, buckets: list[_FlatBucketSpec]
    ) -> list[list[torch.Tensor]]:
        all_views: list[list[torch.Tensor]] = []
        for bucket in buckets:
            bucket_views: list[torch.Tensor] = []
            for slot in bucket.slots:
                if slot.nbytes == 0:
                    bucket_views.append(
                        torch.empty(slot.shape, dtype=slot.dtype, device=buffer.device)
                    )
                else:
                    bucket_views.append(
                        buffer[slot.offset : slot.offset + slot.nbytes]
                        .view(slot.dtype)
                        .view(slot.shape)
                    )
            all_views.append(bucket_views)
        return all_views

    def _finalize_sender_plan(self, buckets: list[_FlatBucketSpec], is_sender: bool):
        self._sender_buckets = buckets
        self._is_sender = bool(is_sender)
        self._sender_module_id = None
        self._sender_module_refs = []
        if self._is_sender:
            max_bytes = max(bucket.total_bytes for bucket in buckets)
            self._sender_buffer = torch.empty(
                max_bytes, dtype=torch.uint8, device=self.bucket_device
            )
            self._sender_views = self._build_views(self._sender_buffer, buckets)
        else:
            self._sender_buffer = None
            self._sender_views = []
        self._sender_initialized = True

    async def init_sender(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        param_names_need_sync: list[str],
        send: SendFn,
        recv: RecvFn | None = None,
        is_sender: bool = True,
    ) -> None:
        """Build the bucketed flat layout plan and send it once to receivers."""
        del recv
        if not param_names_need_sync:
            raise ValueError("param_names_need_sync must not be empty")
        has_visual = any(
            "visual." in key for key in param_names_need_sync if key in state_dict
        )

        def iter_specs() -> Iterator[tuple[str, torch.Size, torch.dtype]]:
            for key in param_names_need_sync:
                value = state_dict.get(key)
                if value is None:
                    continue
                yield key, value.shape, value.dtype

        buckets = self._build_plan(iter_specs(), has_visual=has_visual)
        self._finalize_sender_plan(buckets, is_sender=is_sender)
        if self._is_sender:
            await send(self._serialize_plan(buckets))

    async def init_sender_from_module(
        self,
        module: torch.nn.Module,
        param_names_need_sync: list[str],
        send: SendFn,
        is_sender: bool = True,
    ) -> None:
        """Initialize the sender directly from an unsharded module without building a state dict."""
        if not param_names_need_sync:
            raise ValueError("param_names_need_sync must not be empty")
        refs = _resolve_module_refs(module)
        has_visual = any(
            "visual." in key for key in param_names_need_sync if key in refs
        )

        def iter_specs() -> Iterator[tuple[str, torch.Size, torch.dtype]]:
            for key in param_names_need_sync:
                ref = refs.get(key)
                if ref is None:
                    continue
                parent, attr = ref
                tensor = _to_local(getattr(parent, attr).detach())
                yield key, tensor.shape, tensor.dtype

        buckets = self._build_plan(iter_specs(), has_visual=has_visual)
        self._finalize_sender_plan(buckets, is_sender=is_sender)
        self._cache_sender_module_refs(module, refs)
        if self._is_sender:
            await send(self._serialize_plan(buckets))

    def _cache_sender_module_refs(
        self,
        module: torch.nn.Module,
        refs: dict[str, tuple[torch.nn.Module, str]] | None = None,
    ) -> None:
        if self._sender_module_id == id(module):
            return
        if refs is None:
            refs = _resolve_module_refs(module)
        resolved_per_bucket: list[list[tuple[torch.nn.Module, str]]] = []
        for bucket in self._sender_buckets:
            bucket_refs: list[tuple[torch.nn.Module, str]] = []
            for slot in bucket.slots:
                ref = refs.get(slot.src_key)
                if ref is None:
                    raise KeyError(
                        f"Expected parameter/buffer {slot.src_key!r} on sender module "
                        f"{type(module).__name__}, but it was not found."
                    )
                bucket_refs.append(ref)
            resolved_per_bucket.append(bucket_refs)
        self._sender_module_refs = resolved_per_bucket
        self._sender_module_id = id(module)

    async def init_receiver(
        self,
        state_dict: dict[str, torch.Tensor | DTensor] | None,
        recv: RecvFn,
        send: SendFn | None = None,
    ) -> None:
        """Receive the bucketed flat layout plan from the sender."""
        del state_dict, send
        wire_plan = await recv()
        self._receiver_buckets = self._deserialize_plan(wire_plan)
        self._receiver_max_bytes = max(
            bucket.total_bytes for bucket in self._receiver_buckets
        )
        self._receiver_buffer = None
        self._receiver_model_id = None
        self._receiver_model_refs = []
        self._receiver_initialized = True

    def _ensure_receiver_buffer(self) -> torch.Tensor:
        if self._receiver_buffer is None:
            self._receiver_buffer = torch.empty(
                self._receiver_max_bytes, dtype=torch.uint8, device=self.bucket_device
            )
        return self._receiver_buffer

    def _ensure_receiver_model_refs(
        self, model: torch.nn.Module
    ) -> list[list[tuple[torch.nn.Module, str] | None]]:
        if self._receiver_model_id == id(model):
            return self._receiver_model_refs
        refs = _resolve_module_refs(model)
        per_bucket: list[list[tuple[torch.nn.Module, str] | None]] = []
        for bucket in self._receiver_buckets:
            per_bucket.append([refs.get(slot.dst_key) for slot in bucket.slots])
        self._receiver_model_refs = per_bucket
        self._receiver_model_id = id(model)
        return per_bucket

    @staticmethod
    def _stamp_version(buffer: torch.Tensor, version: int | torch.Tensor) -> None:
        header = buffer[:8].view(torch.int64)
        if torch.is_tensor(version):
            header.copy_(version.detach().to(dtype=torch.int64).reshape(1))
        else:
            header.fill_(int(version))

    @staticmethod
    def _copy_pairs(
        dsts: list[torch.Tensor], srcs: list[torch.Tensor], non_blocking: bool = False
    ) -> None:
        """Copy ``srcs`` into ``dsts``, batching same-device/same-dtype copies with ``_foreach_copy_``."""
        fast_groups: dict[
            tuple[torch.device, torch.dtype],
            tuple[list[torch.Tensor], list[torch.Tensor]],
        ] = {}
        seen_dsts: set[tuple[torch.device, int, int]] = set()
        for dst, src in zip(dsts, srcs, strict=True):
            if dst.numel() == 0:
                continue
            dst_id = (dst.device, dst.data_ptr(), dst.numel())
            if dst_id in seen_dsts:
                continue
            seen_dsts.add(dst_id)
            if dst.device == src.device and dst.dtype == src.dtype:
                g_dst, g_src = fast_groups.setdefault((dst.device, dst.dtype), ([], []))
                g_dst.append(dst)
                g_src.append(src)
            else:
                dst.copy_(src, non_blocking=non_blocking)
        for g_dst, g_src in fast_groups.values():
            torch._foreach_copy_(g_dst, g_src)

    def _pack_bucket(
        self,
        bucket_idx: int,
        tensors: list[torch.Tensor],
        version: int | torch.Tensor,
    ) -> torch.Tensor:
        assert self._sender_buffer is not None
        bucket = self._sender_buckets[bucket_idx]
        if bucket.has_header:
            self._stamp_version(self._sender_buffer, version)
        self._copy_pairs(self._sender_views[bucket_idx], tensors)
        return self._sender_buffer[: bucket.total_bytes]

    @torch.no_grad()
    async def sync(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        send: SendFn,
        version: int | torch.Tensor,
    ) -> None:
        """Pack and send each bucket from ``state_dict``.

        Materializes one bucket at a time in lockstep across all actor ranks so
        sharded FSDP collectives stay matched and peak memory stays bounded by
        ``bucket_size``. ``send`` must finish transmitting ``payload`` before
        returning because the flat buffer is reused across buckets.
        """
        if not self._sender_initialized:
            raise RuntimeError("FlatBucketWeightSyncer sender is not initialized.")
        num_buckets = len(self._sender_buckets)
        sync_between_buckets = (
            num_buckets > 1
            and Worker.torch_device_type is not None
            and self.bucket_device.type == Worker.torch_device_type
        )
        for bucket_idx, bucket in enumerate(self._sender_buckets):
            materialized = [
                _materialize_local_or_full(state_dict[slot.src_key]).detach()
                for slot in bucket.slots
            ]
            if self._is_sender:
                payload = self._pack_bucket(bucket_idx, materialized, version)
                await send(payload)
                if sync_between_buckets and bucket_idx + 1 < num_buckets:
                    Worker.torch_platform.synchronize()
            del materialized

    @torch.no_grad()
    async def sync_module(
        self,
        module: torch.nn.Module,
        send: SendFn,
        version: int | torch.Tensor,
    ) -> None:
        """Pack and send directly from an unsharded module without building a state dict."""
        if not self._sender_initialized:
            raise RuntimeError("FlatBucketWeightSyncer sender is not initialized.")
        self._cache_sender_module_refs(module)
        num_buckets = len(self._sender_module_refs)
        sync_between_buckets = (
            num_buckets > 1
            and Worker.torch_device_type is not None
            and self.bucket_device.type == Worker.torch_device_type
        )
        for bucket_idx, bucket_refs in enumerate(self._sender_module_refs):
            if self._is_sender:
                tensors = [
                    _to_local(getattr(parent, attr).detach())
                    for parent, attr in bucket_refs
                ]
                payload = self._pack_bucket(bucket_idx, tensors, version)
                await send(payload)
                if sync_between_buckets and bucket_idx + 1 < num_buckets:
                    Worker.torch_platform.synchronize()

    def _unpack_bucket_into_model(
        self,
        bucket_idx: int,
        buf: torch.Tensor,
        model_refs: list[tuple[torch.nn.Module, str] | None],
    ) -> None:
        bucket = self._receiver_buckets[bucket_idx]
        dsts: list[torch.Tensor] = []
        srcs: list[torch.Tensor] = []
        for slot, ref in zip(bucket.slots, model_refs, strict=True):
            if ref is None or slot.nbytes == 0:
                continue
            parent, attr = ref
            dst = _to_local(getattr(parent, attr).detach())
            view = (
                buf[slot.offset : slot.offset + slot.nbytes]
                .view(slot.dtype)
                .view(slot.shape)
            )
            dsts.append(dst)
            srcs.append(view)
        if dsts:
            self._copy_pairs(dsts, srcs)

    @torch.no_grad()
    async def apply(
        self,
        model: torch.nn.Module,
        recv: RecvFn,
        *,
        recv_into: RecvIntoFn | None = None,
    ) -> int:
        """Receive flat buckets and unpack them into ``model``.

        Args:
            model: Receiver module whose parameters and buffers are updated in
                place. Keys absent from ``model`` are skipped.
            recv: Out-of-place receive callback returning one flat ``uint8``
                bucket tensor (used on broadcast transport).
            recv_into: Optional in-place receive callback filling a preallocated
                flat ``uint8`` slice (used on 1:1 P2P ``recv_tensor`` transport).

        Returns:
            int: Weight version stamped into bucket 0 by the sender.
        """
        if not self._receiver_initialized:
            raise RuntimeError("FlatBucketWeightSyncer receiver is not initialized.")

        per_bucket_refs = self._ensure_receiver_model_refs(model)
        accel_device_type = Worker.torch_device_type
        on_accel = (
            accel_device_type is not None
            and self.bucket_device.type == accel_device_type
        )
        need_record = recv_into is None and self.load_instant and on_accel
        fallback_keepalive: list[torch.Tensor] = []
        current_stream = (
            Worker.torch_platform.current_stream(self.bucket_device)
            if on_accel
            else None
        )
        cpu_staged: list[tuple[tuple[torch.nn.Module, str], torch.Tensor]] = []
        applied_version = 0
        num_buckets = len(self._receiver_buckets)

        for bucket_idx, bucket in enumerate(self._receiver_buckets):
            if recv_into is not None:
                recv_buf = self._ensure_receiver_buffer()[: bucket.total_bytes]
                await recv_into(recv_buf)
                buf = recv_buf
            else:
                buf = await recv()
                if buf.numel() != bucket.total_bytes:
                    raise ValueError(
                        f"Bucket {bucket_idx} expected {bucket.total_bytes} bytes, "
                        f"got {buf.numel()}."
                    )

            if bucket.has_header:
                applied_version = int(buf[:8].view(torch.int64).item())

            if self.load_instant:
                self._unpack_bucket_into_model(
                    bucket_idx, buf, per_bucket_refs[bucket_idx]
                )
                if need_record:
                    fallback_keepalive.extend(tensors_record_stream([buf]))
                elif (
                    recv_into is not None
                    and bucket_idx + 1 < num_buckets
                    and current_stream is not None
                ):
                    # Finish reading the shared P2P buffer before the next bucket
                    # overwrites it.
                    current_stream.synchronize()
            else:
                for slot, ref in zip(
                    bucket.slots, per_bucket_refs[bucket_idx], strict=True
                ):
                    if ref is None or slot.nbytes == 0:
                        continue
                    view = (
                        buf[slot.offset : slot.offset + slot.nbytes]
                        .view(slot.dtype)
                        .view(slot.shape)
                    )
                    cpu_staged.append((ref, view.to("cpu", copy=True)))

            if recv_into is None:
                del buf

        if not self.load_instant and cpu_staged:
            dsts = [
                _to_local(getattr(parent, attr).detach())
                for (parent, attr), _ in cpu_staged
            ]
            srcs = [tensor for _, tensor in cpu_staged]
            self._copy_pairs(dsts, srcs)
            del cpu_staged

        if fallback_keepalive and current_stream is not None:
            current_stream.synchronize()
            del fallback_keepalive

        return applied_version


def can_use_p2p_weight_sync(
    syncer: WeightSyncer | None,
    cfg: DictConfig,
    actor_world_size: int,
    rollout_world_size: int,
) -> bool:
    """Whether actor and rollout ranks can sync 1:1 via P2P ``send_tensor``/``recv_tensor``.

    Requires :class:`FlatBucketWeightSyncer` with ``enable_p2p=True``, equal
    actor and rollout world sizes, and unsharded weights on every actor rank
    (single actor rank or ``sharding_strategy: no_shard``).
    """
    if not isinstance(syncer, FlatBucketWeightSyncer) or not syncer.enable_p2p:
        return False
    if actor_world_size != rollout_world_size:
        return False
    sharding = OmegaConf.select(
        cfg, "actor.fsdp_config.sharding_strategy", default="full_shard"
    )
    return actor_world_size == 1 or sharding == "no_shard"
