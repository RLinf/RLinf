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

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator

import torch
from torch.distributed.tensor import DTensor

from rlinf.scheduler import Worker
from rlinf.utils.logging import get_logger
from rlinf.utils.utils import (
    dtype_size,
    materialize_tensor,
    normalize_device,
    normalize_dtype,
    tensors_record_stream,
)

from .base import RecvFn, SendFn, WeightSyncer

logger = get_logger()


def _tensor_fingerprint(name: str, tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    header = json.dumps([name, str(value.dtype), list(value.shape)]).encode()
    digest = hashlib.sha256(header)
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _fingerprints_digest(fingerprints: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(sorted(fingerprints.items())).encode()).hexdigest()


def iter_named_tensor_buckets(
    items: Iterable[tuple[str, torch.Tensor | DTensor]],
    version: int | torch.Tensor,
    *,
    bucket_size: int,
    bucket_device: str | torch.device,
    dtype_resolver: Callable[[str, torch.dtype], torch.dtype] | None = None,
) -> Iterator[dict[str, torch.Tensor]]:
    """
    Iterate transport-ready buckets from already-selected named tensors.

    Args:
        - items (Iterable[tuple[str, torch.Tensor | DTensor]]): An iterable of tuples containing parameter names and their corresponding tensors.
        - version (int | torch.Tensor): The version of the model weights.
        - bucket_size (int): The maximum threshold (in bytes) of each bucket.
        - bucket_device (str | torch.device): The device to which the buckets will be moved, cpu or accelerator device.
        - dtype_resolver (Callable[[str, torch.dtype], torch.dtype] | None): A function that takes a parameter name and its original dtype,
            and returns the dtype to be used for transport. If None, the original dtype is used.

    Yields:
        dict[str, torch.Tensor]: A dictionary representing a bucket of parameters to be synchronized.
    """
    metadata_keys = {
        BucketWeightSyncer._TOTAL_BUCKETS_KEY,
        BucketWeightSyncer._SYNCER_VERSION_KEY,
        BucketWeightSyncer._SHA256_KEY,
    }
    bucket_device = normalize_device(bucket_device)
    prepared_items: list[tuple[str, torch.Tensor | DTensor, torch.dtype]] = []
    currently_hold: int = 0
    bucket_plan: list[list[tuple[str, torch.Tensor | DTensor, torch.dtype]]] = []
    for key, value in items:
        if key in metadata_keys:
            raise ValueError(f"Bucket payload key conflicts with metadata key: {key}")

        transport_dtype = (
            dtype_resolver(key, value.dtype)
            if dtype_resolver is not None
            else value.dtype
        )
        prepared_items.append((key, value, transport_dtype))
        currently_hold += value.numel() * dtype_size(transport_dtype)
        if currently_hold >= bucket_size:
            bucket_plan.append(prepared_items)
            prepared_items = []
            currently_hold = 0

    if currently_hold > 0:
        bucket_plan.append(prepared_items)
        prepared_items = []
        currently_hold = 0

    if len(bucket_plan) == 0:
        raise ValueError("No parameters to sync")

    bucket: dict[str, torch.Tensor] = {
        BucketWeightSyncer._TOTAL_BUCKETS_KEY: torch.tensor(
            len(bucket_plan), dtype=torch.int32, device=bucket_device
        ),
        BucketWeightSyncer._SYNCER_VERSION_KEY: torch.as_tensor(
            version, dtype=torch.int32, device=bucket_device
        ),
    }
    for bucket_items in bucket_plan:
        for key, value, transport_dtype in bucket_items:
            tensor = materialize_tensor(value)

            bucket[key] = tensor.to(
                device=bucket_device,
                dtype=transport_dtype,
                non_blocking=False,
            )
        yield bucket
        bucket = {}

    if bucket:
        yield bucket
        bucket = {}


class BucketWeightSyncer(WeightSyncer):
    """Synchronize model weights by sending state dict tensors in buckets.

    The sender materializes selected tensors, optionally casts floating-point
    tensors to ``bucket_dtype`` for transport, moves them to ``bucket_device``,
    and sends them as dictionaries whose payload entries are parameter or buffer
    names. The receiver applies the received buckets to the target model with
    ``load_state_dict(strict=False)``.

    The first bucket also carries metadata used by the receiver-side protocol:
    ``_TOTAL_BUCKETS_KEY`` stores the number of buckets to receive, and
    ``_SYNCER_VERSION_KEY`` stores the weight version associated with the
    transfer. These keys are reserved and must not collide with state dict keys.

    Attributes:
        bucket_size: Target maximum bucket payload size in bytes. A single tensor
            is never split, so an individual payload may exceed this value.
        bucket_dtype: Optional dtype used to transport floating-point tensors.
            Non-floating tensors keep their original dtype.
        bucket_device: Device where bucket payload tensors are staged before
            sending.
        is_agent: Whether to apply agent-specific language-model key rewriting.
        load_instant: Whether the receiver loads each bucket immediately instead
            of staging all buckets on CPU before one final model load.
        verify_interval: Audit versions divisible by this interval, including zero.
            Zero disables checksums. Audits copy synchronized tensors to CPU.
        last_verification: Version and digests for the latest audited apply, or None.
    """

    _TOTAL_BUCKETS_KEY = "total_buckets"
    _SYNCER_VERSION_KEY = "syncer_version"
    _SHA256_KEY = "syncer_sha256"

    def __init__(
        self,
        bucket_size: int,
        bucket_dtype: torch.dtype | str | None,
        bucket_device: str | torch.device,
        is_agent: bool = False,
        load_instant: bool = True,
        verify_interval: int = 0,
    ):
        super().__init__()
        self.bucket_size = bucket_size
        self.bucket_dtype = normalize_dtype(bucket_dtype)
        self.bucket_device = normalize_device(bucket_device)
        self.is_agent = is_agent
        self.load_instant = load_instant
        if verify_interval < 0:
            raise ValueError("verify_interval must be nonnegative")
        self.verify_interval = verify_interval
        self.last_verification: dict | None = None
        self._verified_fingerprints: dict[str, str] = {}
        self._is_sender = True

    def _bucket_key(self, key: str, has_visual: bool) -> str | None:
        """
        Handle special cases for parameter names when syncing weights.
        If the key contains "_extra_state", it is ignored (returns None).
        If the model has visual components and is an agent, and the key starts with
        "model.language_model.", it is transformed to start with "model." instead.
        Otherwise, the key is returned unchanged.

        Args:
            key (str): The original parameter name.
            has_visual (bool): Indicates if the model has visual components.

        Returns:
            str | None: The transformed parameter name, or None if it should be ignored.
        """
        if "_extra_state" in key:
            return None
        if has_visual and self.is_agent and key.startswith("model.language_model."):
            return "model." + key[len("model.language_model.") :]
        return key

    def _transport_dtype(self, dtype: torch.dtype) -> torch.dtype:
        """
        Determine the dtype for transporting tensors in buckets.
        If it's floating point and a bucket_dtype is specified, use the bucket_dtype.
        Otherwise, use the original dtype.

        Args:
            dtype (torch.dtype): The original dtype of the tensor.

        Returns:
            torch.dtype: The dtype to be used for transporting the tensor.
        """
        if self.bucket_dtype is not None and dtype.is_floating_point:
            return self.bucket_dtype
        return dtype

    def iter_buckets(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        version: int | torch.Tensor,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """
        Iterate over the state_dict and yield buckets of parameters that need to be synchronized.

        Args:
            state_dict (dict[str, torch.Tensor | DTensor]): The model's state dictionary. If the tensor is a DTensor,
                it will be materialized to a regular torch.Tensor before being added to the bucket.
            version (int | torch.Tensor): The version of the model weights being synchronized.

        Yields:
            dict[str, torch.Tensor]: A dictionary representing a bucket of parameters to be synchronized.
        """

        has_visual = any(
            "visual." in key
            for key in self.param_names_need_sync_set
            if key in state_dict
        )

        def iter_named_items() -> Iterator[tuple[str, torch.Tensor | DTensor]]:
            for key in self.param_names_need_sync:
                value = state_dict.get(key)
                if value is None:
                    continue
                bucket_key = self._bucket_key(key, has_visual)
                if bucket_key is not None:
                    yield bucket_key, value

        yield from iter_named_tensor_buckets(
            iter_named_items(),
            version,
            bucket_size=self.bucket_size,
            bucket_device=self.bucket_device,
            dtype_resolver=lambda _, dtype: self._transport_dtype(dtype),
        )

    async def init_sender(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        param_names_need_sync: list[str],
        send: SendFn,
        recv: RecvFn | None = None,
        is_sender: bool = True,
    ) -> None:
        """
        Initialize the sender for weight synchronization.

        Args:
            - state_dict (dict[str, torch.Tensor | DTensor]): The model's state dictionary. For BucketWeightSyncer,
                it's not used, just to keep the interface consistent with other syncers.
            - param_names_need_sync (list[str]): A list of parameter names that need to be synchronized.
            - send (SendFn): The function for sender to communicate with the receiver.
            - recv (RecvFn | None): The function for receiver to communicate with the sender.
            - is_sender (bool): Whether this worker sends payloads. Inactive workers
                still materialize distributed tensors but skip checksum computation.
        """

        del state_dict, send, recv
        self._is_sender = is_sender
        self.param_names_need_sync = param_names_need_sync
        self.param_names_need_sync_set = set(param_names_need_sync)
        if not self.param_names_need_sync_set:
            raise ValueError("param_names_need_sync must not be empty")
        self._sender_initialized = True

    async def sync(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        send: SendFn,
        version: int | torch.Tensor,
    ) -> None:
        """
        Synchronize the model weights by sending buckets of parameters to the receiver.
        Called by the sender every time the model weights should update after sender/receiver is initialized.

        Args:
            - state_dict (dict[str, torch.Tensor | DTensor]): The model's state dictionary
            - send (SendFn): The function to send synchronized buckets. it should define who sends, how to send, and where to send.
            - version (int | torch.Tensor): The version of the model weights being synchronized.
        """
        verify = (
            self._is_sender
            and self.verify_interval > 0
            and int(version) % self.verify_interval == 0
        )
        fingerprints: dict[str, str] = {}
        if verify:
            missing = set(self.param_names_need_sync) - state_dict.keys()
            if missing:
                raise RuntimeError(
                    f"Weight sync source is missing tensors: {sorted(missing)}"
                )
        for bucket in self.iter_buckets(state_dict, version):
            if verify:
                bucket_fingerprints = {
                    key: _tensor_fingerprint(key, value)
                    for key, value in bucket.items()
                    if key not in (self._TOTAL_BUCKETS_KEY, self._SYNCER_VERSION_KEY)
                }
                fingerprints.update(bucket_fingerprints)
                digest = bytes.fromhex(_fingerprints_digest(bucket_fingerprints))
                bucket[self._SHA256_KEY] = torch.tensor(
                    list(digest), dtype=torch.uint8, device=self.bucket_device
                )
            await send(bucket)
            del bucket
        if verify:
            logger.info(
                f"Weight sync verified sender version={int(version)} "
                f"tensors={len(fingerprints)} sha256={_fingerprints_digest(fingerprints)}"
            )

    def _verify_loaded_bucket(
        self,
        model: torch.nn.Module,
        bucket: dict[str, torch.Tensor],
        fingerprints: dict[str, str],
        incompatible_keys,
    ) -> None:
        if incompatible_keys.unexpected_keys:
            raise RuntimeError(
                f"Weight sync has unexpected target keys: {incompatible_keys.unexpected_keys}"
            )
        target_state = model.state_dict()
        for key, value in bucket.items():
            if key not in target_state:
                raise RuntimeError(f"Weight sync target is missing tensor: {key}")
            target = target_state[key]
            expected = fingerprints[key]
            if target.dtype != value.dtype:
                expected = _tensor_fingerprint(key, value.to(dtype=target.dtype))
            actual = _tensor_fingerprint(key, target)
            if actual != expected:
                raise RuntimeError(
                    f"Weight sync loaded tensor differs from source: {key}"
                )
            self._verified_fingerprints[key] = actual

    def verify_applied_weights(self, model: torch.nn.Module) -> dict | None:
        """Recheck the last audited snapshot against the model used for evaluation.

        Returns None if the latest synchronization was not audited. Otherwise
        returns its version, tensor count and transport/model SHA256 digests.
        """
        if self.last_verification is None:
            return None
        target_state = model.state_dict()
        for key, expected in self._verified_fingerprints.items():
            if (
                key not in target_state
                or _tensor_fingerprint(key, target_state[key]) != expected
            ):
                raise RuntimeError(
                    f"Weight sync tensor changed before evaluation: {key}"
                )
        return dict(self.last_verification)

    async def apply(self, model: torch.nn.Module, recv: RecvFn) -> int:
        """
        Apply the synchronized weights to the model by receiving buckets of parameters from the sender.
        Called by the receiver every time the model weights should update after sender/receiver is initialized.

        Args:
            - model (torch.nn.Module): The model to which the synchronized weights will be applied.
            - recv (RecvFn): The function to receive synchronized buckets. it should define who receives, how to receive, and where to receive.

        Returns:
            int: The version of the model weights that have been applied.
        """

        self.last_verification = None
        self._verified_fingerprints = {}
        bucket: dict[str, torch.Tensor] = await recv()
        total_buckets = int(bucket.pop(self._TOTAL_BUCKETS_KEY).item())
        applied_version = int(bucket.pop(self._SYNCER_VERSION_KEY).item())
        verify_required = (
            self.verify_interval > 0 and applied_version % self.verify_interval == 0
        )
        if verify_required or self._SHA256_KEY in bucket:
            return await self._apply_verified(
                model, recv, bucket, total_buckets, applied_version
            )

        fallback_keepalive: list[torch.Tensor] = []
        current_stream: torch.Stream | None = None
        # NOTE:
        # actually only accel -> accel could return without finishing copy (if non_blocking=False)
        # to simpilify code and keep extensibility, we judge by bucket's device only.
        # we record buckets because bucket's deleted but tensor comes with another stream.
        need_record = (
            self.load_instant and self.bucket_device.type == Worker.torch_device_type
        )

        if self.load_instant:
            model.load_state_dict(bucket, strict=False)
        else:
            cpu_buffer: dict[str, torch.Tensor] = {}
            for key, value in bucket.items():
                cpu_buffer[key] = value.to("cpu")
        if need_record:
            current_stream = Worker.torch_platform.current_stream(self.bucket_device)
            fallback_keepalive.extend(tensors_record_stream(bucket.values()))
        del bucket

        for _ in range(total_buckets - 1):
            bucket = await recv()
            if self.load_instant:
                model.load_state_dict(bucket, strict=False)
            else:
                for key, value in bucket.items():
                    cpu_buffer[key] = value.to("cpu")

            if need_record:
                fallback_keepalive.extend(tensors_record_stream(bucket.values()))
            del bucket

        if not self.load_instant:
            model.load_state_dict(cpu_buffer, strict=False)
            del cpu_buffer

        if fallback_keepalive:
            current_stream.synchronize()
            del fallback_keepalive

        return applied_version

    async def _apply_verified(
        self,
        model: torch.nn.Module,
        recv: RecvFn,
        first_bucket: dict[str, torch.Tensor],
        total_buckets: int,
        version: int,
    ) -> int:
        fingerprints: dict[str, str] = {}
        cpu_buffer: dict[str, torch.Tensor] = {}
        for index in range(total_buckets):
            bucket = first_bucket if index == 0 else await recv()
            first_bucket = None
            checksum = bucket.pop(self._SHA256_KEY, None)
            if (
                checksum is None
                or checksum.dtype != torch.uint8
                or checksum.shape != (32,)
            ):
                raise RuntimeError(
                    f"Weight sync version {version} bucket {index} has no valid SHA256"
                )
            bucket_fingerprints = {
                key: _tensor_fingerprint(key, value) for key, value in bucket.items()
            }
            actual = _fingerprints_digest(bucket_fingerprints)
            expected = bytes(checksum.cpu().tolist()).hex()
            if actual != expected:
                raise RuntimeError(
                    f"Weight sync payload SHA256 mismatch at version {version} bucket {index}"
                )
            duplicate = fingerprints.keys() & bucket_fingerprints.keys()
            if duplicate:
                raise RuntimeError(
                    f"Weight sync contains duplicate tensors: {sorted(duplicate)}"
                )
            fingerprints.update(bucket_fingerprints)
            if self.load_instant:
                incompatible = model.load_state_dict(bucket, strict=False)
                self._verify_loaded_bucket(
                    model, bucket, bucket_fingerprints, incompatible
                )
            else:
                cpu_buffer.update({key: value.cpu() for key, value in bucket.items()})
            # Fingerprinting copied tensors synchronously; no device work retains the bucket.
            bucket.clear()
        if not self.load_instant:
            incompatible = model.load_state_dict(cpu_buffer, strict=False)
            self._verify_loaded_bucket(model, cpu_buffer, fingerprints, incompatible)
        self.last_verification = {
            "version": version,
            "tensor_count": len(fingerprints),
            "transport_sha256": _fingerprints_digest(fingerprints),
            "model_sha256": _fingerprints_digest(self._verified_fingerprints),
        }
        logger.info(
            f"Weight sync verified receiver version={version} tensors={len(fingerprints)} "
            f"transport_sha256={self.last_verification['transport_sha256']} "
            f"model_sha256={self.last_verification['model_sha256']}"
        )
        return version
