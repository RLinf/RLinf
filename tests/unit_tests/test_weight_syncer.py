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

import asyncio
import copy
from collections import OrderedDict

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.hybrid_engines.weight_syncer import (
    BucketWeightSyncer,
    PatchWeightSyncer,
    WeightSyncer,
)
from rlinf.hybrid_engines.weight_syncer.patch_syncer import (
    as_coo_2d_view,
    downscale_nonnegative_indices,
)


class _TinyWeightSyncModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)
        self.tensor3d = torch.nn.Parameter(
            torch.arange(24, dtype=torch.float32).view(2, 3, 4).clone()
        )
        self.register_buffer("scalar_buf", torch.tensor(1.5, dtype=torch.float32))
        self.register_buffer(
            "vector_buf", torch.tensor([2.0, 4.0, 8.0], dtype=torch.float32)
        )


class _MixedDtypeWeightSyncModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fp32_param = torch.nn.Parameter(
            torch.full((2, 3), 1.0, dtype=torch.float32)
        )
        self.bf16_param = torch.nn.Parameter(
            torch.arange(6, dtype=torch.bfloat16).view(2, 3).clone()
        )


class _BucketDtypeWeightSyncModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fp32_param = torch.nn.Parameter(
            torch.arange(6, dtype=torch.float32).view(2, 3).clone()
        )
        self.bf16_param = torch.nn.Parameter(
            torch.arange(6, dtype=torch.bfloat16).view(2, 3).clone()
        )
        self.register_buffer(
            "int64_buf",
            torch.tensor([2**40 + 123, -(2**39 + 17)], dtype=torch.int64),
        )
        self.register_buffer("bool_buf", torch.tensor([True, False, True]))


class _InMemoryTransport:
    def __init__(self):
        self._queue: list[object] = []

    async def send(self, data):
        self._queue.append(data)

    async def recv(self):
        assert self._queue, "Transport queue is empty"
        return self._queue.pop(0)


class _InMemoryDuplexTransport:
    def __init__(self):
        self._sender_to_receiver: asyncio.Queue[object] = asyncio.Queue()
        self._receiver_to_sender: asyncio.Queue[object] = asyncio.Queue()

    async def sender_send(self, data):
        await self._sender_to_receiver.put(data)

    async def sender_recv(self):
        return await self._receiver_to_sender.get()

    async def receiver_send(self, data):
        await self._receiver_to_sender.put(data)

    async def receiver_recv(self):
        return await self._sender_to_receiver.get()


def _clone_state_dict(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (key, value.detach().clone()) for key, value in model.state_dict().items()
    )


def _make_model(device: torch.device | str = "cpu") -> _TinyWeightSyncModel:
    return _TinyWeightSyncModel().to(device)


def _make_mixed_dtype_model(
    device: torch.device | str = "cpu",
) -> _MixedDtypeWeightSyncModel:
    return _MixedDtypeWeightSyncModel().to(device)


def _make_bucket_dtype_model(
    device: torch.device | str = "cpu",
) -> _BucketDtypeWeightSyncModel:
    return _BucketDtypeWeightSyncModel().to(device)


def _get_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA tests require at least 1 CUDA GPU.")
    return torch.device("cuda:0")


def _assert_state_dict_equal(
    lhs: OrderedDict[str, torch.Tensor], rhs: OrderedDict[str, torch.Tensor]
) -> None:
    assert list(lhs.keys()) == list(rhs.keys())
    for key in lhs.keys():
        torch.testing.assert_close(lhs[key], rhs[key], msg=f"Mismatch at key={key}")


def _assert_state_dict_equal_on_cpu(
    lhs: OrderedDict[str, torch.Tensor], rhs: OrderedDict[str, torch.Tensor]
) -> None:
    assert list(lhs.keys()) == list(rhs.keys())
    for key in lhs.keys():
        torch.testing.assert_close(
            lhs[key].cpu(), rhs[key].cpu(), msg=f"Mismatch at key={key}"
        )


async def _init_patch_syncers(
    sender_syncer: PatchWeightSyncer,
    receiver_syncer: PatchWeightSyncer,
    sender_model: torch.nn.Module,
    receiver_model: torch.nn.Module,
    transport: _InMemoryDuplexTransport,
) -> None:
    await asyncio.gather(
        sender_syncer.init_sender(
            _clone_state_dict(sender_model),
            transport.sender_send,
            transport.sender_recv,
        ),
        receiver_syncer.init_receiver(
            _clone_state_dict(receiver_model),
            transport.receiver_recv,
            transport.receiver_send,
        ),
    )


def test_as_coo_2d_view_for_supported_ranks():
    scalar = torch.tensor(3.0)
    scalar_view, scalar_shape = as_coo_2d_view(scalar)
    assert scalar_view.shape == (1, 1)
    assert scalar_shape == torch.Size([])

    vector = torch.arange(5, dtype=torch.float32)
    vector_view, vector_shape = as_coo_2d_view(vector)
    assert vector_view.shape == (1, 5)
    assert vector_shape == torch.Size([5])

    matrix = torch.arange(6, dtype=torch.float32).view(2, 3)
    matrix_view, matrix_shape = as_coo_2d_view(matrix)
    assert matrix_view.shape == (2, 3)
    assert matrix_shape == torch.Size([2, 3])

    tensor3d = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
    tensor3d_view, tensor3d_shape = as_coo_2d_view(tensor3d)
    assert tensor3d_view.shape == (2, 12)
    assert tensor3d_shape == torch.Size([2, 3, 4])


def test_as_coo_2d_view_raises_for_nonviewable_high_rank_tensor():
    tensor = torch.arange(24, dtype=torch.float32).view(2, 3, 4).transpose(1, 2)
    with pytest.raises(ValueError, match="can be flattened as a view"):
        as_coo_2d_view(tensor)


def test_downscale_nonnegative_indices_selects_expected_dtype():
    empty = downscale_nonnegative_indices(torch.empty(0, dtype=torch.int64))
    assert empty.dtype == torch.uint8

    small = downscale_nonnegative_indices(torch.tensor([0, 7, 255], dtype=torch.int64))
    assert small.dtype == torch.uint8

    medium = downscale_nonnegative_indices(
        torch.tensor([0, 256, 1024], dtype=torch.int64)
    )
    assert medium.dtype == torch.int32

    large = downscale_nonnegative_indices(
        torch.tensor([0, torch.iinfo(torch.int32).max + 1], dtype=torch.int64)
    )
    assert large.dtype == torch.int64


def test_patch_delta_encode_decode_roundtrip():
    syncer = PatchWeightSyncer(
        snapshot_device="cpu",
        transport_device="cpu",
        delta_encoding=True,
        compression_algorithm="none",
    )
    rows = torch.tensor([0, 0, 0, 2, 2, 5], dtype=torch.int64)
    cols = torch.tensor([1, 4, 7, 0, 8, 3], dtype=torch.int64)

    row_deltas, col_deltas = syncer.delta_encode(rows, cols)
    decoded_rows, decoded_cols = syncer.delta_decode(row_deltas, col_deltas)

    torch.testing.assert_close(decoded_rows, rows)
    torch.testing.assert_close(decoded_cols, cols)


def test_patch_weight_syncer_roundtrip_delta_enabled():
    device = _get_cuda_device()
    sender_model = _make_model(device)
    receiver_model = copy.deepcopy(sender_model)
    transport = _InMemoryDuplexTransport()

    sender_syncer = PatchWeightSyncer(
        snapshot_device="cpu",
        transport_device="cpu",
        delta_encoding=True,
        compression_algorithm="none",
    )
    receiver_syncer = PatchWeightSyncer(
        snapshot_device="cpu",
        transport_device="cpu",
        delta_encoding=True,
        compression_algorithm="none",
    )

    async def _run() -> int:
        await _init_patch_syncers(
            sender_syncer,
            receiver_syncer,
            sender_model,
            receiver_model,
            transport,
        )

        with torch.no_grad():
            sender_model.linear.weight[0, 0] += 3.0
            sender_model.linear.bias[2] -= 1.25
            sender_model.tensor3d[1, 2, 3] = -99.0
            sender_model.scalar_buf.add_(4.0)
            sender_model.vector_buf[1] = 123.0

        await sender_syncer.sync(
            sender_model.state_dict(), transport.sender_send, version=11
        )
        return await receiver_syncer.apply(receiver_model, transport.receiver_recv)

    applied_version = asyncio.run(_run())

    assert applied_version == 11
    _assert_state_dict_equal(
        _clone_state_dict(sender_model), _clone_state_dict(receiver_model)
    )


def test_patch_weight_syncer_roundtrip_delta_disabled():
    device = _get_cuda_device()
    sender_model = _make_model(device)
    receiver_model = copy.deepcopy(sender_model)
    transport = _InMemoryDuplexTransport()

    sender_syncer = PatchWeightSyncer(
        snapshot_device="cpu",
        transport_device="cpu",
        delta_encoding=False,
        compression_algorithm="none",
    )
    receiver_syncer = PatchWeightSyncer(
        snapshot_device="cpu",
        transport_device="cpu",
        delta_encoding=False,
        compression_algorithm="none",
    )

    async def _run() -> int:
        await _init_patch_syncers(
            sender_syncer,
            receiver_syncer,
            sender_model,
            receiver_model,
            transport,
        )

        with torch.no_grad():
            sender_model.linear.weight[1, 3] = 77.0
            sender_model.tensor3d[0, 1, 2] += 5.0
            sender_model.vector_buf[0] = -5.0

        await sender_syncer.sync(
            sender_model.state_dict(), transport.sender_send, version=3
        )
        return await receiver_syncer.apply(receiver_model, transport.receiver_recv)

    applied_version = asyncio.run(_run())

    assert applied_version == 3
    _assert_state_dict_equal(
        _clone_state_dict(sender_model), _clone_state_dict(receiver_model)
    )


def test_patch_weight_syncer_roundtrip_cuda_delta_enabled():
    device = _get_cuda_device()
    sender_model = _make_model(device)
    receiver_model = copy.deepcopy(sender_model)
    transport = _InMemoryDuplexTransport()

    sender_syncer = PatchWeightSyncer(
        snapshot_device=device,
        transport_device=device,
        delta_encoding=True,
        compression_algorithm="none",
    )
    receiver_syncer = PatchWeightSyncer(
        snapshot_device=device,
        transport_device=device,
        delta_encoding=True,
        compression_algorithm="none",
    )

    async def _run() -> int:
        await _init_patch_syncers(
            sender_syncer,
            receiver_syncer,
            sender_model,
            receiver_model,
            transport,
        )

        with torch.no_grad():
            sender_model.linear.weight[0, 0] += 1.0
            sender_model.linear.bias[1] = -7.0
            sender_model.tensor3d[1, 1, 2] += 9.0
            sender_model.scalar_buf.mul_(3.0)
            sender_model.vector_buf[2] = -11.0

        await sender_syncer.sync(
            sender_model.state_dict(), transport.sender_send, version=23
        )
        return await receiver_syncer.apply(receiver_model, transport.receiver_recv)

    applied_version = asyncio.run(_run())

    assert applied_version == 23
    _assert_state_dict_equal(
        _clone_state_dict(sender_model), _clone_state_dict(receiver_model)
    )


def test_patch_weight_syncer_roundtrip_cuda_delta_disabled():
    device = _get_cuda_device()
    sender_model = _make_model(device)
    receiver_model = copy.deepcopy(sender_model)
    transport = _InMemoryDuplexTransport()

    sender_syncer = PatchWeightSyncer(
        snapshot_device=device,
        transport_device=device,
        delta_encoding=False,
        compression_algorithm="none",
    )
    receiver_syncer = PatchWeightSyncer(
        snapshot_device=device,
        transport_device=device,
        delta_encoding=False,
        compression_algorithm="none",
    )

    async def _run() -> int:
        await _init_patch_syncers(
            sender_syncer,
            receiver_syncer,
            sender_model,
            receiver_model,
            transport,
        )

        with torch.no_grad():
            sender_model.linear.weight[2, 3] = 55.0
            sender_model.tensor3d[0, 0, 1] = -3.5
            sender_model.vector_buf[0] += 10.0

        await sender_syncer.sync(
            sender_model.state_dict(), transport.sender_send, version=29
        )
        return await receiver_syncer.apply(receiver_model, transport.receiver_recv)

    applied_version = asyncio.run(_run())

    assert applied_version == 29
    _assert_state_dict_equal(
        _clone_state_dict(sender_model), _clone_state_dict(receiver_model)
    )


def test_patch_weight_syncer_cpu_snapshot_cuda_state_roundtrip():
    device = _get_cuda_device()
    sender_model = _make_model(device)
    receiver_model = copy.deepcopy(sender_model)
    transport = _InMemoryDuplexTransport()

    sender_syncer = PatchWeightSyncer(
        snapshot_device="cpu",
        transport_device="cpu",
        delta_encoding=True,
        compression_algorithm="none",
    )
    receiver_syncer = PatchWeightSyncer(
        snapshot_device="cpu",
        transport_device="cpu",
        delta_encoding=True,
        compression_algorithm="none",
    )

    async def _run() -> int:
        await _init_patch_syncers(
            sender_syncer,
            receiver_syncer,
            sender_model,
            receiver_model,
            transport,
        )

        with torch.no_grad():
            sender_model.linear.weight[0, 2] = 123.0
            sender_model.linear.bias[0] -= 6.0
            sender_model.tensor3d[1, 0, 3] += 13.0
            sender_model.scalar_buf.add_(2.5)
            sender_model.vector_buf[1] = -42.0

        await sender_syncer.sync(
            sender_model.state_dict(), transport.sender_send, version=37
        )
        first_applied_version = await receiver_syncer.apply(
            receiver_model, transport.receiver_recv
        )

        await sender_syncer.sync(
            sender_model.state_dict(), transport.sender_send, version=38
        )
        second_applied_version = await receiver_syncer.apply(
            receiver_model, transport.receiver_recv
        )
        return first_applied_version, second_applied_version

    first_applied_version, second_applied_version = asyncio.run(_run())

    assert first_applied_version == 37
    assert second_applied_version == 38
    _assert_state_dict_equal(
        _clone_state_dict(sender_model), _clone_state_dict(receiver_model)
    )


def test_patch_weight_syncer_uses_receiver_dtypes_for_snapshot():
    device = _get_cuda_device()
    sender_model = _make_mixed_dtype_model(device)
    receiver_model = copy.deepcopy(sender_model)
    transport = _InMemoryDuplexTransport()

    sender_syncer = PatchWeightSyncer(
        snapshot_device="cpu",
        transport_device="cpu",
        delta_encoding=True,
        compression_algorithm="none",
    )
    receiver_syncer = PatchWeightSyncer(
        snapshot_device="cpu",
        transport_device="cpu",
        delta_encoding=True,
        compression_algorithm="none",
    )

    async def _run() -> int:
        await _init_patch_syncers(
            sender_syncer,
            receiver_syncer,
            sender_model,
            receiver_model,
            transport,
        )
        assert sender_syncer.snapshot is not None
        assert sender_syncer.snapshot["fp32_param"].dtype == torch.float32
        assert sender_syncer.snapshot["bf16_param"].dtype == torch.bfloat16

        with torch.no_grad():
            sender_model.fp32_param[0, 0] += 1e-4
            sender_model.bf16_param[1, 2] += torch.tensor(
                2.0, dtype=torch.bfloat16, device=device
            )

        await sender_syncer.sync(
            sender_model.state_dict(), transport.sender_send, version=41
        )
        return await receiver_syncer.apply(receiver_model, transport.receiver_recv)

    applied_version = asyncio.run(_run())

    assert applied_version == 41
    _assert_state_dict_equal(
        _clone_state_dict(sender_model), _clone_state_dict(receiver_model)
    )


def test_patch_weight_syncer_preserves_nonfloating_buffers():
    device = _get_cuda_device()
    sender_model = _make_bucket_dtype_model(device)
    receiver_model = copy.deepcopy(sender_model)
    transport = _InMemoryDuplexTransport()

    sender_syncer = PatchWeightSyncer(
        snapshot_device="cpu",
        transport_device="cpu",
        delta_encoding=True,
        compression_algorithm="none",
    )
    receiver_syncer = PatchWeightSyncer(
        snapshot_device="cpu",
        transport_device="cpu",
        delta_encoding=True,
        compression_algorithm="none",
    )

    async def _run() -> int:
        await _init_patch_syncers(
            sender_syncer,
            receiver_syncer,
            sender_model,
            receiver_model,
            transport,
        )

        with torch.no_grad():
            sender_model.fp32_param[0, 0] = 123.25
            sender_model.bf16_param[1, 2] += torch.tensor(
                3.0, dtype=torch.bfloat16, device=device
            )
            sender_model.int64_buf[0] = 2**42 + 999
            sender_model.bool_buf.logical_not_()

        await sender_syncer.sync(
            sender_model.state_dict(), transport.sender_send, version=43
        )
        return await receiver_syncer.apply(receiver_model, transport.receiver_recv)

    applied_version = asyncio.run(_run())

    assert applied_version == 43
    _assert_state_dict_equal(
        _clone_state_dict(sender_model), _clone_state_dict(receiver_model)
    )


def test_patch_weight_syncer_roundtrip_cuda_nvcomp():
    device = _get_cuda_device()
    pytest.importorskip("nvidia.nvcomp")

    sender_model = _make_model(device)
    receiver_model = copy.deepcopy(sender_model)
    transport = _InMemoryDuplexTransport()

    sender_syncer = PatchWeightSyncer(
        snapshot_device=device,
        transport_device=device,
        delta_encoding=True,
        compression_algorithm="nvcomp_lz4",
    )
    receiver_syncer = PatchWeightSyncer(
        snapshot_device=device,
        transport_device=device,
        delta_encoding=True,
        compression_algorithm="nvcomp_lz4",
    )

    async def _run() -> int:
        await _init_patch_syncers(
            sender_syncer,
            receiver_syncer,
            sender_model,
            receiver_model,
            transport,
        )

        with torch.no_grad():
            sender_model.linear.weight[1, 2] -= 4.0
            sender_model.linear.bias[0] += 2.25
            sender_model.tensor3d[1, 2, 0] = 101.0
            sender_model.scalar_buf -= 0.75
            sender_model.vector_buf[1] = 88.0

        await sender_syncer.sync(
            sender_model.state_dict(), transport.sender_send, version=31
        )
        return await receiver_syncer.apply(receiver_model, transport.receiver_recv)

    applied_version = asyncio.run(_run())

    assert applied_version == 31
    _assert_state_dict_equal(
        _clone_state_dict(sender_model), _clone_state_dict(receiver_model)
    )


def test_patch_weight_syncer_empty_patch_still_applies_version():
    device = _get_cuda_device()
    sender_model = _make_model(device)
    receiver_model = copy.deepcopy(sender_model)
    transport = _InMemoryDuplexTransport()

    sender_syncer = PatchWeightSyncer(
        snapshot_device="cpu",
        transport_device="cpu",
        delta_encoding=True,
        compression_algorithm="none",
    )
    receiver_syncer = PatchWeightSyncer(
        snapshot_device="cpu",
        transport_device="cpu",
        delta_encoding=True,
        compression_algorithm="none",
    )

    async def _run() -> int:
        await _init_patch_syncers(
            sender_syncer,
            receiver_syncer,
            sender_model,
            receiver_model,
            transport,
        )
        await sender_syncer.sync(
            sender_model.state_dict(), transport.sender_send, version=19
        )
        return await receiver_syncer.apply(receiver_model, transport.receiver_recv)

    applied_version = asyncio.run(_run())

    assert applied_version == 19
    _assert_state_dict_equal(
        _clone_state_dict(sender_model), _clone_state_dict(receiver_model)
    )


def test_patch_weight_syncer_uses_receiver_key_order():
    device = _get_cuda_device()
    model = _make_model(device)
    syncer = PatchWeightSyncer(
        snapshot_device="cpu",
        transport_device="cpu",
        delta_encoding=True,
        compression_algorithm="none",
    )
    receiver_syncer = PatchWeightSyncer(
        snapshot_device="cpu",
        transport_device="cpu",
        delta_encoding=True,
        compression_algorithm="none",
    )
    receiver_model = copy.deepcopy(model)
    transport = _InMemoryDuplexTransport()

    async def _init() -> None:
        await _init_patch_syncers(
            syncer,
            receiver_syncer,
            model,
            receiver_model,
            transport,
        )

    asyncio.run(_init())

    reversed_state_dict = OrderedDict(reversed(list(_clone_state_dict(model).items())))
    syncer.create_patch(reversed_state_dict, version=1)

    mismatched_state_dict = _clone_state_dict(model)
    mismatched_state_dict.pop(next(iter(mismatched_state_dict)))
    with pytest.raises(ValueError, match="State dict keys do not match snapshot keys"):
        syncer.create_patch(mismatched_state_dict, version=1)


def test_bucket_weight_syncer_roundtrip_load_instant_true():
    sender_model = _TinyWeightSyncModel()
    receiver_model = copy.deepcopy(sender_model)
    transport = _InMemoryTransport()
    syncer = BucketWeightSyncer(
        bucket_size=32,
        bucket_dtype=torch.float32,
        bucket_device="cpu",
        load_instant=True,
    )

    with torch.no_grad():
        sender_model.linear.weight[0, 1] = 42.0
        sender_model.tensor3d[1, 0, 0] = -17.0
        sender_model.scalar_buf.mul_(2.0)

    async def _run() -> int:
        await syncer.sync(sender_model.state_dict(), transport.send, version=5)
        return await syncer.apply(receiver_model, transport.recv)

    applied_version = asyncio.run(_run())

    assert applied_version == 5
    _assert_state_dict_equal(
        _clone_state_dict(sender_model), _clone_state_dict(receiver_model)
    )


def test_bucket_weight_syncer_roundtrip_load_instant_false():
    sender_model = _TinyWeightSyncModel()
    receiver_model = copy.deepcopy(sender_model)
    transport = _InMemoryTransport()
    syncer = BucketWeightSyncer(
        bucket_size=32,
        bucket_dtype=torch.float32,
        bucket_device="cpu",
        load_instant=False,
    )

    with torch.no_grad():
        sender_model.linear.bias[1] = 8.5
        sender_model.tensor3d[0, 2, 1] -= 3.0
        sender_model.vector_buf[2] = 256.0

    async def _run() -> int:
        await syncer.sync(sender_model.state_dict(), transport.send, version=9)
        return await syncer.apply(receiver_model, transport.recv)

    applied_version = asyncio.run(_run())

    assert applied_version == 9
    _assert_state_dict_equal(
        _clone_state_dict(sender_model), _clone_state_dict(receiver_model)
    )


def test_bucket_weight_syncer_preserves_original_dtypes_when_bucket_dtype_none():
    sender_model = _make_bucket_dtype_model()
    receiver_model = copy.deepcopy(sender_model)
    transport = _InMemoryTransport()
    syncer = BucketWeightSyncer(
        bucket_size=32,
        bucket_dtype=None,
        bucket_device="cpu",
        load_instant=False,
    )

    with torch.no_grad():
        sender_model.fp32_param[0, 0] = 123.25
        sender_model.bf16_param[1, 2] += torch.tensor(3.0, dtype=torch.bfloat16)
        sender_model.int64_buf[0] = 2**42 + 999
        sender_model.bool_buf.logical_not_()

    buckets = syncer.divide_into_buckets(sender_model.state_dict(), version=11)
    payload = {
        key: value
        for bucket in buckets
        for key, value in bucket.items()
        if key not in {"total_buckets", "syncer_version"}
    }
    assert payload["fp32_param"].dtype == torch.float32
    assert payload["bf16_param"].dtype == torch.bfloat16
    assert payload["int64_buf"].dtype == torch.int64
    assert payload["bool_buf"].dtype == torch.bool

    async def _run() -> int:
        await syncer.sync(sender_model.state_dict(), transport.send, version=11)
        return await syncer.apply(receiver_model, transport.recv)

    applied_version = asyncio.run(_run())

    assert applied_version == 11
    _assert_state_dict_equal(
        _clone_state_dict(sender_model), _clone_state_dict(receiver_model)
    )


def test_bucket_weight_syncer_preserves_nonfloating_dtypes_when_bucket_dtype_set():
    model = _make_bucket_dtype_model()
    syncer = BucketWeightSyncer(
        bucket_size=32,
        bucket_dtype=torch.bfloat16,
        bucket_device="cpu",
        load_instant=True,
    )

    buckets = syncer.divide_into_buckets(model.state_dict(), version=12)
    payload = {
        key: value
        for bucket in buckets
        for key, value in bucket.items()
        if key not in {"total_buckets", "syncer_version"}
    }

    assert payload["fp32_param"].dtype == torch.bfloat16
    assert payload["bf16_param"].dtype == torch.bfloat16
    assert payload["int64_buf"].dtype == torch.int64
    assert payload["bool_buf"].dtype == torch.bool


def test_bucket_weight_syncer_loads_across_model_and_bucket_devices():
    device = _get_cuda_device()

    async def _run_case(
        model_device: torch.device | str,
        bucket_device: torch.device | str,
        version: int,
    ) -> tuple[int, torch.nn.Module, torch.nn.Module]:
        sender_model = _make_bucket_dtype_model(model_device)
        receiver_model = copy.deepcopy(sender_model)
        transport = _InMemoryTransport()
        syncer = BucketWeightSyncer(
            bucket_size=32,
            bucket_dtype=None,
            bucket_device=bucket_device,
            load_instant=True,
        )

        with torch.no_grad():
            sender_model.fp32_param[0, 1] = 99.0
            sender_model.bf16_param[0, 2] += torch.tensor(
                2.0, dtype=torch.bfloat16, device=model_device
            )
            sender_model.int64_buf[1] = -(2**41 + 7)
            sender_model.bool_buf[1] = True

        await syncer.sync(sender_model.state_dict(), transport.send, version=version)
        applied_version = await syncer.apply(receiver_model, transport.recv)
        return applied_version, sender_model, receiver_model

    cpu_bucket_version, cuda_sender, cuda_receiver = asyncio.run(
        _run_case(device, "cpu", version=21)
    )
    assert cpu_bucket_version == 21
    _assert_state_dict_equal_on_cpu(
        _clone_state_dict(cuda_sender), _clone_state_dict(cuda_receiver)
    )

    cuda_bucket_version, cpu_sender, cpu_receiver = asyncio.run(
        _run_case("cpu", device, version=22)
    )
    assert cuda_bucket_version == 22
    _assert_state_dict_equal_on_cpu(
        _clone_state_dict(cpu_sender), _clone_state_dict(cpu_receiver)
    )


def test_bucket_weight_syncer_rejects_metadata_key_collision():
    model = _TinyWeightSyncModel()
    syncer = BucketWeightSyncer(
        bucket_size=32,
        bucket_dtype=None,
        bucket_device="cpu",
        load_instant=True,
    )
    state_dict = _clone_state_dict(model)
    state_dict["total_buckets"] = torch.tensor(1)

    with pytest.raises(ValueError, match="conflicts with metadata key"):
        syncer.divide_into_buckets(state_dict, version=1)


def test_bucket_weight_syncer_sync_streams_without_prebuilding_buckets(monkeypatch):
    model = _make_bucket_dtype_model()
    transport = _InMemoryTransport()
    syncer = BucketWeightSyncer(
        bucket_size=32,
        bucket_dtype=None,
        bucket_device="cpu",
        load_instant=True,
    )

    def _raise_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("sync should not prebuild all buckets")

    monkeypatch.setattr(syncer, "divide_into_buckets", _raise_if_called)

    asyncio.run(syncer.sync(model.state_dict(), transport.send, version=3))

    assert len(transport._queue) == int(transport._queue[0]["total_buckets"].item())


def test_bucket_weight_syncer_metadata_dtypes_are_nccl_safe():
    model = _TinyWeightSyncModel()
    syncer = BucketWeightSyncer(
        bucket_size=32,
        bucket_dtype=torch.bfloat16,
        bucket_device="cpu",
        load_instant=True,
    )

    buckets = syncer.divide_into_buckets(model.state_dict(), version=13)

    assert buckets
    assert buckets[0]["total_buckets"].dtype == torch.int32
    assert buckets[0]["syncer_version"].dtype == torch.int64


def test_weight_syncer_factory_builds_patch_and_bucket():
    patch_cfg = OmegaConf.create(
        {
            "type": "patch",
            "patch": {
                "snapshot_device": "cpu",
                "transport_device": "cpu",
                "delta_encoding": True,
                "compression": "none",
            },
        }
    )
    patch_syncer = WeightSyncer.create(patch_cfg)
    assert isinstance(patch_syncer, PatchWeightSyncer)

    bucket_cfg = OmegaConf.create(
        {
            "type": "bucket",
            "bucket": {
                "bucket_size": 128,
                "bucket_dtype": "fp32",
                "bucket_device": "cpu",
                "is_agent": False,
                "load_instant": True,
            },
        }
    )
    bucket_syncer = WeightSyncer.create(bucket_cfg)
    assert isinstance(bucket_syncer, BucketWeightSyncer)


def test_weight_syncer_factory_rejects_unknown_type():
    cfg = OmegaConf.create({"type": "unknown"})
    with pytest.raises(ValueError, match="Unsupported weight syncer type"):
        WeightSyncer.create(cfg)
