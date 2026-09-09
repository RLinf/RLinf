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
import inspect
import logging
import unittest
from collections import OrderedDict
from types import MethodType
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.config import validate_weight_sync_overlap_cfg
from rlinf.data.schema.embodied_types import EnvOutput
from rlinf.hybrid_engines.weight_syncer import (
    BucketWeightSyncer,
    PatchWeightSyncer,
    WeightSyncer,
)
from rlinf.hybrid_engines.weight_syncer.bucket_syncer import (
    iter_named_tensor_buckets,
)
from rlinf.hybrid_engines.weight_syncer.patch_syncer import (
    CPUSnapshotPatchBuilder,
    EmptyWeightPatch,
    GPUSnapshotPatchBuilder,
    WeightPatch,
    as_coo_2d_view,
    downscale_nonnegative_indices,
)
from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
from rlinf.runners.async_ppo_embodied_runner import AsyncPPOEmbodiedRunner
from rlinf.runners.async_weight_sync_mixin import AsyncWeightSyncMixin
from rlinf.scheduler import AcceleratorType, Worker
from rlinf.utils.env_helpers import SmoothInterveneController  # noqa: E402
from rlinf.utils.utils import collect_param_names_need_sync
from rlinf.workers.env.env_worker import EnvWorker  # noqa: E402
from rlinf.workers.rollout.hf.async_huggingface_worker import (
    AsyncMultiStepRolloutWorker,
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


class _ValueHeadWeightSyncModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(4, 3),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 1),
        )


class _TiedParamWeightSyncModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        shared = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
        self.embed = shared
        self.lm_head = shared
        self.frozen = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
        self.frozen.requires_grad = False
        shared_buffer = torch.tensor([1.0], dtype=torch.float32)
        self.register_buffer("persistent_buf", shared_buffer)
        self.register_buffer("persistent_buf_alias", shared_buffer)
        self.register_buffer(
            "non_persistent_buf",
            torch.tensor([2.0], dtype=torch.float32),
            persistent=False,
        )


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


def _make_value_head_model(
    device: torch.device | str = "cpu",
) -> _ValueHeadWeightSyncModel:
    return _ValueHeadWeightSyncModel().to(device)


def _get_cuda_device() -> torch.device:
    if (
        Worker.torch_platform is None
        or not hasattr(Worker.torch_platform, "is_available")
        or not Worker.torch_platform.is_available()
    ):
        pytest.skip("Accelerator tests require at least 1 accelerator.")
    return torch.device(f"{Worker.torch_device_type}:0")


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


def _assert_patch_equal(
    lhs: EmptyWeightPatch | WeightPatch, rhs: EmptyWeightPatch | WeightPatch
) -> None:
    assert type(lhs) is type(rhs)
    for field_name, lhs_value in vars(lhs).items():
        rhs_value = getattr(rhs, field_name)
        torch.testing.assert_close(
            lhs_value.cpu(),
            rhs_value.cpu(),
            msg=f"Mismatch at patch field={field_name}",
        )


def _stress_cuda_allocator(device: torch.device, num_tensors: int = 128) -> None:
    streams = [Worker.torch_platform.Stream(device=device) for _ in range(2)]
    tensors: list[torch.Tensor] = []
    for stream in streams:
        with Worker.torch_platform.stream(stream):
            for _ in range(num_tensors):
                tensors.append(
                    torch.empty((256, 256), device=device, dtype=torch.float32)
                )
    for stream in streams:
        stream.synchronize()
    del tensors


def _get_param_names_need_sync(model: torch.nn.Module) -> list[str]:
    return collect_param_names_need_sync(model)


def test_collect_param_names_need_sync_keeps_tied_aliases_and_persistent_buffers():
    model = _TiedParamWeightSyncModel()

    param_names_need_sync = collect_param_names_need_sync(model)

    assert param_names_need_sync == [
        "embed",
        "lm_head",
        "persistent_buf",
        "persistent_buf_alias",
    ]


async def _init_patch_syncers(
    sender_syncer: PatchWeightSyncer,
    receiver_syncer: PatchWeightSyncer,
    sender_model: torch.nn.Module,
    receiver_model: torch.nn.Module,
    transport: _InMemoryDuplexTransport,
) -> None:
    await asyncio.gather(
        sender_syncer.init_sender(
            sender_model.state_dict(),
            _get_param_names_need_sync(sender_model),
            transport.sender_send,
            transport.sender_recv,
        ),
        receiver_syncer.init_receiver(
            receiver_model.state_dict(),
            transport.receiver_recv,
            transport.receiver_send,
        ),
    )


async def _init_bucket_syncer(
    syncer: BucketWeightSyncer,
    sender_model: torch.nn.Module,
    *,
    param_names_need_sync: list[str] | None = None,
) -> None:
    async def _unused_send(_data):
        return None

    await syncer.init_sender(
        sender_model.state_dict(),
        (
            _get_param_names_need_sync(sender_model)
            if param_names_need_sync is None
            else param_names_need_sync
        ),
        _unused_send,
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


def test_patch_weight_syncer_inactive_sender_returns_empty_patch():
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

    async def _run() -> None:
        await asyncio.gather(
            sender_syncer.init_sender(
                sender_model.state_dict(),
                _get_param_names_need_sync(sender_model),
                transport.sender_send,
                transport.sender_recv,
                is_sender=False,
            ),
            receiver_syncer.init_receiver(
                receiver_model.state_dict(),
                transport.receiver_recv,
                transport.receiver_send,
            ),
        )

        assert sender_syncer.sender_initialized()
        assert sender_syncer.snapshot is None
        assert sender_syncer.patch_builder is not None
        assert sender_syncer.patch_builder.snapshot is None

        patch = sender_syncer.create_patch(sender_model.state_dict(), version=11)
        assert isinstance(patch, EmptyWeightPatch)
        assert int(patch.version.item()) == 11

        async def _unused_send(_data):
            return None

        await sender_syncer.sync(sender_model.state_dict(), _unused_send, version=12)

    asyncio.run(_run())


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


def test_cpu_snapshot_patch_builder_matches_gpu_snapshot_under_allocator_pressure():
    device = _get_cuda_device()
    model = _make_model(device)
    state_dict = model.state_dict()
    ordered_keys = list(state_dict.keys())
    param_names_need_sync = _get_param_names_need_sync(model)
    original_shapes = {
        key: as_coo_2d_view(value)[1] for key, value in state_dict.items()
    }

    cpu_snapshot = {
        key: as_coo_2d_view(value.detach())[0].cpu().pin_memory()
        for key, value in state_dict.items()
        if key in param_names_need_sync
    }
    gpu_snapshot = {
        key: as_coo_2d_view(value.detach())[0].clone()
        for key, value in state_dict.items()
        if key in param_names_need_sync
    }
    cpu_builder = CPUSnapshotPatchBuilder(
        cpu_snapshot,
        ordered_keys,
        param_names_need_sync,
        original_shapes,
        torch.device("cpu"),
        delta_encoding=True,
    )
    gpu_builder = GPUSnapshotPatchBuilder(
        gpu_snapshot,
        ordered_keys,
        param_names_need_sync,
        original_shapes,
        torch.device("cpu"),
        delta_encoding=True,
    )

    for step in range(1, 25):
        with torch.no_grad():
            for key in param_names_need_sync:
                value_2dview, _ = as_coo_2d_view(state_dict[key])
                row = step % value_2dview.shape[0]
                col = (step * 7 + len(key)) % value_2dview.shape[1]
                value_2dview[row, col] += (step % 5 + 1) * 0.125
                if value_2dview.numel() > 1:
                    row2 = (row + 1) % value_2dview.shape[0]
                    col2 = (col + 3) % value_2dview.shape[1]
                    value_2dview[row2, col2] -= (step % 3 + 1) * 0.25

        cpu_patch = cpu_builder.create_patch(state_dict, version=step)
        assert all(tensor.device.type == "cpu" for tensor in cpu_patch.tensors())
        _stress_cuda_allocator(device)
        gpu_patch = gpu_builder.create_patch(state_dict, version=step)
        assert all(tensor.device.type == "cpu" for tensor in gpu_patch.tensors())

        _assert_patch_equal(cpu_patch, gpu_patch)
        for key in param_names_need_sync:
            torch.testing.assert_close(
                cpu_snapshot[key],
                gpu_snapshot[key].cpu(),
                msg=f"Mismatch at snapshot key={key}, step={step}",
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


def test_patch_weight_syncer_init_sync_bootstraps_selected_prefixes():
    device = _get_cuda_device()
    torch.manual_seed(0)
    sender_model = _make_value_head_model(device)
    torch.manual_seed(1)
    receiver_model = _make_value_head_model(device)
    transport = _InMemoryDuplexTransport()

    sender_syncer = PatchWeightSyncer(
        snapshot_device="cpu",
        transport_device="cpu",
        delta_encoding=True,
        compression_algorithm="none",
        init_sync_enabled=True,
        init_sync_prefixes=["value_head"],
        init_sync_bucket_size=32,
    )
    receiver_syncer = PatchWeightSyncer(
        snapshot_device="cpu",
        transport_device="cpu",
        delta_encoding=True,
        compression_algorithm="none",
        init_sync_enabled=True,
        init_sync_prefixes=["value_head"],
        init_sync_bucket_size=32,
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
            sender_model.state_dict(), transport.sender_send, version=5
        )
        return await receiver_syncer.apply(receiver_model, transport.receiver_recv)

    applied_version = asyncio.run(_run())

    assert applied_version == 5
    torch.testing.assert_close(
        sender_model.value_head[0].weight, receiver_model.value_head[0].weight
    )
    torch.testing.assert_close(
        sender_model.value_head[2].weight, receiver_model.value_head[2].weight
    )
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            sender_model.backbone.weight, receiver_model.backbone.weight
        )


def test_patch_weight_syncer_init_sync_bootstraps_full_state_dict():
    device = _get_cuda_device()
    sender_model = _make_bucket_dtype_model(device)
    receiver_model = copy.deepcopy(sender_model)
    transport = _InMemoryDuplexTransport()

    with torch.no_grad():
        receiver_model.fp32_param[0, 0] = -17.5
        receiver_model.bf16_param[1, 1] += torch.tensor(
            9.0, dtype=torch.bfloat16, device=device
        )
        receiver_model.int64_buf[0] = -(2**41 + 3)
        receiver_model.bool_buf.logical_not_()

    sender_syncer = PatchWeightSyncer(
        snapshot_device="cpu",
        transport_device="cpu",
        delta_encoding=True,
        compression_algorithm="none",
        init_sync_enabled=True,
        init_sync_prefixes=None,
        init_sync_bucket_size=32,
    )
    receiver_syncer = PatchWeightSyncer(
        snapshot_device="cpu",
        transport_device="cpu",
        delta_encoding=True,
        compression_algorithm="none",
        init_sync_enabled=True,
        init_sync_prefixes=None,
        init_sync_bucket_size=32,
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
            sender_model.state_dict(), transport.sender_send, version=7
        )
        return await receiver_syncer.apply(receiver_model, transport.receiver_recv)

    applied_version = asyncio.run(_run())

    assert applied_version == 7
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
    if Worker.accelerator_type != AcceleratorType.NV_GPU:
        pytest.skip("CUDA nvcomp tests require NV_GPU.")
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
        await _init_bucket_syncer(syncer, sender_model)
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
        await _init_bucket_syncer(syncer, sender_model)
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

    asyncio.run(_init_bucket_syncer(syncer, sender_model))
    buckets = list(syncer.iter_buckets(sender_model.state_dict(), version=11))
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
        await _init_bucket_syncer(syncer, sender_model)
        await syncer.sync(sender_model.state_dict(), transport.send, version=11)
        return await syncer.apply(receiver_model, transport.recv)

    applied_version = asyncio.run(_run())

    assert applied_version == 11
    _assert_state_dict_equal(
        _clone_state_dict(sender_model), _clone_state_dict(receiver_model)
    )


def test_iter_named_tensor_buckets_supports_custom_dtype_resolver():
    model = _make_bucket_dtype_model()
    buckets = list(
        iter_named_tensor_buckets(
            model.state_dict().items(),
            version=17,
            bucket_size=32,
            bucket_device="cpu",
            dtype_resolver=lambda key, dtype: (
                torch.float16 if key == "fp32_param" else dtype
            ),
        )
    )
    payload = {
        key: value
        for bucket in buckets
        for key, value in bucket.items()
        if key not in {"total_buckets", "syncer_version"}
    }

    assert payload["fp32_param"].dtype == torch.float16
    assert payload["bf16_param"].dtype == torch.bfloat16
    assert payload["int64_buf"].dtype == torch.int64
    assert payload["bool_buf"].dtype == torch.bool
    assert buckets[0]["total_buckets"].dtype == torch.int32
    assert buckets[0]["syncer_version"].dtype == torch.int32


def test_bucket_weight_syncer_preserves_nonfloating_dtypes_when_bucket_dtype_set():
    model = _make_bucket_dtype_model()
    syncer = BucketWeightSyncer(
        bucket_size=32,
        bucket_dtype=torch.bfloat16,
        bucket_device="cpu",
        load_instant=True,
    )

    asyncio.run(_init_bucket_syncer(syncer, model))
    buckets = list(syncer.iter_buckets(model.state_dict(), version=12))
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

        await _init_bucket_syncer(syncer, sender_model)
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
    asyncio.run(
        _init_bucket_syncer(
            syncer,
            model,
            param_names_need_sync=list(state_dict.keys()),
        )
    )

    with pytest.raises(ValueError, match="conflicts with metadata key"):
        list(syncer.iter_buckets(state_dict, version=1))


def test_bucket_weight_syncer_metadata_dtypes_are_nccl_safe():
    model = _TinyWeightSyncModel()
    syncer = BucketWeightSyncer(
        bucket_size=32,
        bucket_dtype=torch.bfloat16,
        bucket_device="cpu",
        load_instant=True,
    )

    asyncio.run(_init_bucket_syncer(syncer, model))
    buckets = list(syncer.iter_buckets(model.state_dict(), version=13))

    assert buckets
    assert buckets[0]["total_buckets"].dtype == torch.int32
    assert buckets[0]["syncer_version"].dtype == torch.int32


def test_bucket_weight_syncer_skips_frozen_params_but_syncs_persistent_buffers():
    sender_model = _make_model()
    receiver_model = copy.deepcopy(sender_model)
    transport = _InMemoryTransport()
    syncer = BucketWeightSyncer(
        bucket_size=32,
        bucket_dtype=torch.float32,
        bucket_device="cpu",
        load_instant=True,
    )

    sender_model.linear.weight.requires_grad_(False)

    async def _run() -> int:
        await _init_bucket_syncer(syncer, sender_model)

        with torch.no_grad():
            sender_model.linear.weight.fill_(77.0)
            sender_model.linear.bias.add_(5.0)
            sender_model.scalar_buf.mul_(3.0)

        await syncer.sync(sender_model.state_dict(), transport.send, version=15)
        return await syncer.apply(receiver_model, transport.recv)

    applied_version = asyncio.run(_run())

    assert applied_version == 15
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            sender_model.linear.weight, receiver_model.linear.weight
        )
    torch.testing.assert_close(sender_model.linear.bias, receiver_model.linear.bias)
    torch.testing.assert_close(sender_model.scalar_buf, receiver_model.scalar_buf)


def test_weight_syncer_factory_builds_patch_and_bucket():
    patch_cfg = OmegaConf.create(
        {
            "type": "patch",
            "patch": {
                "snapshot_device": "cpu",
                "transport_device": "cpu",
                "delta_encoding": True,
                "compression": "none",
                "init_sync": {
                    "enabled": True,
                    "prefixes": ["value_head"],
                    "bucket_size": 4096,
                },
            },
        }
    )
    patch_syncer = WeightSyncer.create(patch_cfg)
    assert isinstance(patch_syncer, PatchWeightSyncer)
    assert patch_syncer.comm_options is None
    assert patch_syncer.init_sync_enabled is True
    assert patch_syncer.init_sync_prefixes == ["value_head"]
    assert patch_syncer.init_sync_bucket_size == 4096

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
    assert bucket_syncer.comm_options is None


def test_weight_syncer_factory_builds_shared_comm_options():
    base_patch_cfg = {
        "type": "patch",
        "patch": {
            "snapshot_device": "cpu",
            "transport_device": "cpu",
            "delta_encoding": True,
            "compression": "none",
        },
    }
    base_bucket_cfg = {
        "type": "bucket",
        "bucket": {
            "bucket_size": 128,
            "bucket_dtype": None,
            "bucket_device": "cpu",
        },
    }
    shared_options = {
        "use_ring_sync": True,
        "nccl_max_ctas": 8,
        "nccl_min_ctas": 2,
    }

    for base_cfg in (base_patch_cfg, base_bucket_cfg):
        cfg = OmegaConf.create({**base_cfg, **shared_options})

        syncer = WeightSyncer.create(cfg)

        comm_options = syncer.comm_options
        assert comm_options is not None
        assert comm_options.use_ring_broadcast is True
        assert comm_options.accel_max_ctas == 8
        assert comm_options.accel_min_ctas == 2


def test_weight_syncer_factory_rejects_unknown_type():
    cfg = OmegaConf.create({"type": "unknown"})
    with pytest.raises(ValueError, match="Unsupported weight syncer type"):
        WeightSyncer.create(cfg)


# Both async runners must pick the overlap up from the mixin, not from their own
# copy of it.
RUNNER_CLASSES = (AsyncEmbodiedRunner, AsyncPPOEmbodiedRunner)


class _Handle:
    def __init__(self, done: bool = False) -> None:
        self.is_done = done
        self.wait_calls = 0

    def done(self) -> bool:
        return self.is_done

    def wait(self) -> None:
        self.wait_calls += 1
        self.is_done = True


class _Actor:
    def __init__(self, handles: list[_Handle]) -> None:
        self.handles = handles
        self.sync_calls = 0

    def sync_model_to_rollout(self) -> _Handle:
        handle = self.handles[self.sync_calls]
        self.sync_calls += 1
        return handle


class _Rollout:
    def __init__(
        self,
        blocking_handles: list[_Handle],
        background_handles: list[_Handle],
    ) -> None:
        self.blocking_handles = blocking_handles
        self.background_handles = background_handles
        self.blocking_calls = 0
        self.background_calls = 0

    def sync_model_from_actor(self) -> _Handle:
        handle = self.blocking_handles[self.blocking_calls]
        self.blocking_calls += 1
        return handle

    def request_actor_sync_model(self) -> _Handle:
        handle = self.background_handles[self.background_calls]
        self.background_calls += 1
        return handle


def _make_runner(runner_cls, actor: _Actor, rollout: _Rollout, no_wait: bool = False):
    """Build a runner with only the attributes the weight-sync path touches."""
    runner = object.__new__(runner_cls)
    runner.actor = actor
    runner.rollout = rollout
    runner.logger = logging.getLogger("test_async_weight_sync")
    runner.sync_weight_no_wait = no_wait
    runner._pending_rollout_weight_sync = None
    runner._weight_sync_request_total = 0
    runner._weight_sync_coalesced_total = 0
    return runner


class _LifecycleWorker:
    """Worker-group stub for a zero-step runner lifecycle."""

    def interact(self, **kwargs) -> _Handle:
        return _Handle()

    def generate(self, **kwargs) -> _Handle:
        return _Handle()

    def recv_rollout_trajectories(self, **kwargs) -> _Handle:
        return _Handle()

    def stop(self) -> _Handle:
        return _Handle()


def _make_async_rollout_worker(
    apply_gates: list[asyncio.Event],
) -> tuple[AsyncMultiStepRolloutWorker, list[asyncio.Event]]:
    """Build only the rollout-side background-sync state machine."""
    worker = object.__new__(AsyncMultiStepRolloutWorker)
    worker._background_weight_sync_active = True
    worker._weight_sync_requested = False
    worker._weight_sync_work = None
    worker._weight_sync_apply_total = 0
    worker._weight_sync_coalesced_total = 0
    worker._weight_sync_request_total = 0
    worker.version = 0
    apply_started = [asyncio.Event() for _ in apply_gates]

    async def recv_and_apply(worker_self) -> int:
        apply_index = worker_self.version
        apply_started[apply_index].set()
        await apply_gates[apply_index].wait()
        worker_self.version += 1
        return worker_self.version

    worker._recv_and_apply_actor_sync = MethodType(recv_and_apply, worker)
    return worker, apply_started


async def _request_actor_sync_model(worker: AsyncMultiStepRolloutWorker) -> int:
    """Call the undecorated worker method without constructing a Ray worker."""
    request = inspect.unwrap(AsyncMultiStepRolloutWorker.request_actor_sync_model)
    return await request(worker)


def _weight_sync_cfg(no_wait: bool, syncer_type: str) -> OmegaConf:
    return OmegaConf.create(
        {
            "actor": {"sync_weight_no_wait": no_wait},
            "weight_syncer": {"type": syncer_type},
        }
    )


@pytest.mark.parametrize("syncer_type", ["patch", "bucket"])
def test_mixin_reads_the_flag_for_either_syncer(syncer_type) -> None:
    """The mixin only reads the flag; the config layer rejects bad combos."""
    runner = object.__new__(AsyncWeightSyncMixin)
    runner.cfg = _weight_sync_cfg(no_wait=True, syncer_type=syncer_type)

    runner.init_weight_sync_state()

    assert runner.sync_weight_no_wait
    assert runner._pending_rollout_weight_sync is None


def test_nonblocking_weight_sync_requires_patch_syncer() -> None:
    with pytest.raises(AssertionError, match="weight_syncer.type=patch"):
        validate_weight_sync_overlap_cfg(
            _weight_sync_cfg(no_wait=True, syncer_type="bucket")
        )


def test_nonblocking_weight_sync_accepts_patch_syncer() -> None:
    validate_weight_sync_overlap_cfg(
        _weight_sync_cfg(no_wait=True, syncer_type="patch")
    )


def test_blocking_weight_sync_allows_bucket_syncer() -> None:
    validate_weight_sync_overlap_cfg(
        _weight_sync_cfg(no_wait=False, syncer_type="bucket")
    )


def test_async_embodied_runner_startup_sync_is_blocking() -> None:
    runner = object.__new__(AsyncEmbodiedRunner)
    runner.global_step = 0
    runner.max_steps = 0
    runner.actor = _LifecycleWorker()
    runner.rollout = _LifecycleWorker()
    runner.env = _LifecycleWorker()
    runner.reward = None
    runner.env_channel = None
    runner.rollout_channel = None
    runner.reward_channel = None
    runner.actor_channel = None
    runner.env_metric_channel = None
    runner.rollout_metric_channel = None
    runner.update_rollout_weights = MagicMock()
    runner.drain_pending_rollout_weight_sync = MagicMock()

    runner.run()

    runner.update_rollout_weights.assert_called_once_with()
    runner.drain_pending_rollout_weight_sync.assert_called_once_with()


def test_rollout_request_completes_only_after_weights_are_applied() -> None:
    async def run_test() -> None:
        apply_gate = asyncio.Event()
        worker, apply_started = _make_async_rollout_worker([apply_gate])

        request_task = asyncio.create_task(_request_actor_sync_model(worker))
        await asyncio.wait_for(apply_started[0].wait(), timeout=5.0)
        await asyncio.sleep(0)

        assert not request_task.done()
        assert worker._weight_sync_apply_total == 0
        assert worker.version == 0

        apply_gate.set()

        assert await asyncio.wait_for(request_task, timeout=5.0) == 1
        assert worker._weight_sync_apply_total == 1
        assert worker._weight_sync_work is None
        assert worker.version == 1

    asyncio.run(run_test())


def test_rollout_requests_coalesce_without_reordering_apply() -> None:
    async def run_test() -> None:
        first_apply_gate = asyncio.Event()
        second_apply_gate = asyncio.Event()
        worker, apply_started = _make_async_rollout_worker(
            [first_apply_gate, second_apply_gate]
        )

        first_request = asyncio.create_task(_request_actor_sync_model(worker))
        await asyncio.wait_for(apply_started[0].wait(), timeout=5.0)
        second_request = asyncio.create_task(_request_actor_sync_model(worker))
        await asyncio.sleep(0)

        assert worker._weight_sync_coalesced_total == 1
        assert not first_request.done()
        assert not second_request.done()

        first_apply_gate.set()
        assert await asyncio.wait_for(first_request, timeout=5.0) == 1
        await asyncio.wait_for(apply_started[1].wait(), timeout=5.0)
        assert not second_request.done()
        assert worker.version == 1

        second_apply_gate.set()
        assert await asyncio.wait_for(second_request, timeout=5.0) == 2
        assert worker._weight_sync_apply_total == 2
        assert worker._weight_sync_work is None
        assert worker.version == 2

    asyncio.run(run_test())


@pytest.mark.parametrize("runner_cls", RUNNER_CLASSES)
def test_runner_inherits_the_shared_mixin(runner_cls) -> None:
    assert issubclass(runner_cls, AsyncWeightSyncMixin)
    # The overlap must resolve to the mixin, i.e. no runner-local reimplementation.
    assert (
        runner_cls.update_rollout_weights is AsyncWeightSyncMixin.update_rollout_weights
    )


@pytest.mark.parametrize("runner_cls", RUNNER_CLASSES)
def test_blocking_weight_sync_waits_for_both_sides(runner_cls) -> None:
    actor_handle = _Handle()
    rollout_handle = _Handle()
    actor = _Actor([actor_handle])
    rollout = _Rollout([rollout_handle], [])
    runner = _make_runner(runner_cls, actor, rollout)

    runner.update_rollout_weights()

    assert actor.sync_calls == 1
    assert rollout.blocking_calls == 1
    assert rollout.background_calls == 0
    assert actor_handle.wait_calls == 1
    assert rollout_handle.wait_calls == 1
    assert runner._pending_rollout_weight_sync is None


@pytest.mark.parametrize("runner_cls", RUNNER_CLASSES)
def test_nonblocking_weight_sync_does_not_wait(runner_cls) -> None:
    actor_handle = _Handle()
    rollout_handle = _Handle()
    actor = _Actor([actor_handle])
    rollout = _Rollout([], [rollout_handle])
    runner = _make_runner(runner_cls, actor, rollout, no_wait=True)

    runner.update_rollout_weights(no_wait=True)

    assert rollout.background_calls == 1
    assert actor.sync_calls == 1
    assert actor_handle.wait_calls == 0
    assert rollout_handle.wait_calls == 0
    assert runner._pending_rollout_weight_sync == (rollout_handle, actor_handle)


@pytest.mark.parametrize("runner_cls", RUNNER_CLASSES)
def test_nonblocking_weight_sync_coalesces_until_previous_sync_finishes(
    runner_cls,
) -> None:
    first_actor_handle = _Handle()
    first_rollout_handle = _Handle(done=True)
    second_actor_handle = _Handle()
    second_rollout_handle = _Handle(done=True)
    actor = _Actor([first_actor_handle, second_actor_handle])
    rollout = _Rollout([], [first_rollout_handle, second_rollout_handle])
    runner = _make_runner(runner_cls, actor, rollout, no_wait=True)

    runner.update_rollout_weights(no_wait=True)
    runner.update_rollout_weights(no_wait=True)

    # The second request is dropped, not queued: still one sync in flight.
    assert actor.sync_calls == 1
    assert rollout.background_calls == 1
    assert first_actor_handle.wait_calls == 0
    assert first_rollout_handle.wait_calls == 0
    assert runner._weight_sync_coalesced_total == 1
    assert runner._weight_sync_request_total == 2

    first_actor_handle.is_done = True
    runner.update_rollout_weights(no_wait=True)

    assert first_actor_handle.wait_calls == 1
    assert first_rollout_handle.wait_calls == 1
    assert actor.sync_calls == 2
    assert rollout.background_calls == 2
    assert runner._weight_sync_coalesced_total == 1
    assert runner._weight_sync_request_total == 3
    assert runner._pending_rollout_weight_sync == (
        second_rollout_handle,
        second_actor_handle,
    )


@pytest.mark.parametrize("runner_cls", RUNNER_CLASSES)
def test_blocking_weight_sync_drains_an_inflight_background_sync(runner_cls) -> None:
    pending_actor_handle = _Handle()
    pending_rollout_handle = _Handle()
    blocking_actor_handle = _Handle()
    blocking_rollout_handle = _Handle()
    actor = _Actor([blocking_actor_handle])
    rollout = _Rollout([blocking_rollout_handle], [])
    runner = _make_runner(runner_cls, actor, rollout)
    runner._pending_rollout_weight_sync = (
        pending_rollout_handle,
        pending_actor_handle,
    )

    runner.update_rollout_weights(no_wait=False)

    assert pending_actor_handle.wait_calls == 1
    assert pending_rollout_handle.wait_calls == 1
    assert blocking_actor_handle.wait_calls == 1
    assert blocking_rollout_handle.wait_calls == 1
    assert runner._pending_rollout_weight_sync is None


@pytest.mark.parametrize("runner_cls", RUNNER_CLASSES)
def test_teardown_drains_an_inflight_background_sync(runner_cls) -> None:
    actor_handle = _Handle()
    rollout_handle = _Handle()
    runner = _make_runner(runner_cls, _Actor([]), _Rollout([], []))
    runner._pending_rollout_weight_sync = (rollout_handle, actor_handle)

    runner.drain_pending_rollout_weight_sync()

    assert actor_handle.wait_calls == 1
    assert rollout_handle.wait_calls == 1
    assert runner._pending_rollout_weight_sync is None


@pytest.mark.parametrize("runner_cls", RUNNER_CLASSES)
def test_teardown_is_a_noop_without_a_pending_sync(runner_cls) -> None:
    runner = _make_runner(runner_cls, _Actor([]), _Rollout([], []))

    runner.drain_pending_rollout_weight_sync()

    assert runner._pending_rollout_weight_sync is None


class TestOverlapEnvBootstrap(unittest.TestCase):
    def setUp(self):
        self.cfg = OmegaConf.create(
            {
                "env": {
                    "train": {
                        "total_num_envs": 2,
                        "max_steps_per_rollout_epoch": 8,
                        "env_type": "dummy",
                        "auto_reset": True,
                        "video_cfg": {"save_video": False},
                        "max_episode_steps": 10,
                    },
                    "eval": {
                        "total_num_envs": 2,
                        "max_steps_per_rollout_epoch": 8,
                        "env_type": "dummy",
                        "video_cfg": {"save_video": False},
                        "max_episode_steps": 10,
                    },
                },
                "actor": {
                    "model": {
                        "model_type": "dummy",
                        "num_action_chunks": 4,
                        "action_dim": 7,
                    }
                },
                "rollout": {
                    "group_name": "RolloutGroup",
                    "pipeline_stage_num": 1,
                    "collect_transitions": False,
                },
                "runner": {
                    "val_check_interval": -1,
                },
                "algorithm": {
                    "rollout_epoch": 1,
                },
                "cluster": {},
            }
        )

        # Create EnvWorker instance without calling __init__
        self.worker = object.__new__(EnvWorker)

        # Manually set required attributes
        self.worker.cfg = self.cfg
        self.worker._rank = 0
        self.worker._world_size = 1
        self.worker._group_name = "EnvGroup"
        self.worker._timer_metrics = {}
        self.worker.stage_num = 1
        self.worker.train_num_envs_per_stage = 2
        self.worker.n_train_chunk_steps = 2
        self.worker.rollout_epoch = 1
        self.worker.enable_online_lerobot = False
        self.worker.enable_offload = False
        self.worker.train_enable_offload = False
        self.worker.use_training_pipeline = False
        self.worker.collect_transitions = False
        self.worker.enable_rlt = False
        self.worker.collect_prev_infos = True
        self.worker.reward_mode = self.cfg.get("reward", {}).get(
            "reward_mode", "per_step"
        )
        self.worker.history_reward_assign = self.cfg.get("reward", {}).get(
            "history_reward_assign", True
        )
        self.worker._accelerator_type = AcceleratorType.NO_ACCEL
        self.worker._prefetched_train_bootstrap = None
        self.worker.smooth_intervene = SmoothInterveneController(
            stage_num=self.worker.stage_num, enabled=False
        )

        # Mock env_list
        mock_env = MagicMock(
            wait_delay=AsyncMock(),
            insert_delay_metrics=MagicMock(return_value=torch.empty(0)),
        )
        self.worker.env_list = [mock_env]

        # Initialize last_obs_list for auto_reset=True
        self.worker.last_obs_list = [{"main_images": torch.zeros(2, 3, 224, 224)}]
        self.worker.last_intervened_info_list = [(None, None)]
        self.worker.only_eval = False
        self.worker.model_cfg = self.cfg.actor.model
        self.worker.train_batch_size = (
            self.cfg.env.train.total_num_envs // self.worker.stage_num
        )
        self.worker.env_decoupled_mode = False
        self.worker.send_to = MagicMock()

    def test_prefetch_consumption(self):
        """Test that prefetched bootstrap is correctly consumed in interact()."""
        rollout_channel = MagicMock()
        input_channel = MagicMock()

        # Mock recv_from to return a dummy PolicyOutput
        mock_policy_output = MagicMock()
        mock_policy_output.actions = torch.zeros(2, 28)
        mock_policy_output.bootstrap_values = None
        mock_policy_output.forward_inputs = {"action": torch.zeros(2, 28)}
        mock_policy_output.versions = torch.zeros(2, 1)
        mock_policy_output.intervene_flags = None

        # Patch methods on the instance
        self.worker.recv_from = MagicMock(return_value=mock_policy_output)
        self.worker.env_interact_step = MagicMock(
            return_value=(
                EnvOutput(
                    obs={"main_images": torch.zeros(2, 3, 224, 224)},
                    dones=torch.zeros(2, 4, dtype=torch.bool),
                    truncations=torch.zeros(2, 4, dtype=torch.bool),
                    terminations=torch.zeros(2, 4, dtype=torch.bool),
                ),
                {},
                {},
            )
        )
        self.worker.send_env_batch = MagicMock()
        self.worker.store_last_obs_and_intervened_info = MagicMock()
        self.worker.finish_rollout = MagicMock()
        self.worker.compute_bootstrap_rewards = MagicMock(
            return_value=torch.zeros(2, 4)
        )
        self.worker.record_env_metrics = MagicMock()

        # 1. Prefetch
        # We need to mock _bootstrap_and_send_train as it's called by prefetch_train_bootstrap
        dummy_bootstrap = [
            EnvOutput(obs={"m": torch.zeros(1)}, dones=torch.zeros(1, 4))
        ]
        self.worker._bootstrap_and_send_train = MagicMock(return_value=dummy_bootstrap)

        self.worker.prefetch_train_bootstrap(rollout_channel)
        self.assertEqual(self.worker._prefetched_train_bootstrap, dummy_bootstrap)

        # 2. Interact (should consume the prefetch)
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Mock send_rollout_trajectories as it's awaited
            self.worker.send_rollout_trajectories = MagicMock(
                return_value=asyncio.Future()
            )
            self.worker.send_rollout_trajectories.return_value.set_result(None)

            loop.run_until_complete(
                self.worker.interact(input_channel, rollout_channel, None, None)
            )
        finally:
            asyncio.set_event_loop(None)
            loop.close()

        self.assertIsNone(self.worker._prefetched_train_bootstrap)
        # Verify that _bootstrap_and_send_train was NOT called during interact
        # (it was only called once during prefetch)
        self.worker._bootstrap_and_send_train.assert_called_once()
        self.assertEqual(self.worker.record_env_metrics.call_count, 2)

    def test_duplicate_prefetch_protection(self):
        """Test that multiple prefetch calls raise RuntimeError."""
        rollout_channel = MagicMock()
        self.worker._bootstrap_and_send_train = MagicMock()

        # First prefetch
        self.worker.prefetch_train_bootstrap(rollout_channel)

        # Second prefetch should raise RuntimeError
        with self.assertRaises(RuntimeError) as cm:
            self.worker.prefetch_train_bootstrap(rollout_channel)

        self.assertIn("A prefetched train bootstrap already exists", str(cm.exception))

    def test_record_env_metrics_appends_values(self):
        """record_env_metrics should append env info tensors as-is."""
        env_metrics = {}

        self.worker.record_env_metrics(
            env_metrics, {"episode_len": torch.tensor([5, 6])}
        )
        self.worker.record_env_metrics(
            env_metrics, {"episode_len": torch.tensor([7, 8])}
        )

        self.assertEqual(len(env_metrics["episode_len"]), 2)
        self.assertTrue(
            torch.equal(env_metrics["episode_len"][0], torch.tensor([5, 6]))
        )
        self.assertTrue(
            torch.equal(env_metrics["episode_len"][1], torch.tensor([7, 8]))
        )

    def test_interact_records_metrics_only_on_final_chunk_when_not_auto_reset(self):
        """Non-auto-reset training should record episode metrics only once per rollout epoch."""
        self.worker.cfg.env.train.auto_reset = False
        self.worker.cfg.env.train.ignore_terminations = False
        self.worker.env_list[0].reset.return_value = (
            {"main_images": torch.zeros(2, 3, 224, 224)},
            {},
        )
        self.worker.record_env_metrics = MagicMock()

        rollout_channel = MagicMock()
        input_channel = MagicMock()

        mock_policy_output = MagicMock()
        mock_policy_output.actions = torch.zeros(2, 28)
        mock_policy_output.bootstrap_values = None
        mock_policy_output.forward_inputs = {"action": torch.zeros(2, 28)}
        mock_policy_output.versions = torch.zeros(2, 1)
        mock_policy_output.intervene_flags = None

        self.worker.recv_from = MagicMock(return_value=mock_policy_output)
        self.worker.env_interact_step = MagicMock(
            return_value=(
                EnvOutput(
                    obs={"main_images": torch.zeros(2, 3, 224, 224)},
                    dones=torch.zeros(2, 4, dtype=torch.bool),
                    truncations=torch.zeros(2, 4, dtype=torch.bool),
                    terminations=torch.zeros(2, 4, dtype=torch.bool),
                ),
                {"episode_len": torch.tensor([1, 2])},
                {},
            )
        )
        self.worker.send_env_batch = MagicMock()
        self.worker.store_last_obs_and_intervened_info = MagicMock()
        self.worker.finish_rollout = MagicMock()
        self.worker.compute_bootstrap_rewards = MagicMock(
            return_value=torch.zeros(2, 4)
        )
        self.worker._bootstrap_and_send_train = MagicMock(
            return_value=[EnvOutput(obs={"m": torch.zeros(1)}, dones=torch.zeros(1, 4))]
        )
        self.worker.send_rollout_trajectories = MagicMock(
            return_value=MagicMock(wait=MagicMock(return_value=None))
        )

        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                self.worker.interact(input_channel, rollout_channel, None, None)
            )
        finally:
            asyncio.set_event_loop(None)
            loop.close()

        self.assertEqual(self.worker.record_env_metrics.call_count, 1)


if __name__ == "__main__":
    unittest.main()
