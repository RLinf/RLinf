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

import asyncio
from dataclasses import dataclass

import pytest
import torch

from rlinf.utils import pinned_feature_stream
from rlinf.utils.backbone_cache import ROLLOUT_BACKBONE_SAMPLE_ID_STRIDE


@dataclass(frozen=True)
class _Stats:
    bytes: int
    samples: int
    allocation_seconds: float = 0.01
    d2h_seconds: float = 0.02


class _FakeCache:
    instances = []

    def __init__(self, *, total_samples, feature_example, mask_example, device) -> None:
        del feature_example, mask_example, device
        self.samples = total_samples
        self.stored = 0
        self.byte_count = 0
        self.finalized = False
        self.cleared = False
        self.__class__.instances.append(self)

    def store_blocks(self, *, offset, blocks) -> None:
        assert offset == self.stored
        for feature, mask in blocks:
            self.stored += feature.shape[0]
            self.byte_count += sum(
                tensor.numel() * tensor.element_size() for tensor in (feature, mask)
            )

    def finalize(self) -> None:
        if self.stored != self.samples:
            raise RuntimeError("incomplete fake cache")
        self.finalized = True

    def stats(self) -> _Stats:
        return _Stats(bytes=self.byte_count, samples=self.samples)

    def clear(self) -> None:
        self.cleared = True


class _ImmediateWork:
    def __init__(self, value) -> None:
        self.value = value

    async def async_wait(self):
        return self.value


class _NeverWork:
    async def async_wait(self):
        await asyncio.Event().wait()


class _FakePlatform:
    def __init__(self) -> None:
        self.collects = 0

    @staticmethod
    def current_device():
        return 0

    def ipc_collect(self) -> None:
        self.collects += 1


class _FakeWorker:
    def __init__(self, received) -> None:
        self.received = list(received)
        self.sent = []
        self.logs = []
        self.torch_platform = _FakePlatform()

    def recv(self, **kwargs):
        assert kwargs["async_op"] is True
        assert kwargs["borrowed_ipc"] is True
        return _ImmediateWork(self.received.pop(0))

    def send(self, payload, **kwargs) -> None:
        self.sent.append((payload, kwargs))

    def log_info(self, message) -> None:
        self.logs.append(message)


class _TimeoutWorker(_FakeWorker):
    def __init__(self) -> None:
        super().__init__([])

    def recv(self, **kwargs):
        assert kwargs["async_op"] is True
        assert kwargs["borrowed_ipc"] is True
        return _NeverWork()


def _block(size: int):
    return (
        torch.arange(size * 3, dtype=torch.float32).reshape(size, 3),
        torch.ones(size, 3, dtype=torch.bool),
    )


def _batch(
    *,
    batch_index: int,
    start_block_index: int,
    offset: int,
    block_sizes: list[int],
    total_blocks: int = 3,
    total_samples: int = 6,
    source_rank: int = 2,
    consumer_rank: int = 5,
    model_version: int = 7,
    lease_id: str = "lease-1",
):
    blocks = [_block(size) for size in block_sizes]
    tensors = [tensor for block in blocks for tensor in block]
    byte_count = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    metadata = {
        "schema": 3,
        "lease_id": lease_id,
        "batch_index": batch_index,
        "start_block_index": start_block_index,
        "total_blocks": total_blocks,
        "total_samples": total_samples,
        "producer_rank": source_rank,
        "consumer_rank": consumer_rank,
        "model_version": model_version,
        "sample_id_base": source_rank * ROLLOUT_BACKBONE_SAMPLE_ID_STRIDE,
        "offset": offset,
        "block_sizes": block_sizes,
        "bytes": byte_count,
    }
    return tensors, metadata


def _receive(
    worker,
    *,
    expected_blocks=3,
    expected_samples=6,
    blocks_per_batch=2,
    timeout_seconds=1.0,
):
    return asyncio.run(
        pinned_feature_stream.receive_pinned_rollout_backbone_stream(
            worker,
            source_rank=2,
            consumer_rank=5,
            model_version=7,
            expected_blocks=expected_blocks,
            expected_samples=expected_samples,
            blocks_per_batch=blocks_per_batch,
            rollout_group_name="rollout",
            timeout_seconds=timeout_seconds,
        )
    )


@pytest.fixture(autouse=True)
def fake_cache(monkeypatch):
    _FakeCache.instances.clear()
    monkeypatch.setattr(pinned_feature_stream, "PinnedRolloutBackboneCache", _FakeCache)


def test_receive_stream_validates_counts_and_acks_each_batch():
    worker = _FakeWorker(
        [
            _batch(
                batch_index=0,
                start_block_index=0,
                offset=0,
                block_sizes=[2, 2],
            ),
            _batch(
                batch_index=1,
                start_block_index=2,
                offset=4,
                block_sizes=[2],
            ),
        ]
    )

    cache, metadata, metrics = _receive(worker)

    assert cache.finalized
    assert metadata["lease_id"] == "lease-1"
    assert [entry[0]["batch_index"] for entry in worker.sent] == [0, 1]
    assert [entry[0]["completed_blocks"] for entry in worker.sent] == [2, 3]
    assert all(entry[0]["status"] == "ok" for entry in worker.sent)
    assert worker.torch_platform.collects == 2
    assert metrics["actor/pinned_feature_samples"] == 6.0
    assert metrics["actor/pinned_feature_batches"] == 2.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", 2, "unsupported pinned backbone metadata"),
        ("batch_index", 1, "batches arrived out of order"),
        ("start_block_index", 1, "block range is invalid"),
        ("total_blocks", 4, "total block count changed"),
        ("total_samples", 7, "total sample count changed"),
        ("producer_rank", 3, "producer rank does not match route"),
        ("consumer_rank", 4, "consumer rank does not match Actor"),
        ("model_version", 8, "model version does not match Actor"),
        ("sample_id_base", 0, "sample-ID namespace is invalid"),
        ("offset", 1, "sample offset is invalid"),
        ("block_sizes", [2], "block count mismatch"),
        ("bytes", 1, "byte count mismatch"),
    ],
)
def test_receive_stream_rejects_malformed_first_batch(field, value, message):
    tensors, metadata = _batch(
        batch_index=0,
        start_block_index=0,
        offset=0,
        block_sizes=[2, 2],
    )
    metadata[field] = value
    worker = _FakeWorker([(tensors, metadata)])

    with pytest.raises(RuntimeError, match=message):
        _receive(worker)

    assert len(worker.sent) == 1
    assert worker.sent[0][0]["status"] == "error"
    assert message in worker.sent[0][0]["error"]


def test_receive_stream_rejects_lease_change_before_second_ack():
    worker = _FakeWorker(
        [
            _batch(
                batch_index=0,
                start_block_index=0,
                offset=0,
                block_sizes=[3],
                total_blocks=2,
            ),
            _batch(
                batch_index=1,
                start_block_index=1,
                offset=3,
                block_sizes=[3],
                total_blocks=2,
                lease_id="lease-2",
            ),
        ]
    )

    with pytest.raises(RuntimeError, match="lease changed"):
        _receive(worker, expected_blocks=2, blocks_per_batch=1)

    assert len(worker.sent) == 2
    assert worker.sent[0][0]["lease_id"] == "lease-1"
    assert worker.sent[0][0]["status"] == "ok"
    assert worker.sent[1][0]["status"] == "error"
    assert _FakeCache.instances[0].cleared


def test_receive_stream_rejects_incomplete_sample_count():
    worker = _FakeWorker(
        [
            _batch(
                batch_index=0,
                start_block_index=0,
                offset=0,
                block_sizes=[2, 2],
            ),
            _batch(
                batch_index=1,
                start_block_index=2,
                offset=4,
                block_sizes=[1],
            ),
        ]
    )

    with pytest.raises(RuntimeError, match="incomplete counts"):
        _receive(worker)

    assert worker.sent[-1][0]["status"] == "error"
    assert _FakeCache.instances[0].cleared


@pytest.mark.parametrize(
    ("expected_blocks", "expected_samples", "blocks_per_batch"),
    [(0, 1, 1), (1, 0, 1), (1, 1, 0)],
)
def test_receive_stream_requires_positive_contract_counts(
    expected_blocks, expected_samples, blocks_per_batch
):
    worker = _FakeWorker([])
    with pytest.raises(ValueError):
        _receive(
            worker,
            expected_blocks=expected_blocks,
            expected_samples=expected_samples,
            blocks_per_batch=blocks_per_batch,
        )


def test_receive_stream_requires_positive_timeout():
    with pytest.raises(ValueError, match="timeout"):
        _receive(_FakeWorker([]), timeout_seconds=0)


def test_receive_stream_times_out_without_a_producer_batch():
    worker = _TimeoutWorker()

    with pytest.raises(TimeoutError, match="Rollout rank 2"):
        _receive(worker, timeout_seconds=0.01)

    assert not worker.sent
    assert worker.torch_platform.collects == 1
