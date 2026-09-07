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

import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.scheduler.cluster.cluster import Cluster
from rlinf.scheduler.cluster.config import (
    ClusterConfig,
    CollectiveConfig,
    TensorBufferPoolConfig,
    TensorCompressionConfig,
    TensorCompressionManager,
)
from rlinf.scheduler.collective.collective_group import (
    CollectiveGroup,
    CollectiveGroupOptions,
    TensorData,
)
from rlinf.scheduler.collective.tensor_buffer_pool import TensorBufferPool
from rlinf.scheduler.collective.tensor_compression import (
    LZ4CodecProvider,
    LZ4CompressionConfig,
    LZ4TensorCodec,
    TensorCompressionWireMetadata,
    ZstdCodecProvider,
    ZstdCompressionConfig,
)
from rlinf.scheduler.worker.worker import Worker


def test_buffer_pool_uses_best_fit_buffers_independent_of_tensor_order():
    """The Worker-wide pool chooses the smallest tensor buffer that fits."""
    pool = TensorBufferPool(TensorBufferPoolConfig(max_bytes=1024))
    large_lease = pool.try_acquire(512)
    small_lease = pool.try_acquire(128)
    assert large_lease is not None
    assert small_lease is not None
    large_buffer = large_lease.tensor
    small_buffer = small_lease.tensor
    large_lease.release()
    small_lease.release()

    small_reuse = pool.try_acquire(100)
    large_reuse = pool.try_acquire(300)
    assert small_reuse is not None
    assert large_reuse is not None
    assert small_reuse.tensor.data_ptr() == small_buffer.data_ptr()
    assert large_reuse.tensor.data_ptr() == large_buffer.data_ptr()
    small_reuse.release()
    large_reuse.release()


def test_buffer_pool_reuses_same_size_bucket_and_tracks_cached_bytes():
    """Equal-sized buffers share a bucket and update cached accounting eagerly."""
    pool = TensorBufferPool(TensorBufferPoolConfig(max_bytes=1024))
    leases = [pool.try_acquire(128), pool.try_acquire(128), pool.try_acquire(256)]
    assert all(lease is not None for lease in leases)
    pointers = {lease.tensor.data_ptr() for lease in leases}

    for lease in leases:
        lease.release()
    assert pool.cached_bytes == 512

    first = pool.try_acquire(100)
    second = pool.try_acquire(200)
    assert first is not None
    assert second is not None
    assert first.tensor.data_ptr() in pointers
    assert second.tensor.data_ptr() in pointers
    assert pool.cached_bytes == 128
    first.release(cache=False)
    second.release(cache=False)


def test_buffer_pool_never_exceeds_its_worker_budget():
    """Active buffers make later acquisitions fall back without overallocating."""
    pool = TensorBufferPool(TensorBufferPoolConfig(max_bytes=512))
    buffer = pool.try_acquire(400)
    assert buffer is not None
    assert pool.try_acquire(200) is None
    assert pool.allocated_bytes == 400

    buffer.release(cache=False)
    assert pool.allocated_bytes == 0


def test_buffer_pool_evicts_idle_buffers_to_fit_a_new_shape():
    """Historical shapes cannot make the bounded cache grow indefinitely."""
    pool = TensorBufferPool(TensorBufferPoolConfig(max_bytes=512))
    old_buffer = pool.try_acquire(128)
    assert old_buffer is not None
    old_buffer.release()

    replacement = pool.try_acquire(512)
    assert replacement is not None
    assert pool.allocated_bytes == 512
    replacement.release()


def test_buffer_pool_evicts_an_entire_size_bucket_in_one_step():
    """A new large allocation can replace many equal-sized idle buffers."""
    pool = TensorBufferPool(TensorBufferPoolConfig(max_bytes=512))
    leases = [pool.try_acquire(128) for _ in range(4)]
    assert all(lease is not None for lease in leases)
    for lease in leases:
        lease.release()

    replacement = pool.try_acquire(512)
    assert replacement is not None
    assert pool.allocated_bytes == 512
    assert pool.cached_bytes == 0
    replacement.release()


def test_buffer_pool_preserves_a_large_buffer_when_a_small_one_can_fit():
    """A speculative small compression cannot consume a much larger buffer."""
    pool = TensorBufferPool(TensorBufferPoolConfig(max_bytes=256))
    large_lease = pool.try_acquire(128)
    assert large_lease is not None
    large_buffer = large_lease.tensor
    large_lease.release()

    small_lease = pool.try_acquire(1)
    assert small_lease is not None
    assert small_lease.tensor.data_ptr() != large_buffer.data_ptr()
    small_lease.release(cache=False)
    reused_large = pool.try_acquire(100)
    assert reused_large is not None
    assert reused_large.tensor.data_ptr() == large_buffer.data_ptr()
    reused_large.release()


def test_zstd_provider_never_waits_for_a_busy_compressor():
    """Saturated Zstd context pools do not block a sender."""
    provider = ZstdCodecProvider(ZstdCompressionConfig(max_inflight=1))

    codec = provider.try_acquire_compressor()
    assert codec is not None
    assert provider.try_acquire_compressor() is None
    provider.release(codec)

    reused_codec = provider.try_acquire_compressor()
    assert reused_codec is not None
    provider.release(reused_codec)


def test_lz4_provider_supports_concurrent_round_trips():
    """The shared stateless LZ4 instance is safe across Worker threads."""
    provider = LZ4CodecProvider(LZ4CompressionConfig())

    def round_trip(value: int) -> bool:
        source = torch.full((128 * 1024,), value, dtype=torch.uint8)
        compressor = provider.try_acquire_compressor()
        assert compressor is not None
        try:
            capacity = compressor.compress_bound(source.numel())
            assert capacity is not None
            compressed = torch.empty(capacity, dtype=torch.uint8)
            compressed_numel = compressor.compress_into(source, compressed)
        finally:
            provider.release(compressor)

        restored = torch.empty_like(source)
        decompressor = provider.acquire_decompressor()
        try:
            decompressor.decompress_into(compressed, compressed_numel, restored)
        finally:
            provider.release(decompressor)
        return torch.equal(restored, source)

    with ThreadPoolExecutor(max_workers=8) as executor:
        assert all(executor.map(round_trip, range(16)))


@pytest.mark.parametrize("codec", ["lz4", "zstd"])
def test_codec_provider_compresses_and_restores_a_tensor(codec):
    """A provider's codec writes and restores tensor bytes."""
    compression_config = (
        LZ4CompressionConfig() if codec == "lz4" else ZstdCompressionConfig()
    )
    codec_provider = compression_config.create_codec_provider()
    assert codec_provider.codec_name == codec
    buffer_pool = TensorBufferPool(TensorBufferPoolConfig())

    source = torch.zeros(128 * 1024, dtype=torch.uint8)
    compressor = codec_provider.try_acquire_compressor()
    assert compressor is not None
    try:
        buffer = buffer_pool.try_acquire(compressor.compress_bound(source.numel()))
        assert buffer is not None
        compressed_numel = compressor.compress_into(source, buffer.tensor)
        assert compressed_numel < source.numel()
    finally:
        codec_provider.release(compressor)
    try:
        restored = torch.empty_like(source)
        decompressor = codec_provider.acquire_decompressor()
        try:
            decompressor.decompress_into(
                buffer.tensor[:compressed_numel], compressed_numel, restored
            )
        finally:
            codec_provider.release(decompressor)
        assert torch.equal(restored, source)
    finally:
        buffer.release()


def test_collective_group_prepares_compressed_cpu_tensors():
    """Prepared tensor data keeps raw entries and replaces compressed entries."""
    options = LZ4CompressionConfig()
    codec_provider = options.create_codec_provider()
    buffer_pool = TensorBufferPool(TensorBufferPoolConfig())
    group = object.__new__(CollectiveGroup)
    group._worker = SimpleNamespace(
        _tensor_compression_config=options,
        _tensor_buffer_pool=buffer_pool,
        _get_tensor_codec_provider=lambda: codec_provider,
    )
    fp32_tensor = torch.zeros(4096, dtype=torch.float32)
    uint8_tensor = torch.zeros(16 * 1024, dtype=torch.uint8)
    tensor_data = TensorData(
        cpu_tensor_mask=[True, True],
        cpu_tensors=[fp32_tensor, uint8_tensor],
        accel_tensors=[],
    )

    wire_data, buffers = group._compress_tensor_data(tensor_data)

    assert wire_data.compression is not None
    assert wire_data.compression.compressed_numel[0] is None
    assert wire_data.compression.compressed_numel[1] is not None
    assert wire_data.cpu_tensors[0] is fp32_tensor
    assert wire_data.cpu_tensors[1] is not uint8_tensor
    assert tensor_data.cpu_tensors[0] is fp32_tensor
    assert tensor_data.cpu_tensors[1] is uint8_tensor
    for buffer in buffers:
        buffer.release()


def test_collective_group_restores_a_compressed_cpu_tensor():
    """Tensor-list metadata restores a compressed CPU payload in place."""
    options = LZ4CompressionConfig(min_bytes=1)
    codec_provider = options.create_codec_provider()
    source = torch.zeros(128 * 1024, dtype=torch.uint8)
    compressor = codec_provider.try_acquire_compressor()
    assert compressor is not None
    capacity = compressor.compress_bound(source.numel())
    assert capacity is not None
    wire_tensor = torch.empty(capacity, dtype=torch.uint8)
    try:
        wire_numel = compressor.compress_into(source, wire_tensor)
    finally:
        codec_provider.release(compressor)

    metadata = {
        "meta": [(source.shape, source.dtype)],
        "pb": "payload",
        "cpu_tensor_mask": [True],
        "compression": TensorCompressionWireMetadata(
            codec=options.codec,
            compressed_numel=(wire_numel,),
        ),
    }
    incoming = iter(
        [
            torch.tensor([1], dtype=torch.long),
            torch.zeros(1, dtype=torch.uint8),
            wire_tensor[:wire_numel],
        ]
    )

    group = object.__new__(CollectiveGroup)
    group._worker = SimpleNamespace(
        _tensor_compression_config=options,
        _tensor_buffer_pool=TensorBufferPool(TensorBufferPoolConfig()),
        _get_tensor_codec_provider=lambda: codec_provider,
    )
    group._peer_rank = 0
    group._group_info = SimpleNamespace(group_name="test")
    group._logger = SimpleNamespace(debug=lambda *_args: None)
    group._tensor_to_object = lambda *_args: metadata
    group._recv = lambda tensor, *_args, **_kwargs: tensor.copy_(next(incoming))

    tensors, piggyback_payload = group._recv_tensor_list(comm_id=0)

    assert piggyback_payload == "payload"
    assert torch.equal(tensors[0], source)


def test_float32_compression_can_be_explicitly_enabled():
    """An empty exclusion list restores dtype-agnostic compression."""
    options = LZ4CompressionConfig(min_bytes=1, excluded_dtypes=[])

    assert options.should_compress(torch.zeros(1, dtype=torch.float32))


def test_worker_lazily_shares_one_codec_provider():
    """Concurrent CollectiveGroups share one Worker-wide codec provider."""
    worker = object.__new__(Worker)
    worker._tensor_compression_config = LZ4CompressionConfig()
    worker._tensor_buffer_pool = TensorBufferPool(TensorBufferPoolConfig())
    worker._tensor_codec_provider = None
    worker._tensor_codec_provider_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=8) as executor:
        codec_providers = list(
            executor.map(lambda _: worker._get_tensor_codec_provider(), range(16))
        )

    assert all(provider is codec_providers[0] for provider in codec_providers)
    assert worker._tensor_codec_provider is codec_providers[0]


@pytest.mark.parametrize(
    "tensor",
    [
        pytest.param(torch.zeros(1, dtype=torch.uint8), id="below-min-bytes"),
        pytest.param(
            torch.zeros(16 * 1024 // 4, dtype=torch.float32),
            id="excluded-dtype",
        ),
    ],
)
def test_ineligible_tensors_do_not_initialize_the_codec_provider(tensor):
    """Raw CPU transfers do not require the configured codec library."""
    options = LZ4CompressionConfig()
    group = object.__new__(CollectiveGroup)
    group._worker = SimpleNamespace(
        _tensor_compression_config=options,
        _get_tensor_codec_provider=lambda: pytest.fail(
            "ineligible tensors initialized the codec provider"
        ),
    )
    tensor_data = TensorData(
        cpu_tensor_mask=[True],
        cpu_tensors=[tensor],
        accel_tensors=[],
    )

    wire_data, buffers = group._compress_tensor_data(tensor_data)

    assert wire_data is tensor_data
    assert buffers == []


def test_lz4_compress_bound_returns_none_for_an_unsupported_input_size():
    """An input-size limit is a normal no-compression outcome."""
    codec = LZ4TensorCodec()

    assert codec.compress_bound(LZ4TensorCodec._MAX_INPUT_SIZE + 1) is None


def test_collective_group_options_exclude_tensor_compression():
    """Tensor compression is not a per-call collective option."""
    with pytest.raises(TypeError, match="tensor_compression"):
        CollectiveGroupOptions(tensor_compression=LZ4CompressionConfig())


def test_tensor_container_helpers_keep_async_send_without_unused_options():
    """Private send helpers retain their baseline async contract only."""
    send_helpers = [
        CollectiveGroup._send_tensor_list,
        CollectiveGroup._send_tensor_dict,
        CollectiveGroup._send_tensor_dataclass,
    ]
    recv_helpers = [
        CollectiveGroup._recv_tensor_list,
        CollectiveGroup._recv_tensor_dict,
        CollectiveGroup._recv_tensor_dataclass,
    ]

    for helper in send_helpers:
        parameters = inspect.signature(helper).parameters
        assert "async_op" in parameters
        assert "options" not in parameters
    for helper in recv_helpers:
        assert "options" not in inspect.signature(helper).parameters


def _cluster_config(collective):
    """Build a ClusterConfig from the public ``cluster`` YAML mapping."""
    return ClusterConfig.from_dict_cfg(
        OmegaConf.create(
            {"num_nodes": 1, "component_placement": [], "collective": collective}
        )
    )


@pytest.mark.parametrize(
    ("collective", "message"),
    [
        ({"tensor_compression": {"enabled": "yes"}}, "codec must be specified"),
        ({"tensor_compression": {"codec": "invalid"}}, "Unknown tensor compression"),
        ({"tensor_compression": {"codec": "lz4", "acceleration": 0}}, "acceleration"),
        ({"tensor_compression": {"codec": "zstd", "level": 0}}, "level"),
        ({"tensor_compression": {"codec": "lz4", "min_bytes": 0}}, "min_bytes"),
        ({"tensor_compression": {"codec": "zstd", "max_inflight": 0}}, "max_inflight"),
        (
            {"tensor_compression": {"codec": "lz4", "excluded_dtypes": "float32"}},
            "must be a list",
        ),
        (
            {"tensor_compression": {"codec": "lz4", "excluded_dtypes": ["nope"]}},
            "Unknown torch dtype",
        ),
        (
            {
                "tensor_compression": {
                    "codec": "lz4",
                    "excluded_dtypes": ["float32", "float32"],
                }
            },
            "duplicates",
        ),
        ({"tensor_compression": {"codec": "lz4", "min_bytes": 1.5}}, "integer"),
        ({"tensor_compression": {"codec": "lz4", "min_bytes": True}}, "integer"),
        ({"tensor_compression": {"codec": "lz4", "acceleration": 1.5}}, "integer"),
        ({"tensor_compression": {"codec": "zstd", "level": True}}, "integer"),
        ({"tensor_compression": {"codec": "zstd", "max_inflight": 4.0}}, "integer"),
        ({"tensor_buffer_pool": {"max_bytes": 0}}, "max_bytes must be >= 1"),
        ({"tensor_buffer_pool": {"max_bytes": 1.5}}, "integer"),
    ],
)
def test_cluster_config_validates_collective_settings(collective, message):
    """Invalid collective settings fail while the driver parses cluster yaml."""
    with pytest.raises(ValueError, match=message):
        _cluster_config(collective)


@pytest.mark.parametrize(
    ("collective", "message"),
    [
        ({"tensor_compresion": {"codec": "lz4"}}, "in cluster collective yaml config"),
        (
            {"tensor_compression": {"codec": "lz4", "min_byte": 1024}},
            "in cluster collective tensor_compression yaml config",
        ),
        (
            {"tensor_compression": {"codec": "lz4", "max_inflight": 4}},
            "in cluster collective tensor_compression yaml config",
        ),
        (
            {"tensor_compression": {"codec": "zstd", "acceleration": 1}},
            "in cluster collective tensor_compression yaml config",
        ),
        (
            {"tensor_buffer_pool": {"max_byte": 1024}},
            "in cluster collective tensor_buffer_pool yaml config",
        ),
    ],
)
def test_cluster_config_rejects_unknown_collective_keys(collective, message):
    """Unknown keys are reported the same way as any other cluster yaml typo."""
    with pytest.raises(AssertionError, match=message):
        _cluster_config(collective)


@pytest.mark.parametrize(
    ("collective", "message"),
    [
        ({"tensor_compression": True}, "tensor_compression must be a dictionary"),
        ({"tensor_buffer_pool": True}, "tensor_buffer_pool must be a dictionary"),
    ],
)
def test_cluster_config_requires_collective_mappings(collective, message):
    """Each collective sub-config must be a yaml mapping."""
    with pytest.raises(AssertionError, match=message):
        _cluster_config(collective)


def test_codec_configs_register_under_their_codec_type():
    """Every codec is discoverable by the name that selects it in yaml."""
    assert TensorCompressionManager.codec_config_register == {
        LZ4CompressionConfig.CODEC_TYPE: LZ4CompressionConfig,
        ZstdCompressionConfig.CODEC_TYPE: ZstdCompressionConfig,
    }
    assert LZ4CompressionConfig().create_codec_provider().codec_name == "lz4"
    assert ZstdCompressionConfig().create_codec_provider().codec_name == "zstd"


def test_registering_a_codec_config_requires_a_codec_type():
    """A config with no CODEC_TYPE could never be selected, so it cannot register."""
    with pytest.raises(AssertionError, match="CODEC_TYPE"):

        @TensorCompressionManager.register_codec_config
        @dataclass
        class UnnamedCompressionConfig(TensorCompressionConfig):
            """A codec config that forgot to name itself."""


def test_cluster_config_builds_the_selected_codec_config():
    """``codec`` selects the config class that owns the codec's parameters."""
    cluster_config = _cluster_config(
        {
            "tensor_buffer_pool": {"max_bytes": 4096},
            "tensor_compression": {
                "enabled": False,
                "codec": "zstd",
                "min_bytes": 1024,
                "excluded_dtypes": ["float32", "float64"],
                "level": 3,
                "max_inflight": 2,
            },
        }
    )

    assert cluster_config.collective == CollectiveConfig(
        tensor_compression=ZstdCompressionConfig(
            enabled=False,
            min_bytes=1024,
            excluded_dtypes=["float32", "float64"],
            level=3,
            max_inflight=2,
        ),
        tensor_buffer_pool=TensorBufferPoolConfig(max_bytes=4096),
    )
    assert cluster_config.collective.tensor_compression.codec == "zstd"


def test_cluster_config_supplies_the_default_tensor_buffer_pool():
    """Compression without an explicit pool block still gets the default budget."""
    cluster_config = _cluster_config({"tensor_compression": {"codec": "lz4"}})

    assert cluster_config.collective.tensor_buffer_pool == TensorBufferPoolConfig()
    assert cluster_config.collective.tensor_compression == LZ4CompressionConfig()


def test_cluster_config_without_collective_keeps_the_raw_wire_path():
    """Omitting ``cluster.collective`` leaves compression unconfigured."""
    cluster_config = ClusterConfig.from_dict_cfg(
        OmegaConf.create({"num_nodes": 1, "component_placement": []})
    )

    assert cluster_config.collective is None


def test_worker_loads_and_probes_collective_resources(monkeypatch):
    """Workers take their shared resources from the job-wide ClusterConfig."""
    worker = object.__new__(Worker)
    cluster_config = _cluster_config(
        {
            "tensor_buffer_pool": {"max_bytes": 4096},
            "tensor_compression": {"codec": "zstd", "min_bytes": 1024, "level": 3},
        }
    )
    monkeypatch.setattr(
        Cluster,
        "__new__",
        lambda _cls: SimpleNamespace(collective_config=cluster_config.collective),
    )
    probes = []
    monkeypatch.setattr(
        "rlinf.scheduler.collective.tensor_compression.probe_tensor_codec_library",
        probes.append,
    )

    worker._setup_collective_resources()

    assert worker._tensor_compression_config == ZstdCompressionConfig(
        min_bytes=1024, level=3
    )
    assert worker._tensor_buffer_pool.config == TensorBufferPoolConfig(max_bytes=4096)
    assert probes == ["zstd"]
    assert worker._tensor_codec_provider is None


def test_worker_without_collective_config_uses_pool_defaults(monkeypatch):
    """A job that never configured collectives still gets a usable buffer pool."""
    worker = object.__new__(Worker)
    monkeypatch.setattr(
        Cluster, "__new__", lambda _cls: SimpleNamespace(collective_config=None)
    )
    probes = []
    monkeypatch.setattr(
        "rlinf.scheduler.collective.tensor_compression.probe_tensor_codec_library",
        probes.append,
    )

    worker._setup_collective_resources()

    assert worker._tensor_compression_config is None
    assert worker._tensor_buffer_pool.config == TensorBufferPoolConfig()
    assert probes == []


def test_worker_skips_codec_resources_when_compression_is_disabled(monkeypatch):
    """Disabled compression neither probes nor creates codec resources."""
    worker = object.__new__(Worker)
    cluster_config = _cluster_config(
        {"tensor_compression": {"codec": "zstd", "enabled": False}}
    )
    monkeypatch.setattr(
        Cluster,
        "__new__",
        lambda _cls: SimpleNamespace(collective_config=cluster_config.collective),
    )
    probes = []
    monkeypatch.setattr(
        "rlinf.scheduler.collective.tensor_compression.probe_tensor_codec_library",
        probes.append,
    )

    worker._setup_collective_resources()

    assert probes == []
    with pytest.raises(ValueError, match="not enabled"):
        worker._get_tensor_codec_provider()
    assert worker._tensor_codec_provider is None


def test_net_emulation_uses_the_compressed_wire_size():
    """Compression finishes before a point-to-point bandwidth reservation."""
    group = object.__new__(CollectiveGroup)
    tensor = torch.zeros(1024, dtype=torch.uint8)
    wire_tensor = torch.zeros(64, dtype=torch.uint8)
    tensor_data = TensorData(
        cpu_tensor_mask=[True],
        cpu_tensors=[tensor],
        accel_tensors=[],
    )
    metadata = TensorCompressionWireMetadata(codec="lz4", compressed_numel=(64,))
    wire_data = TensorData(
        cpu_tensor_mask=[True],
        cpu_tensors=[wire_tensor],
        accel_tensors=[],
        compression=metadata,
    )
    events = []

    group._init_process_group = lambda **_kwargs: None
    group._compress_tensor_data = lambda _tensor_data: (
        events.append("compress") or wire_data,
        [],
    )
    group._wait_for_net_emulation = lambda *_payloads, size_bytes=None: events.append(
        ("reserve", size_bytes)
    )
    group._send = lambda *_args, **_kwargs: None
    group._send_tensor_list = lambda *_args, **_kwargs: events.append("send")
    group._cur_worker_address = SimpleNamespace(get_name=lambda: "Src:0")
    group._group_info = SimpleNamespace(group_name="test")
    group._logger = SimpleNamespace(debug=lambda *_args: None)
    group._net_emu_manager = object()

    group._atomic_send(
        work=None,
        object=tensor,
        comm_id=0,
        object_type=CollectiveGroup.TENSOR,
        tensor_data=tensor_data,
    )

    raw_size = group._estimate_payload_size((tensor, None))
    metadata_size = group._estimate_payload_size((metadata,))
    assert events == [
        "compress",
        ("reserve", raw_size - tensor.numel() + wire_tensor.numel() + metadata_size),
        "send",
    ]


def test_compressed_send_skips_size_estimation_without_net_emulation():
    """Disabled network emulation adds no payload-estimation overhead."""
    group = object.__new__(CollectiveGroup)
    tensor = torch.zeros(1024, dtype=torch.uint8)
    metadata = TensorCompressionWireMetadata(codec="lz4", compressed_numel=(64,))
    tensor_data = TensorData(
        cpu_tensor_mask=[True],
        cpu_tensors=[tensor],
        accel_tensors=[],
    )
    wire_data = TensorData(
        cpu_tensor_mask=[True],
        cpu_tensors=[torch.zeros(64, dtype=torch.uint8)],
        accel_tensors=[],
        compression=metadata,
    )

    group._net_emu_manager = None
    group._init_process_group = lambda **_kwargs: None
    group._compress_tensor_data = lambda _tensor_data: (wire_data, [])
    group._estimate_payload_size = lambda *_args: pytest.fail(
        "payload size was estimated with network emulation disabled"
    )
    group._send = lambda *_args, **_kwargs: None
    group._send_tensor_list = lambda *_args, **_kwargs: None
    group._cur_worker_address = SimpleNamespace(get_name=lambda: "Src:0")
    group._group_info = SimpleNamespace(group_name="test")
    group._logger = SimpleNamespace(debug=lambda *_args: None)

    group._atomic_send(
        work=None,
        object=tensor,
        comm_id=0,
        object_type=CollectiveGroup.TENSOR,
        tensor_data=tensor_data,
    )
