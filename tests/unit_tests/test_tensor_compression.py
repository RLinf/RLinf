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

import json
import os
import threading
from types import SimpleNamespace

import pytest
import torch

from rlinf.scheduler.cluster.cluster import Cluster, ClusterEnvVar
from rlinf.scheduler.collective.collective_group import (
    CollectiveGroup,
    CollectiveGroupOptions,
)
from rlinf.scheduler.collective.tensor_compression import (
    TensorCodecPool,
    TensorCompressionOptions,
    TensorCompressionWireMetadata,
)
from rlinf.scheduler.worker.worker import Worker
from rlinf.utils.tensor_codec import LZ4TensorCodec


def test_codec_pool_prefers_a_reused_compressor_and_its_buffer():
    """A released slot outranks an unused slot and retains its workspace."""
    pool = TensorCodecPool(TensorCompressionOptions(max_inflight=2))

    first_lease = pool.try_acquire_compressor()
    assert first_lease is not None
    first_slot = first_lease.slot
    first_buffer = first_slot.acquire_buffer(128)
    first_lease.release()

    reused_lease = pool.try_acquire_compressor()
    assert reused_lease is not None
    assert reused_lease.slot is first_slot
    assert reused_lease.slot.acquire_buffer(64).data_ptr() == first_buffer.data_ptr()
    reused_lease.release()


def test_codec_slot_uses_best_fit_workspaces_independent_of_tensor_order():
    """A slot chooses the smallest retained workspace that can hold each output."""
    pool = TensorCodecPool(TensorCompressionOptions())
    lease = pool.try_acquire_compressor()
    assert lease is not None
    large_buffer = lease.slot.acquire_buffer(512)
    small_buffer = lease.slot.acquire_buffer(128)
    lease.release()

    reused_lease = pool.try_acquire_compressor()
    assert reused_lease is not None
    assert reused_lease.slot.acquire_buffer(100).data_ptr() == small_buffer.data_ptr()
    assert reused_lease.slot.acquire_buffer(300).data_ptr() == large_buffer.data_ptr()
    reused_lease.release()


def test_codec_pool_never_waits_for_a_busy_compressor():
    """Saturated pools report no slot instead of blocking a sender."""
    pool = TensorCodecPool(TensorCompressionOptions(max_inflight=1))

    lease = pool.try_acquire_compressor()
    assert lease is not None
    assert pool.try_acquire_compressor() is None

    lease.release()


@pytest.mark.parametrize("codec", ["lz4", "zstd"])
def test_codec_pool_compresses_and_restores_a_tensor(codec):
    """A slot's codec writes into its reusable buffer without altering bytes."""
    options = TensorCompressionOptions(codec=codec)
    pool = TensorCodecPool(options)
    lease = pool.try_acquire_compressor()
    assert lease is not None

    source = torch.zeros(128 * 1024, dtype=torch.uint8)
    try:
        destination = lease.slot.acquire_buffer(
            lease.slot.codec.compress_bound(source.numel())
        )
        compressed_numel = lease.slot.codec.compress_into(source, destination)
        assert compressed_numel < source.numel()

        restored = torch.empty_like(source)
        decompressor = pool.acquire_decompressor(
            TensorCompressionWireMetadata(
                codec=codec,
                level=options.level,
                compressed_numel=(compressed_numel,),
            )
        )
        try:
            assert decompressor.codec is not lease.slot.codec
            decompressor.codec.decompress_into(
                destination[:compressed_numel], compressed_numel, restored
            )
        finally:
            decompressor.release()
        assert torch.equal(restored, source)
    finally:
        lease.release()


def test_codec_pool_rejects_mismatched_decompression_metadata():
    """A collective has one codec configuration for both directions."""
    pool = TensorCodecPool(TensorCompressionOptions(codec="lz4", level=1))

    with pytest.raises(ValueError, match="does not match"):
        pool.acquire_decompressor(
            TensorCompressionWireMetadata(codec="zstd", level=1, compressed_numel=(1,))
        )


def test_tensor_over_codec_bound_skips_compression(monkeypatch):
    """A codec input-limit error leaves that tensor on the raw wire path."""
    group = object.__new__(CollectiveGroup)
    options = TensorCompressionOptions(min_bytes=1)
    pool = TensorCodecPool(options)
    group._tensor_codec_pool = pool
    group._tensor_codec_pool_lock = threading.Lock()
    group._tensor_compression = options
    tensor = torch.zeros(128, dtype=torch.uint8)

    monkeypatch.setattr(
        pool._fresh_compressors.queue[0].codec,
        "compress_bound",
        lambda _source_bytes: None,
    )
    wire_tensors, metadata, lease = group._compress_cpu_tensors(
        [tensor],
        [tensor],
    )

    assert wire_tensors[0] is tensor
    assert metadata is None
    assert lease is None


def test_lz4_compress_bound_returns_none_for_an_unsupported_input_size():
    """An input-size limit is a normal no-compression outcome."""
    codec = LZ4TensorCodec()

    assert codec.compress_bound(LZ4TensorCodec._MAX_INPUT_SIZE + 1) is None


def test_collective_group_options_exclude_tensor_compression():
    """Tensor compression is not a per-call collective option."""
    with pytest.raises(TypeError, match="tensor_compression"):
        CollectiveGroupOptions(tensor_compression=TensorCompressionOptions())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"enabled": "yes"}, "boolean"),
        ({"codec": "invalid"}, "Unsupported"),
        ({"level": 0}, "level"),
        ({"min_bytes": 0}, "Minimum"),
        ({"max_inflight": 0}, "inflight"),
    ],
)
def test_compression_options_validate_positive_limits(kwargs, message):
    """Invalid compression limits fail before a collective starts."""
    with pytest.raises(ValueError, match=message):
        TensorCompressionOptions(**kwargs)


def test_compression_options_reject_unknown_cluster_yaml_key():
    """Typos in the public YAML interface fail clearly."""
    with pytest.raises(ValueError, match="Unsupported"):
        TensorCompressionOptions.from_dict({"min_byte": 1024})


def test_disabled_compression_preserves_the_original_cpu_tensors():
    """``enabled=False`` takes the raw path before acquiring any workspace."""
    group = object.__new__(CollectiveGroup)
    group._tensor_compression = TensorCompressionOptions(enabled=False, min_bytes=1)
    tensors = [torch.zeros(128 * 1024, dtype=torch.uint8)]
    wire_tensors, metadata, lease = group._compress_cpu_tensors(
        tensors,
        tensors,
    )

    assert wire_tensors is tensors
    assert metadata is None
    assert lease is None


def test_cluster_serializes_validated_tensor_compression_config(monkeypatch):
    """The public YAML config becomes the validated Worker configuration."""
    env_var_name = Cluster.get_full_env_var_name(
        ClusterEnvVar.COLLECTIVE_TENSOR_COMPRESSION
    )
    monkeypatch.delenv(env_var_name, raising=False)

    Cluster._configure_tensor_compression_env(
        {
            "collective": {
                "tensor_compression": {
                    "enabled": False,
                    "codec": "zstd",
                    "level": 3,
                    "min_bytes": 1024,
                    "max_inflight": 2,
                }
            }
        }
    )

    assert json.loads(os.environ[env_var_name]) == {
        "enabled": False,
        "codec": "zstd",
        "level": 3,
        "min_bytes": 1024,
        "max_inflight": 2,
    }


def test_worker_loads_the_job_wide_tensor_compression_config(monkeypatch):
    """Workers load the shared configuration propagated by Cluster."""
    worker = object.__new__(Worker)
    env_var_name = Cluster.get_full_env_var_name(
        ClusterEnvVar.COLLECTIVE_TENSOR_COMPRESSION
    )
    monkeypatch.setenv(env_var_name, '{"codec": "zstd", "min_bytes": 1024}')

    assert worker._load_tensor_compression_options() == TensorCompressionOptions(
        codec="zstd", min_bytes=1024
    )


def test_cluster_yaml_overrides_the_internal_worker_environment(monkeypatch):
    """Workers receive the YAML setting rather than a manually supplied value."""
    env_var_name = Cluster.get_full_env_var_name(
        ClusterEnvVar.COLLECTIVE_TENSOR_COMPRESSION
    )
    cluster = object.__new__(Cluster)
    node = SimpleNamespace(env_vars={env_var_name: "manual-value"})
    cluster._nodes = [node]

    monkeypatch.setenv(env_var_name, '{"enabled": true}')
    cluster._set_scheduler_env_vars()
    assert node.env_vars[env_var_name] == '{"enabled": true}'

    monkeypatch.delenv(env_var_name)
    cluster._set_scheduler_env_vars()
    assert env_var_name not in node.env_vars


@pytest.mark.parametrize(
    "scheduler_value",
    ['{"codec": "lz4"}', None],
    ids=["configured", "disabled"],
)
def test_worker_env_cannot_override_tensor_compression_config(
    scheduler_value,
):
    """User environment merging cannot replace the job-wide codec settings."""
    env_var_name = Cluster.get_full_env_var_name(
        ClusterEnvVar.COLLECTIVE_TENSOR_COMPRESSION
    )
    scheduler_env_vars = (
        {env_var_name: scheduler_value} if scheduler_value is not None else {}
    )
    worker_env_vars = {
        env_var_name: '{"codec": "zstd"}',
        "USER_SETTING": "preserved",
    }

    merged_env_vars = Cluster._enforce_scheduler_owned_env_vars(
        worker_env_vars,
        scheduler_env_vars,
    )

    assert merged_env_vars.get(env_var_name) == scheduler_value
    assert merged_env_vars["USER_SETTING"] == "preserved"
