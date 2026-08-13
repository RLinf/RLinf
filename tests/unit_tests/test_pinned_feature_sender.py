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
from types import SimpleNamespace

import pytest
import torch

from rlinf.data.schema.embodied_types import PolicyOutput
from rlinf.utils.backbone_cache import (
    ROLLOUT_BACKBONE_FEATURE_KEY,
    ROLLOUT_BACKBONE_MASK_KEY,
)
from rlinf.workers.rollout.hf.async_huggingface_worker import (
    AsyncMultiStepRolloutWorker,
)
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


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

    def ipc_collect(self) -> None:
        self.collects += 1


class _FakeSender:
    _begin_pinned_feature_lease = MultiStepRolloutWorker._begin_pinned_feature_lease
    _abort_pinned_feature_stream = MultiStepRolloutWorker._abort_pinned_feature_stream

    def __init__(self, ack) -> None:
        self._rank = 2
        self._world_size = 4
        self.version = 7
        self.cfg = SimpleNamespace(
            env=SimpleNamespace(
                train=SimpleNamespace(
                    rollout_epoch=1,
                    max_steps_per_rollout_epoch=1,
                    total_num_envs=8,
                )
            )
        )
        self.actor_group_name = "actor"
        self.torch_platform = _FakePlatform()
        self.sent = []
        self.logs = []
        self.ack = ack
        self._pinned_feature_verify_trajectory = False
        self._pinned_feature_ipc_enabled = True
        self._pinned_feature_ipc_batch_blocks = 1
        self._pinned_feature_ipc_timeout_seconds = 0.05
        self._pinned_feature_tensors = [
            torch.ones(2, 3),
            torch.ones(2, 3, dtype=torch.bool),
        ]
        self._pinned_feature_block_sizes = [2]
        self._pinned_feature_samples = 2
        self._pinned_feature_active_lease = "lease-1"
        self._pinned_feature_lease_seq = 1
        self._pinned_feature_consumer_rank = 2
        self._pinned_stream_expected_blocks = 1
        self._pinned_stream_expected_samples = 2
        self._pinned_stream_expected_batches = 1
        self._pinned_stream_batches = 0
        self._pinned_stream_blocks = 0
        self._pinned_stream_bytes = 0
        self._pinned_stream_wait_seconds = 0.0

    def send(self, tensors, **kwargs) -> None:
        self.sent.append((tensors, kwargs))

    def recv(self, **kwargs):
        assert kwargs["async_op"] is True
        if self.ack is None:
            return _NeverWork()
        return _ImmediateWork(self.ack)

    def log_info(self, message) -> None:
        self.logs.append(message)


class _FakePolicyOutputBuilder:
    _build_policy_output = MultiStepRolloutWorker._build_policy_output
    _build_policy_output_with_transport = (
        MultiStepRolloutWorker._build_policy_output_with_transport
    )

    def __init__(self, *, pinned_feature_ipc_enabled: bool) -> None:
        self.collect_prev_infos = True
        self.enable_dagger = False
        self.version = 3
        self._pinned_feature_ipc_enabled = pinned_feature_ipc_enabled
        self.retained_forward_inputs = None

    @staticmethod
    def get_bootstrap_values(_final_obs):
        return None

    async def _retain_pinned_feature_block(self, forward_inputs) -> None:
        self.retained_forward_inputs = forward_inputs


def _flush(sender):
    return asyncio.run(MultiStepRolloutWorker._flush_pinned_feature_batch(sender))


def _ok_ack():
    return {
        "schema": 3,
        "status": "ok",
        "lease_id": "lease-1",
        "batch_index": 0,
        "completed_blocks": 1,
    }


def _policy_result():
    return {
        "prev_logprobs": torch.zeros(2, 1),
        "prev_values": torch.ones(2, 1),
        "forward_inputs": {
            ROLLOUT_BACKBONE_FEATURE_KEY: torch.ones(2, 3),
            ROLLOUT_BACKBONE_MASK_KEY: torch.ones(2, 3, dtype=torch.bool),
        },
    }


def test_policy_output_builder_preserves_synchronous_api():
    builder = _FakePolicyOutputBuilder(pinned_feature_ipc_enabled=False)

    output = builder._build_policy_output(torch.zeros(2, 1), _policy_result())

    assert isinstance(output, PolicyOutput)
    assert output.versions.tolist() == [[3.0], [3.0]]


def test_policy_output_transport_wrapper_awaits_pinned_retention():
    builder = _FakePolicyOutputBuilder(pinned_feature_ipc_enabled=True)

    output = asyncio.run(
        builder._build_policy_output_with_transport(
            torch.zeros(2, 1),
            _policy_result(),
        )
    )

    assert isinstance(output, PolicyOutput)
    assert builder.retained_forward_inputs is output.forward_inputs


def test_async_rollout_rejects_pinned_feature_transport(monkeypatch):
    def _fake_base_init(self, _cfg):
        self._pinned_feature_ipc_enabled = True

    monkeypatch.setattr(MultiStepRolloutWorker, "__init__", _fake_base_init)

    with pytest.raises(ValueError, match="only supports the synchronous"):
        AsyncMultiStepRolloutWorker(None)


def test_sender_releases_batch_only_after_valid_ack():
    sender = _FakeSender(_ok_ack())

    _flush(sender)

    assert sender._pinned_stream_batches == 1
    assert sender._pinned_stream_blocks == 1
    assert sender._pinned_feature_tensors == []
    assert sender._pinned_feature_block_sizes == []
    assert sender._pinned_feature_active_lease == "lease-1"
    assert sender.torch_platform.collects == 1


def test_sender_aborts_lease_when_actor_nacks_batch():
    ack = _ok_ack()
    ack.update(status="error", error="RuntimeError: wrong model version")
    sender = _FakeSender(ack)

    with pytest.raises(RuntimeError, match="wrong model version"):
        _flush(sender)

    assert sender._pinned_feature_tensors == []
    assert sender._pinned_feature_block_sizes == []
    assert sender._pinned_feature_active_lease is None
    assert sender._pinned_feature_consumer_rank is None
    assert sender.torch_platform.collects == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", 2),
        ("lease_id", "other-lease"),
        ("batch_index", 1),
        ("completed_blocks", 0),
    ],
)
def test_sender_aborts_lease_on_malformed_ack(field, value):
    ack = _ok_ack()
    ack[field] = value
    sender = _FakeSender(ack)

    with pytest.raises(RuntimeError, match="ACK mismatch"):
        _flush(sender)

    assert sender._pinned_feature_tensors == []
    assert sender._pinned_feature_block_sizes == []
    assert sender._pinned_feature_active_lease is None
    assert sender._pinned_feature_consumer_rank is None
    assert sender.torch_platform.collects == 1


def test_sender_times_out_and_aborts_lease_when_actor_does_not_ack():
    sender = _FakeSender(None)

    with pytest.raises(TimeoutError, match="lease lease-1"):
        _flush(sender)

    assert sender._pinned_feature_tensors == []
    assert sender._pinned_feature_block_sizes == []
    assert sender._pinned_feature_active_lease is None
    assert sender._pinned_feature_consumer_rank is None
    assert sender.torch_platform.collects == 1


def test_sender_can_start_a_new_lease_after_nack_cleanup():
    rejected = _ok_ack()
    rejected.update(status="error", error="injected failure")
    sender = _FakeSender(rejected)

    with pytest.raises(RuntimeError, match="injected failure"):
        _flush(sender)

    sender._begin_pinned_feature_lease()
    assert sender._pinned_feature_active_lease == "pinned-r2-l2"
    assert sender._pinned_stream_expected_blocks == 1
    assert sender._pinned_stream_expected_samples == 2

    sender._pinned_feature_tensors = [
        torch.ones(2, 3),
        torch.ones(2, 3, dtype=torch.bool),
    ]
    sender._pinned_feature_block_sizes = [2]
    sender._pinned_feature_samples = 2
    sender.ack = {
        **_ok_ack(),
        "lease_id": "pinned-r2-l2",
    }
    _flush(sender)

    assert sender._pinned_stream_batches == 1
    assert sender._pinned_stream_blocks == 1
    assert sender._pinned_feature_active_lease == "pinned-r2-l2"
