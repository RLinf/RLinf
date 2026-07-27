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

import logging
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d

from rlinf.scheduler.collective.collective_group import CollectiveGroup
from rlinf.scheduler.collective.multi_channel_pg import MultiChannelProcessGroup


class _BroadcastFailureGroup:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def broadcast(self, tensors: list[torch.Tensor], options: object) -> None:
        del tensors, options
        raise self._error

    def __repr__(self) -> str:
        return "_BroadcastFailureGroup()"


class _WaitFailureWork:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def wait(self) -> None:
        raise self._error


class _WaitFailureGroup:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def broadcast(
        self, tensors: list[torch.Tensor], options: object
    ) -> _WaitFailureWork:
        del tensors, options
        return _WaitFailureWork(self._error)

    def __repr__(self) -> str:
        return "_WaitFailureGroup()"


@pytest.fixture
def multi_channel_group(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> MultiChannelProcessGroup:
    monkeypatch.setattr(distributed_c10d, "BroadcastOptions", SimpleNamespace)
    monkeypatch.setattr(
        distributed_c10d, "_check_single_tensor", lambda tensor, name: None
    )
    monkeypatch.setattr(distributed_c10d, "_rank_not_in_group", lambda group: False)
    monkeypatch.setattr(distributed_c10d, "get_group_rank", lambda group, rank: rank)
    monkeypatch.setattr(dist, "_get_process_group_name", lambda group: "test-group")

    logger = logging.getLogger(__name__)
    caplog.set_level(logging.ERROR, logger=logger.name)
    process_group = object.__new__(MultiChannelProcessGroup)
    process_group._cur_rank = 1
    process_group._peer_rank = 0
    process_group._num_channels = 1
    process_group._is_initialized = True
    process_group._no_accel_ccl = False
    process_group._logger = logger
    return process_group


def _assert_failure_log(caplog: pytest.LogCaptureFixture, expected_error: str) -> None:
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "ProcessGroup test-group rank 1" in message
    assert expected_error in message


def test_recv_propagates_process_group_failure(
    multi_channel_group: MultiChannelProcessGroup,
    caplog: pytest.LogCaptureFixture,
) -> None:
    error = RuntimeError("connection closed by peer")
    multi_channel_group._recv_gloo_process_groups = [_BroadcastFailureGroup(error)]

    with pytest.raises(RuntimeError) as exc_info:
        multi_channel_group.recv(
            torch.empty(1),
            device=CollectiveGroup.CPU,
            channel_id=0,
        )

    assert exc_info.value is error
    _assert_failure_log(caplog, str(error))


def test_recv_propagates_synchronous_wait_failure(
    multi_channel_group: MultiChannelProcessGroup,
    caplog: pytest.LogCaptureFixture,
) -> None:
    error = RuntimeError("timed out waiting for recv")
    multi_channel_group._recv_gloo_process_groups = [_WaitFailureGroup(error)]

    with pytest.raises(RuntimeError) as exc_info:
        multi_channel_group.recv(
            torch.empty(1),
            device=CollectiveGroup.CPU,
            channel_id=0,
        )

    assert exc_info.value is error
    _assert_failure_log(caplog, str(error))
