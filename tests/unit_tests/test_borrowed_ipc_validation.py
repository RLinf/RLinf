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

from types import MethodType, SimpleNamespace

import pytest
import torch

from rlinf.scheduler.collective.collective_group import CollectiveGroup
from rlinf.scheduler.worker.worker import Worker


def _bare_group():
    group = object.__new__(CollectiveGroup)
    group._logger = SimpleNamespace(debug=lambda *args, **kwargs: None)
    group._peer_rank = 0
    group._group_info = SimpleNamespace(group_name="test")
    return group


@pytest.mark.parametrize(
    "payload",
    [
        torch.ones(1),
        [torch.ones(1)],
        [],
    ],
)
def test_borrowed_ipc_rejects_non_cuda_payload_before_send(payload):
    group = _bare_group()

    with pytest.raises(ValueError, match="CUDA tensor payloads"):
        group.send(payload, borrowed_ipc=True)


def test_borrowed_ipc_rejects_unsupported_container_before_send():
    group = _bare_group()

    with pytest.raises(TypeError, match="tensor or tensor list"):
        group.send({"value": torch.ones(1)}, borrowed_ipc=True)


def test_borrowed_mode_mismatch_drains_ipc_payload_before_error(monkeypatch):
    group = _bare_group()
    received_modes = []

    def recv(self, tensor, device, comm_id):
        del self, device, comm_id
        if tensor.dtype == torch.long:
            tensor.fill_(1)

    def recv_ipc(self, comm_id, *, borrowed=False):
        del self, comm_id
        received_modes.append(borrowed)
        return [torch.ones(1)]

    group._recv = MethodType(recv, group)
    group._tensor_to_object = MethodType(
        lambda self, tensor, size: {
            "meta": [(torch.Size([1]), torch.float32)],
            "pb": None,
            "cpu_tensor_mask": [False],
            "borrowed_ipc": True,
        },
        group,
    )
    group._check_same_device_with_peer = MethodType(lambda self: 1, group)
    group._recv_tensor_list_via_ipc = MethodType(recv_ipc, group)

    class Platform:
        collects = 0

        @staticmethod
        def ipc_collect():
            Platform.collects += 1

    monkeypatch.setattr(Worker, "torch_platform", Platform)

    with pytest.raises(RuntimeError, match="enabled explicitly on both"):
        group._recv_tensor_list(comm_id=0, borrowed_ipc=False)

    assert received_modes == [True]
    assert Platform.collects == 1


def test_borrowed_ipc_receiver_rejects_non_colocated_sender_before_payload():
    group = _bare_group()
    recv_calls = []

    def recv(self, tensor, device, comm_id):
        del self, comm_id
        recv_calls.append(device)
        if tensor.dtype == torch.long:
            tensor.fill_(1)

    group._recv = MethodType(recv, group)
    group._tensor_to_object = MethodType(
        lambda self, tensor, size: {
            "meta": [(torch.Size([1]), torch.float32)],
            "pb": None,
            "cpu_tensor_mask": [False],
            "borrowed_ipc": True,
        },
        group,
    )
    group._check_same_device_with_peer = MethodType(lambda self: -1, group)

    with pytest.raises(RuntimeError, match="same CUDA device"):
        group._recv_tensor_list(comm_id=0, borrowed_ipc=True)

    # Only metadata size and metadata were consumed; no tensor payload receive ran.
    assert recv_calls == [CollectiveGroup.CPU, CollectiveGroup.CPU]
