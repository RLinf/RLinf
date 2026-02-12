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
import gc
import os
import threading
import time
from dataclasses import dataclass

import pytest
import torch

from rlinf.scheduler import (
    Cluster,
    CollectiveGroupOptions,
    NodePlacementStrategy,
    PackedPlacementStrategy,
    Worker,
    WorkerAddress,
)

SENDER_GROUP_NAME = "sender_worker_group"
RECEIVER_GROUP_NAME = "receiver_worker_group"

# --- Helper Functions ---


@dataclass
class TensorMessage:
    """Simple dataclass with a tensor field for testing direct tensor send/recv/broadcast."""

    id: int
    payload: torch.Tensor
    note: str


@dataclass
class TensorListMessage:
    """Dataclass with a list of tensors for testing send/recv/broadcast."""

    id: int
    payload_list: list
    note: str


@dataclass
class TensorDictMessage:
    """Dataclass with a dict of tensors for testing send/recv/broadcast."""

    id: int
    payload_dict: dict
    note: str


@dataclass
class PlainMessage:
    """Plain dataclass without tensor fields (sent as Python object)."""

    id: int
    name: str
    value: float


def get_device():
    """Returns the appropriate torch device."""
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return torch.cuda.current_device() if torch.cuda.is_available() else "cpu"


def get_send_peer_rank(rank, world_size):
    """Calculates the rank of the peer worker."""
    return (rank + 1) % world_size


def get_recv_peer_rank(rank, world_size):
    """Calculates the rank of the peer worker."""
    return (rank - 1) % world_size


NON_CONTIGUOUS_ERR = "must be contiguous when using P2P communication"


def make_non_contiguous_tensor(device):
    """Returns a non-contiguous CUDA tensor (e.g. from .t())."""
    t = torch.ones(2, 3, device=device)
    return t.t()  # transpose is non-contiguous


# --- Worker Definitions ---
class SenderWorker(Worker):
    """Worker responsible for sending data in tests."""

    def __init__(self):
        super().__init__()
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    def _send_data(self, data, async_op, use_send_tensor=False):
        """Generic data sending method."""
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        peer_rank = get_send_peer_rank(self._rank, self._world_size)
        if use_send_tensor:
            work = self.send_tensor(
                data, RECEIVER_GROUP_NAME, peer_rank, async_op=async_op
            )
        else:
            work = self.send(
                data,
                RECEIVER_GROUP_NAME,
                peer_rank,
                async_op=async_op,
                options=CollectiveGroupOptions(accel_max_ctas=1),
            )

        if async_op:
            work.wait()
        return True

    async def _send_data_asyncio(self, data_factory, use_send_tensor=False):
        """Generic data sending method using asyncio."""

        async def _send():
            if torch.cuda.is_available():
                torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            data = data_factory()
            peer_rank = get_send_peer_rank(self._rank, self._world_size)
            if use_send_tensor:
                work = self.send_tensor(
                    data, RECEIVER_GROUP_NAME, peer_rank, async_op=True
                )
            else:
                work = self.send(data, RECEIVER_GROUP_NAME, peer_rank, async_op=True)
            await work.async_wait()
            return True

        return await _send()

    # Sync Tests
    def test_send_object(self, async_op=False):
        return self._send_data({"message": f"Hello from rank {self._rank}"}, async_op)

    def test_send_plain_dataclass(self, async_op=False):
        msg = PlainMessage(
            id=self._rank,
            name=f"rank_{self._rank}",
            value=3.14 * (self._rank + 1),
        )
        return self._send_data(msg, async_op)

    def test_send_tensor(self, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        tensor = torch.ones(2, 2, device=device) * self._rank
        return self._send_data(tensor, async_op)

    def test_send_tensor_list(self, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        tensor_list = [torch.ones(2, 2, device=device) * i for i in range(4)]
        return self._send_data(tensor_list, async_op)

    def test_send_tensor_dict(self, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        tensor_dict = {f"t{i}": torch.ones(2, 2, device=device) * i for i in range(4)}
        return self._send_data(tensor_dict, async_op)

    def test_send_tensor_inplace(self, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        tensor = torch.ones(3, 3, device=device) * self._rank
        return self._send_data(tensor, async_op, use_send_tensor=True)

    def test_send_tensor_dataclass(self, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        payload = torch.ones(2, 2, device=device) * self._rank
        msg = TensorMessage(
            id=self._rank, payload=payload, note=f"from rank {self._rank}"
        )
        return self._send_data(msg, async_op)

    def test_send_tensor_list_dataclass(self, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        payload_list = [
            torch.ones(2, 2, device=device) * (self._rank * 10 + i) for i in range(3)
        ]
        msg = TensorListMessage(
            id=self._rank,
            payload_list=payload_list,
            note=f"list from rank {self._rank}",
        )
        return self._send_data(msg, async_op)

    def test_send_tensor_dict_dataclass(self, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        payload_dict = {
            f"k{i}": torch.ones(2, 2, device=device) * (self._rank * 10 + i)
            for i in range(3)
        }
        msg = TensorDictMessage(
            id=self._rank,
            payload_dict=payload_dict,
            note=f"dict from rank {self._rank}",
        )
        return self._send_data(msg, async_op)

    def test_send_non_contiguous_tensor(self):
        try:
            device = get_device()
            data = make_non_contiguous_tensor(device)
            return self._send_data(data, False)
        except ValueError as e:
            return e

    def test_send_non_contiguous_tensor_list(self):
        try:
            device = get_device()
            data = [make_non_contiguous_tensor(device) for _ in range(2)]
            return self._send_data(data, False)
        except ValueError as e:
            return e

    def test_send_non_contiguous_tensor_dict(self):
        try:
            device = get_device()
            data = {
                "a": make_non_contiguous_tensor(device),
                "b": make_non_contiguous_tensor(device),
            }
            return self._send_data(data, False)
        except ValueError as e:
            return e

    def test_send_non_contiguous_tensor_dataclass(self):
        try:
            device = get_device()
            data = TensorMessage(
                id=1, payload=make_non_contiguous_tensor(device), note="non-contiguous"
            )
            return self._send_data(data, False)
        except ValueError as e:
            return e

    def test_send_non_contiguous_tensor_list_dataclass(self):
        try:
            device = get_device()
            data = TensorListMessage(
                id=1,
                payload_list=[make_non_contiguous_tensor(device) for _ in range(2)],
                note="non-contiguous list",
            )
            return self._send_data(data, False)
        except ValueError as e:
            return e

    def test_send_non_contiguous_tensor_dict_dataclass(self):
        try:
            device = get_device()
            data = TensorDictMessage(
                id=1,
                payload_dict={
                    "a": make_non_contiguous_tensor(device),
                    "b": make_non_contiguous_tensor(device),
                },
                note="non-contiguous dict",
            )
            return self._send_data(data, False)
        except ValueError as e:
            return e

    def test_send_tensor_non_contiguous_inplace(self):
        try:
            device = get_device()
            data = make_non_contiguous_tensor(device)
            return self._send_data(data, False, use_send_tensor=True)
        except ValueError as e:
            return e

    # Asyncio Tests
    async def test_send_tensor_asyncio(self, on_cpu):
        device = "cpu" if on_cpu else get_device()
        return await self._send_data_asyncio(
            lambda: torch.ones(4, 4, device=device) * self._rank
        )

    async def test_send_tensor_dataclass_asyncio(self, on_cpu):
        device = "cpu" if on_cpu else get_device()
        return await self._send_data_asyncio(
            lambda: TensorMessage(
                id=self._rank,
                payload=torch.ones(4, 4, device=device) * self._rank,
                note=f"async from rank {self._rank}",
            )
        )

    async def test_send_tensor_list_dataclass_asyncio(self, on_cpu):
        device = "cpu" if on_cpu else get_device()
        return await self._send_data_asyncio(
            lambda: TensorListMessage(
                id=self._rank,
                payload_list=[
                    torch.ones(2, 2, device=device) * (self._rank * 10 + i)
                    for i in range(3)
                ],
                note=f"async list from rank {self._rank}",
            )
        )

    async def test_send_tensor_dict_dataclass_asyncio(self, on_cpu):
        device = "cpu" if on_cpu else get_device()
        return await self._send_data_asyncio(
            lambda: TensorDictMessage(
                id=self._rank,
                payload_dict={
                    f"k{i}": torch.ones(2, 2, device=device) * (self._rank * 10 + i)
                    for i in range(3)
                },
                note=f"async dict from rank {self._rank}",
            )
        )

    def test_unaligned_send_recv(self, on_cpu):
        """Test unaligned sending and receiving of tensors."""
        device = "cpu" if on_cpu else get_device()
        tensor = torch.ones(5, 5, device=device) * self._rank
        peer_rank = get_send_peer_rank(self._rank, self._world_size)
        recv_work = self.recv(RECEIVER_GROUP_NAME, peer_rank, async_op=True)
        self.send(tensor, RECEIVER_GROUP_NAME, peer_rank)
        recv_work.wait()

        recv_tensor = torch.zeros(5, 5, device=device) * self._rank
        recv_work = self.recv_tensor(
            recv_tensor, RECEIVER_GROUP_NAME, peer_rank, async_op=True
        )
        self.send_tensor(tensor, RECEIVER_GROUP_NAME, peer_rank)
        return recv_work.wait()

    def test_consecutive_send_recv(self, on_cpu):
        """Test sending and receiving tensors in a consecutive manner."""
        device = "cpu" if on_cpu else get_device()
        send_tensor = torch.ones(5, 5, device=device) * self._rank
        recv_tensor = torch.zeros(5, 5, device=device)
        send_works = []
        recv_works = []
        peer_rank = get_send_peer_rank(self._rank, self._world_size)
        for _ in range(100):
            send_works.append(
                self.send(send_tensor, RECEIVER_GROUP_NAME, peer_rank, async_op=True)
            )
            recv_works.append(self.recv(RECEIVER_GROUP_NAME, peer_rank, async_op=True))
            send_works.append(
                self.send_tensor(
                    send_tensor, RECEIVER_GROUP_NAME, peer_rank, async_op=True
                )
            )
            recv_works.append(
                self.recv_tensor(
                    recv_tensor, RECEIVER_GROUP_NAME, peer_rank, async_op=True
                )
            )
        for work in send_works:
            work.wait()
        for work in recv_works:
            work.wait()
        return None

    async def test_memory_leak(self):
        """A test to check for memory leaks during send operations."""
        device = get_device()
        tensor_size = 1024
        large_tensor = torch.randn(tensor_size, dtype=torch.float16, device=device)
        peer_rank = get_send_peer_rank(self._rank, self._world_size)

        self.send(large_tensor, RECEIVER_GROUP_NAME, peer_rank)
        self.send(large_tensor, RECEIVER_GROUP_NAME, peer_rank, async_op=True).wait()
        self.send_tensor(large_tensor, RECEIVER_GROUP_NAME, peer_rank)
        self.send_tensor(
            large_tensor, RECEIVER_GROUP_NAME, peer_rank, async_op=True
        ).wait()

        async def _async_send():
            await self.send(
                large_tensor, RECEIVER_GROUP_NAME, peer_rank, async_op=True
            ).async_wait()
            await self.send_tensor(
                large_tensor, RECEIVER_GROUP_NAME, peer_rank, async_op=True
            ).async_wait()

        await _async_send()

        large_tensor = None
        gc.collect()
        torch.cuda.empty_cache()
        assert torch.cuda.memory_allocated() == 0
        return True


class ReceiverWorker(Worker):
    """Worker responsible for receiving data in tests."""

    def __init__(self):
        super().__init__()
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    def _recv_data(self, async_op, recv_tensor_inplace_shape=None):
        """Generic data receiving method."""
        peer_rank = get_recv_peer_rank(self._rank, self._world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        if recv_tensor_inplace_shape:
            on_cpu, shape = recv_tensor_inplace_shape
            device = "cpu" if on_cpu else get_device()
            tensor = torch.empty(shape, device=device)
            work = self.recv_tensor(
                tensor, SENDER_GROUP_NAME, peer_rank, async_op=async_op
            )
            if async_op:
                work.wait()
            return tensor
        else:
            work = self.recv(
                SENDER_GROUP_NAME,
                peer_rank,
                async_op=async_op,
                options=CollectiveGroupOptions(accel_max_ctas=1),
            )
            if async_op:
                return work.wait()
            return work

    async def _recv_data_asyncio(self, recv_tensor_inplace_shape=None):
        """Generic data receiving method using asyncio."""

        async def _recv():
            if torch.cuda.is_available():
                torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            peer_rank = get_recv_peer_rank(self._rank, self._world_size)
            if recv_tensor_inplace_shape:
                on_cpu, shape = recv_tensor_inplace_shape
                device = "cpu" if on_cpu else get_device()
                tensor = torch.empty(shape, device=device)
                work = self.recv_tensor(
                    tensor, SENDER_GROUP_NAME, peer_rank, async_op=True
                )
                await work.async_wait()
                return tensor
            else:
                work = self.recv(SENDER_GROUP_NAME, peer_rank, async_op=True)
                return await work.async_wait()

        return await _recv()

    def test_unaligned_send_recv(self, on_cpu):
        """Test unaligned sending and receiving of tensors."""
        device = "cpu" if on_cpu else get_device()
        tensor = torch.ones(5, 5, device=device) * self._rank
        peer_rank = get_recv_peer_rank(self._rank, self._world_size)
        recv_work = self.recv(SENDER_GROUP_NAME, peer_rank, async_op=True)
        self.send(tensor, SENDER_GROUP_NAME, peer_rank)
        recv_work.wait()

        recv_tensor = torch.zeros(5, 5, device=device) * self._rank
        recv_work = self.recv_tensor(
            recv_tensor, SENDER_GROUP_NAME, peer_rank, async_op=True
        )
        self.send_tensor(tensor, SENDER_GROUP_NAME, peer_rank)
        recv_work.wait()
        return recv_tensor

    def test_consecutive_send_recv(self, on_cpu):
        """Test sending and receiving tensors in a consecutive manner."""
        device = "cpu" if on_cpu else get_device()
        send_tensor = torch.ones(5, 5, device=device) * self._rank
        recv_tensor = torch.zeros(5, 5, device=device)
        send_works = []
        recv_works = []
        peer_rank = get_recv_peer_rank(self._rank, self._world_size)
        for _ in range(100):
            recv_works.append(self.recv(SENDER_GROUP_NAME, peer_rank, async_op=True))
            send_works.append(
                self.send(send_tensor, SENDER_GROUP_NAME, peer_rank, async_op=True)
            )
            recv_works.append(
                self.recv_tensor(
                    recv_tensor, SENDER_GROUP_NAME, peer_rank, async_op=True
                )
            )
            send_works.append(
                self.send_tensor(
                    send_tensor, SENDER_GROUP_NAME, peer_rank, async_op=True
                )
            )
        for work in send_works:
            work.wait()
        tensors = [work.wait() for work in recv_works]
        return tensors[0]

    # Sync/Async Wait Tests
    def test_recv_object(self, async_op=False):
        return self._recv_data(async_op)

    def test_recv_plain_dataclass(self, async_op=False):
        return self._recv_data(async_op)

    def test_recv_tensor(self, async_op=False):
        return self._recv_data(async_op)

    def test_recv_tensor_list(self, async_op=False):
        return self._recv_data(async_op)

    def test_recv_tensor_dict(self, async_op=False):
        return self._recv_data(async_op)

    def test_recv_tensor_inplace(self, on_cpu, async_op=False):
        return self._recv_data(async_op, recv_tensor_inplace_shape=(on_cpu, (3, 3)))

    def test_recv_tensor_dataclass(self, async_op=False):
        return self._recv_data(async_op)

    def test_recv_tensor_list_dataclass(self, async_op=False):
        return self._recv_data(async_op)

    def test_recv_tensor_dict_dataclass(self, async_op=False):
        return self._recv_data(async_op)

    # Asyncio Tests
    async def test_recv_tensor_asyncio(self, on_cpu):
        return await self._recv_data_asyncio()

    async def test_recv_tensor_dataclass_asyncio(self):
        return await self._recv_data_asyncio()

    async def test_recv_tensor_list_dataclass_asyncio(self):
        return await self._recv_data_asyncio()

    async def test_recv_tensor_dict_dataclass_asyncio(self):
        return await self._recv_data_asyncio()

    async def test_memory_leak(self):
        """A test to check for memory leaks during send operations."""
        peer_rank = get_recv_peer_rank(self._rank, self._world_size)
        recv_tensor_size = 1024
        device = get_device()
        recv_tensor = torch.randn(recv_tensor_size, dtype=torch.float16, device=device)

        self.recv(SENDER_GROUP_NAME, peer_rank)
        self.recv(SENDER_GROUP_NAME, peer_rank, async_op=True).wait()
        self.recv_tensor(recv_tensor, SENDER_GROUP_NAME, peer_rank)
        self.recv_tensor(
            recv_tensor, SENDER_GROUP_NAME, peer_rank, async_op=True
        ).wait()

        async def _async_recv():
            await self.recv(SENDER_GROUP_NAME, peer_rank, async_op=True).async_wait()
            await self.recv_tensor(
                recv_tensor, SENDER_GROUP_NAME, peer_rank, async_op=True
            ).async_wait()

        await _async_recv()

        recv_tensor = None
        gc.collect()
        torch.cuda.empty_cache()
        assert torch.cuda.memory_allocated() == 0

    async def test_async_wait_yields_control(self):
        """Run recv(async_op=True) and await async_wait() concurrently with another
        asyncio task. Assert the other task ran while waiting, proving async_wait()
        yields control to the event loop."""
        peer_rank = get_recv_peer_rank(self._rank, self._world_size)

        async def recv_task():
            work = self.recv(
                SENDER_GROUP_NAME,
                peer_rank,
                async_op=True,
                options=CollectiveGroupOptions(accel_max_ctas=1),
            )
            return await work.async_wait()

        async def yield_check_task():
            count = 0
            for _ in range(30):
                count += 1
                await asyncio.sleep(0.01)
            return count

        asyncio.create_task(recv_task())
        count = await yield_check_task()
        return count


class CommCollectiveWorker(Worker):
    """Worker for collective communication tests."""

    def __init__(self):
        super().__init__()
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    def _broadcast_data(self, data, async_op):
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        groups = [(self._group_name, list(range(self._world_size)))]
        payload = data if self._rank == 0 else None
        result = self.broadcast(payload, groups=groups, async_op=async_op)
        if async_op:
            return result.wait()
        return result

    def test_broadcast_object(self, async_op=False):
        payload = {"message": "Hello from rank 0", "rank": 0}
        return self._broadcast_data(payload, async_op)

    def test_broadcast_plain_dataclass(self, async_op=False):
        payload = (
            PlainMessage(id=0, name="broadcast_src", value=2.71)
            if self._rank == 0
            else None
        )
        return self._broadcast_data(payload, async_op)

    def test_broadcast_tensor(self, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        payload = torch.ones(2, 2, device=device) * 7
        return self._broadcast_data(payload, async_op)

    def test_broadcast_tensor_list(self, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        payload = [torch.ones(2, 2, device=device) * i for i in range(4)]
        return self._broadcast_data(payload, async_op)

    def test_broadcast_tensor_dict(self, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        payload = {f"t{i}": torch.ones(2, 2, device=device) * i for i in range(4)}
        return self._broadcast_data(payload, async_op)

    def test_broadcast_tensor_dataclass(self, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        payload = (
            TensorMessage(
                id=0,
                payload=torch.ones(2, 2, device=device) * 7,
                note="broadcast from rank 0",
            )
            if self._rank == 0
            else None
        )
        return self._broadcast_data(payload, async_op)

    def test_broadcast_tensor_list_dataclass(self, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        payload = (
            TensorListMessage(
                id=0,
                payload_list=[torch.ones(2, 2, device=device) * i for i in range(4)],
                note="broadcast list from rank 0",
            )
            if self._rank == 0
            else None
        )
        return self._broadcast_data(payload, async_op)

    def test_broadcast_tensor_dict_dataclass(self, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        payload = (
            TensorDictMessage(
                id=0,
                payload_dict={
                    f"t{i}": torch.ones(2, 2, device=device) * i for i in range(4)
                },
                note="broadcast dict from rank 0",
            )
            if self._rank == 0
            else None
        )
        return self._broadcast_data(payload, async_op)

    async def test_broadcast_tensor_asyncio(self, on_cpu):
        async def _broadcast():
            if torch.cuda.is_available():
                torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            device = "cpu" if on_cpu else get_device()
            groups = [(self._group_name, list(range(self._world_size)))]
            payload = torch.ones(3, 3, device=device) * 5
            result = self.broadcast(
                payload if self._rank == 0 else None, groups=groups, async_op=True
            )
            await result.async_wait()
            return result.wait()

        return await _broadcast()

    async def test_cross_group_broadcast_tensor_asyncio(self, groups, on_cpu):
        async def _broadcast():
            if torch.cuda.is_available():
                torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            device = "cpu" if on_cpu else get_device()
            src_group_name, src_ranks = groups[0]
            if isinstance(src_ranks, list):
                src_rank = src_ranks[0]
            else:
                src_rank = src_ranks
            is_src = self._worker_address == WorkerAddress(
                src_group_name, ranks=src_rank
            )
            payload = torch.ones(3, 3, device=device) * 9
            result = self.broadcast(
                payload if is_src else None, groups=groups, async_op=True
            )
            await result.async_wait()
            return result.wait()

        return await _broadcast()

    def _cross_group_broadcast(self, groups, payload, async_op):
        src_group_name, src_ranks = groups[0]
        if isinstance(src_ranks, list):
            src_rank = src_ranks[0]
        else:
            src_rank = src_ranks
        is_src = self._worker_address == WorkerAddress(src_group_name, ranks=src_rank)
        result = self.broadcast(
            payload if is_src else None, groups=groups, async_op=async_op
        )
        if async_op:
            return result.wait()
        return result

    def test_cross_group_broadcast_object(self, groups, async_op=False):
        payload = {"message": "Hello from cross-group src", "rank": 0}
        return self._cross_group_broadcast(groups, payload, async_op)

    def test_cross_group_broadcast_tensor(self, groups, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        payload = torch.ones(2, 2, device=device) * 11
        return self._cross_group_broadcast(groups, payload, async_op)

    def test_broadcast_object_with_src(self, groups, src, async_op=False):
        payload = {"message": "Hello from explicit src", "rank": 0}
        result = self.broadcast(
            payload if self._worker_address == src else None,
            groups=groups,
            src=(src.root_group_name, src.rank),
            async_op=async_op,
        )
        if async_op:
            return result.wait()
        return result

    def test_broadcast_tensor_with_src(self, groups, src, on_cpu, async_op=False):
        device = "cpu" if on_cpu else get_device()
        payload = torch.ones(2, 2, device=device) * 13
        result = self.broadcast(
            payload if self._worker_address == src else None,
            groups=groups,
            src=(src.root_group_name, src.rank),
            async_op=async_op,
        )
        if async_op:
            return result.wait()
        return result

    def test_broadcast_tensor_dataclass_with_src(
        self, groups, src, on_cpu, async_op=False
    ):
        device = "cpu" if on_cpu else get_device()
        payload = (
            TensorMessage(
                id=13,
                payload=torch.ones(2, 2, device=device) * 13,
                note="broadcast with src",
            )
            if self._worker_address == src
            else None
        )
        result = self.broadcast(
            payload,
            groups=groups,
            src=(src.root_group_name, src.rank),
            async_op=async_op,
        )
        if async_op:
            return result.wait()
        return result

    def test_cross_group_broadcast_tensor_dataclass(
        self, groups, on_cpu, async_op=False
    ):
        device = "cpu" if on_cpu else get_device()
        src_group_name, src_ranks = groups[0]
        if isinstance(src_ranks, list):
            src_rank = src_ranks[0]
        else:
            src_rank = src_ranks
        is_src = self._worker_address == WorkerAddress(src_group_name, ranks=src_rank)
        payload = (
            TensorMessage(
                id=11,
                payload=torch.ones(2, 2, device=device) * 11,
                note="cross-group broadcast dataclass",
            )
            if is_src
            else None
        )
        return self._cross_group_broadcast(groups, payload, async_op)

    async def test_broadcast_tensor_dataclass_asyncio(self, on_cpu):
        async def _broadcast():
            if torch.cuda.is_available():
                torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            device = "cpu" if on_cpu else get_device()
            groups = [(self._group_name, list(range(self._world_size)))]
            payload = TensorMessage(
                id=5,
                payload=torch.ones(3, 3, device=device) * 5,
                note="async broadcast from rank 0",
            )
            result = self.broadcast(
                payload if self._rank == 0 else None,
                groups=groups,
                async_op=True,
            )
            await result.async_wait()
            return result.wait()

        return await _broadcast()


# --- Pytest Setup ---


@pytest.fixture(scope="module")
def cluster():
    """Provides a ClusterResource instance for the tests."""
    return Cluster(num_nodes=1)


@pytest.fixture(scope="class")
def worker_groups(cluster: Cluster):
    """Creates and yields the sender and receiver worker groups."""
    if cluster.num_accelerators > 0:
        sender_group = SenderWorker.create_group().launch(
            cluster=cluster, name=SENDER_GROUP_NAME
        )
        receiver_group = ReceiverWorker.create_group().launch(
            cluster=cluster, name=RECEIVER_GROUP_NAME
        )
    else:
        placement = NodePlacementStrategy([0] * 8)
        sender_group = SenderWorker.create_group().launch(
            cluster=cluster, placement_strategy=placement, name=SENDER_GROUP_NAME
        )
        receiver_group = ReceiverWorker.create_group().launch(
            cluster=cluster, placement_strategy=placement, name=RECEIVER_GROUP_NAME
        )
    yield sender_group, receiver_group
    sender_group._close()
    receiver_group._close()


@pytest.fixture(scope="class")
def collective_group(cluster: Cluster):
    """Creates and yields the collective worker group."""
    if cluster.num_accelerators > 0:
        group = CommCollectiveWorker.create_group().launch(
            cluster=cluster, name="collective_group"
        )
    else:
        placement = NodePlacementStrategy([0] * 8)
        group = CommCollectiveWorker.create_group().launch(
            cluster=cluster, placement_strategy=placement, name="collective_group"
        )
    yield group
    group._close()


@pytest.fixture(scope="class")
def cross_collective_groups(cluster: Cluster):
    """Creates and yields two collective worker groups for cross-group tests."""
    if torch.cuda.is_available():
        if cluster.num_accelerators < 2:
            pytest.skip("Skipping cross-group tests with insufficient accelerators.")
        if cluster.num_accelerators >= 4:
            group_a_size = 2
            group_b_size = 2
        else:
            group_a_size = 1
            group_b_size = 1
        placement_a = PackedPlacementStrategy(0, group_a_size - 1)
        placement_b = PackedPlacementStrategy(
            group_a_size, group_a_size + group_b_size - 1
        )
    else:
        group_a_size = 2
        group_b_size = 2
        placement_a = NodePlacementStrategy([0] * group_a_size)
        placement_b = NodePlacementStrategy([0] * group_b_size)

    group_a = CommCollectiveWorker.create_group().launch(
        cluster=cluster, placement_strategy=placement_a, name="collective_group_a"
    )
    group_b = CommCollectiveWorker.create_group().launch(
        cluster=cluster, placement_strategy=placement_b, name="collective_group_b"
    )
    yield group_a, group_b, group_a_size, group_b_size
    group_a._close()
    group_b._close()


# --- Test Class ---


@pytest.mark.usefixtures("worker_groups")
class TestCommunication:
    """A suite of tests for send/recv communication APIs."""

    def _run_test(
        self,
        worker_groups,
        sender_method,
        receiver_method,
        sender_args=(),
        receiver_args=(),
    ):
        """Helper to run a sender/receiver test pair."""
        sender_group, receiver_group = worker_groups
        sender_results = getattr(sender_group, sender_method)(*sender_args)
        receiver_results = getattr(receiver_group, receiver_method)(*receiver_args)
        results = sender_results.wait()
        results = receiver_results.wait()
        return results

    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_object_communication(self, worker_groups, async_op):
        """Tests sending and receiving a Python object."""
        results = self._run_test(
            worker_groups,
            "test_send_object",
            "test_recv_object",
            (async_op,),
            (async_op,),
        )
        for i, res in enumerate(results):
            peer_rank = get_recv_peer_rank(i, len(results))
            assert res == {"message": f"Hello from rank {peer_rank}"}

    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_plain_dataclass_communication(self, worker_groups, async_op):
        """Tests sending and receiving a plain dataclass without tensor fields."""
        results = self._run_test(
            worker_groups,
            "test_send_plain_dataclass",
            "test_recv_plain_dataclass",
            (async_op,),
            (async_op,),
        )
        for i, res in enumerate(results):
            peer_rank = get_recv_peer_rank(i, len(results))
            assert isinstance(res, PlainMessage)
            assert res.id == peer_rank
            assert res.name == f"rank_{peer_rank}"
            assert res.value == pytest.approx(3.14 * (peer_rank + 1))

    @pytest.mark.parametrize("on_cpu", [True, False], ids=["cpu", "cuda"])
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_tensor_communication(self, worker_groups, on_cpu, async_op):
        """Tests sending and receiving a single tensor."""
        if not on_cpu and not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test on CPU-only environment.")
        results = self._run_test(
            worker_groups,
            "test_send_tensor",
            "test_recv_tensor",
            (on_cpu, async_op),
            (async_op,),
        )
        for i, res in enumerate(results):
            peer_rank = get_recv_peer_rank(i, len(results))
            expected = torch.ones(2, 2) * peer_rank
            assert torch.equal(res.cpu(), expected)

    @pytest.mark.parametrize("on_cpu", [True, False], ids=["cpu", "cuda"])
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_tensor_list_communication(self, worker_groups, on_cpu, async_op):
        """Tests sending and receiving a list of tensors."""
        if not on_cpu and not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test on CPU-only environment.")
        results = self._run_test(
            worker_groups,
            "test_send_tensor_list",
            "test_recv_tensor_list",
            (on_cpu, async_op),
            (async_op,),
        )
        for res_list in results:
            assert isinstance(res_list, list)
            for i, tensor in enumerate(res_list):
                expected = torch.ones(2, 2) * i
                assert torch.equal(tensor.cpu(), expected)

    @pytest.mark.parametrize("on_cpu", [True, False], ids=["cpu", "cuda"])
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_tensor_dict_communication(self, worker_groups, on_cpu, async_op):
        """Tests sending and receiving a dictionary of tensors."""
        if not on_cpu and not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test on CPU-only environment.")
        results = self._run_test(
            worker_groups,
            "test_send_tensor_dict",
            "test_recv_tensor_dict",
            (on_cpu, async_op),
            (async_op,),
        )
        for res_dict in results:
            assert isinstance(res_dict, dict)
            for i, key in enumerate(sorted(res_dict.keys())):
                assert key == f"t{i}"
                expected = torch.ones(2, 2) * i
                assert torch.equal(res_dict[key].cpu(), expected)

    @pytest.mark.parametrize("on_cpu", [True, False], ids=["cpu", "cuda"])
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_inplace_tensor_communication(self, worker_groups, on_cpu, async_op):
        """Tests send_tensor/recv_tensor for in-place tensor communication."""
        if not on_cpu and not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test on CPU-only environment.")
        results = self._run_test(
            worker_groups,
            "test_send_tensor_inplace",
            "test_recv_tensor_inplace",
            (on_cpu, async_op),
            (on_cpu, async_op),
        )
        for i, res in enumerate(results):
            peer_rank = get_recv_peer_rank(i, len(results))
            expected = torch.ones(3, 3) * peer_rank
            assert torch.equal(res.cpu(), expected)

    @pytest.mark.parametrize("on_cpu", [True, False], ids=["cpu", "cuda"])
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_tensor_dataclass_communication(self, worker_groups, on_cpu, async_op):
        """Tests sending and receiving a dataclass containing torch tensors."""
        if not on_cpu and not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test on CPU-only environment.")
        results = self._run_test(
            worker_groups,
            "test_send_tensor_dataclass",
            "test_recv_tensor_dataclass",
            (on_cpu, async_op),
            (async_op,),
        )
        for i, res in enumerate(results):
            peer_rank = get_recv_peer_rank(i, len(results))
            assert isinstance(res, TensorMessage)
            assert res.id == peer_rank
            assert res.note == f"from rank {peer_rank}"
            assert torch.equal(res.payload.cpu(), torch.ones(2, 2) * peer_rank)

    @pytest.mark.parametrize("on_cpu", [True, False], ids=["cpu", "cuda"])
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_tensor_list_dataclass_communication(self, worker_groups, on_cpu, async_op):
        """Tests sending and receiving a dataclass containing a list of tensors."""
        if not on_cpu and not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test on CPU-only environment.")
        results = self._run_test(
            worker_groups,
            "test_send_tensor_list_dataclass",
            "test_recv_tensor_list_dataclass",
            (on_cpu, async_op),
            (async_op,),
        )
        for i, res in enumerate(results):
            peer_rank = get_recv_peer_rank(i, len(results))
            assert isinstance(res, TensorListMessage)
            assert res.id == peer_rank
            assert res.note == f"list from rank {peer_rank}"
            assert len(res.payload_list) == 3
            for j, t in enumerate(res.payload_list):
                expected = torch.ones(2, 2) * (peer_rank * 10 + j)
                assert torch.equal(t.cpu(), expected)

    @pytest.mark.parametrize("on_cpu", [True, False], ids=["cpu", "cuda"])
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_tensor_dict_dataclass_communication(self, worker_groups, on_cpu, async_op):
        """Tests sending and receiving a dataclass containing a dict of tensors."""
        if not on_cpu and not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test on CPU-only environment.")
        results = self._run_test(
            worker_groups,
            "test_send_tensor_dict_dataclass",
            "test_recv_tensor_dict_dataclass",
            (on_cpu, async_op),
            (async_op,),
        )
        for i, res in enumerate(results):
            peer_rank = get_recv_peer_rank(i, len(results))
            assert isinstance(res, TensorDictMessage)
            assert res.id == peer_rank
            assert res.note == f"dict from rank {peer_rank}"
            assert list(res.payload_dict.keys()) == ["k0", "k1", "k2"]
            for j, key in enumerate(sorted(res.payload_dict.keys())):
                expected = torch.ones(2, 2) * (peer_rank * 10 + j)
                assert torch.equal(res.payload_dict[key].cpu(), expected)

    @pytest.mark.parametrize("on_cpu", [True, False], ids=["cpu", "cuda"])
    def test_asyncio_communication(self, worker_groups, on_cpu):
        """Tests async communication with asyncio.run and async_wait."""
        if not on_cpu and not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test on CPU-only environment.")
        results = self._run_test(
            worker_groups,
            "test_send_tensor_asyncio",
            "test_recv_tensor_asyncio",
            (on_cpu,),
            (on_cpu,),
        )
        for i, res in enumerate(results):
            peer_rank = get_recv_peer_rank(i, len(results))
            expected = torch.ones(4, 4) * peer_rank
            assert torch.equal(res.cpu(), expected)

    @pytest.mark.parametrize("on_cpu", [True, False], ids=["cpu", "cuda"])
    def test_tensor_dataclass_asyncio_communication(self, worker_groups, on_cpu):
        """Tests async send/recv of dataclass containing torch tensors."""
        if not on_cpu and not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test on CPU-only environment.")
        results = self._run_test(
            worker_groups,
            "test_send_tensor_dataclass_asyncio",
            "test_recv_tensor_dataclass_asyncio",
            (on_cpu,),
            (),
        )
        for i, res in enumerate(results):
            peer_rank = get_recv_peer_rank(i, len(results))
            assert isinstance(res, TensorMessage)
            assert res.id == peer_rank
            assert res.note == f"async from rank {peer_rank}"
            assert torch.equal(res.payload.cpu(), torch.ones(4, 4) * peer_rank)

    @pytest.mark.parametrize("on_cpu", [True, False], ids=["cpu", "cuda"])
    def test_tensor_list_dataclass_asyncio_communication(self, worker_groups, on_cpu):
        """Tests async send/recv of dataclass containing list of tensors."""
        if not on_cpu and not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test on CPU-only environment.")
        results = self._run_test(
            worker_groups,
            "test_send_tensor_list_dataclass_asyncio",
            "test_recv_tensor_list_dataclass_asyncio",
            (on_cpu,),
            (),
        )
        for i, res in enumerate(results):
            peer_rank = get_recv_peer_rank(i, len(results))
            assert isinstance(res, TensorListMessage)
            assert res.id == peer_rank
            assert res.note == f"async list from rank {peer_rank}"
            assert len(res.payload_list) == 3
            for j, t in enumerate(res.payload_list):
                assert torch.equal(t.cpu(), torch.ones(2, 2) * (peer_rank * 10 + j))

    @pytest.mark.parametrize("on_cpu", [True, False], ids=["cpu", "cuda"])
    def test_tensor_dict_dataclass_asyncio_communication(self, worker_groups, on_cpu):
        """Tests async send/recv of dataclass containing dict of tensors."""
        if not on_cpu and not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test on CPU-only environment.")
        results = self._run_test(
            worker_groups,
            "test_send_tensor_dict_dataclass_asyncio",
            "test_recv_tensor_dict_dataclass_asyncio",
            (on_cpu,),
            (),
        )
        for i, res in enumerate(results):
            peer_rank = get_recv_peer_rank(i, len(results))
            assert isinstance(res, TensorDictMessage)
            assert res.id == peer_rank
            assert res.note == f"async dict from rank {peer_rank}"
            for j, key in enumerate(sorted(res.payload_dict.keys())):
                assert torch.equal(
                    res.payload_dict[key].cpu(),
                    torch.ones(2, 2) * (peer_rank * 10 + j),
                )

    @pytest.mark.parametrize("on_cpu", [True, False], ids=["cpu", "cuda"])
    def test_unaligned_send_recv(self, worker_groups, on_cpu):
        """Tests unaligned sending and receiving of tensors."""
        if not on_cpu and not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test on CPU-only environment.")
        results = self._run_test(
            worker_groups,
            "test_unaligned_send_recv",
            "test_unaligned_send_recv",
            (on_cpu,),
            (on_cpu,),
        )
        for i, res in enumerate(results):
            expected = torch.ones(5, 5) * get_recv_peer_rank(i, len(results))
            assert torch.equal(res.cpu(), expected)

    @pytest.mark.parametrize("on_cpu", [True, False], ids=["cpu", "cuda"])
    def test_consecutive_send_recv(self, worker_groups, on_cpu):
        """Tests sending and receiving tensors in a consecutive manner."""
        if not on_cpu and not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test on CPU-only environment.")
        results = self._run_test(
            worker_groups,
            "test_consecutive_send_recv",
            "test_consecutive_send_recv",
            (on_cpu,),
            (on_cpu,),
        )
        for i, res in enumerate(results):
            expected = torch.ones(5, 5) * get_recv_peer_rank(i, len(results))
            assert torch.equal(res.cpu(), expected)

    def test_memory_leak(self, worker_groups):
        """Tests unaligned sending and receiving of tensors."""
        if not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test on CPU-only environment.")
        self._run_test(
            worker_groups,
            "test_memory_leak",
            "test_memory_leak",
        )

    def test_async_wait_yields_control(self, worker_groups):
        """Ensures async_wait() of async comm correctly yields control so other
        asyncio tasks can run while waiting."""
        sender_group, receiver_group = worker_groups
        # Run on rank 0 only to avoid multi-worker timing; receiver waits, sender sends after delay.
        recv_ref = receiver_group.execute_on(1).test_async_wait_yields_control()

        def delayed_send():
            time.sleep(0.15)
            sender_group.execute_on(0).test_send_object(False).wait()

        t = threading.Thread(target=delayed_send)
        try:
            results = recv_ref.wait()
            t.start()
        finally:
            t.join()
        for i, yield_count in enumerate(results):
            assert yield_count >= 1, (
                f"async_wait() did not yield: yield_check task ran {yield_count} times"
            )

    @pytest.mark.parametrize(
        "sender_method",
        [
            "test_send_non_contiguous_tensor",
            "test_send_non_contiguous_tensor_list",
            "test_send_non_contiguous_tensor_dict",
            "test_send_non_contiguous_tensor_dataclass",
            "test_send_non_contiguous_tensor_list_dataclass",
            "test_send_non_contiguous_tensor_dict_dataclass",
            "test_send_tensor_non_contiguous_inplace",
        ],
    )
    def test_non_contiguous_send_raises_value_error(self, worker_groups, sender_method):
        """Sending non-contiguous CUDA tensors (any struct) must raise ValueError."""
        if not torch.cuda.is_available():
            pytest.skip("Skipping non-contiguous tests on CPU-only environment.")
        sender_group, _ = worker_groups
        results = getattr(sender_group.execute_on(0), sender_method)().wait()
        err = results[0]
        assert isinstance(err, ValueError), (
            f"Expected ValueError, got {type(err)}: {err}"
        )
        assert NON_CONTIGUOUS_ERR in str(err), (
            f"Expected message containing {NON_CONTIGUOUS_ERR!r}, got: {err}"
        )


@pytest.mark.usefixtures("collective_group")
class TestCollective:
    """A suite of tests for collective communication APIs."""

    def _run_collective_test(self, collective_group, method, *args):
        results = getattr(collective_group, method)(*args).wait()
        return results

    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_broadcast_object(self, collective_group, async_op):
        results = self._run_collective_test(
            collective_group, "test_broadcast_object", async_op
        )
        for res in results:
            assert res == {"message": "Hello from rank 0", "rank": 0}

    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_broadcast_plain_dataclass(self, collective_group, async_op):
        """Tests broadcasting a plain dataclass without tensor fields."""
        results = self._run_collective_test(
            collective_group, "test_broadcast_plain_dataclass", async_op
        )
        for res in results:
            assert isinstance(res, PlainMessage)
            assert res.id == 0
            assert res.name == "broadcast_src"
            assert res.value == pytest.approx(2.71)

    @pytest.mark.parametrize("on_cpu", [True, False], ids=["cpu", "cuda"])
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_broadcast_tensor(self, collective_group, on_cpu, async_op):
        if not on_cpu and not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test on CPU-only environment.")
        results = self._run_collective_test(
            collective_group, "test_broadcast_tensor", on_cpu, async_op
        )
        expected = torch.ones(2, 2) * 7
        for res in results:
            assert torch.equal(res.cpu(), expected)

    @pytest.mark.parametrize("on_cpu", [True, False], ids=["cpu", "cuda"])
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_broadcast_tensor_list(self, collective_group, on_cpu, async_op):
        if not on_cpu and not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test on CPU-only environment.")
        results = self._run_collective_test(
            collective_group, "test_broadcast_tensor_list", on_cpu, async_op
        )
        for res_list in results:
            assert isinstance(res_list, list)
            for i, tensor in enumerate(res_list):
                expected = torch.ones(2, 2) * i
                assert torch.equal(tensor.cpu(), expected)

    @pytest.mark.parametrize("on_cpu", [True, False], ids=["cpu", "cuda"])
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_broadcast_tensor_dict(self, collective_group, on_cpu, async_op):
        if not on_cpu and not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test on CPU-only environment.")
        results = self._run_collective_test(
            collective_group, "test_broadcast_tensor_dict", on_cpu, async_op
        )
        for res_dict in results:
            assert isinstance(res_dict, dict)
            for i, key in enumerate(sorted(res_dict.keys())):
                assert key == f"t{i}"
                expected = torch.ones(2, 2) * i
                assert torch.equal(res_dict[key].cpu(), expected)

    @pytest.mark.parametrize("on_cpu", [True, False], ids=["cpu", "cuda"])
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_broadcast_tensor_dataclass(self, collective_group, on_cpu, async_op):
        if not on_cpu and not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test on CPU-only environment.")
        results = self._run_collective_test(
            collective_group, "test_broadcast_tensor_dataclass", on_cpu, async_op
        )
        expected_payload = torch.ones(2, 2) * 7
        for res in results:
            assert isinstance(res, TensorMessage)
            assert res.id == 0
            assert res.note == "broadcast from rank 0"
            assert torch.equal(res.payload.cpu(), expected_payload)

    @pytest.mark.parametrize("on_cpu", [True, False], ids=["cpu", "cuda"])
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_broadcast_tensor_list_dataclass(self, collective_group, on_cpu, async_op):
        if not on_cpu and not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test on CPU-only environment.")
        results = self._run_collective_test(
            collective_group,
            "test_broadcast_tensor_list_dataclass",
            on_cpu,
            async_op,
        )
        for res in results:
            assert isinstance(res, TensorListMessage)
            assert res.id == 0
            assert res.note == "broadcast list from rank 0"
            assert len(res.payload_list) == 4
            for i, t in enumerate(res.payload_list):
                assert torch.equal(t.cpu(), torch.ones(2, 2) * i)

    @pytest.mark.parametrize("on_cpu", [True, False], ids=["cpu", "cuda"])
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_broadcast_tensor_dict_dataclass(self, collective_group, on_cpu, async_op):
        if not on_cpu and not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test on CPU-only environment.")
        results = self._run_collective_test(
            collective_group,
            "test_broadcast_tensor_dict_dataclass",
            on_cpu,
            async_op,
        )
        for res in results:
            assert isinstance(res, TensorDictMessage)
            assert res.id == 0
            assert res.note == "broadcast dict from rank 0"
            assert list(res.payload_dict.keys()) == ["t0", "t1", "t2", "t3"]
            for i, key in enumerate(sorted(res.payload_dict.keys())):
                assert torch.equal(res.payload_dict[key].cpu(), torch.ones(2, 2) * i)

    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_cross_group_broadcast_object(self, cross_collective_groups, async_op):
        group_a, group_b, group_a_size, group_b_size = cross_collective_groups
        groups = [
            ("collective_group_a", list(range(group_a_size))),
            ("collective_group_b", list(range(group_b_size))),
        ]
        handle_a = group_a.test_cross_group_broadcast_object(groups, async_op)
        handle_b = group_b.test_cross_group_broadcast_object(groups, async_op)
        results = handle_a.wait() + handle_b.wait()
        for res in results:
            assert res == {"message": "Hello from cross-group src", "rank": 0}

    @pytest.mark.parametrize("on_cpu", [True, False], ids=["cpu", "cuda"])
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_cross_group_broadcast_tensor(
        self, cross_collective_groups, on_cpu, async_op
    ):
        if not on_cpu and not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test on CPU-only environment.")
        group_a, group_b, group_a_size, group_b_size = cross_collective_groups
        groups = [
            ("collective_group_a", list(range(group_a_size))),
            ("collective_group_b", list(range(group_b_size))),
        ]
        handle_a = group_a.test_cross_group_broadcast_tensor(groups, on_cpu, async_op)
        handle_b = group_b.test_cross_group_broadcast_tensor(groups, on_cpu, async_op)
        results = handle_a.wait() + handle_b.wait()
        expected = torch.ones(2, 2) * 11
        for res in results:
            assert torch.equal(res.cpu(), expected)

    @pytest.mark.parametrize("on_cpu", [True, False], ids=["cpu", "cuda"])
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_cross_group_broadcast_tensor_dataclass(
        self, cross_collective_groups, on_cpu, async_op
    ):
        if not on_cpu and not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test on CPU-only environment.")
        group_a, group_b, group_a_size, group_b_size = cross_collective_groups
        groups = [
            ("collective_group_a", list(range(group_a_size))),
            ("collective_group_b", list(range(group_b_size))),
        ]
        handle_a = group_a.test_cross_group_broadcast_tensor_dataclass(
            groups, on_cpu, async_op
        )
        handle_b = group_b.test_cross_group_broadcast_tensor_dataclass(
            groups, on_cpu, async_op
        )
        results = handle_a.wait() + handle_b.wait()
        expected_payload = torch.ones(2, 2) * 11
        for res in results:
            assert isinstance(res, TensorMessage)
            assert res.id == 11
            assert res.note == "cross-group broadcast dataclass"
            assert torch.equal(res.payload.cpu(), expected_payload)

    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_broadcast_src_ignores_group_order(self, cross_collective_groups, async_op):
        group_a, group_b, group_a_size, group_b_size = cross_collective_groups
        groups_a_first = [
            ("collective_group_a", list(range(group_a_size))),
            ("collective_group_b", list(range(group_b_size))),
        ]
        groups_b_first = [
            ("collective_group_b", list(range(group_b_size))),
            ("collective_group_a", list(range(group_a_size))),
        ]
        src_addr = WorkerAddress("collective_group_a", ranks=0)
        handle_a = group_a.test_broadcast_object_with_src(
            groups_a_first, src_addr, async_op
        )
        handle_b = group_b.test_broadcast_object_with_src(
            groups_a_first, src_addr, async_op
        )
        results_a = handle_a.wait() + handle_b.wait()
        handle_a = group_a.test_broadcast_object_with_src(
            groups_b_first, src_addr, async_op
        )
        handle_b = group_b.test_broadcast_object_with_src(
            groups_b_first, src_addr, async_op
        )
        results_b = handle_a.wait() + handle_b.wait()
        assert results_a == results_b

    @pytest.mark.parametrize("on_cpu", [True, False], ids=["cpu", "cuda"])
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_broadcast_src_tensor_order_independent(
        self, cross_collective_groups, on_cpu, async_op
    ):
        if not on_cpu and not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test on CPU-only environment.")
        group_a, group_b, group_a_size, group_b_size = cross_collective_groups
        groups_a_first = [
            ("collective_group_a", list(range(group_a_size))),
            ("collective_group_b", list(range(group_b_size))),
        ]
        groups_b_first = [
            ("collective_group_b", list(range(group_b_size))),
            ("collective_group_a", list(range(group_a_size))),
        ]
        src_addr = WorkerAddress("collective_group_a", ranks=0)
        handle_a = group_a.test_broadcast_tensor_with_src(
            groups_a_first, src_addr, on_cpu, async_op
        )
        handle_b = group_b.test_broadcast_tensor_with_src(
            groups_a_first, src_addr, on_cpu, async_op
        )
        results_a = handle_a.wait() + handle_b.wait()
        handle_a = group_a.test_broadcast_tensor_with_src(
            groups_b_first, src_addr, on_cpu, async_op
        )
        handle_b = group_b.test_broadcast_tensor_with_src(
            groups_b_first, src_addr, on_cpu, async_op
        )
        results_b = handle_a.wait() + handle_b.wait()
        expected = torch.ones(2, 2) * 13
        for res in results_a + results_b:
            assert torch.equal(res.cpu(), expected)

    @pytest.mark.parametrize("on_cpu", [True, False], ids=["cpu", "cuda"])
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_broadcast_tensor_dataclass_with_src(
        self, cross_collective_groups, on_cpu, async_op
    ):
        if not on_cpu and not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test on CPU-only environment.")
        group_a, group_b, group_a_size, group_b_size = cross_collective_groups
        groups_a_first = [
            ("collective_group_a", list(range(group_a_size))),
            ("collective_group_b", list(range(group_b_size))),
        ]
        groups_b_first = [
            ("collective_group_b", list(range(group_b_size))),
            ("collective_group_a", list(range(group_a_size))),
        ]
        src_addr = WorkerAddress("collective_group_a", ranks=0)
        handle_a = group_a.test_broadcast_tensor_dataclass_with_src(
            groups_a_first, src_addr, on_cpu, async_op
        )
        handle_b = group_b.test_broadcast_tensor_dataclass_with_src(
            groups_a_first, src_addr, on_cpu, async_op
        )
        results_a = handle_a.wait() + handle_b.wait()
        handle_a = group_a.test_broadcast_tensor_dataclass_with_src(
            groups_b_first, src_addr, on_cpu, async_op
        )
        handle_b = group_b.test_broadcast_tensor_dataclass_with_src(
            groups_b_first, src_addr, on_cpu, async_op
        )
        results_b = handle_a.wait() + handle_b.wait()
        expected_payload = torch.ones(2, 2) * 13
        for res in results_a + results_b:
            assert isinstance(res, TensorMessage)
            assert res.id == 13
            assert res.note == "broadcast with src"
            assert torch.equal(res.payload.cpu(), expected_payload)

    @pytest.mark.parametrize("on_cpu", [True, False], ids=["cpu", "cuda"])
    def test_collective_asyncio_broadcast(self, collective_group, on_cpu):
        if not on_cpu and not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test on CPU-only environment.")
        results = self._run_collective_test(
            collective_group, "test_broadcast_tensor_asyncio", on_cpu
        )
        expected = torch.ones(3, 3) * 5
        for res in results:
            assert torch.equal(res.cpu(), expected)

    @pytest.mark.parametrize("on_cpu", [True, False], ids=["cpu", "cuda"])
    def test_collective_asyncio_broadcast_tensor_dataclass(
        self, collective_group, on_cpu
    ):
        if not on_cpu and not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test on CPU-only environment.")
        results = self._run_collective_test(
            collective_group, "test_broadcast_tensor_dataclass_asyncio", on_cpu
        )
        expected_payload = torch.ones(3, 3) * 5
        for res in results:
            assert isinstance(res, TensorMessage)
            assert res.id == 5
            assert res.note == "async broadcast from rank 0"
            assert torch.equal(res.payload.cpu(), expected_payload)

    @pytest.mark.parametrize("on_cpu", [True, False], ids=["cpu", "cuda"])
    def test_collective_asyncio_cross_group_broadcast(
        self, cross_collective_groups, on_cpu
    ):
        if not on_cpu and not torch.cuda.is_available():
            pytest.skip("Skipping CUDA test on CPU-only environment.")
        group_a, group_b, group_a_size, group_b_size = cross_collective_groups
        groups = [
            ("collective_group_a", list(range(group_a_size))),
            ("collective_group_b", list(range(group_b_size))),
        ]
        handle_a = group_a.test_cross_group_broadcast_tensor_asyncio(groups, on_cpu)
        handle_b = group_b.test_cross_group_broadcast_tensor_asyncio(groups, on_cpu)
        results = handle_a.wait() + handle_b.wait()
        expected = torch.ones(3, 3) * 9
        for res in results:
            assert torch.equal(res.cpu(), expected)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
