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
import os

import pytest
import torch

from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker

SENDER_GROUP_NAME = "sender_worker_group"
RECEIVER_GROUP_NAME = "receiver_worker_group"

# --- Helper Functions ---


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
            work = self.send(data, RECEIVER_GROUP_NAME, peer_rank, async_op=async_op)

        if async_op:
            work.wait()
        return True

    def _send_data_asyncio(self, data_factory, use_send_tensor=False):
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

        return asyncio.run(_send())

    # Sync Tests
    def test_send_object(self, async_op=False):
        return self._send_data({"message": f"Hello from rank {self._rank}"}, async_op)

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

    # Asyncio Tests
    def test_send_tensor_asyncio(self, on_cpu):
        device = "cpu" if on_cpu else get_device()
        return self._send_data_asyncio(
            lambda: torch.ones(4, 4, device=device) * self._rank
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
            work = self.recv(SENDER_GROUP_NAME, peer_rank, async_op=async_op)
            if async_op:
                return work.wait()
            return work

    def _recv_data_asyncio(self, recv_tensor_inplace_shape=None):
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

        return asyncio.run(_recv())

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

    def test_recv_tensor(self, async_op=False):
        return self._recv_data(async_op)

    def test_recv_tensor_list(self, async_op=False):
        return self._recv_data(async_op)

    def test_recv_tensor_dict(self, async_op=False):
        return self._recv_data(async_op)

    def test_recv_tensor_inplace(self, on_cpu, async_op=False):
        return self._recv_data(async_op, recv_tensor_inplace_shape=(on_cpu, (3, 3)))

    # Asyncio Tests
    def test_recv_tensor_asyncio(self, on_cpu):
        return self._recv_data_asyncio()


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
    # No explicit cleanup needed, Ray handles actor termination on shutdown.


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


if __name__ == "__main__":
    pytest.main(["-v", __file__])
