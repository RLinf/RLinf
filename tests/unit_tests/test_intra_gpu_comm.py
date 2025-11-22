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

from rlinf.scheduler import Cluster, PackedPlacementStrategy, Worker

SENDER_GROUP_NAME = "sender_ipc_worker_group"
RECEIVER_GROUP_NAME = "receiver_ipc_worker_group"

# --- Helper Functions ---


def get_device(rank=0):
    """Returns the appropriate torch device, setting it for the current process."""
    if torch.cuda.is_available():
        # In a real worker, LOCAL_RANK would be set. We simulate it.
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


# --- Worker Definitions ---
class SenderWorker(Worker):
    """Worker responsible for sending data in IPC tests."""

    def __init__(self):
        super().__init__()
        get_device()

    def async_wait(self, work):
        """Waits for an async operation to complete."""

        async def wait(work):
            if work:
                return await work.async_wait()

        return asyncio.run(wait(work))

    def send_single_tensor(self, on_cpu, async_op, group_name):
        """Sends a single tensor using send_tensor."""
        device = "cpu" if on_cpu else get_device()
        tensor = torch.ones(3, 3, device=device) * self._rank
        is_async = async_op > 0
        work = self.send_tensor(
            tensor, group_name, dst_rank=self._rank, async_op=is_async
        )
        if is_async and work:
            if async_op == 1:
                work.wait()
            else:
                self.async_wait(work)
        return True

    def send_tensor_list(self, on_cpu, async_op, group_name):
        """Sends a list of tensors using send."""
        device = "cpu" if on_cpu else get_device()
        tensors = [torch.ones(2, 2, device=device) * (self._rank + i) for i in range(3)]
        is_async = async_op > 0
        work = self.send(tensors, group_name, dst_rank=self._rank, async_op=is_async)
        if is_async and work:
            if async_op == 1:
                work.wait()
            else:
                self.async_wait(work)
        return True

    def send_mixed_gpu_tensor_list(self, async_op, group_name):
        """Sends a list of tensors from different GPUs."""
        num_gpus = torch.cuda.device_count()
        tensors = [
            torch.ones(2, 2, device=get_device(i % num_gpus)) * (self._rank + i)
            for i in range(num_gpus)
        ]
        is_async = async_op > 0
        work = self.send(tensors, group_name, dst_rank=self._rank, async_op=is_async)
        if is_async and work:
            if async_op == 1:
                work.wait()
            else:
                self.async_wait(work)
        return True


class ReceiverWorker(Worker):
    """Worker responsible for receiving data in IPC tests."""

    def __init__(self):
        super().__init__()
        get_device()

    def async_wait(self, work):
        """Waits for an async operation to complete."""

        async def wait(work):
            if work:
                return await work.async_wait()

        return asyncio.run(wait(work))

    def recv_single_tensor(self, on_cpu, async_op, group_name):
        """Receives a single tensor using recv_tensor."""
        device = "cpu" if on_cpu else get_device()
        tensor = torch.empty(3, 3, device=device)
        is_async = async_op > 0
        work = self.recv_tensor(
            tensor, group_name, src_rank=self._rank, async_op=is_async
        )
        if is_async and work:
            if async_op == 1:
                work.wait()
            else:
                self.async_wait(work)
        return tensor

    def recv_tensor_list(self, async_op, group_name):
        """Receives a list of tensors using recv."""
        is_async = async_op > 0
        work = self.recv(group_name, src_rank=self._rank, async_op=is_async)
        if is_async and work:
            if async_op == 1:
                return work.wait()
            else:
                return self.async_wait(work)
        return work


# --- Pytest Setup ---


@pytest.fixture(scope="module")
def cluster():
    """Provides a Cluster instance for the tests."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        pytest.skip("IPC/Uncertain Peer tests require at least 1 CUDA GPU.")
    # Use all GPUs on one node to test same-node communication
    return Cluster(num_nodes=1)


def create_worker_groups(cluster, sender_gpus, receiver_gpus):
    """Helper to create worker groups with specific GPU assignments."""
    sender_placement = PackedPlacementStrategy(
        start_hardware_rank=sender_gpus[0], end_hardware_rank=sender_gpus[-1]
    )
    sender_group = SenderWorker.create_group().launch(
        cluster=cluster,
        name=SENDER_GROUP_NAME,
        placement_strategy=sender_placement,
    )

    receiver_placement = PackedPlacementStrategy(
        start_hardware_rank=receiver_gpus[0], end_hardware_rank=receiver_gpus[-1]
    )
    receiver_group = ReceiverWorker.create_group().launch(
        cluster=cluster,
        name=RECEIVER_GROUP_NAME,
        placement_strategy=receiver_placement,
    )
    return sender_group, receiver_group


@pytest.fixture(scope="class")
def single_shared_gpu_groups(cluster):
    """Workers on the exact same single GPU."""
    global SENDER_GROUP_NAME, RECEIVER_GROUP_NAME
    SENDER_GROUP_NAME = "sender_ipc_worker_group_single"
    RECEIVER_GROUP_NAME = "receiver_ipc_worker_group_single"
    yield create_worker_groups(cluster, sender_gpus=[0], receiver_gpus=[0])


@pytest.fixture(scope="class")
def multi_shared_gpu_groups(cluster):
    """Workers with access to the same pool of multiple GPUs."""
    global SENDER_GROUP_NAME, RECEIVER_GROUP_NAME
    SENDER_GROUP_NAME = "sender_ipc_worker_group_multi"
    RECEIVER_GROUP_NAME = "receiver_ipc_worker_group_multi"
    if torch.cuda.device_count() < 2:
        pytest.skip("Multi-GPU tests require at least 2 GPUs.")
    all_gpus = list(range(torch.cuda.device_count()))
    yield create_worker_groups(cluster, sender_gpus=all_gpus, receiver_gpus=all_gpus)


# --- Test Class ---


class TestSameDeviceCommunication:
    """
    Tests for send/recv when sender and receiver might share GPU resources,
    triggering IPC or uncertain peer logic.
    """

    def _run_test(
        self, worker_groups, sender_method, receiver_method, sender_args, receiver_args
    ):
        sender_group, receiver_group = worker_groups
        sender_results = getattr(sender_group, sender_method)(*sender_args)
        receiver_results = getattr(receiver_group, receiver_method)(*receiver_args)
        # Wait for both to complete
        results = sender_results.wait()
        results = receiver_results.wait()
        # Return only the receiver's result for verification
        return results

    @pytest.mark.parametrize("async_op", [0, 1, 2], ids=["sync", "async", "asyncio"])
    def test_single_tensor_on_single_shared_gpu(
        self, single_shared_gpu_groups, async_op
    ):
        """Tests send_tensor/recv_tensor on one shared GPU (triggers direct IPC)."""
        result = self._run_test(
            single_shared_gpu_groups,
            "send_single_tensor",
            "recv_single_tensor",
            (False, async_op, RECEIVER_GROUP_NAME),
            (False, async_op, SENDER_GROUP_NAME),
        )
        result = result[0]
        expected = torch.ones(3, 3) * 0  # Sender rank is 0
        assert torch.equal(result.cpu(), expected)

    @pytest.mark.parametrize("async_op", [0, 1, 2], ids=["sync", "async", "asyncio"])
    def test_tensor_list_on_single_shared_gpu(self, single_shared_gpu_groups, async_op):
        """Tests send/recv for a tensor list on one shared GPU (triggers direct IPC)."""
        results = self._run_test(
            single_shared_gpu_groups,
            "send_tensor_list",
            "recv_tensor_list",
            (False, async_op, RECEIVER_GROUP_NAME),
            (async_op, SENDER_GROUP_NAME),
        )
        results = results[0]
        assert isinstance(results, list)
        for i, tensor in enumerate(results):
            expected = torch.ones(2, 2) * i  # Sender rank 0 + i
            assert torch.equal(tensor.cpu(), expected)

    @pytest.mark.parametrize("async_op", [0, 1, 2], ids=["sync", "async", "asyncio"])
    def test_single_tensor_on_multi_shared_gpu(self, multi_shared_gpu_groups, async_op):
        """Tests send_tensor/recv_tensor with overlapping GPUs (triggers uncertain peer)."""
        result = self._run_test(
            multi_shared_gpu_groups,
            "send_single_tensor",
            "recv_single_tensor",
            (False, async_op, RECEIVER_GROUP_NAME),
            (False, async_op, SENDER_GROUP_NAME),
        )
        result = result[0]
        expected = torch.ones(3, 3) * 0  # Sender rank is 0
        assert torch.equal(result.cpu(), expected)

    @pytest.mark.parametrize("async_op", [0, 1, 2], ids=["sync", "async", "asyncio"])
    def test_mixed_gpu_tensor_list_on_multi_shared_gpu(
        self, multi_shared_gpu_groups, async_op
    ):
        """Tests send/recv with a list of tensors on different GPUs from a shared pool."""
        results = self._run_test(
            multi_shared_gpu_groups,
            "send_mixed_gpu_tensor_list",
            "recv_tensor_list",
            (
                async_op,
                RECEIVER_GROUP_NAME,
            ),
            (
                async_op,
                SENDER_GROUP_NAME,
            ),
        )
        assert isinstance(results, list)
        num_gpus = torch.cuda.device_count()
        assert len(results) == num_gpus
        for i, tensor in enumerate(results):
            tensor = tensor[0]
            expected = torch.ones(2, 2) * i  # Sender rank 0 + i
            assert torch.equal(tensor.cpu(), expected)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
