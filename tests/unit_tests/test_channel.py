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

# ruff: noqa: D103
import asyncio
import logging
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, fields, is_dataclass
from functools import partial
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import Mock

import pytest
import ray
import torch

from rlinf.scheduler import (
    Channel,
    Cluster,
    NodePlacementStrategy,
    PackedPlacementStrategy,
    Worker,
)
from rlinf.scheduler.channel.channel import DEFAULT_KEY
from rlinf.scheduler.channel.channel_worker import ChannelWorker, PeekQueue
from rlinf.scheduler.channel.hooks import (
    COLLECTOR_REGISTRY,
    DISPATCHER_REGISTRY,
    ChannelContext,
    Collector,
    Dispatcher,
    LeastLoadedDispatcher,
    LeastLoadedStealingDispatcher,
    RoundRobinDispatcher,
    SharedDispatcher,
    iter_collected,
    register_collector,
    register_dispatcher,
    resolve_collector,
    resolve_dispatcher,
)
from rlinf.scheduler.cluster import (
    get_group_world_size,
    resolve_colocation_node_rank,
    resolve_group_names,
    resolve_group_sizes,
    resolve_worker_names,
)
from rlinf.scheduler.collective import AsyncChannelCommWork, AsyncChannelWork
from rlinf.scheduler.collective.async_work import (
    AsyncFuncWork,
    PeerCommunicationError,
)
from rlinf.scheduler.collective.collective_group import CollectiveGroup
from rlinf.scheduler.worker import WorkerAddress, WorkerGroup

# --- Constants ---
PRODUCER_GROUP_NAME = "producer_group"
CONSUMER_GROUP_NAME = "consumer_group"
TEST_CHANNEL_NAME = "my_test_channel"
group_count = 0
channel_count = 0


@dataclass
class TensorMessage:
    """Simple dataclass with a tensor field for testing direct tensor send/recv/broadcast."""

    id: int
    payload: torch.Tensor
    note: str


@dataclass
class TensorListMessage:
    """Dataclass with a list of tensors for testing channel put/get."""

    id: int
    payload_list: list
    note: str


@dataclass
class TensorDictMessage:
    """Dataclass with a dict of tensors for testing channel put/get."""

    id: int
    payload_dict: dict
    note: str


@dataclass
class PlainMessage:
    """Plain dataclass without tensor fields (serialized as Python object)."""

    id: int
    name: str
    value: float


def accelerator_is_available():
    """Return whether the Worker accelerator backend is available."""
    return (
        Worker.torch_platform is not None
        and hasattr(Worker.torch_platform, "is_available")
        and Worker.torch_platform.is_available()
    )


def get_device():
    """Returns the appropriate torch device."""
    if accelerator_is_available():
        return torch.device(
            f"{Worker.torch_device_type}:{Worker.torch_platform.current_device()}"
        )
    return torch.device("cpu")


# --- Test Worker Definitions ---
class ProducerWorker(Worker):
    """Worker responsible for creating channels and putting items."""

    def __init__(self):
        super().__init__()

    def put_item(
        self,
        channel: Channel,
        item: Any,
        weight: int,
        maxsize: int,
        async_op: bool,
        key: Optional[str] = None,
    ):
        if key:
            put_work = channel.put(item, weight, key=key, async_op=async_op)
        else:
            put_work = channel.put(item, weight, async_op=async_op)
        if async_op:
            put_work.wait()
        return True

    async def put_item_asyncio(
        self, channel: Channel, item: Any, weight: int, maxsize: int
    ):
        put_work = channel.put(item, weight, async_op=True)
        if put_work:
            await put_work.async_wait()
        return True

    async def put_get_ood(self, channel: Channel):
        channel.put(key="q2", item="World")

    def put_nowait(self, channel: Channel, item: Any):
        channel.put_nowait(item, key="nowait")

    def test_memory(self, channel: Channel):
        large_tensor = torch.randn(512, 1024, 1024, device=get_device())
        channel.put(large_tensor)
        channel.put(large_tensor, async_op=True).wait()
        channel.put_nowait(large_tensor)
        channel.put(large_tensor, weight=1)

    async def stress(self, channel: Channel, num_items: int):
        data = []
        for i in range(num_items):
            channel.put(item=i, key="stress_key", async_op=True)

        for i in range(num_items):
            data.append(await channel.get(async_op=True, key="stress_key").async_wait())

        return data

    async def ordered_puts_then_get_nowait(
        self, channel: Channel, key: str, items: list
    ):
        """Issue same-key puts without waiting for them, await only the last
        one, then get_nowait every item. Returns the items in issue order."""
        works = [channel.put(item, key=key, async_op=True) for item in items]
        await works[-1].async_wait()
        return [channel.get_nowait(key=key) for _ in items]

    async def stress_multiple_queues(self, channel: Channel, num_items: int):
        works = []
        for i in range(num_items):
            channel.put(item=i, key=f"stress_key{i}", async_op=True)

        for i in range(num_items):
            works.append(channel.get(async_op=True, key=f"stress_key{i}"))

        works = [work.async_wait() for work in works]
        return await asyncio.gather(*works)

    def create_with_affinity(self, channel_name: str):
        channel = self.create_channel(
            channel_name=channel_name,
            node_rank=0,
        )
        channel.put("affinity_item", 1)
        return True

    def get_qsize(self, channel: Channel):
        return channel.qsize()

    def put_mixed_tensor_list(
        self, channel: Channel, async_op: bool, key: Optional[str] = None
    ):
        if not accelerator_is_available():
            raise RuntimeError("Accelerator is required for mixed tensor tests.")
        mixed_item = [
            torch.ones(2, 2, device="cpu") * 1,
            torch.ones(2, 2, device=get_device()) * 2,
            torch.ones(2, 2, device="cpu") * 3,
        ]
        put_work = channel.put(mixed_item, async_op=async_op, key=key)
        if async_op:
            put_work.wait()
        return True

    def put_mixed_tensor_dict(
        self, channel: Channel, async_op: bool, key: Optional[str] = None
    ):
        if not accelerator_is_available():
            raise RuntimeError("Accelerator is required for mixed tensor tests.")
        mixed_item = {
            "cpu_a": torch.ones(2, 2, device="cpu") * 1,
            "cuda_b": torch.ones(2, 2, device=get_device()) * 2,
            "cpu_c": torch.ones(2, 2, device="cpu") * 3,
        }
        put_work = channel.put(mixed_item, async_op=async_op, key=key)
        if async_op:
            put_work.wait()
        return True

    def put_mixed_tensor_list_dataclass(
        self, channel: Channel, async_op: bool, key: Optional[str] = None
    ):
        if not accelerator_is_available():
            raise RuntimeError("Accelerator is required for mixed tensor tests.")
        item = TensorListMessage(
            id=10,
            payload_list=[
                torch.ones(2, 2, device="cpu") * 1,
                torch.ones(2, 2, device=get_device()) * 2,
                torch.ones(2, 2, device="cpu") * 3,
            ],
            note="channel mixed list dataclass",
        )
        put_work = channel.put(item, async_op=async_op, key=key)
        if async_op:
            put_work.wait()
        return True

    def put_mixed_tensor_dict_dataclass(
        self, channel: Channel, async_op: bool, key: Optional[str] = None
    ):
        if not accelerator_is_available():
            raise RuntimeError("Accelerator is required for mixed tensor tests.")
        item = TensorDictMessage(
            id=20,
            payload_dict={
                "cpu_a": torch.ones(2, 2, device="cpu") * 1,
                "cuda_b": torch.ones(2, 2, device=get_device()) * 2,
                "cpu_c": torch.ones(2, 2, device="cpu") * 3,
            },
            note="channel mixed dict dataclass",
        )
        put_work = channel.put(item, async_op=async_op, key=key)
        if async_op:
            put_work.wait()
        return True


class ConsumerWorker(Worker):
    """Worker responsible for connecting to channels and getting items."""

    def get_item(self, channel: Channel, async_op: bool, key: Optional[str] = None):
        if key:
            result = channel.get(key=key, async_op=async_op)
        else:
            result = channel.get(async_op=async_op)
        if async_op:
            return result.wait()
        return result

    async def get_item_asyncio(self, channel: Channel):
        result = channel.get(async_op=True)
        if result:
            return await result.async_wait()
        return None

    def get_batch(self, channel: Channel, batch_weight: int, async_op: bool):
        result = channel.get_batch(target_weight=batch_weight, async_op=async_op)
        if async_op:
            return result.wait()
        return result

    async def get_batch_asyncio(self, channel: Channel, batch_weight: int):
        result = channel.get_batch(target_weight=batch_weight, async_op=True)
        if result:
            return await result.async_wait()
        return None

    async def put_get_ood(self, channel: Channel):
        channel.put(key="q1", item="Hello")
        handle2 = channel.get(key="q2", async_op=True)
        handle1 = channel.get(key="q1", async_op=True)
        data1 = await handle1.async_wait()
        data2 = await handle2.async_wait()
        return data1, data2

    def get_nowait(self, channel: Channel):
        try:
            data = channel.get_nowait(key="nowait")
        except asyncio.QueueEmpty:
            data = None
        return data

    def get_qsize(self, channel: Channel):
        return channel.qsize()

    def test_memory(self, channel: Channel):
        channel.get()
        channel.get(async_op=True).wait()
        while channel.empty():
            pass
        channel.get_nowait()
        channel.get_batch(target_weight=1)

    def is_empty(self, channel: Channel):
        return channel.empty()

    def is_full(self, channel: Channel):
        return channel.full()

    def get_cluster_node_rank(self):
        """Get the cluster node rank of this worker."""
        return self._cluster_node_rank

    async def test_async_wait_beyond_executor_pool(
        self,
        channel: Channel,
        blocked_keys: list,
        free_key: str,
        timeout: float,
    ):
        """Hold len(blocked_keys) pending async_wait() calls, then fetch an item
        put on a different key. Returns (free_timed_out, free_item,
        free_elapsed, blocked_items).

        free_elapsed is measured from just before the free_key put, so it
        covers the full wait latency while the blocked waits are pending. On
        timeout the blocked keys are still put so parked waits are released
        and the worker method returns instead of raising; the caller asserts
        on the returned values.
        """

        async def get_task(key: str):
            work = channel.get(async_op=True, key=key)
            return await work.async_wait()

        blocked_tasks = [asyncio.ensure_future(get_task(key)) for key in blocked_keys]
        free_task = asyncio.ensure_future(get_task(free_key))
        # Let every get enter async_wait before any data is put.
        await asyncio.sleep(1.0)

        start = time.time()
        channel.put("free_item", key=free_key)
        free_timed_out = False
        try:
            await asyncio.wait_for(asyncio.shield(free_task), timeout)
        except asyncio.TimeoutError:
            free_timed_out = True
        free_elapsed = time.time() - start

        # Unblock the parked waits even when the free get timed out. The free
        # get finishes last on a timeout: it only proceeds once the blocked
        # waits have drained the resource they are serialized behind.
        for i, key in enumerate(blocked_keys):
            channel.put(i, key=key)
        results = await asyncio.gather(*blocked_tasks, free_task)
        blocked_items = results[:-1]
        free_item = results[-1]
        return free_timed_out, free_item, free_elapsed, blocked_items

    async def test_async_wait_cancelled(
        self, channel: Channel, key: str, followup_key: str
    ):
        """Cancel an async_wait() mid-wait, put the data afterwards, and verify
        no error surfaces and the channel still serves a fresh put/get.

        Returns the event-loop exceptions the default handler would have
        logged (e.g. InvalidStateError from a done future), for the caller
        to assert on."""
        loop = asyncio.get_running_loop()
        loop_errors = []
        original_handler = loop.get_exception_handler()

        def record_error(loop, context):
            loop_errors.append(context.get("exception") or context.get("message"))

        loop.set_exception_handler(record_error)
        try:
            work = channel.get(async_op=True, key=key)
            task = asyncio.ensure_future(work.async_wait())
            # Let the get enter async_wait before cancelling it.
            await asyncio.sleep(1.0)
            task.cancel()

            channel.put("late_item", key=key)
            await asyncio.sleep(1.0)

            channel.put("after_cancel", key=followup_key)
            item = channel.get(key=followup_key)
        finally:
            loop.set_exception_handler(original_handler)
        return item, loop_errors

    async def test_async_wait_yields_control(
        self, channel: Channel, key: str = "async_wait_yields_test"
    ):
        """Run get(async_op=True) and await async_wait() concurrently with another
        asyncio task. Assert the other task ran while waiting, proving async_wait()
        yields control to the event loop. Returns (yield_count, received_item)."""

        async def get_task():
            work = channel.get(async_op=True, key=key)
            return await work.async_wait()

        async def yield_check_task():
            count = 0
            for _ in range(30):
                count += 1
                await asyncio.sleep(0.01)
            return count

        async def main():
            self.get_fut = asyncio.create_task(get_task())
            return await yield_check_task()

        return await main()


# --- Pytest Fixtures ---
@pytest.fixture(scope="module")
def cluster():
    c = Cluster(num_nodes=1)
    yield c


@pytest.fixture(scope="module")
def worker_groups(cluster):
    if accelerator_is_available():
        placement = PackedPlacementStrategy(start_hardware_rank=0, end_hardware_rank=0)
    else:
        placement = NodePlacementStrategy([0])
    global \
        group_count, \
        channel_count, \
        PRODUCER_GROUP_NAME, \
        CONSUMER_GROUP_NAME, \
        TEST_CHANNEL_NAME
    group_count += 1
    channel_count += 1
    PRODUCER_GROUP_NAME = f"producer_group_{group_count}"
    CONSUMER_GROUP_NAME = f"consumer_group_{group_count}"
    TEST_CHANNEL_NAME = f"my_test_channel_{channel_count}"
    producer = ProducerWorker.create_group().launch(
        cluster, name=PRODUCER_GROUP_NAME, placement_strategy=placement
    )
    consumer = ConsumerWorker.create_group().launch(
        cluster, name=CONSUMER_GROUP_NAME, placement_strategy=placement
    )
    return producer, consumer


# Number of simulated nodes (channel workers) for distributed testing
NUM_SIMULATED_NODES = 3


# --- Distributed Channel Class ---
class DistributedChannel(Channel):
    """A Channel subclass that creates multiple channel workers on the same node for testing."""

    @classmethod
    def create(
        cls,
        name: str,
        maxsize: int = 0,
        distributed: bool = True,  # Always distributed for these tests
        node_rank: int = 0,
        local: bool = False,
    ) -> "DistributedChannel":
        """Create a distributed channel with multiple workers on the same node.

        This simulates a multi-node setup by launching multiple channel workers
        on node 0, but treating them as if they were on different nodes.
        """
        from rlinf.scheduler.channel.channel_worker import LocalChannel

        cluster = Cluster()
        channel = cls()
        if local:
            local_channel = LocalChannel(maxsize=maxsize)
            channel._initialize(
                name,
                None,
                None,
                Worker.current_worker,
                local_channel=local_channel,
                maxsize=maxsize,
            )
            return channel

        # Launch multiple channel workers on the same node (node 0)
        # This simulates having multiple nodes, but all on the same physical node
        placement = NodePlacementStrategy(node_ranks=[0] * NUM_SIMULATED_NODES)
        try:
            channel_worker_group = ChannelWorker.create_group(maxsize=maxsize).launch(
                cluster=cluster,
                name=name,
                placement_strategy=placement,
                max_concurrency=2**31 - 1,
            )
        except ValueError:
            Worker.logger.warning(f"Channel {name} already exists, connecting to it.")
            return cls.connect(name, Worker.current_worker)

        # Distributed channel actors
        import ray.actor

        channel_actors: dict[int, ray.actor.ActorHandle] = {
            worker.rank: worker.worker
            for worker in channel_worker_group.worker_info_list
        }

        # Verify we actually created multiple channel workers
        assert len(channel_actors) == NUM_SIMULATED_NODES, (
            f"DistributedChannel.create() should create {NUM_SIMULATED_NODES} channel workers, "
            f"but created {len(channel_actors)}"
        )

        channel._initialize(
            channel_name=name,
            channel_worker_group=channel_worker_group,
            channel_worker_actor=channel_actors[0],
            current_worker=Worker.current_worker,
            maxsize=maxsize,
            channel_actors=channel_actors,
        )

        # Verify the channel is marked as distributed after initialization
        assert channel._distributed, (
            "DistributedChannel should be marked as distributed after initialization"
        )
        assert len(channel._channel_actors_by_rank) == NUM_SIMULATED_NODES, (
            f"DistributedChannel should have {NUM_SIMULATED_NODES} channel workers after initialization, "
            f"but has {len(channel._channel_actors_by_rank)}"
        )

        return channel


@pytest.fixture(scope="module")
def regular_channel():
    """Create a regular (non-distributed) channel once per module."""
    return Channel.create(f"{TEST_CHANNEL_NAME}_regular_{uuid.uuid4().hex[:8]}")


@pytest.fixture(scope="module")
def distributed_channel():
    """Create a distributed channel once per module."""
    channel_name = f"distributed_test_channel_{uuid.uuid4().hex[:8]}"
    dist_channel = DistributedChannel.create(channel_name)
    # Verify it's actually distributed (has multiple channel workers)
    assert dist_channel._distributed, (
        "DistributedChannel should be marked as distributed"
    )
    assert len(dist_channel._channel_actors_by_rank) == NUM_SIMULATED_NODES, (
        f"DistributedChannel should have {NUM_SIMULATED_NODES} channel workers, "
        f"but has {len(dist_channel._channel_actors_by_rank)}"
    )
    return dist_channel


@pytest.fixture
def channel_type(request):
    """Fixture that provides the channel type parameter."""
    return request.param if hasattr(request, "param") else "regular"


@pytest.fixture
def channel(channel_type, regular_channel, distributed_channel):
    """Select channel based on channel_type parameter from test function."""
    if channel_type == "distributed":
        return distributed_channel
    else:
        return regular_channel


# --- Test Data Generation ---
def get_test_data():
    device = get_device()
    return [
        ("python_string", "hello world"),
        ("torch_tensor", torch.randn(2, 2, device=device)),
        (
            "list_of_tensors",
            [torch.ones(1, device=device), torch.zeros(1, device=device)],
        ),
        (
            "dict_of_tensors",
            {
                "a": torch.tensor([1], device=device),
                "b": torch.tensor([2], device=device),
            },
        ),
        (
            "dataclass_with_tensor",
            TensorMessage(
                id=42,
                payload=torch.ones(2, 2, device=device) * 3,
                note="channel test",
            ),
        ),
        (
            "dataclass_with_list_of_tensors",
            TensorListMessage(
                id=10,
                payload_list=[torch.ones(2, 2, device=device) * i for i in range(3)],
                note="channel list test",
            ),
        ),
        (
            "dataclass_with_dict_of_tensors",
            TensorDictMessage(
                id=20,
                payload_dict={
                    "x": torch.ones(2, 2, device=device) * 1,
                    "y": torch.ones(2, 2, device=device) * 2,
                },
                note="channel dict test",
            ),
        ),
        (
            "plain_dataclass",
            PlainMessage(id=1, name="channel_plain", value=3.14),
        ),
    ]


# --- Test Class ---
class TestChannel:
    """Comprehensive tests for the Channel class."""

    def _run_test(
        self,
        producer,
        consumer,
        producer_method,
        producer_args,
        consumer_method,
        consumer_args,
    ):
        """Helper to run producer and consumer workers and get results."""
        getattr(producer, producer_method)(*producer_args)
        consumer_ref = getattr(consumer, consumer_method)(*consumer_args)
        results = consumer_ref.wait()
        return results[0]  # Return only consumer result

    def _run_async_test(
        self,
        producer,
        consumer,
        producer_method,
        producer_args,
        consumer_method,
        consumer_args,
    ):
        """Helper to run async producer/consumer workers."""
        getattr(producer, producer_method)(*producer_args).wait()
        consumer_worker = getattr(consumer, consumer_method)(*consumer_args).wait()
        return consumer_worker[0]

    @pytest.mark.parametrize("channel_type", ["regular", "distributed"], indirect=True)
    @pytest.mark.parametrize("data_name, item_to_send", get_test_data())
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_put_get_single_item(
        self, worker_groups, channel, channel_type, data_name, item_to_send, async_op
    ):
        """Tests a single put/get for various data types with sync and async_wait."""
        producer, consumer = worker_groups
        received_item = self._run_test(
            producer,
            consumer,
            "put_item",
            (channel, item_to_send, 1, 0, async_op),
            "get_item",
            (
                channel,
                async_op,
            ),
        )
        self._assert_equal(received_item, item_to_send)

    @pytest.mark.parametrize("channel_type", ["regular", "distributed"], indirect=True)
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_put_get_mixed_tensor_list(
        self, worker_groups, channel, channel_type, async_op
    ):
        if not accelerator_is_available():
            pytest.skip("Skipping mixed tensor test without an accelerator.")
        producer, consumer = worker_groups
        key = "mixed_tensor_list"
        received_item = self._run_test(
            producer,
            consumer,
            "put_mixed_tensor_list",
            (channel, async_op, key),
            "get_item",
            (channel, async_op, key),
        )
        expected_vals = [1, 2, 3]
        expected_devices = ["cpu", Worker.torch_device_type, "cpu"]
        for tensor, expected_val, expected_device in zip(
            received_item, expected_vals, expected_devices
        ):
            assert tensor.device.type == expected_device
            assert torch.equal(tensor.cpu(), torch.ones(2, 2) * expected_val)

    @pytest.mark.parametrize("channel_type", ["regular", "distributed"], indirect=True)
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_put_get_mixed_tensor_dict(
        self, worker_groups, channel, channel_type, async_op
    ):
        if not accelerator_is_available():
            pytest.skip("Skipping mixed tensor test without an accelerator.")
        producer, consumer = worker_groups
        key = "mixed_tensor_dict"
        received_item = self._run_test(
            producer,
            consumer,
            "put_mixed_tensor_dict",
            (channel, async_op, key),
            "get_item",
            (channel, async_op, key),
        )
        assert received_item["cpu_a"].device.type == "cpu"
        assert received_item["cuda_b"].device.type == Worker.torch_device_type
        assert received_item["cpu_c"].device.type == "cpu"
        assert torch.equal(received_item["cpu_a"].cpu(), torch.ones(2, 2) * 1)
        assert torch.equal(received_item["cuda_b"].cpu(), torch.ones(2, 2) * 2)
        assert torch.equal(received_item["cpu_c"].cpu(), torch.ones(2, 2) * 3)

    @pytest.mark.parametrize("channel_type", ["regular", "distributed"], indirect=True)
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_put_get_mixed_tensor_list_dataclass(
        self, worker_groups, channel, channel_type, async_op
    ):
        if not accelerator_is_available():
            pytest.skip("Skipping mixed tensor test without an accelerator.")
        producer, consumer = worker_groups
        key = "mixed_tensor_list_dataclass"
        received_item = self._run_test(
            producer,
            consumer,
            "put_mixed_tensor_list_dataclass",
            (channel, async_op, key),
            "get_item",
            (channel, async_op, key),
        )
        assert isinstance(received_item, TensorListMessage)
        assert received_item.id == 10
        assert received_item.note == "channel mixed list dataclass"
        expected_vals = [1, 2, 3]
        expected_devices = ["cpu", Worker.torch_device_type, "cpu"]
        for tensor, expected_val, expected_device in zip(
            received_item.payload_list, expected_vals, expected_devices
        ):
            assert tensor.device.type == expected_device
            assert torch.equal(tensor.cpu(), torch.ones(2, 2) * expected_val)

    @pytest.mark.parametrize("channel_type", ["regular", "distributed"], indirect=True)
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_put_get_mixed_tensor_dict_dataclass(
        self, worker_groups, channel, channel_type, async_op
    ):
        if not accelerator_is_available():
            pytest.skip("Skipping mixed tensor test without an accelerator.")
        producer, consumer = worker_groups
        key = "mixed_tensor_dict_dataclass"
        received_item = self._run_test(
            producer,
            consumer,
            "put_mixed_tensor_dict_dataclass",
            (channel, async_op, key),
            "get_item",
            (channel, async_op, key),
        )
        assert isinstance(received_item, TensorDictMessage)
        assert received_item.id == 20
        assert received_item.note == "channel mixed dict dataclass"
        assert received_item.payload_dict["cpu_a"].device.type == "cpu"
        assert (
            received_item.payload_dict["cuda_b"].device.type == Worker.torch_device_type
        )
        assert received_item.payload_dict["cpu_c"].device.type == "cpu"
        assert torch.equal(
            received_item.payload_dict["cpu_a"].cpu(), torch.ones(2, 2) * 1
        )
        assert torch.equal(
            received_item.payload_dict["cuda_b"].cpu(), torch.ones(2, 2) * 2
        )
        assert torch.equal(
            received_item.payload_dict["cpu_c"].cpu(), torch.ones(2, 2) * 3
        )

    @pytest.mark.parametrize("channel_type", ["regular", "distributed"], indirect=True)
    @pytest.mark.parametrize("data_name, item_to_send", get_test_data())
    def test_put_get_single_item_asyncio(
        self, worker_groups, channel, channel_type, data_name, item_to_send
    ):
        """Tests a single put/get for various data types with native asyncio."""
        producer, consumer = worker_groups
        received_item = self._run_async_test(
            producer,
            consumer,
            "put_item_asyncio",
            (channel, item_to_send, 1, 0),
            "get_item_asyncio",
            (channel,),
        )
        self._assert_equal(received_item, item_to_send)

    @pytest.mark.parametrize("channel_type", ["regular", "distributed"], indirect=True)
    def test_async_wait_concurrency_not_limited_by_executor(
        self, worker_groups, channel, channel_type
    ):
        """Ensures concurrent async_wait() calls are not serialized by the event
        loop's default executor thread pool.

        64 gets on keys with no data pending occupy 64 concurrent async_wait()
        calls. A 65th get on a freshly put key must still complete promptly:
        if each waiting async_wait() parked an executor thread, the 64 waits
        would exhaust the default pool (32 threads) and the 65th wait would
        queue behind them until timeout.
        """
        consumer = worker_groups[1]
        blocked_keys = [f"blocked_key_{i}" for i in range(64)]
        free_key = "free_key"

        results = consumer.test_async_wait_beyond_executor_pool(
            channel, blocked_keys, free_key, timeout=10.0
        ).wait()[0]
        free_timed_out, free_item, free_elapsed, blocked_items = results
        assert not free_timed_out, (
            f"Get on the newly put key timed out after {free_elapsed:.1f}s; "
            "concurrent async_wait() calls appear serialized behind the "
            "default executor"
        )
        assert free_item == "free_item", (
            f"Get on the newly put key returned {free_item!r}, expected 'free_item'"
        )
        assert len(blocked_items) == len(blocked_keys), (
            f"Expected {len(blocked_keys)} blocked gets to finish, got "
            f"{len(blocked_items)}"
        )
        assert set(blocked_items) == set(range(len(blocked_keys))), (
            "Blocked gets did not each receive their own item"
        )

    @pytest.mark.parametrize("channel_type", ["regular", "distributed"], indirect=True)
    def test_async_wait_cancelled_before_data_arrives(
        self, worker_groups, channel, channel_type
    ):
        """A cancelled async_wait() must not error when the data arrives later,
        and the channel must keep working for subsequent put/get."""
        consumer = worker_groups[1]
        key = "cancel_key"
        followup_key = "cancel_followup_key"

        item, loop_errors = consumer.test_async_wait_cancelled(
            channel, key, followup_key
        ).wait()[0]
        assert item == "after_cancel", (
            f"Get after cancellation returned {item!r}, expected 'after_cancel'"
        )
        assert not loop_errors, (
            f"Cancelling async_wait() surfaced event-loop errors: {loop_errors}"
        )

    def test_await_torch_future_surfaces_exception(self):
        """A torch Future ending in an exception must surface that exception
        when the caller reads the value after awaiting, matching the
        executor-wait behavior where future.wait() raises."""
        from torch.futures import Future

        from rlinf.scheduler.collective.async_work import _await_torch_future

        async def scenario():
            future = Future()
            error = RuntimeError("future failed")
            waiter = asyncio.ensure_future(_await_torch_future(future))
            await asyncio.sleep(0.1)
            future.set_exception(error)
            await waiter
            return future

        future = asyncio.run(scenario())
        with pytest.raises(RuntimeError, match="future failed"):
            future.value()

    def test_last_future_per_key_is_dropped(self, cluster):
        """A key leaves last_future_per_key once its operation completes.

        Keys are per group on the payload channels, so a table that keeps
        them grows for the whole run.
        """

        @ray.remote
        class _RecordingActor:
            def __init__(self):
                self.calls = []

            def work(self, value, delay):
                time.sleep(delay)
                self.calls.append(value)
                return value

            def called(self):
                return list(self.calls)

        actor = _RecordingActor.remote()
        channel_key = f"last_future_{uuid.uuid4().hex[:8]}"
        table_key = f"order_channel:{channel_key}"

        work = AsyncChannelWork("order_channel", channel_key, actor, "work", 1, 0.3)

        assert work.wait() == 1
        assert ray.get(actor.called.remote()) == [1]
        assert table_key not in AsyncChannelWork.last_future_per_key

    def test_a_finished_call_leaves_no_entry_in_the_table(self):
        """A remote call that has already finished leaves no entry behind.

        _complete runs in the thread that registers it; the entry is written
        before the call is issued, so the callback finds its own future and
        removes it.
        """
        channel_key = uuid.uuid4().hex[:8]
        table_key = f"completed_channel:{channel_key}"

        work = AsyncChannelWork(
            "completed_channel", channel_key, _CompletedHandle(), "work"
        )

        assert work.wait() == "ok"
        assert table_key not in AsyncChannelWork.last_future_per_key

    def test_two_finished_calls_on_one_key_both_return(self):
        """A second operation on a key whose first call has already finished
        still runs, and the key leaves the table.

        The first call completes inside its own construction, so the second
        reads an empty table instead of the first call's finished future.
        """
        channel_key = uuid.uuid4().hex[:8]
        table_key = f"completed_channel:{channel_key}"
        handle = _CompletedHandle()

        first = AsyncChannelWork("completed_channel", channel_key, handle, "work", 1)
        second = AsyncChannelWork("completed_channel", channel_key, handle, "work", 2)

        assert first.wait() == "ok"
        assert second.wait() == "ok"
        assert table_key not in AsyncChannelWork.last_future_per_key

    def test_async_channel_work_remote_failure_raises(self, cluster):
        """A failed remote call must raise at the waiter instead of leaving
        the work forever pending, and work issued later on the same channel
        key must still execute instead of queueing behind the failed work."""

        @ray.remote
        class FailingActor:
            def ok(self, value):
                return value

            def boom(self):
                raise ValueError("remote boom")

        actor = FailingActor.remote()
        channel_key = f"failure_key_{uuid.uuid4().hex[:8]}"

        sync_work = AsyncChannelWork("failure_channel", channel_key, actor, "boom")
        with pytest.raises(Exception, match="remote boom"):
            sync_work.wait()

        async_work = AsyncChannelWork("failure_channel", channel_key, actor, "boom")
        with pytest.raises(Exception, match="remote boom"):
            asyncio.run(async_work.async_wait())

        follow_up = AsyncChannelWork("failure_channel", channel_key, actor, "ok", 42)
        assert follow_up.wait() == 42

    def test_recv_failure_fails_pending_gets_of_the_same_peer(self, monkeypatch):
        """A worker's recvs from one channel worker are not keyed: each takes
        the next message, and the store returns the data to the get its
        query_id names. When one recv of a peer fails after an earlier
        recv already delivered another get's data, every get still pending
        on that peer fails instead of hanging, and gets on other peers are
        untouched.

        The recvs are real ``AsyncFuncWork`` objects, called the way the work
        queue thread calls them, so a failure that escapes the call ends the
        thread and every later recv hangs.
        """

        # The queue thread runs inside a worker; this test process has none.
        monkeypatch.setattr(
            Worker, "current_worker", SimpleNamespace(has_accelerator=False)
        )

        def _boom():
            raise ValueError("peer boom")

        peer = ("chan", 0)
        other_peer = ("chan", 1)
        # A's recv delivers B's data (arrival order is not get order).
        recv_a = AsyncFuncWork(lambda: ("data_b", 200))
        recv_b = AsyncFuncWork(_boom)
        work_a = AsyncChannelCommWork(recv_a, 100, None, peer=peer)
        work_b = AsyncChannelCommWork(recv_b, 200, None, peer=peer)

        # A's recv delivers B's data; B is done, A still waits.
        recv_a(None)
        assert work_b.wait() == "data_b"

        # B's recv fails: A, whose data never arrived, must fail with the
        # original error instead of hanging, and the call must return so the
        # queue thread keeps running.
        recv_b(None)
        with pytest.raises(ValueError, match="peer boom"):
            asyncio.run(asyncio.wait_for(work_a.async_wait(), timeout=5))
        with pytest.raises(ValueError, match="peer boom"):
            work_a.wait()

        # A later get on that peer fails as well: the failed recv may have
        # lost part of the stream, so the data the peer sends afterwards
        # cannot be matched to the get it belongs to.
        recv_d = AsyncFuncWork(_raise_peer_error)
        work_d = AsyncChannelCommWork(recv_d, 400, None, peer=peer)
        recv_d(None)
        with pytest.raises(PeerCommunicationError):
            work_d.wait()

        # A get on another peer is unaffected: its data still arrives.
        recv_c = AsyncFuncWork(lambda: ("data_c", 300))
        work_c = AsyncChannelCommWork(recv_c, 300, None, peer=other_peer)
        recv_c(None)
        assert work_c.wait() == "data_c"

    def test_done_callback_runs_after_a_failure(self):
        """A callback added with add_done_callback runs whether the work
        succeeded or failed, unlike a work chained with then().

        A step that has to hand the outcome to its callers, such as failing
        the gets waiting on a channel rank after a recv error, is skipped by
        then() on failure and must register here instead.
        """
        called = []

        def _boom():
            raise ValueError("call back failed")

        work = AsyncFuncWork(_boom)
        work.add_done_callback(lambda future: called.append(future))
        work(None)

        assert len(called) == 1
        with pytest.raises(ValueError, match="call back failed"):
            called[0].wait()

    def test_work_returning_a_work_completes_with_the_chain(self, monkeypatch):
        """A work whose function returns another work completes once that
        work does, and its waiter sees the chain's failure.

        Nested works are how the collective helpers chain one transfer with
        the next one it needs (a send with its follow-up recv, a broadcast
        with the copy out of the receive buffer). An outer work that is never
        completed hangs its caller even when every step succeeded.
        """
        monkeypatch.setattr(
            Worker, "current_worker", SimpleNamespace(has_accelerator=False)
        )
        created = []

        def _outer():
            inner = AsyncFuncWork(lambda: "inner result")
            created.append(inner)
            return inner

        outer = AsyncFuncWork(_outer)
        outer(None)
        assert not outer.done()
        created[0](None)
        assert outer.done()
        assert outer.wait() == "inner result"

    def test_work_returning_a_failing_work_completes_with_its_error(self, monkeypatch):
        """The chain's failure reaches the outer work's waiter.

        The inner work here is the one that fails, so the outer work is
        completed by the failure, not by a result.
        """
        monkeypatch.setattr(
            Worker, "current_worker", SimpleNamespace(has_accelerator=False)
        )
        created = []

        def _boom():
            raise ValueError("inner failed")

        def _outer():
            inner = AsyncFuncWork(_boom)
            created.append(inner)
            return inner

        outer = AsyncFuncWork(_outer)
        outer(None)
        created[0](None)
        assert outer.done()
        with pytest.raises(ValueError, match="inner failed"):
            outer.wait()

    def test_chained_work_skips_its_function_when_the_previous_failed(self):
        """A work chained with then() does not run its function when the work
        before it failed, and its waiter gets the original error.

        Without this, a chained step runs on missing data, or its waiter
        waits forever on a future that is never completed.
        """
        calls = []

        def _boom():
            raise ValueError("first failed")

        first = AsyncFuncWork(_boom)
        next_work = first.then(lambda: calls.append("ran"))
        first(None)

        assert calls == []
        with pytest.raises(ValueError, match="first failed"):
            next_work.wait()

    @pytest.mark.parametrize("channel_type", ["regular", "distributed"], indirect=True)
    def test_remove_key_drops_items_and_keeps_the_channel_usable(
        self, worker_groups, channel, channel_type
    ):
        """remove_key drops a key's queued items and queues in place, without
        pulling them to the caller, and the channel keeps serving the key."""
        producer, consumer = worker_groups
        key = f"remove_key_{uuid.uuid4().hex[:8]}"

        for i in range(3):
            producer.put_item(channel, i, 1, 0, True, key=key).wait()
        assert channel.qsize(key) == 3

        assert channel.remove_key(key) == 3
        assert channel.qsize(key) == 0
        assert channel.remove_key(f"never_put_{uuid.uuid4().hex[:8]}") == 0

        producer.put_item(channel, "after_remove", 1, 0, True, key=key).wait()
        assert consumer.get_item(channel, False, key=key).wait()[0] == "after_remove"
        assert channel.qsize(key) == 0

        with pytest.raises(ValueError, match="default queue"):
            channel.remove_key(DEFAULT_KEY)

    def test_local_channel_remove_key(self):
        """remove_key on a local channel drops the key's items and queue, and
        later puts on the key still work."""
        channel = Channel.create(
            name=f"local_remove_{uuid.uuid4().hex[:8]}", local=True
        )
        key = "local_key"
        channel.put(1, key=key)
        channel.put(2, key=key)

        assert channel.remove_key(key) == 2
        assert channel.qsize(key) == 0

        channel.put(3, key=key)
        assert channel.get_nowait(key) == 3

    @pytest.mark.parametrize("channel_type", ["regular", "distributed"], indirect=True)
    def test_same_key_puts_execute_in_issue_order(
        self, worker_groups, channel, channel_type
    ):
        """Same-key put(async_op=True) calls from one process must run in the
        order they were issued: once the last put's handle completes, every
        item is already queued and get_nowait returns them in that order."""
        producer = worker_groups[0]
        key = f"ordered_key_{uuid.uuid4().hex[:8]}"
        items = list(range(5))

        received = producer.ordered_puts_then_get_nowait(channel, key, items).wait()[0]
        assert received == items

    @pytest.mark.parametrize("channel_type", ["regular", "distributed"], indirect=True)
    def test_async_wait_yields_control(self, worker_groups, channel, channel_type):
        """Ensures channel get(async_op=True).async_wait() yields control so other
        asyncio tasks can run while waiting."""
        producer, consumer = worker_groups
        key = "async_wait_yields_test"
        recv_ref = consumer.test_async_wait_yields_control(channel, key)
        producer_done = []

        def delayed_put():
            time.sleep(0.1)
            producer.put_item(channel, "yield_test_item", 1, 0, False, key=key).wait()
            producer_done.append(True)

        t = threading.Thread(target=delayed_put)
        try:
            results = recv_ref.wait()
            t.start()
        finally:
            t.join()
        yield_count = results[0]
        assert yield_count >= 1, (
            f"async_wait() did not yield: yield_check task ran {yield_count} times"
        )

    @pytest.mark.parametrize("channel_type", ["regular", "distributed"], indirect=True)
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_get_batch(self, worker_groups, channel, channel_type, async_op):
        """Tests getting a batch of items based on weight."""
        producer, consumer = worker_groups
        items = [("item1", 1), ("item2", 2), ("item3", 3)]

        # Producer puts all items
        for item, weight in items:
            producer.put_item(channel, item, weight, 10, async_op).wait()

        # Consumer gets a batch with total weight 3
        batch = consumer.get_batch(channel, 3, async_op).wait()[0]
        channel.get()

        assert len(batch) == 2
        assert batch[0] == items[0][0]
        assert batch[1] == items[1][0]

    @pytest.mark.parametrize("channel_type", ["regular", "distributed"], indirect=True)
    def test_get_batch_asyncio(self, worker_groups, channel, channel_type):
        """Tests getting a batch of items with native asyncio."""
        producer, consumer = worker_groups
        items = [("item1", 1), ("item2", 2), ("item3", 3)]

        # Producer puts all items
        for item, weight in items:
            producer.put_item_asyncio(channel, item, weight, 10).wait()

        # Consumer gets a batch with total weight 3
        batch = consumer.get_batch_asyncio(channel, 3).wait()
        channel.get()

        assert len(batch[0]) == 2
        assert batch[0][0] == items[0][0]
        assert batch[0][1] == items[1][0]

    def test_queued_summary_sums_over_every_channel_rank(self):
        """queued_summary counts every channel worker, not one rank.

        A key is hosted on the rank of the node it is first produced from.
        This test runs on one node, so it asks the channel worker for the
        two keys the way callers from nodes 0 and 1 would, which is what a
        key's own lookup does on first use. Reading the queue of one rank
        would report only that rank's share.
        """
        channel = DistributedChannel.create(
            f"QUEUED_SUMMARY_TEST_{uuid.uuid4().hex[:8]}"
        )
        for key, src_node_rank in (("q_rank0", 0), ("q_rank1", 1)):
            channel._channel_worker_group.execute_on(0).ensure_key_replica(
                key=key, src_node_rank=src_node_rank
            ).wait()
        channel.put("first", weight=2, key="q_rank0")
        channel.put("second", weight=3, key="q_rank1")

        assert channel.queued_summary() == (2, 5)

    @pytest.mark.parametrize("channel_type", ["regular", "distributed"], indirect=True)
    def test_qsize_empty_full(self, worker_groups, channel, channel_type):
        """Tests the qsize, empty, and full methods of the channel."""
        producer, consumer = worker_groups
        maxsize = 2
        # Create a new channel with maxsize for this specific test
        # Use the same type as the fixture channel (regular or distributed)
        is_distributed = hasattr(channel, "_distributed") and channel._distributed
        if is_distributed:
            test_channel = DistributedChannel.create(
                f"EMPTY_FULL_TEST_{uuid.uuid4().hex[:8]}", maxsize=maxsize
            )
        else:
            test_channel = Channel.create(
                f"EMPTY_FULL_TEST_{uuid.uuid4().hex[:8]}", maxsize=maxsize
            )
        channel = test_channel

        # Initial state
        producer.put_item(
            channel, "dummy", 1, maxsize, False
        ).wait()  # Creates the channel
        consumer.get_item(channel, False).wait()  # Clears it
        assert consumer.is_empty(channel).wait()[0]
        assert not consumer.is_full(channel).wait()[0]
        assert consumer.get_qsize(channel).wait()[0] == 0

        # Add one item
        producer.put_item(channel, "item1", 1, maxsize, False).wait()
        assert not consumer.is_empty(channel).wait()[0]
        assert not consumer.is_full(channel).wait()[0]
        assert consumer.get_qsize(channel).wait()[0] == 1

        # Fill the channel
        producer.put_item(channel, "item2", 1, maxsize, False).wait()
        assert not consumer.is_empty(channel).wait()[0]
        assert consumer.is_full(channel).wait()[0]
        assert consumer.get_qsize(channel).wait()[0] == 2

    @pytest.mark.parametrize("channel_type", ["regular", "distributed"], indirect=True)
    def test_channel_multiple_queues(self, worker_groups, channel, channel_type):
        """Tests creating multiple queues in a single channel."""
        producer, consumer = worker_groups

        # Put items in different queues
        producer.put_item(channel, "item1", 1, 10, False, key="queue1").wait()
        producer.put_item(channel, "item2", 2, 10, False, key="queue2").wait()

        # Get items from different queues
        item1 = consumer.get_item(channel, False, key="queue1").wait()[0]
        item2 = consumer.get_item(channel, False, key="queue2").wait()[0]

        assert item1 == "item1"
        assert item2 == "item2"

    @pytest.mark.parametrize("channel_type", ["regular", "distributed"], indirect=True)
    def test_channel_multiple_queues_order(self, worker_groups, channel, channel_type):
        """Tests the order of items in multiple queues."""
        producer: ProducerWorker = worker_groups[0]
        consumer: ConsumerWorker = worker_groups[1]

        # Put items in different queues
        handle = consumer.put_get_ood(channel)
        producer.put_get_ood(channel)
        item1, item2 = handle.wait()[0]

        assert item1 == "Hello"
        assert item2 == "World"

    @pytest.mark.parametrize("channel_type", ["regular", "distributed"], indirect=True)
    def test_channel_nowait(self, worker_groups, channel, channel_type):
        """Tests the channel under heavy load."""
        producer: ProducerWorker = worker_groups[0]
        consumer: ConsumerWorker = worker_groups[1]

        data = consumer.get_nowait(channel).wait()[0]
        assert data is None

        producer.put_nowait(channel, "item_100").wait()
        data = consumer.get_nowait(channel).wait()[0]

        assert data == "item_100"

    @pytest.mark.parametrize("channel_type", ["regular", "distributed"], indirect=True)
    def test_stress(self, worker_groups, channel, channel_type):
        """Tests the channel under heavy load."""
        producer: ProducerWorker = worker_groups[0]
        num_items = 1000

        data = producer.stress(channel, num_items).wait()[0]
        assert data == list(range(num_items))

        data = producer.stress_multiple_queues(channel, num_items).wait()[0]
        assert data == list(range(num_items))

    @pytest.mark.parametrize("channel_type", ["regular", "distributed"], indirect=True)
    def test_peek_all(self, channel: Channel, channel_type):
        """Tests the peek_all method of the channel."""
        while channel.qsize() > 0:
            channel.get()

        test_items = ["item1", "item2", "item3"]
        for item in test_items:
            channel.put(item)

        item_str = str(channel)
        assert all(item in item_str for item in test_items)

    def _assert_equal(self, received: Any, expected: Any):
        """Helper to compare various data types."""
        assert type(received) is type(expected)
        if isinstance(expected, torch.Tensor):
            assert torch.equal(received.cpu(), expected.cpu())
        elif isinstance(expected, list):
            assert len(received) == len(expected)
            for r, e in zip(received, expected):
                self._assert_equal(r, e)
        elif isinstance(expected, dict):
            assert received.keys() == expected.keys()
            for key in expected:
                self._assert_equal(received[key], expected[key])
        elif is_dataclass(expected):
            for f in fields(type(expected)):
                self._assert_equal(getattr(received, f.name), getattr(expected, f.name))
        else:
            assert received == expected

    # --- Distributed Channel Specific Tests ---

    @pytest.mark.parametrize("channel_type", ["distributed"], indirect=True)
    def test_distributed_channel_creation(self, worker_groups, channel, channel_type):
        """Test that distributed channel is actually created with multiple workers."""

        # Verify the channel has multiple channel workers
        assert channel._distributed, "Channel should be marked as distributed"
        assert len(channel._channel_actors_by_rank) == NUM_SIMULATED_NODES, (
            f"DistributedChannel should have {NUM_SIMULATED_NODES} channel workers, "
            f"but has {len(channel._channel_actors_by_rank)}"
        )
        # Verify all ranks are present
        for rank in range(NUM_SIMULATED_NODES):
            assert rank in channel._channel_actors_by_rank, (
                f"Channel worker rank {rank} should exist"
            )

    @pytest.mark.parametrize("channel_type", ["distributed"], indirect=True)
    def test_channel_worker_routing(self, worker_groups, channel, channel_type):
        """Test that keys are routed to the correct channel worker based on source node."""
        producer, consumer = worker_groups

        # Put items with different keys from the producer
        # The channel should route them to the channel worker matching the producer's node rank
        test_keys = ["key1", "key2", "key3"]
        test_items = ["item1", "item2", "item3"]

        for key, item in zip(test_keys, test_items):
            producer.put_item(channel, item, 1, 0, False, key=key).wait()

        # Verify items can be retrieved
        for key, expected_item in zip(test_keys, test_items):
            received = consumer.get_item(channel, False, key=key).wait()[0]
            assert received == expected_item

    @pytest.mark.parametrize("channel_type", ["distributed"], indirect=True)
    def test_channel_worker_isolation(self, worker_groups, channel, channel_type):
        """Test that different keys can be routed to different channel workers."""
        producer, consumer = worker_groups

        # Put items with different keys
        # Each key should be assigned to a channel worker based on routing logic
        keys = [f"key_{i}" for i in range(NUM_SIMULATED_NODES * 2)]
        items = [f"item_{i}" for i in range(NUM_SIMULATED_NODES * 2)]

        for key, item in zip(keys, items):
            producer.put_item(channel, item, 1, 0, False, key=key).wait()

        # Verify all items can be retrieved correctly
        for key, expected_item in zip(keys, items):
            received = consumer.get_item(channel, False, key=key).wait()[0]
            assert received == expected_item

    @pytest.mark.parametrize("channel_type", ["distributed"], indirect=True)
    def test_channel_worker_rank_assignment(self, worker_groups, channel, channel_type):
        """Test that ensure_key_replica assigns ranks correctly."""
        producer, consumer = worker_groups

        # Put an item - this will trigger ensure_key_replica
        test_key = "routing_test_key"
        test_item = "routing_test_item"
        producer.put_item(channel, test_item, 1, 0, False, key=test_key).wait()

        # Verify the item is accessible
        received = consumer.get_item(channel, False, key=test_key).wait()[0]
        assert received == test_item

        # Verify the key is cached (subsequent operations should use cached rank)
        producer.put_item(channel, "second_item", 1, 0, False, key=test_key).wait()
        received = consumer.get_item(channel, False, key=test_key).wait()[0]
        assert received == "second_item"

    @pytest.mark.parametrize("channel_type", ["distributed"], indirect=True)
    def test_channel_worker_selection(self, worker_groups, channel, channel_type):
        """Test that the channel correctly selects the channel worker for a given key."""
        producer, consumer = worker_groups

        # Put items with different keys and verify they are routed correctly
        # The first key should be assigned to the channel worker matching producer's node rank
        test_key1 = "selection_key1"
        test_item1 = "selection_item1"
        producer.put_item(channel, test_item1, 1, 0, False, key=test_key1).wait()

        # Verify the item is on the correct channel worker by checking qsize
        # The qsize should be 1 on the target channel worker
        qsize = channel.qsize(key=test_key1)
        assert qsize == 1, f"Expected qsize 1, got {qsize}"

        # Get the item and verify qsize becomes 0
        received = consumer.get_item(channel, False, key=test_key1).wait()[0]
        assert received == test_item1
        qsize = channel.qsize(key=test_key1)
        assert qsize == 0, f"Expected qsize 0 after get, got {qsize}"

        # Test with another key - should be routed to the same or different worker
        test_key2 = "selection_key2"
        test_item2 = "selection_item2"
        producer.put_item(channel, test_item2, 1, 0, False, key=test_key2).wait()

        # Both keys should work independently
        qsize1 = channel.qsize(key=test_key1)
        qsize2 = channel.qsize(key=test_key2)
        assert qsize1 == 0, f"Key1 qsize should be 0, got {qsize1}"
        assert qsize2 == 1, f"Key2 qsize should be 1, got {qsize2}"

        received = consumer.get_item(channel, False, key=test_key2).wait()[0]
        assert received == test_item2


if __name__ == "__main__":
    pytest.main(["-v", __file__])


ACTORS = ("actor:0", "actor:1", "actor:2")


def _ctx(consumers=ACTORS, **kwargs):
    return ChannelContext(name="test", consumers=tuple(consumers), **kwargs)


def _stub_worker(collector=None, dispatcher=None, ctx=None):
    """A ChannelWorker with hooks wired but no Ray actor behind it."""
    worker = object.__new__(ChannelWorker)
    worker._queue_map = {DEFAULT_KEY: PeekQueue(maxsize=0)}
    worker._consumer_queues = {}
    worker._context = ctx or _ctx()
    worker._collector = resolve_collector(collector)
    worker._dispatcher = resolve_dispatcher(dispatcher)
    worker._collector.setup(worker._context)
    worker._dispatcher.setup(worker._context)
    worker._deals = worker._dispatcher.deals
    return worker


# --- defaults reproduce today's channel -------------------------------------


def test_default_hooks_pass_items_through_a_single_shared_queue():
    worker = _stub_worker()

    assert worker._deals is False
    assert list(iter_collected(worker._collector, "payload", "k")) == [("k", "payload")]
    assert worker._dispatcher.route("payload", "k") is None

    asyncio.run(worker._enqueue("k", "payload", 0))
    # One shared queue, and no per-consumer queues were created.
    assert worker._queue_map["k"].qsize() == 1
    assert worker._consumer_queues == {}


def test_default_dispatcher_serves_every_consumer_from_the_shared_queue():
    async def run():
        worker = _stub_worker()
        await worker._enqueue("k", "first", 0)
        await worker._enqueue("k", "second", 0)
        a = await worker._take("k", "actor:0", nowait=False)
        b = await worker._take("k", "actor:1", nowait=False)
        return a.item, b.item

    assert asyncio.run(run()) == ("first", "second")


# --- collector ---------------------------------------------------------------


class FanOutCollector(Collector):
    def setup(self, ctx):
        self.saw_ctx = ctx

    def collect(self, item, key):
        for index, part in enumerate(item):
            yield f"{key}:{index}", part


def test_collector_can_fan_one_item_out_across_several_keys():
    async def run():
        worker = _stub_worker(collector=FanOutCollector())
        for out_key, out_item in iter_collected(worker._collector, "abc", "k"):
            await worker._enqueue(out_key, out_item, 0)
        return {
            key: queue.qsize()
            for key, queue in worker._queue_map.items()
            if key != DEFAULT_KEY
        }

    assert asyncio.run(run()) == {"k:0": 1, "k:1": 1, "k:2": 1}


def test_collector_can_drop_items():
    class DropAll(Collector):
        def collect(self, item, key):
            return iter(())

    async def run():
        worker = _stub_worker(collector=DropAll())
        for out_key, out_item in iter_collected(worker._collector, "x", "k"):
            await worker._enqueue(out_key, out_item, 0)
        return [key for key in worker._queue_map if key != DEFAULT_KEY]

    assert asyncio.run(run()) == []


def test_collector_receives_the_channel_context_once():
    collector = FanOutCollector()
    ctx = _ctx(producers=("env",), options={"tuning": 3})
    _stub_worker(collector=collector, ctx=ctx)

    assert collector.saw_ctx is ctx
    assert collector.saw_ctx.producers == ("env",)
    assert collector.saw_ctx.options == {"tuning": 3}


def test_a_collector_yielding_a_bad_shape_is_rejected_with_its_name():
    class Broken(Collector):
        def collect(self, item, key):
            yield item

    with pytest.raises(TypeError, match="Broken.collect must yield"):
        list(iter_collected(Broken(), "x", "k"))


# --- dispatcher / load balancing ---------------------------------------------


@pytest.mark.parametrize(
    "dispatcher", ["round_robin", "least_loaded", "least_loaded_stealing"]
)
def test_dealing_dispatchers_split_an_even_stream_exactly_evenly(dispatcher):
    async def run():
        worker = _stub_worker(dispatcher=dispatcher)
        for index in range(9):
            await worker._enqueue("k", index, 0)
        return {
            consumer: queue.qsize()
            for consumer, queue in worker._consumer_queues["k"].items()
        }

    assert asyncio.run(run()) == {"actor:0": 3, "actor:1": 3, "actor:2": 3}


def test_dealing_keeps_a_polling_consumer_from_starving_its_peers():
    """The hungry-consumer bug: a nowait poll loop must not drain peers' work."""

    async def run():
        worker = _stub_worker(dispatcher="least_loaded")
        for index in range(9):
            await worker._enqueue("k", index, 0)

        drained = []
        while True:
            try:
                drained.append((await worker._take("k", "actor:0", nowait=True)).item)
            except asyncio.QueueEmpty:
                break

        remaining = {
            consumer: queue.qsize()
            for consumer, queue in worker._consumer_queues["k"].items()
        }
        return drained, remaining

    drained, remaining = asyncio.run(run())
    assert len(drained) == 3
    assert remaining["actor:1"] == 3 and remaining["actor:2"] == 3


def test_a_shared_queue_lets_one_polling_consumer_take_everything():
    """Contrast case: this is exactly what the dealing dispatcher prevents."""

    async def run():
        worker = _stub_worker()  # SharedDispatcher
        for index in range(9):
            await worker._enqueue("k", index, 0)
        drained = []
        while True:
            try:
                drained.append((await worker._take("k", "actor:0", nowait=True)).item)
            except asyncio.QueueEmpty:
                break
        return drained

    assert len(asyncio.run(run())) == 9


def test_least_loaded_deals_to_the_shallowest_queue_after_consumption():
    async def run():
        worker = _stub_worker(dispatcher="least_loaded")
        for index in range(3):
            await worker._enqueue("k", index, 0)
        # actor:0 consumes; the next items should still round out fairly by
        # total assignment, not by current depth.
        await worker._take("k", "actor:0", nowait=True)
        for index in range(3, 6):
            await worker._enqueue("k", index, 0)
        return {
            consumer: queue.qsize()
            for consumer, queue in worker._consumer_queues["k"].items()
        }

    assert asyncio.run(run()) == {"actor:0": 1, "actor:1": 2, "actor:2": 2}


def test_least_loaded_splits_each_dispatch_group_evenly():
    async def run():
        worker = _stub_worker(dispatcher="least_loaded")
        # Dealt by count alone, this order gives actor:0 two "a" items and
        # actor:2 two "b" items.
        for group in ["a", "a", "b", "a", "b", "b"]:
            await worker._enqueue("k", SimpleNamespace(dispatch_group=group), 0)
        return {
            consumer: sorted(
                queue.get_nowait().item.dispatch_group for _ in range(queue.qsize())
            )
            for consumer, queue in worker._consumer_queues["k"].items()
        }

    assert asyncio.run(run()) == dict.fromkeys(ACTORS, ["a", "b"])


def test_stealing_lets_a_starved_consumer_take_from_the_deepest_peer():
    async def run():
        worker = _stub_worker(dispatcher="least_loaded_stealing")
        # Deal everything to one consumer by hand, then starve another.
        worker._consumer_queue("k", "actor:1")
        for index in range(3):
            queue = worker._consumer_queue("k", "actor:2")
            queue.put_nowait(Mock(item=index))
        taken = await worker._take("k", "actor:1", nowait=False)
        return taken.item, worker._consumer_queues["k"]["actor:2"].qsize()

    item, donor_left = asyncio.run(run())
    assert item == 0
    assert donor_left == 2


def test_non_stealing_dispatcher_does_not_take_from_a_peer():
    dispatcher = LeastLoadedDispatcher()
    dispatcher.setup(_ctx())
    assert dispatcher.rebalance("k", "actor:1", {"actor:2": 5}) is None


def test_stealing_dispatcher_waits_when_no_peer_has_work():
    dispatcher = LeastLoadedStealingDispatcher()
    dispatcher.setup(_ctx())
    assert dispatcher.rebalance("k", "actor:1", {"actor:2": 0}) is None


@pytest.mark.parametrize("dispatcher", ["round_robin", "least_loaded"])
def test_each_key_is_dealt_independently(dispatcher):
    """Traffic on one key must not move another key's split.

    Both dispatchers self-balance, so with per-round counts that are multiples
    of the consumer count a shared cursor happens to stay even too. Keeping the
    state per key makes the property hold by construction instead of by that
    coincidence, which is what a consumer waiting on a fixed count relies on.
    """

    async def run():
        worker = _stub_worker(dispatcher=dispatcher)
        # An odd number on one key leaves its own split uneven, by construction.
        for index in range(4):
            await worker._enqueue("other", index, 0)
        for index in range(3):
            await worker._enqueue("k", index, 0)
        return {
            consumer: queue.qsize()
            for consumer, queue in worker._consumer_queues["k"].items()
        }

    # Three items across three consumers: exactly one each, regardless of the
    # four items already dealt on "other".
    assert asyncio.run(run()) == {"actor:0": 1, "actor:1": 1, "actor:2": 1}


def test_a_consumer_unknown_at_setup_is_still_dealt_to():
    async def run():
        worker = _stub_worker(dispatcher="least_loaded", ctx=_ctx(consumers=()))
        # No consumers declared, so the first items stay on the shared queue.
        await worker._enqueue("k", "early", 0)
        # A get from an undeclared consumer registers it for later dealing.
        worker._queue_for_get("k", "actor:9")
        await worker._enqueue("k", "late", 0)
        return worker._queue_map["k"].qsize(), worker._consumer_queues["k"][
            "actor:9"
        ].qsize()

    shared, dealt = asyncio.run(run())
    assert shared == 1
    assert dealt == 1


# --- qsize semantics ---------------------------------------------------------


def test_qsize_reports_the_total_and_a_single_consumer_share():
    async def run():
        worker = _stub_worker(dispatcher="least_loaded")
        worker.maxsize = Mock(return_value=0)
        for index in range(6):
            await worker._enqueue("k", index, 0)
        return (
            ChannelWorker.qsize(worker, "k"),
            ChannelWorker.qsize(worker, "k", consumer="actor:0"),
            ChannelWorker.empty(worker, "k"),
            ChannelWorker.empty(worker, "missing"),
        )

    total, share, empty, missing_empty = asyncio.run(run())
    assert total == 6
    assert share == 2
    assert empty is False
    assert missing_empty is True


def test_remove_key_clears_dealt_consumer_queues_too():
    async def run():
        worker = _stub_worker(dispatcher="least_loaded")
        for index in range(3):
            await worker._enqueue("k", index, 0)
        dropped = ChannelWorker.remove_key(worker, "k")
        return dropped, "k" in worker._queue_map, "k" in worker._consumer_queues

    dropped, shared_left, dealt_left = asyncio.run(run())
    assert dropped == 3
    assert not shared_left
    assert not dealt_left


# --- resolution ---------------------------------------------------------------


def test_hooks_resolve_from_names_classes_and_instances():
    assert isinstance(resolve_dispatcher(None), SharedDispatcher)
    assert isinstance(resolve_dispatcher("round_robin"), RoundRobinDispatcher)
    assert isinstance(resolve_dispatcher(LeastLoadedDispatcher), LeastLoadedDispatcher)
    instance = LeastLoadedDispatcher()
    assert resolve_dispatcher(instance) is instance
    assert type(resolve_collector(None)) is Collector


def test_a_hook_outside_rlinf_resolves_from_an_import_path():
    collector = resolve_collector("test_channel:FanOutCollector")
    assert isinstance(collector, FanOutCollector)


def test_an_unknown_hook_name_lists_what_is_available_and_the_escape_hatch():
    with pytest.raises(ValueError, match="module:ClassName"):
        resolve_dispatcher("nope")
    with pytest.raises(ValueError, match="least_loaded"):
        resolve_dispatcher("nope")


def test_a_hook_of_the_wrong_base_class_is_rejected():
    with pytest.raises(TypeError, match="not a Dispatcher subclass"):
        resolve_dispatcher(FanOutCollector)


def test_registration_rejects_a_duplicate_name():
    name = "unit_test_duplicate_collector"
    try:
        register_collector(name)(FanOutCollector)
        with pytest.raises(ValueError, match="already registered"):
            register_collector(name)(FanOutCollector)
    finally:
        COLLECTOR_REGISTRY.pop(name, None)


def test_registered_hooks_are_selectable_by_name():
    name = "unit_test_registered_dispatcher"
    try:

        @register_dispatcher(name)
        class Custom(Dispatcher):
            def route(self, item, key):
                return "actor:0"

        resolved = resolve_dispatcher(name)
        assert isinstance(resolved, Custom)
        assert resolved.deals is True
    finally:
        DISPATCHER_REGISTRY.pop(name, None)


def test_overriding_route_is_what_makes_a_dispatcher_deal():
    class Passive(Dispatcher):
        pass

    class Active(Dispatcher):
        def route(self, item, key):
            return "actor:0"

    assert Passive().deals is False
    assert Active().deals is True


def test_peek_queue_and_default_queue_map_are_untouched_by_the_hooks():
    worker = _stub_worker()
    asyncio.run(worker._enqueue("k", "payload", weight=7))
    queue = worker._queue_map["k"]
    assert isinstance(queue, PeekQueue)
    assert queue.peek_all()[0].weight == 7


def test_collector_and_dispatcher_state_is_per_channel_not_global():
    first = _stub_worker(dispatcher="round_robin")
    second = _stub_worker(dispatcher="round_robin")
    assert first._dispatcher is not second._dispatcher

    counts = defaultdict(int)
    for _ in range(3):
        counts[first._dispatcher.route(None, "k")] += 1
    assert counts == {"actor:0": 1, "actor:1": 1, "actor:2": 1}
    assert second._dispatcher.route(None, "k") == "actor:0"


# --- producer / consumer resolution ------------------------------------------


def test_group_specs_accept_worker_groups_names_and_mixtures():
    group = object.__new__(WorkerGroup)
    group._worker_group_name = "ActorGroup"

    assert resolve_group_names(None) == []
    assert resolve_group_names("EnvGroup") == ["EnvGroup"]
    assert resolve_group_names(group) == ["ActorGroup"]
    assert resolve_group_names([group, "EnvGroup"]) == ["ActorGroup", "EnvGroup"]


def test_a_group_spec_of_the_wrong_type_is_rejected():
    with pytest.raises(TypeError, match="WorkerGroup or a group name"):
        resolve_group_names([object()])


def test_consumers_expand_to_one_id_per_rank_of_a_worker_group():
    group = object.__new__(WorkerGroup)
    group._worker_group_name = "ActorGroup"
    group._workers = [Mock(rank=rank) for rank in range(3)]

    assert resolve_worker_names(group) == [
        "ActorGroup:0",
        "ActorGroup:1",
        "ActorGroup:2",
    ]
    assert resolve_worker_names(None) == []


def test_a_bare_group_name_gets_its_size_from_the_worker_manager(monkeypatch):
    proxy = Mock()
    proxy.get_worker_info.return_value = Mock(group_world_size=2)
    monkeypatch.setattr(
        "rlinf.scheduler.manager.WorkerManager.get_proxy", lambda: proxy
    )

    assert resolve_worker_names("ActorGroup") == ["ActorGroup:0", "ActorGroup:1"]
    proxy.get_worker_info.assert_called_once()


def test_group_sizes_mixes_groups_and_names(monkeypatch):
    group = object.__new__(WorkerGroup)
    group._worker_group_name = "ActorGroup"
    group._workers = [Mock(rank=0)]
    proxy = Mock()
    proxy.get_worker_info.return_value = Mock(group_world_size=4)
    monkeypatch.setattr(
        "rlinf.scheduler.manager.WorkerManager.get_proxy", lambda: proxy
    )

    assert resolve_group_sizes([group, "EnvGroup"]) == [
        ("ActorGroup", 1),
        ("EnvGroup", 4),
    ]


def test_naming_an_unlaunched_group_says_what_to_do_about_it(monkeypatch):
    proxy = Mock()
    proxy.get_worker_info.return_value = None
    monkeypatch.setattr(
        "rlinf.scheduler.manager.WorkerManager.get_proxy", lambda: proxy
    )

    with pytest.raises(ValueError, match="launch it before"):
        get_group_world_size("NeverLaunched")


def test_consumer_ids_match_the_ids_the_worker_get_path_reports():
    """The dispatcher keys on dst_addr.get_name(); expansion must agree."""
    monkeypatch_size = 2
    addresses = [
        WorkerAddress(root_group_name="ActorGroup", ranks=rank).get_name()
        for rank in range(monkeypatch_size)
    ]
    assert addresses == ["ActorGroup:0", "ActorGroup:1"]


# Where each named group is placed, for the stubbed manager to report.
NODE_OF_GROUP = {"Rollout": 2, "Actor": 3, "Env": 1}


class _CompletedCall:
    """A channel call whose future is already done when it is returned."""

    def __init__(self, value: Any):
        from concurrent.futures import Future

        self._future = Future()
        self._future.set_result(value)

    def future(self):
        return self._future


class _PendingCall:
    """A channel call whose future is not completed by anyone else."""

    def __init__(self):
        from concurrent.futures import Future

        self._future = Future()

    def future(self):
        return self._future


class _PendingHandle:
    """A channel-actor handle whose calls stay in flight.

    Stands in for a remote call that has not returned: nothing settles it but
    the code under test.
    """

    def __getattr__(self, name):
        def remote(*args, **kwargs):
            return _PendingCall()

        return SimpleNamespace(remote=remote)


class _CompletedHandle:
    """A channel-actor handle whose every call has already finished.

    Stands in for the process boundary: the remote call behind
    ``getattr(handle, method).remote(...)`` is done before the caller
    registers its completion callback.
    """

    def __getattr__(self, name):
        def remote(*args, **kwargs):
            return _CompletedCall("ok")

        return SimpleNamespace(remote=remote)


class _FakeGroup:
    """A worker group as ``resolve_group_names`` sees it: just a name."""

    def __init__(self, name):
        self.worker_group_name = name


@pytest.fixture(autouse=True)
def stub_manager(monkeypatch):
    """Report placements by group name, as the real worker manager does."""

    class _Proxy:
        def get_worker_info(self, address):
            assert address.rank == 0, "placement is read from rank 0"
            node = NODE_OF_GROUP.get(address.root_group_name)
            return None if node is None else SimpleNamespace(cluster_node_rank=node)

    import rlinf.scheduler.manager as manager_mod

    monkeypatch.setattr(
        manager_mod.WorkerManager, "get_proxy", staticmethod(lambda: _Proxy())
    )


@pytest.fixture
def group_type(monkeypatch):
    """Make the name resolver accept _FakeGroup in place of WorkerGroup."""
    import rlinf.scheduler.cluster.utils as utils_mod
    import rlinf.scheduler.worker as worker_mod

    monkeypatch.setattr(worker_mod, "WorkerGroup", _FakeGroup)
    monkeypatch.setattr(utils_mod, "WorkerGroup", _FakeGroup, raising=False)


def test_resolves_group_object_to_its_node(group_type):
    """A group passed as an object resolves through its name."""
    assert resolve_colocation_node_rank(_FakeGroup("Rollout")) == 2


def test_resolves_group_name(group_type):
    """A group passed as a bare name resolves the same way."""
    assert resolve_colocation_node_rank("Actor") == 3


def test_prefers_producer_over_consumer(group_type):
    """Producers win: the collector runs channel-side, before the consumer hop."""
    assert resolve_colocation_node_rank("Rollout", "Actor") == 2


def test_falls_back_to_consumer(group_type):
    """With no producer given, the consumer still pins the channel."""
    assert resolve_colocation_node_rank(None, "Actor") == 3


def test_accepts_iterables_and_skips_unknown_groups(group_type):
    """A group the manager does not know is skipped, not chosen."""
    assert resolve_colocation_node_rank(["NotLaunched", _FakeGroup("Env")]) == 1


def test_returns_none_when_nothing_resolves(group_type):
    """No producers and no consumers leaves the caller to pick a default."""
    assert resolve_colocation_node_rank(None, None) is None


def test_unregistered_group_name_is_skipped(group_type):
    """An unknown name yields None instead of raising, so creation can proceed."""
    assert resolve_colocation_node_rank("Missing") is None


class TestPeerCommunicationFailure:
    """One collective direction failing refuses the operations after it.

    The failure state lives on the group, one per direction, so every pool
    queue of that direction shares it. These cases drive the works the way
    the queue thread and the synchronous path do, without a cluster: the
    group object carries only what the guard and the recorder read.
    """

    @pytest.fixture(autouse=True)
    def _worker_context(self, monkeypatch):
        """The queue thread runs inside a worker; this process has none."""
        monkeypatch.setattr(
            Worker, "current_worker", SimpleNamespace(has_accelerator=False)
        )

    @staticmethod
    def _group(name="direction_group"):
        """A group with only the state the guard and the recorder read."""
        group = CollectiveGroup.__new__(CollectiveGroup)
        group._group_name = name
        group._logger = logging.getLogger(f"test.{name}")
        group._direction_failures = {"send": None, "recv": None}
        return group

    @staticmethod
    def _guarded_work(group, direction, calls, comm_id):
        """A work built like a real send or recv: guarded, and recorded."""

        def _run(work, **kwargs):
            calls.append(kwargs["comm_id"])
            return "ran"

        work = AsyncFuncWork(
            partial(group._run_guarded, direction, _run),
            comm_id=comm_id,
            pass_self=True,
        )
        group._watch_work(work, direction, comm_id)
        return work

    def test_a_failed_send_refuses_later_sends(self):
        """A send issued after a failed send does not run: the failure of the
        first one lost the stream's alignment, so later sends report it
        instead of sending, on every pool queue of that direction."""
        group = self._group()
        calls = []

        def _failing(work, **kwargs):
            calls.append(("failed", kwargs["comm_id"]))
            raise ValueError("link down")

        failed = AsyncFuncWork(
            partial(group._run_guarded, "send", _failing), comm_id=8, pass_self=True
        )
        group._watch_work(failed, "send", 8)
        failed(None)
        with pytest.raises(ValueError, match="link down"):
            failed.wait()

        # The recorder took this failure from the work it was watching.
        assert isinstance(group._direction_failures["send"], ValueError)

        for comm_id in (0, 1):
            work = self._guarded_work(group, "send", calls, comm_id)
            work(None)
            with pytest.raises(PeerCommunicationError) as excinfo:
                work.wait()
            assert isinstance(excinfo.value.__cause__, ValueError)
            assert str(excinfo.value) == (
                "an earlier send in collective group direction_group failed "
                "(ValueError('link down')); later sends in this group are not run"
            )

        assert calls == [("failed", 8)]

    def test_a_failed_recv_refuses_later_recvs_and_sends_still_run(self):
        """The state is per direction: a failed recv refuses the recvs after
        it, and the sends of the same group are untouched."""
        group = self._group()
        group._direction_failures["recv"] = OSError("peer gone")
        calls = []

        send_work = self._guarded_work(group, "send", calls, 0)
        send_work(None)
        assert send_work.wait() == "ran"
        recv_work = self._guarded_work(group, "recv", calls, 1)
        recv_work(None)
        with pytest.raises(PeerCommunicationError) as excinfo:
            recv_work.wait()

        assert calls == [0]
        assert isinstance(excinfo.value.__cause__, OSError)

    def test_the_synchronous_path_raises_the_same_error(self):
        """The synchronous path runs the send in the caller's thread; the
        guard is inside the work, so ``send`` raises there too."""
        group = self._group()
        group._direction_failures["send"] = ValueError("link down")
        group._send_comm_id_iter = iter(range(CollectiveGroup.POOL_SIZE))
        group._send_work_queues = [
            SimpleNamespace(done=True) for _ in range(CollectiveGroup.POOL_SIZE)
        ]
        sent = []
        group._atomic_send = lambda work, **kwargs: sent.append(kwargs["comm_id"])

        with pytest.raises(PeerCommunicationError) as excinfo:
            group.send("item", async_op=False)

        assert sent == []
        assert "an earlier send in collective group direction_group failed" in str(
            excinfo.value
        )

    def test_another_group_is_unaffected(self):
        """The failure state belongs to the group that failed."""
        failed = self._group("failed_group")
        failed._direction_failures["send"] = ValueError("link down")
        other = self._group("other_group")
        calls = []

        other_work = self._guarded_work(other, "send", calls, 0)
        other_work(None)
        assert other_work.wait() == "ran"
        refused = self._guarded_work(failed, "send", calls, 1)
        refused(None)
        with pytest.raises(PeerCommunicationError):
            refused.wait()

        assert calls == [0]

    def test_the_first_failure_is_recorded_and_logged(self, caplog):
        """The recorder keeps the first failure of a direction and logs every
        failure; a later one, from an operation that had already started when
        the first failed, does not replace it."""
        group = self._group()

        def _failing(exc, comm_id):
            def _run():
                raise exc

            work = AsyncFuncWork(_run)
            group._watch_work(work, "send", comm_id)
            return work

        with caplog.at_level(logging.ERROR, logger=group._logger.name):
            first = _failing(ValueError("first"), 3)
            first(None)
            with pytest.raises(ValueError, match="first"):
                first.wait()
            second = _failing(RuntimeError("second"), 4)
            second(None)
            with pytest.raises(RuntimeError, match="second"):
                second.wait()

        assert isinstance(group._direction_failures["send"], ValueError)
        logged = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert any("send 3 failed in group direction_group" in m for m in logged)
        assert any("send 4 failed in group direction_group" in m for m in logged)

    def test_a_refused_work_is_logged_at_debug_only(self, caplog):
        """A work refused because its direction failed was not itself a
        failure: the first one was logged, and this one at debug level."""
        group = self._group()
        group._direction_failures["send"] = ValueError("link down")

        with caplog.at_level(logging.DEBUG, logger=group._logger.name):
            work = self._guarded_work(group, "send", [], 5)
            work(None)
            with pytest.raises(PeerCommunicationError):
                work.wait()

        assert [r.levelno for r in caplog.records] == [logging.DEBUG]

    def test_broadcast_failures_are_logged_but_not_recorded(self, caplog):
        """A broadcast belongs to no direction: its failure is logged, it
        fails nothing, and a broadcast still runs when a direction has
        already failed."""
        group = self._group()

        def _failing(work, **kwargs):
            raise ValueError("broadcast down")

        work = AsyncFuncWork(_failing, comm_id=9, pass_self=True)
        # Broadcast work is watched with no direction.
        with caplog.at_level(logging.ERROR, logger=group._logger.name):
            group._watch_work(work, None, 9)
            work(None)
            with pytest.raises(ValueError, match="broadcast down"):
                work.wait()

        assert group._direction_failures == {"send": None, "recv": None}

        # A failed send refuses the sends after it, not the broadcasts.
        group._direction_failures["send"] = ValueError("link down")
        ran = AsyncFuncWork(lambda **kwargs: "broadcast ran", comm_id=10)
        group._watch_work(ran, None, 10)
        ran(None)

        assert ran.wait() == "broadcast ran"
        assert any(
            "collective broadcast 9 failed in group direction_group" in r.getMessage()
            for r in caplog.records
        )

    def test_a_keyboard_interrupt_is_not_stored(self):
        """Only Exception is stored on a work: an interrupt keeps going out,
        the way it does on main."""
        work = AsyncFuncWork(_raise_keyboard_interrupt)

        with pytest.raises(KeyboardInterrupt):
            work(None)
        assert not work.done()

    def test_a_failed_send_fails_the_put(self):
        """A put whose send failed fails with the error the send reports: the
        item never reached the channel, so the put cannot report success."""
        key = uuid.uuid4().hex[:8]
        channel = _put_channel(AsyncFuncWork(_raise_peer_error))
        table_key = f"chan:{key}"

        work = channel.put("item", key=key, async_op=True)

        assert work.done(), "the put was never failed"
        with pytest.raises(PeerCommunicationError) as excinfo:
            work.wait()
        assert "an earlier send in collective group test_peer failed" in str(
            excinfo.value
        )
        assert table_key not in AsyncChannelWork.last_future_per_key

    def test_a_failed_send_of_another_kind_is_wrapped(self):
        """A send that failed for its own reason fails the put with an error
        naming the channel, chained to the send's error, and leaves no table
        entry behind."""
        key = uuid.uuid4().hex[:8]
        channel = _put_channel(AsyncFuncWork(_raise_zero_division))

        work = channel.put("item", key=key, async_op=True)

        assert work.done(), "the put was never failed"
        with pytest.raises(PeerCommunicationError) as excinfo:
            work.wait()
        assert str(excinfo.value) == (
            "the send of a put to channel chan failed "
            "(ZeroDivisionError('division by zero'))"
        )
        assert isinstance(excinfo.value.__cause__, ZeroDivisionError)
        assert f"chan:{key}" not in AsyncChannelWork.last_future_per_key

    def test_late_data_from_a_failed_peer_is_dropped(self):
        """A recv that was already running when its peer failed keeps no
        entry: its get was failed with the others, so nobody would take it."""
        peer = ("late_chan", 3)

        def _boom():
            raise ValueError("recv failed")

        failed_recv = AsyncFuncWork(_boom)
        failed = AsyncChannelCommWork(failed_recv, 100, None, peer=peer)
        late_recv = AsyncFuncWork(lambda: ("data_late", 200))
        late = AsyncChannelCommWork(late_recv, 200, None, peer=peer)

        failed_recv(None)
        with pytest.raises(ValueError, match="recv failed"):
            failed.wait()
        late_recv(None)
        with pytest.raises(ValueError, match="recv failed"):
            late.wait()

        assert AsyncChannelCommWork.channel_data_store.get(peer, {}) == {}
        AsyncChannelCommWork.failed_peers.discard(peer)
        AsyncChannelCommWork.channel_data_store.pop(peer, None)


def _raise_keyboard_interrupt():
    raise KeyboardInterrupt


def _put_channel(send_work):
    """A Channel whose worker send is ``send_work``, for the put cases.

    Every part a real put needs beyond the send is a stand-in: the channel
    rank and the channel worker's actor, whose call never returns.
    """
    channel = Channel.__new__(Channel)
    channel._local_channel = None
    channel._channel_name = "chan"
    # A send that has already finished when it returns, the way a fast remote
    # call does: the failure is there for the put to pick up.
    send_work(None)
    channel._current_worker = SimpleNamespace(
        worker_address=WorkerAddress("Worker", ranks=0),
        send=lambda *args, **kwargs: send_work,
    )
    channel._get_channel_rank_by_key = lambda key: 0
    channel._get_channel_actor = lambda rank: _PendingHandle()
    return channel


def _raise_peer_error():
    raise PeerCommunicationError("an earlier send in collective group test_peer failed")


def _raise_zero_division():
    raise ZeroDivisionError("division by zero")
