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
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, fields, is_dataclass
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import Mock

import pytest
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
