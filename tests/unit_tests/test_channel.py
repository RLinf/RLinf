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
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Optional

import pytest
import torch

from rlinf.scheduler import (
    Channel,
    Cluster,
    NodePlacementStrategy,
    PackedPlacementStrategy,
    Worker,
)
from rlinf.scheduler.channel.channel_worker import ChannelWorker

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


def get_device():
    """Returns the appropriate torch device."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
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
    if torch.cuda.is_available():
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
