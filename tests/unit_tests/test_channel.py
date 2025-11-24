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

# --- Constants ---
PRODUCER_GROUP_NAME = "producer_group"
CONSUMER_GROUP_NAME = "consumer_group"
TEST_CHANNEL_NAME = "my_test_channel"
group_count = 0
channel_count = 0


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

    def is_empty(self, channel: Channel):
        return channel.empty()

    def is_full(self, channel: Channel):
        return channel.full()


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


@pytest.fixture(scope="module")
def channel():
    return Channel.create(TEST_CHANNEL_NAME)


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

    @pytest.mark.parametrize("data_name, item_to_send", get_test_data())
    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_put_get_single_item(
        self, worker_groups, channel, data_name, item_to_send, async_op
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

    @pytest.mark.parametrize("data_name, item_to_send", get_test_data())
    def test_put_get_single_item_asyncio(
        self, worker_groups, channel, data_name, item_to_send
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

    @pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async_wait"])
    def test_get_batch(self, worker_groups, channel, async_op):
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

    def test_get_batch_asyncio(self, worker_groups, channel):
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

    def test_qsize_empty_full(self, worker_groups, channel):
        """Tests the qsize, empty, and full methods of the channel."""
        producer, consumer = worker_groups
        maxsize = 2
        channel = Channel.create("EMPTY_FULL_TEST", maxsize=maxsize)

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

    def test_channel_multiple_queues(self, worker_groups, channel):
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

    def test_channel_multiple_queues_order(self, worker_groups, channel):
        """Tests the order of items in multiple queues."""
        producer: ProducerWorker = worker_groups[0]
        consumer: ConsumerWorker = worker_groups[1]

        # Put items in different queues
        handle = consumer.put_get_ood(channel)
        producer.put_get_ood(channel)
        item1, item2 = handle.wait()[0]

        assert item1 == "Hello"
        assert item2 == "World"

    def test_channel_nowait(self, worker_groups, channel):
        """Tests the channel under heavy load."""
        producer: ProducerWorker = worker_groups[0]
        consumer: ConsumerWorker = worker_groups[1]

        data = consumer.get_nowait(channel).wait()[0]
        assert data is None

        producer.put_nowait(channel, "item_100").wait()
        data = consumer.get_nowait(channel).wait()[0]

        assert data == "item_100"

    def test_stress(self, worker_groups, channel):
        """Tests the channel under heavy load."""
        producer: ProducerWorker = worker_groups[0]
        num_items = 1000

        data = producer.stress(channel, num_items).wait()[0]
        assert data == list(range(num_items))

        data = producer.stress_multiple_queues(channel, num_items).wait()[0]
        assert data == list(range(num_items))

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
        else:
            assert received == expected


if __name__ == "__main__":
    pytest.main(["-v", __file__])
