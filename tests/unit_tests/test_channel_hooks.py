# Copyright 2026 The RLinf Authors.
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

# ruff: noqa: D101, D102, D103

import asyncio
from collections import defaultdict
from unittest.mock import Mock

import pytest

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
    resolve_group_names,
    resolve_group_sizes,
    resolve_worker_names,
)
from rlinf.scheduler.worker import WorkerAddress, WorkerGroup

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
    collector = resolve_collector("test_channel_hooks:FanOutCollector")
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
