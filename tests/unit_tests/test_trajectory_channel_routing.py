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

from types import SimpleNamespace

import pytest
import torch

from rlinf.scheduler.worker import Worker
from rlinf.workers.trajectory.channel import TrajectoryChannel
from rlinf.workers.trajectory.data import (
    Actions,
    Observations,
    TrajectoryEnvelope,
    merge_trajectory_data,
)
from rlinf.workers.trajectory.route_plan import TrajectoryRoute, TrajectoryRoutePlan


def _channel() -> TrajectoryChannel:
    channel = TrajectoryChannel.__new__(TrajectoryChannel)
    channel._channel_workers_by_rank = {2: object(), 5: object()}
    channel._channel_name = ""
    channel._task_descriptions_by_slot = {}
    return channel


def _actions(rank: int) -> Actions:
    return Actions(
        global_step=0,
        rank=rank,
        current_epoch=0,
        current_step=0,
        actions=torch.zeros(1, 1),
    )


@pytest.fixture
def route_plan() -> TrajectoryRoutePlan:
    return TrajectoryRoutePlan(
        total_slots=10,
        trajectory_world_size=3,
        component_world_sizes={"env": 2, "rollout": 4, "reward": 1, "actor": 2},
    )


def test_routes_source_rank_to_a_stable_trajectory_worker():
    channel = _channel()

    assert channel._route_channel_workers(_actions(0)) == [TrajectoryRoute(2, (), ())]
    assert channel._route_channel_workers(_actions(1)) == [TrajectoryRoute(5, (), ())]
    assert channel._route_channel_workers(_actions(2)) == [TrajectoryRoute(2, (), ())]


def test_unpickled_channel_binds_to_the_receiving_worker(monkeypatch):
    receiving_worker = object()
    channel = TrajectoryChannel.__new__(TrajectoryChannel)
    channel._current_worker = None
    monkeypatch.setattr(Worker, "current_worker", receiving_worker)

    channel.__setstate__({"_current_worker": None})

    assert channel._current_worker is receiving_worker


def test_get_channel_worker_rejects_unknown_rank():
    channel = _channel()

    try:
        channel._get_channel_worker(1)
    except ValueError as error:
        assert str(error) == "Invalid trajectory channel worker rank: 1."
    else:
        raise AssertionError("Expected an invalid worker rank to raise ValueError.")


def test_record_uses_worker_p2p_when_called_from_a_worker(monkeypatch):
    class FakeAsyncChannelWork:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeWorker:
        worker_address = "env-address"

        def __init__(self):
            self.sent = []

        def send(self, *args, **kwargs):
            self.sent.append((args, kwargs))

    monkeypatch.setattr(
        "rlinf.workers.trajectory.channel.AsyncChannelWork", FakeAsyncChannelWork
    )
    channel = _channel()
    channel._channel_name = "Rollout"
    channel._channel_worker_group_name = "TrajectoryGroup"
    channel._channel_workers_by_rank = {0: object()}
    channel._current_worker = current_worker = FakeWorker()
    record = TrajectoryEnvelope(record=_actions(0))

    work = channel._record_shard(0, record)

    assert isinstance(work, FakeAsyncChannelWork)
    assert work.kwargs["method"] == "record"
    assert work.kwargs["channel_name"] == "TrajectoryP2P"
    assert work.kwargs["channel_key"] == ("TrajectoryGroup", 0)
    assert work.kwargs["src_addr"] == "env-address"
    assert current_worker.sent == [((record, "TrajectoryGroup", 0), {"async_op": True})]


def test_put_and_record_share_one_physical_transport_key():
    channel = _channel()
    channel._channel_worker_group_name = "TrajectoryGroup"

    assert channel._transport_channel_key(2) == ("TrajectoryGroup", 2)


def test_route_plan_splits_a_record_and_attaches_global_slot_ids():
    channel = _channel()
    channel._channel_workers_by_rank = {0: object(), 1: object(), 2: object()}
    channel._route_plan = TrajectoryRoutePlan(
        total_slots=10,
        trajectory_world_size=3,
        component_world_sizes={"env": 2, "rollout": 2, "reward": 1},
    )
    observations = Observations(
        global_step=0,
        rank=0,
        current_epoch=0,
        current_step=0,
        obs={"state": torch.arange(10).reshape(5, 2)},
    )

    routes = channel._route_item(observations)

    assert [rank for rank, _ in routes] == [0, 1]
    assert [shard.slot_ids for _, shard in routes] == [(0, 1, 2), (3, 4)]
    assert [shard.obs["state"].shape[0] for _, shard in routes] == [3, 2]


def test_outbound_slot_ids_survive_generic_peer_batch_splitting():
    channel = _channel()
    channel._route_plan = TrajectoryRoutePlan(
        total_slots=4,
        trajectory_world_size=2,
        component_world_sizes={"env": 1, "rollout": 2},
    )
    observations = Observations(
        global_step=0,
        rank=0,
        current_epoch=0,
        obs={"state": torch.arange(4).reshape(4, 1)},
    )

    prepared = channel.prepare_outbound(observations)
    peer_shard = prepared.select([2, 3])
    routes = channel._route_item(peer_shard)

    assert prepared.slot_ids == (0, 1, 2, 3)
    assert peer_shard.slot_ids == (2, 3)
    assert [(rank, shard.slot_ids) for rank, shard in routes] == [(1, (2, 3))]


def test_relay_caches_task_descriptions_until_a_slot_value_changes():
    channel = _channel()
    channel._channel_name = "Rollout"
    channel._route_plan = None
    initial = Observations(
        global_step=0,
        rank=0,
        current_epoch=0,
        current_step=0,
        slot_ids=(4,),
        obs={"state": torch.zeros(1, 1)},
        task_descriptions=["pick"],
    )
    unchanged = Observations(
        global_step=0,
        rank=0,
        current_epoch=0,
        current_step=1,
        slot_ids=(4,),
        obs={"state": torch.zeros(1, 1)},
        task_descriptions=["pick"],
    )
    changed = Observations(
        global_step=0,
        rank=0,
        current_epoch=0,
        current_step=2,
        slot_ids=(4,),
        obs={"state": torch.zeros(1, 1)},
        task_descriptions=["place"],
    )

    assert channel.prepare_outbound(initial).task_descriptions == ["pick"]
    assert channel.prepare_outbound(unchanged).task_descriptions is None
    assert channel.prepare_outbound(changed).task_descriptions == ["place"]


def test_rollout_restores_cached_task_descriptions_by_slot():
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    worker = MultiStepRolloutWorker.__new__(MultiStepRolloutWorker)
    worker._task_descriptions_by_slot = {}
    worker._trajectory_uses_task_descriptions = None
    initial = Observations(
        global_step=0,
        rank=0,
        current_epoch=0,
        slot_ids=(4, 7),
        obs={"state": torch.zeros(2, 1)},
        task_descriptions=["pick", "place"],
    )
    later = Observations(
        global_step=0,
        rank=0,
        current_epoch=0,
        current_step=1,
        slot_ids=(7, 4),
        obs={"state": torch.zeros(2, 1)},
    )

    worker._trajectory_rollout_input(initial)
    restored = worker._trajectory_rollout_input(later)

    assert restored["obs"]["task_descriptions"] == ["place", "pick"]


def test_explicit_placement_strategy_takes_precedence_over_node_shorthand():
    placement = object()

    assert (
        TrajectoryChannel._resolve_placement_strategy(
            object(),
            node_ranks=None,
            placement_strategy=placement,
        )
        is placement
    )

    with pytest.raises(ValueError, match="cannot be provided together"):
        TrajectoryChannel._resolve_placement_strategy(
            object(),
            node_ranks=[0],
            placement_strategy=placement,
        )


def test_relay_shards_merge_back_to_the_component_local_batch():
    first = Observations(
        global_step=0,
        rank=0,
        current_epoch=0,
        slot_ids=(0, 1),
        obs={"state": torch.tensor([[1.0], [2.0]])},
    )
    second = Observations(
        global_step=0,
        rank=0,
        current_epoch=0,
        slot_ids=(2,),
        obs={"state": torch.tensor([[3.0]])},
    )

    merged = merge_trajectory_data([first, second])

    assert merged.slot_ids == (0, 1, 2)
    assert torch.equal(merged.obs["state"], torch.tensor([[1.0], [2.0], [3.0]]))


def test_actions_preserve_the_live_rollout_control_payload():
    from rlinf.data.embodied_io_struct import RolloutResult

    rollout_result = RolloutResult(
        actions=torch.tensor([[1.0], [2.0]]),
        bootstrap_values=torch.tensor([[3.0], [4.0]]),
        intervene_flags=torch.tensor([[True], [False]]),
        forward_inputs={"action": torch.tensor([[1.0], [2.0]])},
    )
    actions = Actions.from_rollout_result(
        rollout_result,
        num_action_chunks=1,
        global_step=0,
        rank=0,
        current_epoch=0,
    )

    restored = actions.to_rollout_result()

    assert torch.equal(restored.bootstrap_values, rollout_result.bootstrap_values)
    assert torch.equal(restored.intervene_flags, rollout_result.intervene_flags)
    assert torch.equal(
        restored.forward_inputs["action"],
        rollout_result.forward_inputs["action"].reshape(2, 1, 1),
    )


def test_envelope_preserves_decoupled_batch_index_when_sharded():
    record = Observations(
        global_step=0,
        rank=0,
        current_epoch=0,
        obs={"state": torch.tensor([[1.0], [2.0]])},
    )
    envelope = TrajectoryEnvelope.from_channel_item(
        {"batch_index": "0_0_train_rollout_results", "batch": record}
    )

    shard = envelope.select([1], slot_ids=[9])

    assert shard.to_channel_item() == {
        "batch_index": "0_0_train_rollout_results",
        "batch": shard.record,
    }
    assert shard.record.slot_ids == (9,)
    assert torch.equal(shard.record.obs["state"], torch.tensor([[2.0]]))


def test_component_rank_can_split_across_trajectory_workers(route_plan):
    assert route_plan.routes_for("env", 0) == [
        TrajectoryRoute(0, (0, 1, 2), (0, 1, 2)),
        TrajectoryRoute(1, (3, 4), (3, 4)),
    ]
    assert route_plan.routes_for("env", 1) == [
        TrajectoryRoute(1, (0,), (5,)),
        TrajectoryRoute(2, (1, 2, 3, 4), (6, 7, 8, 9)),
    ]


def test_slot_owner_mappings_are_consistent(route_plan):
    assert [route_plan.channel_worker_for_slot(slot) for slot in range(10)] == [
        0,
        0,
        0,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
    ]
    assert [
        route_plan.component_rank_for_slot("actor", slot) for slot in range(10)
    ] == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]


def test_channel_worker_ranges_avoid_per_slot_owner_lookups(route_plan):
    assert route_plan.channel_worker_slot_range(1) == (3, 6)
    assert route_plan.component_local_range_on_channel_worker("actor", 0, 1) == (
        0,
        2,
    )
    assert route_plan.component_local_range_on_channel_worker("actor", 1, 1) == (
        2,
        3,
    )
    assert route_plan.component_local_range_on_channel_worker("actor", 0, 2) is None


def test_route_plan_rejects_unknown_components_and_invalid_ranks(route_plan):
    with pytest.raises(ValueError, match="Unknown trajectory component"):
        route_plan.routes_for("missing", 0)
    with pytest.raises(ValueError, match="outside the 'env' world size 2"):
        route_plan.routes_for("env", 2)


def test_route_plan_uses_launched_worker_group_sizes():
    cfg = SimpleNamespace(env=SimpleNamespace(train=SimpleNamespace(total_num_envs=12)))
    trajectory_group = SimpleNamespace(worker_info_list=[object(), object(), object()])
    component_groups = {
        "env": SimpleNamespace(worker_info_list=[object(), object()]),
        "rollout": SimpleNamespace(worker_info_list=[object(), object(), object()]),
    }

    plan = TrajectoryRoutePlan.from_worker_groups(
        cfg,
        trajectory_worker_group=trajectory_group,
        component_worker_groups=component_groups,
    )

    assert plan.total_slots == 12
    assert plan.trajectory_world_size == 3
    assert plan.component_world_sizes == {"env": 2, "rollout": 3}


def test_env_and_rollout_metadata_share_the_runner_global_step():
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    env = EnvWorker.__new__(EnvWorker)
    rollout = MultiStepRolloutWorker.__new__(MultiStepRolloutWorker)
    env._rank = 2
    rollout._rank = 1
    rollout.hf_model = object()

    env.set_global_step(7)
    rollout.set_global_step(7)

    expected = {
        "global_step": 7,
        "current_epoch": 3,
        "current_step": 4,
        "stage_id": 5,
    }
    assert env.trajectory_metadata(current_epoch=3, current_step=4, stage_id=5) == {
        **expected,
        "rank": 2,
    }
    assert rollout.trajectory_metadata(current_epoch=3, current_step=4, stage_id=5) == {
        **expected,
        "rank": 1,
    }
