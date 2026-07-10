import pytest
import torch

from rlinf.scheduler import Channel
from rlinf.workers.trajectory import (
    Actions,
    Observations,
    RolloutBootstrap,
    TrajectoryChannel,
    trajectory_queue_key,
)


def test_trajectory_queue_key_is_shared_by_protocol_endpoints():
    assert trajectory_queue_key("env_to_trajectory", 3) == "env_to_trajectory:3"


def test_trajectory_channel_converts_final_actions_to_bootstrap_record():
    channel = TrajectoryChannel(None, "TrajectoryGroup")
    action_batch = Actions(
        global_step=3,
        rank=1,
        current_epoch=2,
        current_step=4,
        actions=torch.zeros(2, 1, 3),
        prev_values=torch.ones(2, 1),
        is_bootstrap=True,
    )

    records = channel._storage_records(action_batch)

    assert len(records) == 1
    assert isinstance(records[0], RolloutBootstrap)
    assert records[0].prev_values is action_batch.prev_values


def test_actions_merge_preserves_protocol_metadata():
    actions = [
        Actions(
            global_step=1,
            rank=rank,
            current_epoch=0,
            current_step=2,
            stage_id=3,
            actions=torch.full((1, 1, 2), float(rank)),
            prev_values=torch.full((1, 1), float(rank)),
        )
        for rank in range(2)
    ]

    merged = Actions.merge_actions(actions)

    assert merged.global_step == 1
    assert merged.current_step == 2
    assert merged.stage_id == 3
    assert merged.actions.shape == (2, 1, 2)


def test_observation_protocol_split_keeps_storage_metadata_aligned():
    observations = Observations(
        global_step=1,
        rank=0,
        current_epoch=0,
        current_step=1,
        obs={"states": torch.arange(8).reshape(4, 2)},
        storage_obs={"states": torch.arange(8, 16).reshape(4, 2)},
        storage_rewards=torch.arange(4).reshape(4, 1),
    )

    shards = observations.split([1, 3])

    assert [shard.obs["states"].shape[0] for shard in shards] == [1, 3]
    assert [shard.storage_obs["states"].shape[0] for shard in shards] == [1, 3]
    assert [shard.storage_rewards.shape[0] for shard in shards] == [1, 3]


def test_channel_explicit_replica_rank_overrides_key_routing():
    channel = Channel.__new__(Channel)
    channel._local_channel = None
    channel._distributed = True
    channel._channel_actors_by_rank = {0: object(), 1: object()}
    channel._key_to_channel_rank_cache = {"key": 0}

    assert channel._get_channel_rank_by_key("key", channel_rank=1) == 1
    assert channel._get_channel_rank_by_key("key") == 0


def test_channel_rejects_invalid_explicit_replica_rank():
    channel = Channel.__new__(Channel)
    channel._local_channel = None
    channel._distributed = True
    channel._channel_actors_by_rank = {0: object()}

    with pytest.raises(ValueError, match="Invalid channel rank"):
        channel._get_channel_rank_by_key("key", channel_rank=1)
