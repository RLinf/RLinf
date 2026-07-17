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

import asyncio
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.data.embodied_io_struct import ChunkStepResult, EmbodiedRolloutResult
from rlinf.scheduler.channel.channel import DEFAULT_KEY
from rlinf.scheduler.channel.channel_worker import PeekQueue
from rlinf.workers.trajectory import (
    Actions,
    EnvBootstrap,
    Observations,
    Rewards,
    RolloutBootstrap,
    TrajectoryStorage,
    TrajectoryStorageConfig,
)
from rlinf.workers.trajectory.trajectory_worker import TrajectoryChannelWorker


def _trajectory_channel_worker() -> TrajectoryChannelWorker:
    worker = TrajectoryChannelWorker.__new__(TrajectoryChannelWorker)
    worker._queue_map = {DEFAULT_KEY: PeekQueue()}
    worker._storage_queue = asyncio.Queue()
    worker._storage_task = None
    worker.configure_storage(
        TrajectoryStorageConfig(
            num_envs=2,
            rollout_epoch=1,
            max_steps_per_rollout_epoch=1,
            max_episode_length=4,
            requires_values=True,
        ),
        slot_ids=(10, 11),
        actor_slot_indices={0: (0, 1)},
    )
    return worker


def test_channel_worker_publishes_only_a_complete_trajectory():
    worker = _trajectory_channel_worker()

    async def assemble():
        await worker.put_via_ray(
            Observations(
                global_step=3,
                rank=0,
                current_epoch=0,
                slot_ids=(10, 11),
                obs={"state": torch.tensor([[1.0], [2.0]])},
            ),
            weight=0,
        )
        await worker.put_via_ray(
            Actions(
                global_step=3,
                rank=0,
                current_epoch=0,
                slot_ids=(10, 11),
                actions=torch.tensor([[[3.0]], [[4.0]]]),
                prev_values=torch.zeros(2, 1),
            ),
            weight=0,
        )
        await worker.put_via_ray(
            EnvBootstrap(
                global_step=3,
                rank=0,
                current_epoch=0,
                slot_ids=(10, 11),
                rewards=torch.tensor([[5.0], [6.0]]),
                dones=torch.ones(2, 1, dtype=torch.bool),
            ),
            weight=0,
        )
        assert not worker.empty()
        await worker.put_via_ray(
            RolloutBootstrap(
                global_step=3,
                rank=0,
                current_epoch=0,
                slot_ids=(10, 11),
                prev_values=torch.ones(2, 1),
            ),
            weight=0,
        )
        await worker._storage_queue.join()
        return await worker.get_via_ray(key=("Actor", DEFAULT_KEY, 0))

    trajectory = asyncio.run(assemble())
    assert torch.equal(trajectory.actions[0, :, 0, 0], torch.tensor([3.0, 4.0]))
    assert torch.equal(trajectory.rewards[0, :, 0], torch.tensor([5.0, 6.0]))


def test_channel_worker_enqueues_live_data_before_storage_completes():
    worker = _trajectory_channel_worker()
    observation = Observations(
        global_step=3,
        rank=0,
        current_epoch=0,
        slot_ids=(10, 11),
        obs={"state": torch.tensor([[1.0], [2.0]])},
    )

    async def relay():
        await worker.put_via_ray(
            observation,
            weight=0,
            key=("Env", DEFAULT_KEY),
        )
        return await worker.get_via_ray(key=("Env", DEFAULT_KEY))

    assert asyncio.run(relay()) is observation


def test_terminal_live_observation_is_not_written_to_storage():
    worker = _trajectory_channel_worker()
    terminal_observation = Observations(
        global_step=3,
        rank=0,
        current_epoch=0,
        current_step=1,
        slot_ids=(10, 11),
        obs={"state": torch.tensor([[1.0], [2.0]])},
    )

    async def record_live_terminal_observation():
        await worker.put_via_ray(terminal_observation, weight=0)
        await worker._storage_queue.join()

    asyncio.run(record_live_terminal_observation())

    assert worker._storages == {}
    assert not worker.empty()


def test_storage_only_record_does_not_create_a_live_queue_entry():
    worker = _trajectory_channel_worker()
    bootstrap = EnvBootstrap(
        global_step=3,
        rank=0,
        current_epoch=0,
        slot_ids=(10, 11),
        observations={"state": torch.tensor([[1.0], [2.0]])},
        rewards=torch.tensor([[3.0], [4.0]]),
    )

    async def record_bootstrap():
        await worker.record_via_ray(bootstrap)
        await worker._storage_queue.join()

    asyncio.run(record_bootstrap())

    assert (3, 0) in worker._storages
    assert worker.empty()


def test_p2p_storage_only_record_does_not_create_a_live_queue_entry():
    worker = _trajectory_channel_worker()
    bootstrap = EnvBootstrap(
        global_step=3,
        rank=0,
        current_epoch=0,
        slot_ids=(10, 11),
        observations={"state": torch.tensor([[1.0], [2.0]])},
        rewards=torch.tensor([[3.0], [4.0]]),
    )
    worker.recv = lambda *_: bootstrap

    async def record_bootstrap():
        await worker.record(
            SimpleNamespace(root_group_name="Env", rank_path=0),
            transport_key="Rollout:EnvBootstrap",
        )
        await worker._storage_queue.join()

    asyncio.run(record_bootstrap())

    assert (3, 0) in worker._storages
    assert worker.empty()


def test_channel_worker_decodes_a_bootstrapped_tensor_frame():
    worker = _trajectory_channel_worker()
    src_addr = SimpleNamespace(root_group_name="Env", rank_path=0)
    first = Observations(
        global_step=3,
        rank=0,
        current_epoch=0,
        current_step=0,
        slot_ids=(10, 11),
        obs={"state": torch.tensor([[1.0], [2.0]])},
    )
    second = Observations(
        global_step=3,
        rank=0,
        current_epoch=0,
        current_step=1,
        slot_ids=(10, 11),
        obs={"state": torch.tensor([[3.0], [4.0]])},
    )
    worker._bootstrap_transport(src_addr, first, "Rollout:Observations")
    endpoint = worker._transport_endpoint(src_addr, "Rollout:Observations")
    frame = endpoint.encode(second)
    assert frame is not None

    def recv_tensor_frame(buffers, *_):
        for buffer, tensor in zip(buffers, (frame.header, *frame.tensors), strict=True):
            buffer.copy_(tensor)

    worker.recv_tensor_frame = recv_tensor_frame

    received = worker._receive_frame(src_addr, "Rollout:Observations")

    assert received.current_step == 1
    assert torch.equal(received.obs["state"], second.obs["state"])


def test_channel_worker_separates_logical_channel_transport_layouts():
    worker = _trajectory_channel_worker()
    src_addr = SimpleNamespace(root_group_name="Env", rank_path=0)
    rollout_observation = Observations(
        global_step=3,
        rank=0,
        current_epoch=0,
        slot_ids=(10, 11),
        obs={"state": torch.tensor([[1.0], [2.0]])},
    )
    reward_observation = Observations(
        global_step=3,
        rank=0,
        current_epoch=0,
        slot_ids=(10, 11),
        reward_inputs={"images": torch.ones(2, 3, 2, 2)},
    )
    worker._bootstrap_transport(src_addr, rollout_observation, "Rollout:Observations")
    worker._bootstrap_transport(src_addr, reward_observation, "Reward:Observations")

    rollout_endpoint = worker._transport_endpoint(src_addr, "Rollout:Observations")
    reward_endpoint = worker._transport_endpoint(src_addr, "Reward:Observations")
    assert rollout_endpoint is not reward_endpoint

    frame = reward_endpoint.encode(reward_observation)
    assert frame is not None

    def recv_tensor_frame(buffers, *_):
        for buffer, tensor in zip(buffers, (frame.header, *frame.tensors), strict=True):
            buffer.copy_(tensor)

    worker.recv_tensor_frame = recv_tensor_frame
    received = worker._receive_frame(src_addr, "Reward:Observations")

    assert received.reward_inputs is not None
    assert torch.equal(
        received.reward_inputs["images"], reward_observation.reward_inputs["images"]
    )


def test_storage_records_share_one_background_consumer():
    worker = _trajectory_channel_worker()

    async def record_two_items():
        await worker.put_via_ray(
            Observations(
                global_step=3,
                rank=0,
                current_epoch=0,
                slot_ids=(10, 11),
                obs={"state": torch.tensor([[1.0], [2.0]])},
            ),
            weight=0,
        )
        storage_task = worker._storage_task
        await worker.put_via_ray(
            Actions(
                global_step=3,
                rank=0,
                current_epoch=0,
                slot_ids=(10, 11),
                actions=torch.tensor([[[3.0]], [[4.0]]]),
            ),
            weight=0,
        )
        assert worker._storage_task is storage_task
        await worker._storage_queue.join()

    asyncio.run(record_two_items())


def test_pending_intervention_is_applied_after_actions_arrive():
    storage = TrajectoryStorage(
        num_envs=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        max_episode_length=4,
        requires_values=False,
    )

    storage.apply_intervention(
        current_epoch=0,
        current_step=0,
        slot_indices=[0, 1],
        intervene_actions=torch.tensor([[[9.0, 9.0]], [[8.0, 8.0]]]),
        intervene_flags=torch.tensor([[True], [False]]),
    )
    storage.write_actions(
        Actions(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=0,
            actions=torch.tensor([[[1.0, 1.0]], [[2.0, 2.0]]]),
            forward_inputs={
                "action": torch.tensor([[[1.0, 1.0]], [[2.0, 2.0]]]),
                "model_action": torch.tensor([[[1.0, 1.0]], [[2.0, 2.0]]]),
            },
        ),
        [0, 1],
    )

    assert torch.equal(
        storage.actions[0],
        torch.tensor([[[9.0, 9.0]], [[2.0, 2.0]]]),
    )
    assert torch.equal(
        storage.forward_inputs["action"][0],
        torch.tensor([[[9.0, 9.0]], [[2.0, 2.0]]]),
    )
    assert "model_action" not in storage.forward_inputs
    assert torch.equal(
        storage.intervene_flags[0],
        torch.tensor([[[True, True]], [[False, False]]]),
    )


def test_storage_restores_flattened_action_chunks_to_existing_layout():
    storage = TrajectoryStorage(
        num_envs=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        max_episode_length=4,
        requires_values=False,
    )
    first = torch.arange(20, dtype=torch.float32).reshape(2, 2, 5)
    storage.write_actions(
        Actions(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=0,
            actions=first,
        ),
        [0, 1],
    )
    flattened = torch.arange(20, 40, dtype=torch.float32).reshape(2, 10)
    storage.write_actions(
        Actions(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=1,
            actions=flattened,
        ),
        [0, 1],
    )

    assert torch.equal(storage.actions[1], flattened.reshape(2, 2, 5))


def test_storage_casts_later_floating_action_writes_to_established_dtype():
    storage = TrajectoryStorage(
        num_envs=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        max_episode_length=4,
        requires_values=False,
    )
    storage.write_actions(
        Actions(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=0,
            actions=torch.ones(2, 1, 2, dtype=torch.float32),
        ),
        [0, 1],
    )
    storage.write_actions(
        Actions(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=1,
            actions=torch.full((2, 1, 2), 2.0, dtype=torch.float64),
        ),
        [0, 1],
    )

    assert storage.actions.dtype == torch.float32
    assert torch.equal(storage.actions[1], torch.full((2, 1, 2), 2.0))


def test_storage_preallocates_registered_action_schema():
    storage = TrajectoryStorage(
        num_envs=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        max_episode_length=4,
        requires_values=False,
        action_chunk_shape=(5, 7),
    )

    assert storage.actions.shape == (2, 2, 5, 7)
    assert storage.actions.dtype == torch.float32
    assert storage.intervene_flags.dtype == torch.bool


def test_storage_schema_registers_all_action_record_fields_before_writing():
    storage = TrajectoryStorage(
        num_envs=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        max_episode_length=4,
        requires_values=True,
        action_chunk_shape=(5, 7),
    )

    storage.write_actions(
        Actions(
            global_step=0,
            rank=0,
            current_epoch=0,
            actions=torch.ones(2, 35),
            prev_logprobs=torch.full((2, 5, 7), 2.0, dtype=torch.float64),
            prev_values=torch.full((2, 1), 3.0, dtype=torch.float64),
            versions=torch.full((2, 5, 7), 4.0),
            forward_inputs={
                "action": torch.full((2, 35), 5.0),
                "model_action": torch.full((2, 5, 320), 6.0),
            },
        ),
        [0, 1],
    )

    assert storage.actions.shape == (1, 2, 5, 7)
    assert storage.prev_logprobs.shape == (1, 2, 5, 7)
    assert storage.prev_logprobs.dtype == torch.float32
    assert storage.prev_values.shape == (2, 2, 1)
    assert storage.forward_inputs["action"].shape == (1, 2, 5, 7)
    assert storage.forward_inputs["model_action"].shape == (1, 2, 5, 320)
    assert storage.schema.specs["rewards"].shape == (5,)
    assert storage.schema.specs["dones"].shape == (5,)


def test_observation_schema_registration_is_atomic_for_nested_tensor_fields():
    storage = TrajectoryStorage(
        num_envs=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        max_episode_length=4,
        requires_values=False,
    )
    storage.write_observations(
        Observations(
            global_step=0,
            rank=0,
            current_epoch=0,
            obs={
                "state": torch.ones(2, 3),
                "cameras": {"main": torch.ones(2, 3, 4, 4)},
            },
        ),
        [0, 1],
    )

    assert "curr_obs.state" in storage.schema.specs
    assert "curr_obs.cameras.main" in storage.schema.specs
    with pytest.raises(ValueError, match="curr_obs.state"):
        storage.write_observations(
            Observations(
                global_step=0,
                rank=0,
                current_epoch=0,
                current_step=1,
                obs={
                    "new_leaf": torch.ones(2, 1),
                    "state": torch.ones(2, 4),
                },
            ),
            [0, 1],
        )

    assert "curr_obs.new_leaf" not in storage.schema.specs


def test_reward_schema_broadcasts_observation_reward_over_action_chunks():
    storage = TrajectoryStorage(
        num_envs=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        max_episode_length=4,
        requires_values=False,
        action_chunk_shape=(5, 7),
        requires_external_rewards=True,
        env_reward_weight=0.25,
        reward_weight=2.0,
    )
    storage.write_env_bootstrap(
        EnvBootstrap(
            global_step=0,
            rank=0,
            current_epoch=0,
            rewards=torch.full((2, 5), 4.0),
        ),
        [0, 1],
    )
    storage.write_rewards(
        Rewards(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=1,
            rewards=torch.full((2, 1), 3.0),
        ),
        [0, 1],
    )

    assert storage._reward_model_rewards.shape == (1, 2, 1)
    assert storage.rewards.shape == (1, 2, 5)
    assert torch.equal(storage.rewards, torch.full((1, 2, 5), 7.0))


def test_bootstrap_without_reward_marks_reward_ready():
    storage = TrajectoryStorage(
        num_envs=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        max_episode_length=4,
        requires_values=True,
    )

    storage.write_observations(
        Observations(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=0,
            rewards=torch.ones(2, 1),
        ),
        [0, 1],
    )
    storage.write_actions(
        Actions(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=0,
            actions=torch.zeros(2, 1, 2),
            prev_values=torch.zeros(2, 1),
        ),
        [0, 1],
    )
    storage.write_env_bootstrap(
        EnvBootstrap(global_step=0, rank=0, current_epoch=0),
        [0, 1],
    )
    storage.write_rollout_bootstrap(
        RolloutBootstrap(
            global_step=0,
            rank=0,
            current_epoch=0,
            prev_values=torch.zeros(2, 1),
        ),
        [0, 1],
    )

    assert storage.complete()


def test_observation_without_reward_marks_reward_ready():
    storage = TrajectoryStorage(
        num_envs=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        max_episode_length=4,
        requires_values=True,
    )

    storage.write_observations(
        Observations(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=0,
            rewards=None,
        ),
        [0, 1],
    )
    storage.write_actions(
        Actions(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=0,
            actions=torch.zeros(2, 1, 2),
            prev_values=torch.zeros(2, 1),
        ),
        [0, 1],
    )
    storage.write_env_bootstrap(
        EnvBootstrap(global_step=0, rank=0, current_epoch=0),
        [0, 1],
    )
    storage.write_rollout_bootstrap(
        RolloutBootstrap(
            global_step=0,
            rank=0,
            current_epoch=0,
            prev_values=torch.zeros(2, 1),
        ),
        [0, 1],
    )

    assert storage.complete()


def test_storage_omits_live_only_task_descriptions():
    storage = TrajectoryStorage(
        num_envs=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        max_episode_length=4,
        requires_values=False,
    )

    storage.write_observations(
        Observations(
            global_step=0,
            rank=0,
            current_epoch=0,
            obs={
                "state": torch.tensor([[1.0], [2.0]]),
                "task_descriptions": ["pick", "place"],
            },
        ),
        [0, 1],
    )

    assert torch.equal(storage.curr_obs["state"][0], torch.tensor([[1.0], [2.0]]))
    assert "task_descriptions" not in storage.curr_obs


def test_reward_request_is_not_written_to_trajectory_storage():
    reward_request = Observations(
        global_step=0,
        rank=0,
        current_epoch=0,
        reward_inputs={"main_images": torch.zeros(1, 3, 4, 4)},
    )

    assert not TrajectoryChannelWorker._stores_live_item(
        ("Reward", "train_reward_obs"), reward_request
    )
    assert TrajectoryChannelWorker._stores_live_item(
        ("Rollout", "train_rollout_results"), reward_request
    )


def test_external_reward_is_required_before_trajectory_is_complete():
    storage = TrajectoryStorage(
        num_envs=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        max_episode_length=4,
        requires_values=False,
        requires_external_rewards=True,
        env_reward_weight=0.25,
        reward_weight=2.0,
    )
    storage.write_observations(
        Observations(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=0,
            obs={"state": torch.ones(1, 1)},
        ),
        [0],
    )
    storage.write_actions(
        Actions(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=0,
            actions=torch.ones(1, 1),
        ),
        [0],
    )
    storage.write_env_bootstrap(
        EnvBootstrap(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=1,
            observations={"state": torch.ones(1, 1)},
            rewards=torch.tensor([[4.0]]),
        ),
        [0],
    )

    assert not storage.complete()

    storage.write_rewards(
        Rewards(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=1,
            rewards=torch.tensor([[3.0]]),
        ),
        [0],
    )

    assert storage.complete()
    assert torch.equal(storage.rewards, torch.tensor([[[7.0]]]))


def test_reward_micro_batch_preserves_request_metadata_and_slots():
    from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker

    class OutputChannel:
        def __init__(self):
            self.records = []

        def record(self, item, async_op):
            self.records.append((item, async_op))

    worker = EmbodiedRewardWorker.__new__(EmbodiedRewardWorker)
    worker._rank = 1
    worker.cfg = OmegaConf.create({"reward": {"reward_mode": "per_step"}})
    worker.compute_image_rewards = lambda observations: observations["state"] * 2
    requests = [
        Observations(
            global_step=3,
            rank=0,
            current_epoch=0,
            current_step=step,
            stage_id=0,
            slot_ids=slot_ids,
            reward_inputs={"state": torch.tensor([[float(step)], [float(step)]])},
        )
        for step, slot_ids in ((1, (4, 5)), (2, (6, 7)))
    ]
    channel = OutputChannel()

    worker._record_trajectory_reward_batch(requests, channel)

    assert [record.slot_ids for record, _ in channel.records] == [(4, 5), (6, 7)]
    assert [record.current_step for record, _ in channel.records] == [1, 2]
    assert torch.equal(channel.records[0][0].rewards, torch.tensor([[2.0], [2.0]]))
    assert torch.equal(channel.records[1][0].rewards, torch.tensor([[4.0], [4.0]]))


def test_reward_request_normalizes_a_decoupled_trajectory_envelope():
    from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker

    request = Observations(
        global_step=3,
        rank=0,
        current_epoch=0,
        slot_ids=(4,),
        reward_inputs={"state": torch.ones(1, 1)},
    )

    normalized = EmbodiedRewardWorker._as_trajectory_reward_request(
        {"batch_index": "reward-3", "batch": request}
    )

    assert normalized is request


def test_terminal_bootstrap_reward_is_applied_once():
    storage = TrajectoryStorage(
        num_envs=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        max_episode_length=4,
        requires_values=True,
    )
    storage.write_observations(
        Observations(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=0,
        ),
        [0, 1],
    )
    storage.write_actions(
        Actions(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=0,
            actions=torch.zeros(2, 1, 2),
            prev_values=torch.zeros(2, 1),
        ),
        [0, 1],
    )
    storage.write_env_bootstrap(
        EnvBootstrap(
            global_step=0,
            rank=0,
            current_epoch=0,
            rewards=torch.tensor([[1.0], [2.0]]),
            dones=torch.tensor([[False], [True]]),
            truncations=torch.tensor([[False], [True]]),
        ),
        [0, 1],
    )
    storage.write_rollout_bootstrap(
        RolloutBootstrap(
            global_step=0,
            rank=0,
            current_epoch=0,
            prev_values=torch.tensor([[3.0], [4.0]]),
        ),
        [0, 1],
    )

    storage.apply_terminal_bootstrap_reward(
        current_epoch=0,
        gamma=0.5,
        auto_reset=True,
        bootstrap_type="standard",
    )
    storage.apply_terminal_bootstrap_reward(
        current_epoch=0,
        gamma=0.5,
        auto_reset=True,
        bootstrap_type="standard",
    )

    assert torch.equal(storage.rewards[0, :, 0], torch.tensor([1.0, 4.0]))


def test_exported_trajectory_uses_transition_length_for_rewards():
    storage = TrajectoryStorage(
        num_envs=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        max_episode_length=4,
        requires_values=True,
    )

    storage.write_observations(
        Observations(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=0,
            rewards=None,
            dones=torch.zeros(2, 1, dtype=torch.bool),
        ),
        [0, 1],
    )
    storage.write_actions(
        Actions(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=0,
            actions=torch.zeros(2, 1, 2),
            prev_values=torch.zeros(2, 1),
        ),
        [0, 1],
    )
    storage.write_observations(
        Observations(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=1,
            rewards=torch.full((2, 1), 1.0),
            dones=torch.zeros(2, 1, dtype=torch.bool),
        ),
        [0, 1],
    )
    storage.write_actions(
        Actions(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=1,
            actions=torch.zeros(2, 1, 2),
            prev_values=torch.zeros(2, 1),
        ),
        [0, 1],
    )
    storage.write_env_bootstrap(
        EnvBootstrap(
            global_step=0,
            rank=0,
            current_epoch=0,
            rewards=torch.full((2, 1), 3.0),
            dones=torch.ones(2, 1, dtype=torch.bool),
        ),
        [0, 1],
    )
    storage.write_rollout_bootstrap(
        RolloutBootstrap(
            global_step=0,
            rank=0,
            current_epoch=0,
            prev_values=torch.zeros(2, 1),
        ),
        [0, 1],
    )

    trajectory = storage.to_trajectory()

    assert trajectory.actions.shape[:2] == (2, 2)
    assert trajectory.rewards.shape[:2] == (2, 2)
    assert trajectory.dones.shape[:2] == (3, 2)
    assert trajectory.prev_values.shape[:2] == (3, 2)
    assert torch.equal(
        trajectory.rewards[:, :, 0], torch.tensor([[1.0, 1.0], [3.0, 3.0]])
    )
    assert torch.equal(
        trajectory.dones[:, :, 0],
        torch.tensor(
            [[False, False], [False, False], [True, True]],
            dtype=torch.bool,
        ),
    )


def test_storage_matches_embodied_rollout_result_time_alignment():
    storage = TrajectoryStorage(
        num_envs=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        max_episode_length=4,
        requires_values=True,
    )
    rollout_result = EmbodiedRolloutResult(max_episode_length=4)

    obs0 = {"states": torch.tensor([[0.0], [10.0]])}
    obs1 = {"states": torch.tensor([[1.0], [11.0]])}
    obs2 = {"states": torch.tensor([[2.0], [12.0]])}
    action0 = torch.tensor([[[101.0]], [[102.0]]])
    action1 = torch.tensor([[[201.0]], [[202.0]]])
    logprob0 = torch.tensor([[[0.1]], [[0.2]]])
    logprob1 = torch.tensor([[[0.3]], [[0.4]]])
    value0 = torch.tensor([[1.0], [2.0]])
    value1 = torch.tensor([[3.0], [4.0]])
    value2 = torch.tensor([[5.0], [6.0]])
    version0 = torch.tensor([[[7.0]], [[7.0]]])
    version1 = torch.tensor([[[8.0]], [[8.0]]])
    reward0 = torch.tensor([[0.5], [0.6]])
    reward1 = torch.tensor([[1.5], [1.6]])
    done0 = torch.zeros(2, 1, dtype=torch.bool)
    done1 = torch.tensor([[False], [True]])
    done2 = torch.tensor([[True], [True]])

    rollout_result.append_step_result(
        ChunkStepResult(
            actions=action0,
            prev_logprobs=logprob0,
            prev_values=value0,
            forward_inputs={"action": action0},
            versions=version0,
            dones=done0,
            terminations=done0,
            truncations=done0,
            rewards=None,
        )
    )
    rollout_result.append_transitions(obs0.copy(), obs1.copy())
    rollout_result.append_step_result(
        ChunkStepResult(
            actions=action1,
            prev_logprobs=logprob1,
            prev_values=value1,
            forward_inputs={"action": action1},
            versions=version1,
            dones=done1,
            terminations=done1,
            truncations=torch.zeros_like(done1),
            rewards=reward0,
        )
    )
    rollout_result.append_transitions(obs1.copy(), obs2.copy())
    rollout_result.append_step_result(
        ChunkStepResult(
            prev_values=value2,
            dones=done2,
            terminations=done2,
            truncations=torch.zeros_like(done2),
            rewards=reward1,
        )
    )

    storage.write_observations(
        Observations(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=0,
            obs=obs0,
            next_obs=obs1,
            rewards=None,
            dones=done0,
            terminations=done0,
            truncations=done0,
        ),
        [0, 1],
    )
    storage.write_actions(
        Actions(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=0,
            actions=action0,
            prev_logprobs=logprob0,
            prev_values=value0,
            versions=version0,
            forward_inputs={"action": action0},
        ),
        [0, 1],
    )
    storage.write_observations(
        Observations(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=1,
            obs=obs1,
            next_obs=obs2,
            rewards=reward0,
            dones=done1,
            terminations=done1,
            truncations=torch.zeros_like(done1),
        ),
        [0, 1],
    )
    storage.write_actions(
        Actions(
            global_step=0,
            rank=0,
            current_epoch=0,
            current_step=1,
            actions=action1,
            prev_logprobs=logprob1,
            prev_values=value1,
            versions=version1,
            forward_inputs={"action": action1},
        ),
        [0, 1],
    )
    storage.write_env_bootstrap(
        EnvBootstrap(
            global_step=0,
            rank=0,
            current_epoch=0,
            rewards=reward1,
            dones=done2,
            terminations=done2,
            truncations=torch.zeros_like(done2),
        ),
        [0, 1],
    )
    storage.write_rollout_bootstrap(
        RolloutBootstrap(
            global_step=0,
            rank=0,
            current_epoch=0,
            prev_values=value2,
        ),
        [0, 1],
    )

    expected = rollout_result.to_trajectory()
    actual = storage.to_trajectory()

    assert storage.complete()
    for field_name in (
        "actions",
        "rewards",
        "dones",
        "terminations",
        "truncations",
        "prev_logprobs",
        "prev_values",
        "versions",
    ):
        assert torch.equal(getattr(actual, field_name), getattr(expected, field_name))
    assert torch.equal(actual.curr_obs["states"], expected.curr_obs["states"])
    assert torch.equal(actual.next_obs["states"], expected.next_obs["states"])
    assert torch.equal(
        actual.forward_inputs["action"], expected.forward_inputs["action"]
    )
