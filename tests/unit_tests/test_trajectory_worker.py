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
from collections import defaultdict
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.data.schema.embodied_trajectory_builder import (
    EmbodiedLerobotTrajectoryBuilder,
)
from rlinf.data.schema.embodied_types import (
    EmbodiedRolloutResult,
    EnvResult,
    TrajectoryKey,
    TrajectorySource,
)
from rlinf.scheduler.channel.trajectory_channel.assembler import (
    AssembledChunk,
    TrajectoryEventAssembler,
)
from rlinf.scheduler.channel.trajectory_channel.collectors import (
    CollectorOutput,
    OnlineLerobotTrajectoryCollector,
    PipelineTrajectoryCollector,
    RolloutTrajectoryCollector,
    create_trajectory_collector,
)
from rlinf.scheduler.channel.trajectory_channel.data import (
    DummyPolicyStep,
    EnvStepResult,
    PolicyStep,
    TrajectoryStart,
)
from rlinf.scheduler.channel.trajectory_channel.trajectory_worker import (
    TrajectoryWorker,
)
from rlinf.scheduler.worker.worker import WorkerAddress


def _config(**overrides):
    cfg = OmegaConf.create(
        {
            "env": {
                "train": {
                    "auto_reset": True,
                    "ignore_terminations": False,
                    "max_episode_steps": 4,
                    "rollout_epoch": 1,
                    "total_num_envs": 1,
                }
            },
            "rollout": {
                "collect_prev_infos": True,
                "collect_transitions": False,
                "pipeline_stage_num": 1,
            },
            "actor": {
                "micro_batch_size": 1,
                "seed": 1,
                "model": {"action_dim": 1, "num_action_chunks": 1},
            },
            "algorithm": {
                "adv_type": "gae",
                "dagger": {"online_lerobot": {"enabled": False}},
                "gamma": 0.5,
                "group_size": 1,
                "loss_type": "actor_critic",
                "normalize_advantages": False,
                "reward_type": "chunk_level",
                "shuffle_rollout": False,
            },
            "reward": {"env_reward_weight": 1.0, "reward_weight": 1.0},
            "runner": {"task_type": "embodied", "use_training_pipeline": False},
        }
    )
    return OmegaConf.merge(cfg, overrides)


def _source(key: TrajectoryKey, size: int = 1, offset: int = 0):
    return TrajectorySource(key, size, offset)


def _policy(key: TrajectoryKey, value: float = 1.0, **source_kwargs):
    value_tensor = torch.tensor([[value]])
    return PolicyStep(
        sources=[_source(key, **source_kwargs)],
        obs={"states": value_tensor},
        rollout_result=EmbodiedRolloutResult(
            actions=value_tensor,
            forward_inputs={"action": value_tensor},
            prev_logprobs=torch.zeros(1, 1),
            prev_values=torch.zeros(1, 1),
        ),
    )


def _env_step(
    key: TrajectoryKey,
    *,
    episode_data=None,
    value: float = 2.0,
    **source_kwargs,
):
    value_tensor = torch.tensor([[value]])
    return EnvStepResult(
        sources=[_source(key, **source_kwargs)],
        result=EnvResult(
            rewards=torch.ones(1, 1),
            dones=torch.ones(1, 1, dtype=torch.bool),
            truncations=torch.ones(1, 1, dtype=torch.bool),
            episode_data=episode_data,
        ),
        next_obs={"states": value_tensor},
        forward_inputs={"states": value_tensor},
        bootstrap_values=torch.tensor([[4.0]]),
        final_prev_values=torch.tensor([[5.0]]),
    )


def _chunk(
    key: TrajectoryKey,
    *,
    policy=None,
    env=None,
    initial_env_result=None,
):
    return AssembledChunk(
        key=key,
        source=(key.env_rank, key.stage_id),
        policy=policy or _policy(key),
        env=env or _env_step(key),
        initial_env_result=initial_env_result,
    )


def _episode_data():
    return {
        "chunk_actions": torch.zeros(1, 1, 1),
        "obs_list": [{"states": torch.zeros(1, 1)}],
        "terminations": torch.zeros(1, 1, dtype=torch.bool),
        "truncations": torch.zeros(1, 1, dtype=torch.bool),
        "infos_list": [{}],
    }


def test_try_subscribe_returns_false_without_starting_recv():
    worker = object.__new__(TrajectoryWorker)
    worker._receiver_error = None
    worker._output_queues = defaultdict(asyncio.Queue)
    worker.send = Mock()

    ready = asyncio.run(
        worker.try_subscribe(
            WorkerAddress(root_group_name="actor", ranks=0), "default", 7
        )
    )

    assert ready is False
    worker.send.assert_not_called()


def test_try_subscribe_dequeues_and_submits_async_send():
    worker = object.__new__(TrajectoryWorker)
    worker._receiver_error = None
    worker._output_queues = defaultdict(asyncio.Queue)
    worker._output_queues["actor:0"].put_nowait("trajectory")
    worker.send = Mock()
    address = WorkerAddress(root_group_name="actor", ranks=0)

    ready = asyncio.run(worker.try_subscribe(address, "actor:0", 11))

    assert ready is True
    assert worker._output_queues["actor:0"].empty()
    worker.send.assert_called_once_with(
        object="trajectory",
        dst_group_name="actor",
        dst_rank=0,
        piggyback_payload=11,
        async_op=True,
    )


def test_worker_starts_one_persistent_loop_per_producer():
    async def run_test():
        worker = object.__new__(TrajectoryWorker)
        worker._receiver_tasks = {}
        worker._receiver_error = None
        release = asyncio.Event()
        started = []

        async def consume(address):
            started.append(address)
            await release.wait()

        worker._consume = consume
        addresses = [
            WorkerAddress(root_group_name="env", ranks=0),
            WorkerAddress(root_group_name="rollout", ranks=1),
        ]

        await worker.start_receivers(addresses)
        assert started == addresses
        assert set(worker._receiver_tasks) == set(addresses)

        for task in worker._receiver_tasks.values():
            task.cancel()
        await asyncio.gather(*worker._receiver_tasks.values(), return_exceptions=True)

    asyncio.run(run_test())


def test_worker_enqueues_collector_outputs():
    worker = object.__new__(TrajectoryWorker)
    worker._output_queues = defaultdict(asyncio.Queue)
    chunk = Mock()
    worker._assembler = Mock()
    worker._assembler.push.return_value = [chunk]
    worker._collector = Mock()
    worker._collector.collect.return_value = [
        CollectorOutput(queue_key="actor:2", data="batch")
    ]
    event = object()

    worker._apply_event(event)

    worker._assembler.push.assert_called_once_with(event)
    worker._collector.collect.assert_called_once_with(chunk)
    worker._assembler.acknowledge.assert_called_once_with(chunk.key)
    assert worker._output_queues["actor:2"].get_nowait() == "batch"


def test_worker_keeps_assembled_chunk_when_collection_fails():
    worker = object.__new__(TrajectoryWorker)
    worker._output_queues = defaultdict(asyncio.Queue)
    chunk = Mock()
    worker._assembler = Mock()
    worker._assembler.push.return_value = [chunk]
    worker._collector = Mock()
    worker._collector.collect.side_effect = ValueError("invalid collection")

    with pytest.raises(ValueError, match="invalid collection"):
        worker._apply_event(object())

    worker._assembler.acknowledge.assert_not_called()


def test_assembler_joins_out_of_order_source_fragments():
    assembler = TrajectoryEventAssembler(source_batch_size=2)
    key = TrajectoryKey(0, 0, 0, 0, 0)
    assembler.push(TrajectoryStart(source=_source(key, size=2), result=EnvResult()))

    assert assembler.push(_policy(key, 11, size=1, offset=1)) == []
    assert assembler.push(_policy(key, 10, size=1, offset=0)) == []
    chunks = assembler.push(
        EnvStepResult(
            sources=[_source(key, size=2)],
            result=EnvResult(rewards=torch.ones(2, 1)),
            next_obs={"states": torch.tensor([[20], [21]])},
            forward_inputs={"states": torch.tensor([[20], [21]])},
            bootstrap_values=None,
            final_prev_values=None,
        )
    )

    assert len(chunks) == 1
    assert torch.equal(chunks[0].policy.obs["states"], torch.tensor([[10], [11]]))
    assert torch.equal(chunks[0].env.next_obs["states"], torch.tensor([[20], [21]]))


def test_assembler_reuses_complete_source_events():
    assembler = TrajectoryEventAssembler(source_batch_size=2)
    key = TrajectoryKey(0, 0, 0, 0, 0)
    policy = _policy(key, size=2)
    env = EnvStepResult(
        sources=[_source(key, size=2)],
        result=EnvResult(rewards=torch.ones(2, 1)),
        next_obs={"states": torch.ones(2, 1)},
        forward_inputs=None,
        bootstrap_values=None,
        final_prev_values=None,
    )
    assembler.push(TrajectoryStart(source=_source(key, size=2), result=EnvResult()))

    assert assembler.push(policy) == []
    chunks = assembler.push(env)

    assert chunks[0].policy is policy
    assert chunks[0].env is env


def test_later_chunk_does_not_consume_initial_state():
    assembler = TrajectoryEventAssembler(source_batch_size=1)
    first_key = TrajectoryKey(0, 0, 0, 0, 0)
    later_key = TrajectoryKey(0, 0, 0, 0, 1)
    initial = EnvResult(dones=torch.zeros(1, 1, dtype=torch.bool))
    assembler.push(TrajectoryStart(source=_source(first_key), result=initial))

    assembler.push(_policy(later_key))
    later_chunks = assembler.push(_env_step(later_key))
    assembler.push(_policy(first_key))
    first_chunks = assembler.push(_env_step(first_key))

    assert later_chunks[0].initial_env_result is None
    assert first_chunks[0].initial_env_result is initial


def test_trajectory_start_completes_waiting_chunk():
    assembler = TrajectoryEventAssembler(source_batch_size=1)
    key = TrajectoryKey(2, 1, 0, 0, 0)
    assembler.push(_policy(key))
    assert assembler.push(_env_step(key)) == []

    chunks = assembler.push(TrajectoryStart(source=_source(key), result=EnvResult()))

    assert len(chunks) == 1
    assert chunks[0].key == key


def test_rollout_collector_bootstraps_reward_and_emits_trajectory():
    collector = RolloutTrajectoryCollector(
        _config(), source_count=1, chunk_count=1, shards_per_source=1
    )
    key = TrajectoryKey(3, 0, 0, 0, 0)

    outputs = collector.collect(_chunk(key, initial_env_result=EnvResult()))

    assert len(outputs) == 1
    assert outputs[0].queue_key == "default"
    assert torch.equal(outputs[0].data.rewards, torch.tensor([[[3.0]]]))


def test_rollout_collector_preserves_scope_when_conversion_fails():
    collector = RolloutTrajectoryCollector(
        _config(), source_count=1, chunk_count=1, shards_per_source=1
    )
    key = TrajectoryKey(3, 0, 0, 0, 0)
    builder = Mock()
    builder.to_splited_trajectories.side_effect = RuntimeError("invalid trajectory")
    collector._builders = {3: {(0, 0): builder}}

    with pytest.raises(RuntimeError, match="invalid trajectory"):
        collector.collect(_chunk(key))

    assert collector._builders == {3: {(0, 0): builder}}
    assert collector._completed_keys == {key}


def test_rollout_collector_rejects_duplicate_completed_key():
    cfg = _config(env={"train": {"rollout_epoch": 2}})
    collector = RolloutTrajectoryCollector(
        cfg, source_count=1, chunk_count=1, shards_per_source=1
    )
    key = TrajectoryKey(3, 0, 0, 0, 0)
    collector.collect(_chunk(key))

    with pytest.raises(ValueError, match="duplicate trajectory event"):
        collector.collect(_chunk(key))


def test_online_lerobot_collector_accepts_dummy_step_and_emits_shards():
    cfg = _config(
        algorithm={"dagger": {"online_lerobot": {"enabled": True}}},
    )
    collector = OnlineLerobotTrajectoryCollector(
        cfg, source_count=1, chunk_count=1, shards_per_source=2
    )
    key = TrajectoryKey(3, 0, 0, 0, 0)
    dummy = DummyPolicyStep(
        sources=[_source(key)],
        obs={"states": torch.zeros(1, 1)},
        actions=torch.zeros(1, 1, 1),
    )
    env = _env_step(key, episode_data=_episode_data())

    outputs = collector.collect(_chunk(key, policy=dummy, env=env))

    assert len(outputs) == 2
    assert all(output.queue_key == "default" for output in outputs)


def test_pipeline_collector_routes_completed_epoch_to_actor():
    cfg = _config(runner={"use_training_pipeline": True})
    collector = PipelineTrajectoryCollector(
        cfg, source_count=1, chunk_count=1, actor_world_size=1
    )
    collector._prepare_pipeline_batch = lambda trajectory: {"value": trajectory}
    collector._pipeline_micro_batches = lambda batch, actor_rank: [batch]
    key = TrajectoryKey(2, 1, 0, 0, 0)

    outputs = collector.collect(_chunk(key, initial_env_result=EnvResult()))

    assert len(outputs) == 1
    assert outputs[0].queue_key == "actor:0"
    assert (2, 1) not in collector._builders


@pytest.mark.parametrize(
    ("overrides", "expected_type"),
    [
        ({}, RolloutTrajectoryCollector),
        (
            {"runner": {"use_training_pipeline": True}},
            PipelineTrajectoryCollector,
        ),
        (
            {"algorithm": {"dagger": {"online_lerobot": {"enabled": True}}}},
            OnlineLerobotTrajectoryCollector,
        ),
    ],
)
def test_collector_factory_selects_configured_strategy(overrides, expected_type):
    collector = create_trajectory_collector(
        _config(**overrides),
        source_count=1,
        chunk_count=1,
        shards_per_source=1,
        actor_world_size=1,
    )

    assert isinstance(collector, expected_type)


def test_collector_factory_rejects_pipeline_with_online_lerobot():
    cfg = _config(
        runner={"use_training_pipeline": True},
        algorithm={"dagger": {"online_lerobot": {"enabled": True}}},
    )

    with pytest.raises(ValueError, match="does not support online LeRobot"):
        create_trajectory_collector(
            cfg,
            source_count=1,
            chunk_count=1,
            shards_per_source=1,
            actor_world_size=1,
        )


def test_online_lerobot_dummy_step_records_intervened_action():
    collector = EmbodiedLerobotTrajectoryBuilder(
        max_episode_length=4,
        num_envs=1,
        num_action_chunks=1,
        action_dim=2,
    )

    collector.append_chunk_episode_data(
        policy_output=None,
        chunk_actions=torch.tensor([[[1.0, 2.0]]]),
        obs_list=[
            {
                "states": torch.tensor([[0.0, 0.0]]),
                "task_descriptions": ["pick"],
            }
        ],
        terminations=torch.tensor([[False]]),
        truncations=torch.tensor([[False]]),
        infos_list=[
            {
                "intervene_action": torch.tensor([[9.0, 8.0]]),
                "intervene_flag": torch.tensor([True]),
            }
        ],
    )

    frame = collector._env_buffers[0][0]
    assert np.array_equal(frame["actions"], np.array([9.0, 8.0]))
    assert frame["intervene_flag"].item()
