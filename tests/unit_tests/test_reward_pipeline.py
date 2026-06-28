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
from collections import defaultdict, deque

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    EnvOutput,
    RolloutResult,
)
from rlinf.scheduler import CommMapper
from rlinf.scheduler.hardware.accelerators.accelerator import AcceleratorType
from rlinf.workers.env.env_worker import EnvWorker, PendingRolloutStep
from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker


class _FakeAsyncWork:
    def __init__(self, value):
        self._value = value

    async def async_wait(self):
        return self._value


class _FakeKeyedChannel:
    def __init__(self, items=None):
        self.items = {key: list(values) for key, values in (items or {}).items()}
        self.puts = []

    def get(self, key, async_op=False):
        value = self.items[key].pop(0)
        if async_op:
            return _FakeAsyncWork(value)
        return value

    def qsize(self, key):
        return len(self.items.get(key, []))

    def put(self, item, key, async_op=False):
        self.puts.append((key, item, async_op))


def _build_reward_input(batch_size: int, *, mark_last_run: bool = False) -> dict:
    reward_input = {
        "main_images": torch.zeros((batch_size, 2, 2, 3), dtype=torch.uint8),
    }
    if mark_last_run:
        reward_input["last_run"] = torch.ones((batch_size, 1), dtype=torch.bool)
    return reward_input


def test_reward_worker_aggregates_multiple_requests_before_inference():
    worker = object.__new__(EmbodiedRewardWorker)
    worker.src_ranks = {"train": [(0, 2), (1, 1)]}
    worker._rank = 7
    worker.local_num_train_envs = 3
    worker.aggregate_request_count = 4

    channel = _FakeKeyedChannel(
        {
            CommMapper.build_channel_key(0, 7, extra="train_reward_input"): [
                _build_reward_input(2),
                _build_reward_input(2, mark_last_run=True),
            ],
            CommMapper.build_channel_key(1, 7, extra="train_reward_input"): [
                _build_reward_input(1),
                _build_reward_input(1, mark_last_run=True),
            ],
        }
    )

    merged_inputs, batch_sizes, last_run_count = asyncio.run(
        worker.recv_aggregated_reward_inputs(channel, mode="train")
    )

    assert batch_sizes == [3, 3]
    assert last_run_count == 3
    assert merged_inputs["main_images"].shape[0] == 6


def test_reward_worker_sends_aggregated_outputs_back_in_request_order():
    worker = object.__new__(EmbodiedRewardWorker)
    worker.dst_ranks = {"train": [(5, 2), (6, 1)]}
    worker._rank = 7

    output_channel = _FakeKeyedChannel()
    rewards = torch.arange(6, dtype=torch.float32).unsqueeze(-1)

    worker.send_aggregated_reward_output(output_channel, rewards, [3, 3])

    assert len(output_channel.puts) == 4
    first_key, first_tensor, _ = output_channel.puts[0]
    second_key, second_tensor, _ = output_channel.puts[1]
    third_key, third_tensor, _ = output_channel.puts[2]
    fourth_key, fourth_tensor, _ = output_channel.puts[3]

    assert first_key == CommMapper.build_channel_key(7, 5, extra="reward_output")
    assert second_key == CommMapper.build_channel_key(7, 6, extra="reward_output")
    assert third_key == CommMapper.build_channel_key(7, 5, extra="reward_output")
    assert fourth_key == CommMapper.build_channel_key(7, 6, extra="reward_output")
    assert first_tensor.shape[0] == 2
    assert second_tensor.shape[0] == 1
    assert third_tensor.shape[0] == 2
    assert fourth_tensor.shape[0] == 1
    assert torch.equal(first_tensor.flatten(), torch.tensor([0.0, 1.0]))
    assert torch.equal(second_tensor.flatten(), torch.tensor([2.0]))
    assert torch.equal(third_tensor.flatten(), torch.tensor([3.0, 4.0]))
    assert torch.equal(fourth_tensor.flatten(), torch.tensor([5.0]))


def test_reward_worker_records_model_timing_attrs_as_seconds():
    worker = object.__new__(EmbodiedRewardWorker)
    worker._timer_metrics = {}

    class _Model:
        last_timing_ms = {"generate_ms": 25.0, "total_ms": 40.0}

    worker.model = _Model()
    worker._record_reward_model_timings()

    assert worker._timer_metrics["model/generate"] == 0.025
    assert worker._timer_metrics["model/total"] == 0.04


def test_standalone_realworld_reward_worker_initializes_image_reward_model():
    worker = object.__new__(EmbodiedRewardWorker)
    worker._standalone_realworld = True
    worker.device = "cpu"

    class _Model:
        def __init__(self):
            self.to_device = None
            self.eval_called = False
            self.reward_input = None

        def to(self, device):
            self.to_device = device
            return self

        def eval(self):
            self.eval_called = True

        def compute_reward(self, reward_input):
            self.reward_input = reward_input
            return torch.tensor([0.25, 0.75], dtype=torch.float32)

    model = _Model()
    worker.model_provider_func = lambda: model

    EmbodiedRewardWorker.init_worker(worker)

    assert worker.dst_ranks == {"train": [(0, 1)]}
    assert worker.src_ranks == {"train": [(0, 1)]}
    assert worker.local_num_train_envs == 1
    assert worker.model is model
    assert model.to_device == "cpu"
    assert model.eval_called is True

    images = torch.zeros((2, 4, 4, 3), dtype=torch.uint8)
    rewards = worker.compute_image_rewards(images)

    assert model.reward_input["main_images"] is images
    assert rewards.device.type == "cpu"
    assert rewards.shape == (2, 1)
    assert torch.equal(rewards.flatten(), torch.tensor([0.25, 0.75]))


def test_reward_worker_rejects_inconsistent_aggregated_reward_input_keys():
    worker = object.__new__(EmbodiedRewardWorker)
    worker.src_ranks = {"train": [(0, 1)]}
    worker._rank = 7
    worker.local_num_train_envs = 2
    worker.aggregate_request_count = 2

    channel = _FakeKeyedChannel(
        {
            CommMapper.build_channel_key(0, 7, extra="train_reward_input"): [
                _build_reward_input(1),
                {
                    **_build_reward_input(1),
                    "env_infos": {"episode": {"success": torch.ones((1, 1))}},
                },
            ],
        }
    )

    with pytest.raises(AssertionError, match="Inconsistent aggregated reward input"):
        asyncio.run(worker.recv_aggregated_reward_inputs(channel, mode="train"))


def test_standalone_realworld_reward_worker_rejects_sglang_backend(monkeypatch):
    worker = object.__new__(EmbodiedRewardWorker)
    worker._standalone_realworld = True
    worker.device = "cpu"
    worker.local_num_train_envs = 1
    worker.cfg = OmegaConf.create(
        {
            "reward": {
                "model": {
                    "model_type": "history_vlm",
                    "inference_backend": "sglang",
                },
            }
        }
    )

    monkeypatch.setattr(
        "rlinf.workers.reward.reward_worker.get_reward_model_class",
        lambda model_type, inference_backend=None: object,
    )

    with pytest.raises(ValueError, match="standalone_realworld"):
        worker.model_provider_func()


def test_env_terminal_reward_request_requires_final_obs():
    worker = object.__new__(EnvWorker)
    worker.reward_mode = "terminal"

    env_output = EnvOutput(
        obs={"states": torch.zeros((1, 1), dtype=torch.float32)},
        final_obs=None,
        dones=torch.zeros((1, 1), dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="terminal reward request"):
        worker.build_reward_request(env_output)


def test_env_pending_reward_drain_preserves_fifo_order():
    worker = object.__new__(EnvWorker)
    worker.reward_pending_step_window = 2

    finalized = []
    worker.recv_pending_reward_output = (
        lambda recv_channel, env_output, pending_step=None: torch.tensor(
            [1.0], dtype=torch.float32
        )
    )
    worker.finalize_pending_rollout_step = (
        lambda pending_step, reward_model_output, env_metrics: finalized.append(
            (
                pending_step.stage_id,
                pending_step.reward_required,
                reward_model_output is not None,
            )
        )
    )

    pending_steps = deque(
        [
            PendingRolloutStep(
                stage_id=0,
                env_output=None,
                rollout_result=None,
                reward_required=False,
            ),
            PendingRolloutStep(
                stage_id=1,
                env_output=None,
                rollout_result=None,
                reward_required=True,
            ),
            PendingRolloutStep(
                stage_id=2,
                env_output=None,
                rollout_result=None,
                reward_required=True,
            ),
        ]
    )

    remaining_reward_count = worker.drain_pending_rollout_steps(
        pending_steps,
        recv_channel=None,
        env_metrics=defaultdict(list),
        pending_reward_count=2,
    )

    assert finalized == [(0, False, False), (1, True, True)]
    assert remaining_reward_count == 1
    assert len(pending_steps) == 1
    assert pending_steps[0].stage_id == 2


def test_env_no_pending_reward_mode_waits_and_finalizes_immediately():
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "env": {"train": {"max_episode_steps": 8, "auto_reset": False}},
        }
    )
    worker.stage_num = 1
    worker.rollout_epoch = 1
    worker.n_train_chunk_steps = 1
    worker.reward_pending_step_window = 0
    worker.collect_transitions = False
    worker.use_training_pipeline = False
    worker._timer_metrics = {}
    worker._accelerator_type = AcceleratorType.NO_ACCEL
    worker._prefetched_train_bootstrap = None

    env_outputs = [
        EnvOutput(
            obs={"states": torch.zeros((1, 1), dtype=torch.float32)},
            final_obs={"states": torch.zeros((1, 1), dtype=torch.float32)},
            dones=torch.zeros((1, 1), dtype=torch.bool),
            terminations=torch.zeros((1, 1), dtype=torch.bool),
            truncations=torch.zeros((1, 1), dtype=torch.bool),
            rewards=torch.zeros((1, 1), dtype=torch.float32),
        ),
        EnvOutput(
            obs={"states": torch.ones((1, 1), dtype=torch.float32)},
            final_obs={"states": torch.ones((1, 1), dtype=torch.float32)},
            dones=torch.zeros((1, 1), dtype=torch.bool),
            terminations=torch.zeros((1, 1), dtype=torch.bool),
            truncations=torch.zeros((1, 1), dtype=torch.bool),
            rewards=torch.zeros((1, 1), dtype=torch.float32),
        ),
    ]
    rollout_result = RolloutResult(actions=torch.zeros((1, 1), dtype=torch.float32))
    sent_obs = []
    finalized = []

    worker.bootstrap_step = lambda: [env_outputs[0]]
    worker.send_env_batch = lambda channel, batch: sent_obs.append(
        batch["obs"]["states"].clone()
    )
    worker.send_reward_request = (
        lambda env_output,
        send_channel,
        stage_id,
        last_run=False,
        epoch=None,
        chunk_step_idx=None: (
            True,
            None,
        )
    )
    worker.recv_pending_reward_output = (
        lambda recv_channel, env_output, pending_step=None: torch.tensor(
            [[0.5]], dtype=torch.float32
        )
    )
    worker.recv_rollout_results = lambda input_channel, mode: rollout_result
    worker.env_interact_step = lambda actions, stage_id: (env_outputs[1], {})
    worker.record_env_metrics = lambda env_metrics, env_info, epoch: None
    worker.finalize_pending_rollout_step = (
        lambda pending_step, reward_model_output, env_metrics: finalized.append(
            (pending_step.stage_id, pending_step.reward_required, reward_model_output)
        )
    )

    def drain_pending_steps(
        pending_steps,
        recv_channel,
        env_metrics,
        drain_all=False,
        pending_reward_count=0,
    ):
        del recv_channel, env_metrics, drain_all
        pending_steps.clear()
        return pending_reward_count

    worker.drain_pending_rollout_steps = drain_pending_steps
    worker.store_last_obs_and_intervened_info = lambda env_output_list: None
    worker.finish_rollout = lambda: None

    env_metrics = asyncio.run(
        worker._run_interact_once(
            input_channel=None,
            rollout_channel=None,
            reward_channel=object(),
            actor_channel=None,
            cooperative_yield=False,
        )
    )

    assert [obs.item() for obs in sent_obs] == [0.0, 1.0]
    assert len(finalized) == 1
    assert finalized[0][0] == 0
    assert finalized[0][1] is True
    assert torch.equal(finalized[0][2], torch.tensor([[0.5]], dtype=torch.float32))
    assert torch.equal(
        env_metrics["reward_pending_count"],
        torch.tensor([0.0], dtype=torch.float32),
    )


def test_env_bootstrap_pending_step_does_not_append_actions():
    worker = object.__new__(EnvWorker)
    worker.collect_prev_infos = True
    worker.reward_mode = "per_step"
    worker.history_reward_assign = False
    worker.rollout_results = [EmbodiedRolloutResult(max_episode_length=8)]
    worker.compute_bootstrap_rewards = (
        lambda env_output, bootstrap_values, reward_model_output: torch.ones((2, 1))
    )

    pending_step = PendingRolloutStep(
        stage_id=0,
        env_output=EnvOutput(
            obs={"state": torch.zeros((2, 1), dtype=torch.float32)},
            dones=torch.zeros((2, 1), dtype=torch.bool),
            terminations=torch.zeros((2, 1), dtype=torch.bool),
            truncations=torch.zeros((2, 1), dtype=torch.bool),
            rewards=torch.zeros((2, 1), dtype=torch.float32),
        ),
        rollout_result=RolloutResult(
            actions=torch.zeros((2, 8), dtype=torch.float32),
            prev_logprobs=torch.zeros((2, 8), dtype=torch.float32),
            prev_values=torch.zeros((2, 1), dtype=torch.float32),
            bootstrap_values=torch.zeros((2, 1), dtype=torch.float32),
            forward_inputs={"action": torch.zeros((2, 1, 8), dtype=torch.float32)},
            versions=torch.zeros((2, 1), dtype=torch.float32),
        ),
        reward_required=False,
        append_rollout_payload=False,
    )

    worker.finalize_pending_rollout_step(
        pending_step=pending_step,
        reward_model_output=None,
        env_metrics=defaultdict(list),
    )

    rollout_result = worker.rollout_results[0]
    assert rollout_result.actions == []
    assert rollout_result.forward_inputs == []
    assert len(rollout_result.prev_values) == 1
    assert rollout_result.prev_values[0].shape == (2, 1)


def test_env_update_last_logical_step_prefers_pending_step():
    worker = object.__new__(EnvWorker)
    worker.rollout_results = [EmbodiedRolloutResult(max_episode_length=8)]
    worker._latest_pending_step_by_stage = [None]

    worker.rollout_results[0].append_step_result(
        ChunkStepResult(
            actions=torch.zeros((2, 8), dtype=torch.float32),
            forward_inputs={"action": torch.zeros((2, 8), dtype=torch.float32)},
            rewards=torch.zeros((2, 1), dtype=torch.float32),
            dones=torch.zeros((2, 1), dtype=torch.bool),
            terminations=torch.zeros((2, 1), dtype=torch.bool),
            truncations=torch.zeros((2, 1), dtype=torch.bool),
        )
    )

    pending_step = PendingRolloutStep(
        stage_id=0,
        env_output=EnvOutput(
            obs={"state": torch.zeros((2, 1), dtype=torch.float32)},
            dones=torch.zeros((2, 1), dtype=torch.bool),
            terminations=torch.zeros((2, 1), dtype=torch.bool),
            truncations=torch.zeros((2, 1), dtype=torch.bool),
        ),
        rollout_result=RolloutResult(
            actions=torch.ones((2, 8), dtype=torch.float32),
            forward_inputs={
                "action": torch.ones((2, 8), dtype=torch.float32),
                "model_action": torch.full((2, 8), -1.0, dtype=torch.float32),
            },
        ),
        reward_required=True,
    )
    worker._register_pending_logical_step(pending_step)

    intervene_actions = torch.full((2, 8), 5.0, dtype=torch.float32)
    intervene_flags = torch.tensor([[True], [False]])

    worker.update_last_logical_step(0, intervene_actions, intervene_flags)

    assert torch.equal(worker.rollout_results[0].actions[0], torch.zeros((2, 8)))
    expected_actions = torch.tensor(
        [[5.0] * 8, [1.0] * 8],
        dtype=torch.float32,
    )
    assert torch.equal(pending_step.rollout_result.actions, expected_actions)
    assert torch.equal(
        pending_step.rollout_result.forward_inputs["action"], expected_actions
    )
    assert "model_action" not in pending_step.rollout_result.forward_inputs
    assert torch.equal(pending_step.intervene_actions, intervene_actions)
    assert torch.equal(pending_step.intervene_flags, intervene_flags)


def test_env_finalize_pending_step_applies_pending_intervention_after_save_flags():
    worker = object.__new__(EnvWorker)
    worker.collect_prev_infos = True
    worker.reward_mode = "per_step"
    worker.history_reward_assign = False
    worker.rollout_results = [EmbodiedRolloutResult(max_episode_length=8)]
    worker.compute_bootstrap_rewards = (
        lambda env_output, bootstrap_values, reward_model_output: torch.ones((2, 1))
    )

    pending_step = PendingRolloutStep(
        stage_id=0,
        env_output=EnvOutput(
            obs={"state": torch.zeros((2, 1), dtype=torch.float32)},
            dones=torch.zeros((2, 1), dtype=torch.bool),
            terminations=torch.zeros((2, 1), dtype=torch.bool),
            truncations=torch.zeros((2, 1), dtype=torch.bool),
            rewards=torch.zeros((2, 1), dtype=torch.float32),
        ),
        rollout_result=RolloutResult(
            actions=torch.ones((2, 8), dtype=torch.float32),
            prev_logprobs=torch.zeros((2, 8), dtype=torch.float32),
            prev_values=torch.zeros((2, 1), dtype=torch.float32),
            bootstrap_values=torch.zeros((2, 1), dtype=torch.float32),
            forward_inputs={
                "action": torch.ones((2, 8), dtype=torch.float32),
                "model_action": torch.full((2, 8), -1.0, dtype=torch.float32),
            },
            versions=torch.zeros((2, 1), dtype=torch.float32),
        ),
        reward_required=True,
        save_flags=torch.tensor([[False], [True]]),
        intervene_actions=torch.full((2, 8), 9.0, dtype=torch.float32),
        intervene_flags=torch.tensor([[True], [False]]),
        curr_obs={"state": torch.zeros((2, 1), dtype=torch.float32)},
        next_obs={"state": torch.ones((2, 1), dtype=torch.float32)},
    )

    worker.finalize_pending_rollout_step(
        pending_step=pending_step,
        reward_model_output=torch.zeros((2, 1), dtype=torch.float32),
        env_metrics=defaultdict(list),
    )

    rollout_result = worker.rollout_results[0]
    expected_actions = torch.tensor(
        [[9.0] * 8, [1.0] * 8],
        dtype=torch.float32,
    )
    expected_intervene_flags = torch.tensor(
        [[True] * 8, [False] * 8],
        dtype=torch.bool,
    )

    assert torch.equal(rollout_result.actions[0], expected_actions)
    assert torch.equal(rollout_result.intervene_flags[0], expected_intervene_flags)
    assert torch.equal(rollout_result.forward_inputs[0]["action"], expected_actions)
    assert "model_action" not in rollout_result.forward_inputs[0]
    assert torch.equal(rollout_result.curr_obs[0]["state"], torch.zeros((2, 1)))
    assert torch.equal(rollout_result.next_obs[0]["state"], torch.ones((2, 1)))
