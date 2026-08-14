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
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.data.schema import (
    DummyPolicyInput,
    EnvOutput,
    EnvResult,
    PolicyCompletion,
    PolicyInput,
    PolicyOutput,
    TrajectoryKey,
    TrajectorySource,
    merge_policy_inputs,
)
from rlinf.data.schema.embodied_types import EmbodiedRolloutResult
from rlinf.scheduler.channel.trajectory_channel.data import (
    DummyPolicyStep,
    EnvStepResult,
    PolicyStep,
)
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.env.smooth_intervene import SmoothInterveneController
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def _policy_input() -> PolicyInput:
    return PolicyInput(
        obs={"states": torch.zeros(16, 4)},
    )


def test_decoupled_policy_route_round_trip():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.env_decoupled_mode = True
    rollout.cfg = SimpleNamespace(env=SimpleNamespace(group_name="EnvGroup"))
    rollout.train_batch_size = 32
    rollout.rollout_queue_size = 0
    rollout.batch_router = {"policy": ["stale"]}
    rollout.recv_from_and_record_batch_routes_with_timeout = AsyncMock(
        return_value=(_policy_input(), [8, 8])
    )
    rollout.send_to_recorded_batch_routes = Mock()

    policy_input, split_sizes = asyncio.run(
        rollout._receive_policy_input(None, "policy", 0)
    )
    output = PolicyOutput(actions=torch.zeros(16, 4))
    rollout._send_policy_output(None, output, 0, split_sizes)

    assert policy_input.obs["states"].shape == (16, 4)
    rollout.recv_from_and_record_batch_routes_with_timeout.assert_awaited_once_with(
        group_name="EnvGroup",
        channel=None,
        tag="policy",
        batch_size=32,
        merge_fn=merge_policy_inputs,
        infer_batch_size_fn=rollout._infer_policy_input_batch_size,
        timeout_time=0.02,
        recv_queue_size=0,
    )
    rollout.send_to_recorded_batch_routes.assert_called_once_with(
        group_name="EnvGroup",
        channel=None,
        data=output,
        tag="policy",
        split_fn=rollout._split_policy_output,
        split_sizes=[8, 8],
    )


def test_env_uses_mode_qualified_decoupled_response_tag():
    env = object.__new__(EnvWorker)
    env.cfg = SimpleNamespace(rollout=SimpleNamespace(group_name="RolloutGroup"))
    env.env_decoupled_mode = True
    env.train_batch_size = 32
    env.recv_from = Mock(return_value=PolicyOutput(actions=torch.zeros(8, 4)))

    env._recv_policy_output(None, stage_id=0)

    env.recv_from.assert_called_once_with(
        group_name="RolloutGroup",
        channel=None,
        tag="train_policy",
        route_key=None,
        batch_size=32,
        infer_batch_size_fn=env._infer_policy_output_batch_size,
        decoupled_mode=True,
    )


def test_smooth_intervention_builds_dummy_input_without_policy_history():
    controller = SmoothInterveneController(
        stage_num=1,
        num_envs_per_stage=1,
        num_action_chunks=2,
        action_dim=3,
        enabled=True,
    )
    env = SimpleNamespace(get_hold_actions=lambda: [[1.0, 2.0, 3.0]])

    policy_input = controller.build_dummy_policy_input(
        0, env=env, obs={"states": torch.zeros(1, 4)}
    )

    assert isinstance(policy_input, DummyPolicyInput)
    assert torch.equal(
        policy_input.actions,
        torch.tensor([[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]]),
    )


def test_smooth_intervention_requires_online_lerobot_dagger():
    cfg = OmegaConf.create(
        {
            "actor": {"model": {"num_action_chunks": 2, "action_dim": 3}},
            "algorithm": {
                "loss_type": "ppo",
                "dagger": {"online_lerobot": {"enabled": True}},
            },
            "env": {
                "train": {
                    "smooth_intervene": True,
                    "env_type": "realworld",
                    "use_pico": True,
                    "use_spacemouse": False,
                }
            },
        }
    )

    with pytest.raises(ValueError, match="loss_type=embodied_dagger"):
        SmoothInterveneController.from_cfg(
            cfg,
            stage_num=1,
            enable_train=True,
            train_num_envs_per_stage=1,
        )


def test_env_sends_dummy_request_after_intervention_continues():
    env = object.__new__(EnvWorker)
    env._trajectory_step = 0
    env._rank = 0
    env.train_num_envs_per_stage = 1
    env.n_train_chunk_steps = 2
    env.enable_online_lerobot = True
    env.env_decoupled_mode = False
    env.env_list = [SimpleNamespace(get_hold_actions=lambda: [[1.0, 2.0]])]
    env.smooth_intervene = SmoothInterveneController(1, 1, 2, 2, enabled=True)
    env._build_env_result = Mock(return_value=EnvResult())
    env._send_policy_input = Mock()
    env_output = EnvOutput(
        obs={"states": torch.zeros(1, 4)},
        dones=torch.tensor([[False, False]]),
        intervene_flags=torch.tensor([[False, True]]),
    )

    env._publish_step(Mock(), env_output, None, {}, 0, 0, 0)

    policy_input = env._send_policy_input.call_args.args[1]
    assert isinstance(policy_input, DummyPolicyInput)
    assert policy_input.completions[0] is not None
    assert torch.equal(policy_input.actions, torch.tensor([[[1.0, 2.0], [1.0, 2.0]]]))


def test_rollout_completes_each_epoch_through_policy_inputs():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.n_train_chunk_steps = 2
    rollout.num_pipeline_stages = 1
    rollout.env_decoupled_mode = False
    rollout.update_dagger_beta = Mock()
    rollout._send_policy_output = Mock()
    rollout._build_rollout_result = Mock(
        side_effect=[
            EmbodiedRolloutResult(
                actions=torch.zeros(1, 1),
                forward_inputs={"states": torch.tensor([[1]])},
            ),
            EmbodiedRolloutResult(
                actions=torch.zeros(1, 1),
                forward_inputs={"states": torch.tensor([[2]])},
            ),
        ]
    )
    key0 = TrajectoryKey(0, 0, 0, 0, 0)
    key1 = TrajectoryKey(0, 0, 0, 0, 1)
    completion0 = PolicyCompletion(
        sources=[TrajectorySource(key0, 1)],
        env_result=EnvResult(),
        next_obs={"states": torch.tensor([[2]])},
        requires_inference=False,
    )
    completion1 = PolicyCompletion(
        sources=[TrajectorySource(key1, 1)],
        env_result=EnvResult(),
        next_obs={"states": torch.tensor([[3]])},
        requires_inference=True,
    )
    rollout._receive_policy_input = AsyncMock(
        side_effect=[
            (
                PolicyInput(
                    obs={"states": torch.tensor([[1]])},
                    sources=[TrajectorySource(key0, 1)],
                    completions=[None],
                    request_sizes=[1],
                ),
                None,
            ),
            (
                PolicyInput(
                    obs={"states": torch.tensor([[2]])},
                    sources=[TrajectorySource(key1, 1)],
                    completions=[completion0],
                    request_sizes=[1],
                ),
                None,
            ),
            (
                PolicyInput(
                    obs={"states": torch.tensor([[3]])},
                    completions=[completion1],
                    request_sizes=[1],
                    is_last=True,
                ),
                None,
            ),
        ]
    )
    rollout._predict_rollout_actions = Mock(
        side_effect=[
            (torch.zeros(1, 1), {"forward_inputs": {"states": torch.tensor([[1]])}}),
            (torch.zeros(1, 1), {"forward_inputs": {"states": torch.tensor([[2]])}}),
            (
                torch.zeros(1, 1),
                {
                    "forward_inputs": {"states": torch.tensor([[3]])},
                    "prev_values": torch.tensor([[4.0, 5.0]]),
                },
            ),
        ]
    )
    trajectory_channel = Mock()
    generate_one_epoch = MultiStepRolloutWorker.generate_one_epoch
    while hasattr(generate_one_epoch, "__wrapped__"):
        generate_one_epoch = generate_one_epoch.__wrapped__

    asyncio.run(
        generate_one_epoch(
            rollout,
            input_channel=Mock(),
            output_channel=Mock(),
            trajectory_channel=trajectory_channel,
        )
    )

    events = [call.args[0] for call in trajectory_channel.publish.call_args_list]
    assert sum(isinstance(event, PolicyStep) for event in events) == 2
    env_events = [event for event in events if isinstance(event, EnvStepResult)]
    assert len(env_events) == 2
    assert torch.equal(env_events[0].forward_inputs["states"], torch.tensor([[2]]))
    assert torch.equal(env_events[1].bootstrap_values, torch.tensor([[4.0]]))
    assert torch.equal(env_events[1].final_prev_values, torch.tensor([[4.0, 5.0]]))
    assert rollout._predict_rollout_actions.call_count == 3


def test_rollout_routes_dummy_input_without_model_inference():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.n_train_chunk_steps = 2
    rollout.num_pipeline_stages = 1
    rollout.env_decoupled_mode = False
    rollout.update_dagger_beta = Mock()
    rollout._send_policy_output = Mock()
    rollout._build_rollout_result = Mock(
        return_value=EmbodiedRolloutResult(
            actions=torch.ones(1, 2, 3),
            forward_inputs={"action": torch.ones(1, 6)},
        )
    )
    key0 = TrajectoryKey(0, 0, 0, 0, 0)
    key1 = TrajectoryKey(0, 0, 0, 0, 1)
    completion0 = PolicyCompletion(
        sources=[TrajectorySource(key0, 1)],
        env_result=EnvResult(),
        next_obs={"states": torch.ones(1, 4)},
        requires_inference=False,
    )
    completion1 = PolicyCompletion(
        sources=[TrajectorySource(key1, 1)],
        env_result=EnvResult(),
        next_obs={"states": torch.ones(1, 4)},
        requires_inference=False,
    )
    dummy_actions = torch.full((1, 2, 3), 2.0)
    rollout._receive_policy_input = AsyncMock(
        side_effect=[
            (
                PolicyInput(
                    obs={"states": torch.zeros(1, 4)},
                    sources=[TrajectorySource(key0, 1)],
                    completions=[None],
                    request_sizes=[1],
                ),
                None,
            ),
            (
                DummyPolicyInput(
                    obs={"states": torch.ones(1, 4)},
                    actions=dummy_actions,
                    sources=[TrajectorySource(key1, 1)],
                    completions=[completion0],
                    request_sizes=[1],
                ),
                None,
            ),
            (
                PolicyInput(
                    obs={"states": torch.ones(1, 4)},
                    completions=[completion1],
                    request_sizes=[1],
                    is_last=True,
                ),
                None,
            ),
        ]
    )
    rollout._predict_rollout_actions = Mock(
        return_value=(
            torch.ones(1, 2, 3),
            {"forward_inputs": {"action": torch.ones(1, 6)}},
        )
    )
    trajectory_channel = Mock()
    generate_one_epoch = MultiStepRolloutWorker.generate_one_epoch
    while hasattr(generate_one_epoch, "__wrapped__"):
        generate_one_epoch = generate_one_epoch.__wrapped__

    asyncio.run(generate_one_epoch(rollout, Mock(), Mock(), trajectory_channel))

    events = [call.args[0] for call in trajectory_channel.publish.call_args_list]
    assert sum(isinstance(event, PolicyStep) for event in events) == 1
    dummy_event = next(event for event in events if isinstance(event, DummyPolicyStep))
    assert torch.equal(dummy_event.actions, dummy_actions)
    assert rollout._predict_rollout_actions.call_count == 1
    assert torch.equal(
        rollout._send_policy_output.call_args_list[1].args[1].actions, dummy_actions
    )


def test_decoupled_final_completion_uses_next_bootstrap():
    env = object.__new__(EnvWorker)
    env._trajectory_step = 0
    env._rank = 0
    env.stage_num = 1
    env.train_num_envs_per_stage = 1
    env.n_train_chunk_steps = 1
    env.enable_online_lerobot = False
    env.env_decoupled_mode = True
    env.smooth_intervene = SmoothInterveneController(1, 1, 1, 4)
    env._build_env_result = Mock(return_value=EnvResult())
    env._send_policy_input = Mock()
    env_output = EnvOutput(obs={"states": torch.zeros(1, 4)})

    completion = env._publish_step(
        rollout_channel=Mock(),
        env_output=env_output,
        reward_model_output=None,
        chunk_step_data=None,
        epoch_id=0,
        chunk_id=0,
        stage_id=0,
    )

    assert completion is not None
    env._send_policy_input.assert_not_called()

    env._send_train_bootstrap(
        rollout_channel=Mock(),
        trajectory_channel=Mock(),
        env_outputs=[env_output],
        step_id=1,
        epoch_id=0,
        completions={0: completion},
    )

    policy_input = env._send_policy_input.call_args.args[1]
    assert policy_input.completions == [completion]


def test_env_closes_reward_stream_with_final_chunk():
    env = object.__new__(EnvWorker)
    env.rollout_epoch = 2
    env.stage_num = 1
    env.n_train_chunk_steps = 2
    env._prefetched_train_bootstrap = None
    env._trajectory_step = 0
    env._rank = 0
    env.env_list = [SimpleNamespace()]
    env.use_training_pipeline = False
    env.env_decoupled_mode = False
    env.cfg = SimpleNamespace(
        env=SimpleNamespace(
            train=SimpleNamespace(auto_reset=True, ignore_terminations=False)
        )
    )
    env._bootstrap_and_send_train = Mock(
        return_value=[EnvOutput(obs={"states": torch.zeros(1, 4)})]
    )
    env._maybe_wait_env_delay = AsyncMock()
    env._recv_policy_output = Mock(return_value=PolicyOutput(actions=torch.zeros(1, 4)))
    env.env_interact_step = Mock(
        return_value=(
            EnvOutput(
                obs={"states": torch.zeros(1, 4)},
                rewards=torch.zeros(1, 1),
            ),
            {},
            {},
        )
    )
    env.get_reward_model_output = Mock(return_value=torch.zeros(1, 1))
    env._publish_step = Mock()
    env.record_env_metrics = Mock()
    env.store_last_obs_and_intervened_info = Mock()
    env.finish_rollout = Mock()
    trajectory_channel = Mock()

    asyncio.run(
        EnvWorker._run_interact_once.__wrapped__(
            env,
            input_channel=Mock(),
            rollout_channel=Mock(),
            reward_channel=Mock(),
            trajectory_channel=trajectory_channel,
            cooperative_yield=False,
        )
    )

    assert env.get_reward_model_output.call_count == 4
    assert [
        call.kwargs["last_run"] for call in env.get_reward_model_output.call_args_list
    ] == [False, False, False, True]
