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
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch
from omegaconf import OmegaConf

import rlinf.utils.obs_compression as obs_compression
from rlinf.data.schema import (
    EnvOutput,
    EnvPart,
    EnvTransition,
    PolicyInput,
    PolicyOutput,
    PolicyPart,
    TrajectoryKey,
    TrajectorySource,
)
from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
from rlinf.utils.env_helpers import SmoothInterveneController
from rlinf.utils.obs_compression import is_compressed_image
from rlinf.workers.env.async_env_worker import AsyncEnvWorker
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.async_huggingface_worker import (
    AsyncMultiStepRolloutWorker,
)
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def _policy_input() -> PolicyInput:
    return PolicyInput(
        obs={"states": torch.zeros(16, 4)},
    )


def test_policy_input_compression_preserves_attached_env_parts(monkeypatch):
    """Trajectory requests remain lossless across split/compress/merge routing."""
    monkeypatch.setattr(
        obs_compression,
        "_get_backend",
        lambda _codec: (
            lambda raw, _level: raw,
            lambda raw: raw,
        ),
    )
    key = TrajectoryKey(0, 0, 0, 0, 0)
    source = TrajectorySource(key=key, size=4)
    current_images = torch.randint(0, 256, (4, 8, 8, 3), dtype=torch.uint8)
    next_images = torch.randint(0, 256, (4, 8, 8, 3), dtype=torch.uint8)
    policy_input = PolicyInput(
        obs={"main_images": current_images},
        sources=[source],
        env_parts=[
            EnvPart(
                sources=[source],
                transition=EnvTransition(rewards=torch.ones(4, 2)),
                next_obs={"main_images": next_images},
            )
        ],
    )
    env_worker = object.__new__(EnvWorker)
    env_worker.obs_compression_cfg = {
        "enable": True,
        "codec": "lz4",
        "level": 1,
        "xor_delta": True,
    }

    shards = env_worker._split_and_compress_policy_input(policy_input, [1, 3])

    assert is_compressed_image(shards[0].obs["main_images"])
    assert is_compressed_image(shards[0].env_parts[0].next_obs["main_images"])
    merged = MultiStepRolloutWorker._merge_policy_inputs(shards)
    assert torch.equal(merged.obs["main_images"], current_images)
    assert torch.equal(
        torch.cat([part.next_obs["main_images"] for part in merged.env_parts], dim=0),
        next_images,
    )


def test_sync_and_async_workers_share_the_part_channel_contract():
    """Only Rollout publishes parts; Env communicates exclusively with Rollout."""
    assert "actor_channel" not in inspect.signature(EnvWorker.interact).parameters
    assert "actor_channel" not in inspect.signature(AsyncEnvWorker.interact).parameters
    assert (
        "actor_channel" in inspect.signature(MultiStepRolloutWorker.generate).parameters
    )
    assert (
        "actor_channel"
        in inspect.signature(AsyncMultiStepRolloutWorker.generate).parameters
    )


def test_decoupled_evaluation_service_is_reused_and_cancelled_on_stop():
    async def run():
        worker = object.__new__(AsyncMultiStepRolloutWorker)
        worker.env_decoupled_mode = True
        worker._generate_task = None
        worker._evaluate_task = None
        blocker = asyncio.Event()
        calls = 0

        async def serve(_input_channel, _output_channel):
            nonlocal calls
            calls += 1
            await blocker.wait()

        worker._run_evaluate_service = serve
        await worker.ensure_evaluate_service(Mock(), Mock())
        first_task = worker._evaluate_task
        await worker.ensure_evaluate_service(Mock(), Mock())

        assert worker._evaluate_task is first_task
        assert calls == 1

        worker.stop()
        await asyncio.sleep(0)
        assert first_task.cancelled()

    asyncio.run(run())


def test_async_runner_reuses_decoupled_evaluation_service_across_validations():
    runner = object.__new__(AsyncEmbodiedRunner)
    runner.cfg = OmegaConf.create({"runner": {"enable_decoupled_mode": True}})
    runner.env_channel = Mock()
    runner.rollout_channel = Mock()
    runner.rollout = Mock()
    runner.rollout.ensure_evaluate_service.return_value = Mock(
        wait=Mock(return_value=None)
    )
    runner.env = Mock()
    runner.env.evaluate.return_value = Mock(wait=Mock(return_value=[{}]))

    with patch(
        "rlinf.runners.async_embodied_runner.compute_evaluate_metrics",
        return_value={},
    ):
        runner.evaluate()
        runner.evaluate()

    assert runner.rollout.ensure_evaluate_service.call_count == 2
    runner.rollout.evaluate.assert_not_called()


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
    source = torch.zeros(16, 4, requires_grad=True)
    output = source * 2
    rollout._send_actions(None, output, 0, split_sizes)

    assert policy_input.obs["states"].shape == (16, 4)
    rollout.recv_from_and_record_batch_routes_with_timeout.assert_awaited_once_with(
        group_name="EnvGroup",
        channel=None,
        tag="policy",
        batch_size=32,
        merge_fn=rollout._merge_policy_inputs,
        infer_batch_size_fn=rollout._infer_policy_input_batch_size,
        timeout_time=0.02,
        recv_queue_size=0,
    )
    sent_output = rollout.send_to_recorded_batch_routes.call_args.kwargs["data"]
    assert torch.equal(sent_output, output)
    assert not sent_output.requires_grad
    assert sent_output.grad_fn is None
    rollout.send_to_recorded_batch_routes.assert_called_once_with(
        group_name="EnvGroup",
        channel=None,
        data=sent_output,
        tag="policy",
        split_fn=rollout._split_actions,
        split_sizes=[8, 8],
    )


def test_env_uses_mode_qualified_decoupled_response_tag():
    env = object.__new__(EnvWorker)
    env.cfg = SimpleNamespace(rollout=SimpleNamespace(group_name="RolloutGroup"))
    env.env_decoupled_mode = True
    env.train_batch_size = 32
    env.recv_from = Mock(return_value=torch.zeros(8, 4))

    env._recv_actions(None, stage_id=0)

    env.recv_from.assert_called_once_with(
        group_name="RolloutGroup",
        channel=None,
        tag="train_policy",
        route_key=None,
        batch_size=32,
        infer_batch_size_fn=env._infer_action_batch_size,
        decoupled_mode=True,
    )


def test_smooth_intervention_builds_external_input_without_policy_history():
    controller = SmoothInterveneController(
        stage_num=1,
        num_envs_per_stage=1,
        num_action_chunks=2,
        action_dim=3,
        enabled=True,
    )
    env = SimpleNamespace(get_hold_actions=lambda fallback=None: [[1.0, 2.0, 3.0]])

    policy_input = controller.build_external_policy_input(
        0, env=env, obs={"states": torch.zeros(1, 4)}
    )

    assert isinstance(policy_input, PolicyInput)
    assert not policy_input.requires_inference
    assert torch.equal(
        policy_input.external_actions,
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


def test_env_sends_external_request_after_intervention_continues():
    env = object.__new__(EnvWorker)
    env._trajectory_step = 0
    env._rank = 0
    env.train_num_envs_per_stage = 1
    env.n_train_chunk_steps = 2
    env.enable_online_lerobot = True
    env.env_decoupled_mode = False
    env.env_list = [
        SimpleNamespace(get_hold_actions=lambda fallback=None: [[1.0, 2.0]])
    ]
    env.smooth_intervene = SmoothInterveneController(1, 1, 2, 2, enabled=True)
    env._build_env_transition = Mock(return_value=EnvTransition())
    env._send_policy_input = Mock()
    env_output = EnvOutput(
        obs={"states": torch.zeros(1, 4)},
        transition=EnvTransition(
            dones=torch.tensor([[False, False]]),
            intervene_flags=torch.tensor([[False, True]]),
        ),
    )

    env._publish_step(Mock(), env_output, EnvTransition(), None, {}, 0, 0, 0)

    policy_input = env._send_policy_input.call_args.args[1]
    assert not policy_input.requires_inference
    assert policy_input.env_parts[0] is not None
    assert policy_input.env_parts[0].next_obs is None
    assert torch.equal(policy_input.obs["states"], env_output.obs["states"])
    assert torch.equal(
        policy_input.external_actions,
        torch.tensor([[[1.0, 2.0], [1.0, 2.0]]]),
    )


def _publish_step_env(**overrides):
    """Build the minimal EnvWorker used by _publish_step tests."""
    env = object.__new__(EnvWorker)
    env._trajectory_step = 0
    env._rank = 0
    env.train_num_envs_per_stage = 1
    env.n_train_chunk_steps = 2
    env.enable_online_lerobot = False
    env.env_decoupled_mode = False
    env.collect_final_values = True
    env.smooth_intervene = SmoothInterveneController(1, 1, 1, 4)
    env._build_env_transition = Mock(return_value=EnvTransition())
    env._send_policy_input = Mock()
    for name, value in overrides.items():
        setattr(env, name, value)
    return env


def _terminal_env_output(reset_obs, terminal_obs, **transition):
    flags = {"dones": torch.ones(1, 1, dtype=torch.bool), **transition}
    return EnvOutput(
        obs={"states": reset_obs},
        final_obs={"states": terminal_obs},
        transition=EnvTransition(**flags),
    )


def test_env_sends_terminal_observation_only_as_a_boundary_override():
    env = _publish_step_env()
    reset_obs = torch.zeros(1, 4)
    terminal_obs = torch.ones(1, 4)
    env_output = _terminal_env_output(
        reset_obs,
        terminal_obs,
    )

    env._publish_step(Mock(), env_output, EnvTransition(), None, None, 0, 0, 0)

    policy_input = env._send_policy_input.call_args.args[1]
    assert torch.equal(policy_input.obs["states"], reset_obs)
    assert torch.equal(policy_input.env_parts[0].next_obs["states"], terminal_obs)
    assert policy_input.env_parts[0].requires_inference


def test_rollout_completes_each_epoch_through_policy_inputs():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.n_train_chunk_steps = 2
    rollout.num_pipeline_stages = 1
    rollout.env_decoupled_mode = False
    rollout.enable_rlt = False
    rollout.collect_transitions = False
    rollout.hf_model = SimpleNamespace(value_head=object())
    rollout.update_dagger_beta = Mock()
    rollout._send_actions = Mock()
    rollout._build_policy_output = Mock(
        side_effect=[
            PolicyOutput(
                forward_inputs={"states": torch.tensor([[1]])},
            ),
            PolicyOutput(
                forward_inputs={"states": torch.tensor([[2]])},
            ),
        ]
    )
    key0 = TrajectoryKey(0, 0, 0, 0, 0)
    key1 = TrajectoryKey(0, 0, 0, 0, 1)
    completion0 = EnvPart(
        sources=[TrajectorySource(key0, 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.tensor([[2]])},
        requires_inference=False,
        initial_transition=EnvTransition(dones=torch.zeros(1, 1, dtype=torch.bool)),
    )
    completion1 = EnvPart(
        sources=[TrajectorySource(key1, 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.tensor([[3]])},
        requires_inference=True,
    )
    rollout._receive_policy_input = AsyncMock(
        side_effect=[
            (
                PolicyInput(
                    obs={"states": torch.tensor([[1]])},
                    sources=[TrajectorySource(key0, 1)],
                    env_parts=[None],
                    request_sizes=[1],
                ),
                None,
            ),
            (
                PolicyInput(
                    obs={"states": torch.tensor([[2]])},
                    sources=[TrajectorySource(key1, 1)],
                    env_parts=[completion0],
                    request_sizes=[1],
                ),
                None,
            ),
            (
                PolicyInput(
                    obs={"states": torch.tensor([[3]])},
                    env_parts=[completion1],
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
    actor_channel = Mock()
    generate_one_epoch = MultiStepRolloutWorker.generate_one_epoch
    while hasattr(generate_one_epoch, "__wrapped__"):
        generate_one_epoch = generate_one_epoch.__wrapped__

    asyncio.run(
        generate_one_epoch(
            rollout,
            input_channel=Mock(),
            output_channel=Mock(),
            actor_channel=actor_channel,
        )
    )

    parts = [call.args[0] for call in actor_channel.put.call_args_list]
    assert sum(isinstance(part, PolicyPart) for part in parts) == 2
    env_parts = [part for part in parts if isinstance(part, EnvPart)]
    assert len(env_parts) == 2
    assert env_parts[0].initial_transition is completion0.initial_transition
    assert env_parts[0].next_rlt_obs is None
    assert env_parts[0].next_obs is None
    assert torch.equal(env_parts[1].bootstrap_values, torch.tensor([[4.0]]))
    assert torch.equal(env_parts[1].final_prev_values, torch.tensor([[4.0, 5.0]]))
    assert rollout._predict_rollout_actions.call_count == 3


def test_rollout_routes_external_input_without_model_inference():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.n_train_chunk_steps = 2
    rollout.num_pipeline_stages = 1
    rollout.env_decoupled_mode = False
    rollout.enable_rlt = False
    rollout.collect_transitions = False
    rollout.hf_model = SimpleNamespace(value_head=object())
    rollout.update_dagger_beta = Mock()
    rollout._send_actions = Mock()
    rollout._build_policy_output = Mock(
        return_value=PolicyOutput(
            forward_inputs={"action": torch.ones(1, 6)},
        )
    )
    key0 = TrajectoryKey(0, 0, 0, 0, 0)
    key1 = TrajectoryKey(0, 0, 0, 0, 1)
    completion0 = EnvPart(
        sources=[TrajectorySource(key0, 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.ones(1, 4)},
        requires_inference=False,
    )
    completion1 = EnvPart(
        sources=[TrajectorySource(key1, 1)],
        transition=EnvTransition(),
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
                    env_parts=[None],
                    request_sizes=[1],
                ),
                None,
            ),
            (
                PolicyInput(
                    obs={"states": torch.ones(1, 4)},
                    external_actions=dummy_actions,
                    sources=[TrajectorySource(key1, 1)],
                    env_parts=[completion0],
                    request_sizes=[1],
                ),
                None,
            ),
            (
                PolicyInput(
                    obs={"states": torch.ones(1, 4)},
                    env_parts=[completion1],
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
    actor_channel = Mock()
    generate_one_epoch = MultiStepRolloutWorker.generate_one_epoch
    while hasattr(generate_one_epoch, "__wrapped__"):
        generate_one_epoch = generate_one_epoch.__wrapped__

    asyncio.run(generate_one_epoch(rollout, Mock(), Mock(), actor_channel))

    parts = [call.args[0] for call in actor_channel.put.call_args_list]
    inferred_parts = [
        part for part in parts if isinstance(part, PolicyPart) and part.inferred
    ]
    assert len(inferred_parts) == 1
    external_part = next(
        part for part in parts if isinstance(part, PolicyPart) and not part.inferred
    )
    assert torch.equal(external_part.external_actions, dummy_actions)
    assert rollout._predict_rollout_actions.call_count == 1
    assert torch.equal(rollout._send_actions.call_args_list[1].args[1], dummy_actions)


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
    env._build_env_transition = Mock(return_value=EnvTransition())
    env._send_policy_input = Mock()
    env_output = EnvOutput(obs={"states": torch.zeros(1, 4)})

    completion = env._publish_step(
        rollout_channel=Mock(),
        env_output=env_output,
        initial_transition=EnvTransition(),
        reward_model_output=None,
        chunk_step_data=None,
        epoch_id=0,
        chunk_id=0,
        stage_id=0,
    )

    assert completion is not None
    assert completion.initial_transition is not None
    assert torch.equal(completion.next_obs["states"], env_output.obs["states"])
    env._send_policy_input.assert_not_called()

    env._send_train_bootstrap(
        rollout_channel=Mock(),
        env_outputs=[env_output],
        step_id=1,
        epoch_id=0,
        previous_env_parts={0: completion},
    )

    policy_input = env._send_policy_input.call_args.args[1]
    assert policy_input.env_parts == [completion]


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
    env.enable_online_lerobot = False
    env.cfg = SimpleNamespace(
        env=SimpleNamespace(
            train=SimpleNamespace(auto_reset=True, ignore_terminations=False)
        )
    )
    env._bootstrap_and_send_train = Mock(
        return_value=[EnvOutput(obs={"states": torch.zeros(1, 4)})]
    )
    env._maybe_wait_env_delay = AsyncMock()
    env._recv_actions = Mock(return_value=torch.zeros(1, 4))
    env.smooth_intervene = SmoothInterveneController(1, 1, 2, 4)
    env.env_interact_step = Mock(
        return_value=(
            EnvOutput(
                obs={"states": torch.zeros(1, 4)},
                transition=EnvTransition(rewards=torch.zeros(1, 1)),
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
    asyncio.run(
        EnvWorker._run_interact_once.__wrapped__(
            env,
            input_channel=Mock(),
            rollout_channel=Mock(),
            reward_channel=Mock(),
            cooperative_yield=False,
        )
    )

    assert env.get_reward_model_output.call_count == 4
    assert [
        call.kwargs["last_run"] for call in env.get_reward_model_output.call_args_list
    ] == [False, False, False, True]


def test_smooth_intervention_holds_at_the_last_commanded_step():
    controller = SmoothInterveneController(
        stage_num=1,
        num_envs_per_stage=1,
        num_action_chunks=2,
        action_dim=3,
        enabled=True,
    )
    seen = []

    def get_hold_actions(fallback=None):
        seen.append(fallback)
        return [[1.0, 2.0, 3.0]]

    env = SimpleNamespace(get_hold_actions=get_hold_actions)
    controller.remember_actions(0, torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]))

    controller.build_external_policy_input(
        0, env=env, obs={"states": torch.zeros(1, 4)}
    )

    # The wrapper receives the final step of the chunk that was just executed,
    # so it can hold that pose instead of snapping back to its own default.
    assert len(seen) == 1
    assert seen[0].tolist() == [[4.0, 5.0, 6.0]]


def test_rollout_runs_requested_terminal_inference_without_a_value_head():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.enable_rlt = False
    rollout.collect_transitions = False
    rollout._predict_rollout_actions = Mock(
        return_value=(
            torch.zeros(1, 1),
            {
                "prev_values": torch.zeros(1, 1),
                "forward_inputs": {"states": torch.ones(1, 1)},
            },
        )
    )
    actor_channel = Mock()
    completion = EnvPart(
        sources=[TrajectorySource(TrajectoryKey(0, 0, 0, 0, 0), 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.zeros(1, 4)},
        requires_inference=True,
    )

    rollout._publish_env_part(completion, None, None, actor_channel)

    rollout._predict_rollout_actions.assert_called_once_with(completion.next_obs)
    part = actor_channel.put.call_args.args[0]
    assert torch.equal(part.bootstrap_values, torch.zeros(1, 1))
    assert torch.equal(part.final_prev_values, torch.zeros(1, 1))
    assert part.next_rlt_obs is None
    assert part.next_obs is None


def test_rollout_reuses_policy_obs_for_transition_next_obs():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.enable_rlt = False
    rollout.collect_transitions = True
    actor_channel = Mock()
    policy_obs = {"states": torch.ones(1, 4)}
    completion = EnvPart(
        sources=[TrajectorySource(TrajectoryKey(0, 0, 0, 0, 0), 1)],
        transition=EnvTransition(),
    )

    rollout._publish_env_part(
        completion,
        policy_obs,
        {"unused": torch.ones(1, 1)},
        actor_channel,
    )

    part = actor_channel.put.call_args.args[0]
    assert part.next_obs is policy_obs
    assert part.next_rlt_obs is None


def test_rollout_runs_terminal_inference_for_rlt_without_a_value_head():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.enable_rlt = True
    rollout.collect_transitions = True
    rollout.hf_model = SimpleNamespace()
    rollout._predict_rollout_actions = Mock(
        return_value=(
            torch.zeros(1, 1),
            {
                "forward_inputs": {
                    "rlt_transition_z_rl": torch.ones(1, 2),
                    "rlt_transition_proprio": torch.ones(1, 3),
                    "rlt_transition_ref_chunk": torch.ones(1, 4),
                }
            },
        )
    )
    actor_channel = Mock()
    completion = EnvPart(
        sources=[TrajectorySource(TrajectoryKey(0, 0, 0, 0, 0), 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.zeros(1, 4)},
        requires_inference=True,
    )

    rollout._publish_env_part(completion, None, None, actor_channel)

    rollout._predict_rollout_actions.assert_called_once()
    part = actor_channel.put.call_args.args[0]
    assert torch.equal(part.next_rlt_obs["z_rl"], torch.ones(1, 2))


def test_rollout_keeps_next_forward_inputs_only_for_rlt():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.enable_rlt = True
    rollout.collect_transitions = True
    actor_channel = Mock()
    key = TrajectoryKey(0, 0, 0, 0, 0)
    policy_input = PolicyInput(
        obs={"states": torch.zeros(1, 4)},
        env_parts=[
            EnvPart(
                sources=[TrajectorySource(key, 1)],
                transition=EnvTransition(),
            )
        ],
        request_sizes=[1],
    )
    forward_inputs = {
        "rlt_transition_z_rl": torch.ones(1, 2),
        "rlt_transition_proprio": torch.ones(1, 3),
        "rlt_transition_ref_chunk": torch.ones(1, 4),
        "unrelated": torch.ones(1, 8),
    }

    rollout._publish_env_parts(policy_input, forward_inputs, actor_channel)

    part = actor_channel.put.call_args.args[0]
    assert part.next_obs is None
    assert torch.equal(
        part.next_rlt_obs["z_rl"],
        forward_inputs["rlt_transition_z_rl"],
    )
    assert set(part.next_rlt_obs) == {"z_rl", "proprio", "ref_chunk"}
