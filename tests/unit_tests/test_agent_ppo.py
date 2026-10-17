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

"""Synchronous agent PPO targets and placement-independent loss scaling."""

import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rlinf.algorithms.advantages import compute_gae_advantages_and_returns
from rlinf.algorithms.losses import (
    compute_ppo_actor_loss,
    compute_ppo_critic_loss,
)
from rlinf.algorithms.registry import get_adv_and_returns
from rlinf.data.schema.agent_trajectory import prepare_agent_ppo_targets
from rlinf.data.schema.reasoning_results import DynamicRolloutResult
from rlinf.scheduler import Channel, Cluster
from rlinf.scheduler.channel.channel_worker import LocalChannel


def _local_channel():
    channel = Channel()
    channel._initialize("ppo-targets", None, None, None, local_channel=LocalChannel())
    return channel


@pytest.mark.parametrize("truncate", [True, False])
@pytest.mark.parametrize("budget", [12, 100])
@pytest.mark.parametrize("return_logprobs", [True, False])
@pytest.mark.placement
def test_searchr1_response_processing_preserves_behavior_tokens(
    truncate, budget, return_logprobs
):
    from rlinf.agents.searchr1.searchr1_agent_loop import Searchr1AgentLoopWorker
    from rlinf.agents.tokenization import get_agentic_tokenizer

    class Tokenizer:
        def decode(self, ids):
            return "".join(map(chr, ids))

        def encode(self, text):
            raise AssertionError("Generated tokens must not be re-encoded.")

    worker = object.__new__(Searchr1AgentLoopWorker)
    worker.cfg = SimpleNamespace(agentloop=SimpleNamespace(max_turns=2))
    worker.max_resp_len = budget
    worker.truncate_response = truncate
    worker.return_logprobs = return_logprobs
    worker.tokenizer = Tokenizer()
    worker.agentic_tokenizer = get_agentic_tokenizer(worker.tokenizer, "default")
    Cluster(num_nodes=1)
    worker.generate_input_channel = Channel.create(f"ResponseInput-{truncate}-{budget}")
    worker.generate_output_channel = Channel.create(
        f"ResponseOutput-{truncate}-{budget}"
    )
    ids = list(map(ord, "<search>x</search>\n"))
    logprobs = [-0.01 * (i + 1) for i in range(len(ids))]
    context = {"turn": 0, "all_llm_response_ids": []}

    async def engine():
        request = await worker.generate_input_channel.get(async_op=True).async_wait()
        assert request["sampling_params"]["max_new_tokens"] == budget
        await worker.generate_output_channel.put(
            {"output_ids": ids.copy(), "logprobs": logprobs.copy()},
            key=request["channel_key"],
            async_op=True,
        ).async_wait()

    async def run():
        output, _ = await asyncio.gather(
            worker.generate_llm_response(context, [], [1], [1]), engine()
        )
        return output[-1]

    output = asyncio.run(run())
    expected = ids[:budget]
    if truncate and budget == 100:
        expected = ids[:-1]
    assert output.response_ids == expected
    assert output.response_logprobs == (
        logprobs[: len(expected)] if return_logprobs else None
    )
    assert context["all_llm_response_ids"] == output.response_ids


@pytest.mark.parametrize("truncate", [None, True, False])
@pytest.mark.parametrize("recompute", [True, False])
def test_searchr1_response_config_defaults_and_raw_logprobs(truncate, recompute):
    from omegaconf import OmegaConf

    from rlinf.config import validate_searchr1_cfg

    cfg = OmegaConf.load(
        Path(__file__).resolve().parents[2]
        / "examples/agent/searchr1/config/train_qwen2.5_ppo.yaml"
    )
    cfg.agentloop.pop("truncate_response")
    if truncate is not None:
        cfg.agentloop.truncate_response = truncate
    cfg.algorithm.recompute_logprobs = recompute
    cfg.rollout.return_logprobs = not recompute
    validate_searchr1_cfg(cfg)
    assert cfg.agentloop.truncate_response == (truncate is not False)


def _prepare_targets(results, gamma, gae_lambda, normalize_advantages):
    input_channel, output_channel = _local_channel(), _local_channel()
    for result in results:
        input_channel.put(result)
    prepare_agent_ppo_targets(
        input_channel,
        output_channel,
        len(results),
        gamma,
        gae_lambda,
        normalize_advantages,
    )
    for result in results:
        assert output_channel.get() is result


@pytest.mark.parametrize("weight_init", [None, "normal", "zero"])
@pytest.mark.parametrize("init_std", [0.02, 0.003])
def test_critic_value_head_initialization(weight_init, init_std):
    pytest.importorskip("megatron.training")
    from rlinf.hybrid_engines.megatron.megatron_model_manager import LinearForLastLayer

    options = {} if weight_init is None else {"weight_init": weight_init}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        head = LinearForLastLayer(
            16384, 1, sequence_parallel=False, bias=True, init_std=init_std, **options
        )
    if weight_init == "zero":
        assert torch.count_nonzero(head.weight) == 0
    else:
        assert head.weight.mean().item() == pytest.approx(0.0, abs=init_std * 0.05)
        assert head.weight.std().item() == pytest.approx(init_std, rel=0.05)
    assert torch.count_nonzero(head.bias) == 0


def test_critic_value_head_rejects_unknown_initialization():
    pytest.importorskip("megatron.training")
    from rlinf.hybrid_engines.megatron.megatron_model_manager import LinearForLastLayer

    with pytest.raises(ValueError, match="'normal' or 'zero'.*'uniform'"):
        LinearForLastLayer(8, 1, sequence_parallel=False, weight_init="uniform")


def test_checkpoint_context_keeps_resident_state_trainable(tmp_path):
    pytest.importorskip("megatron.training")
    from omegaconf import OmegaConf

    from rlinf.hybrid_engines.megatron.megatron_model_manager import (
        MegatronModelManager,
    )

    manager = object.__new__(MegatronModelManager)
    manager._cfg = OmegaConf.create(
        {"offload_weight": False, "offload_grad": False, "offload_optimizer": False}
    )
    manager.is_weight_offloaded = False
    manager.is_grad_offloaded = False
    manager.is_optimizer_offloaded = False
    model = torch.nn.Linear(2, 1)
    model(torch.ones(1, 2)).sum().backward()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.step()
    manager.model = [model]
    checkpoint = tmp_path / "model.pt"
    pointers = [parameter.grad.data_ptr() for parameter in model.parameters()]
    with manager.ensure_onloaded_and_then_offload():
        torch.save(model.state_dict(), checkpoint)

    assert not manager.is_weight_offloaded
    assert not manager.is_grad_offloaded
    assert not manager.is_optimizer_offloaded
    assert pointers == [parameter.grad.data_ptr() for parameter in model.parameters()]
    saved = torch.load(checkpoint, weights_only=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved[name])
    optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    assert not torch.equal(model.weight, saved["weight"])


def test_masked_gae_carries_across_observations_and_bootstraps_truncation():
    values = torch.tensor([[0.2, 0.1], [99.0, 0.3], [0.4, 99.0]])
    rewards = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    mask = torch.tensor([[True, True], [False, True], [True, False]])
    advantages, returns = get_adv_and_returns("actor_gae")(
        rewards,
        values=values,
        step_mask=mask,
        bootstrap_values=torch.tensor([0.0, 0.5]),
        normalize_advantages=False,
    )
    torch.testing.assert_close(
        advantages, torch.tensor([[0.8, 0.4], [0.0, 0.2], [0.6, 0.0]])
    )
    torch.testing.assert_close(
        returns, torch.tensor([[1.0, 0.5], [0.0, 0.5], [1.0, 0.0]])
    )


def test_legacy_gae_preserves_terminal_contract():
    advantages, returns = compute_gae_advantages_and_returns(
        torch.tensor([[0.0], [1.0]]),
        values=torch.tensor([[0.2], [0.4], [0.0]]),
        dones=torch.tensor([[False], [False], [True]]),
        normalize_advantages=False,
    )
    torch.testing.assert_close(advantages, torch.tensor([[0.8], [0.6]]))
    torch.testing.assert_close(returns, torch.ones(2, 1))


def test_masked_gae_discounts_actions_without_discounting_observations():
    advantages, returns = get_adv_and_returns("actor_gae")(
        torch.tensor([[0.0], [0.0], [1.0]]),
        values=torch.tensor([[0.2], [99.0], [0.4]]),
        step_mask=torch.tensor([[True], [False], [True]]),
        bootstrap_values=torch.zeros(1),
        gamma=0.9,
        gae_lambda=0.8,
        normalize_advantages=False,
    )
    torch.testing.assert_close(advantages, torch.tensor([[0.592], [0.0], [0.6]]))
    torch.testing.assert_close(returns, torch.tensor([[0.792], [0.0], [1.0]]))


@pytest.mark.parametrize("role", ["actor", "critic"])
@pytest.mark.parametrize("dp_size", [2, 4, 8])
@pytest.mark.parametrize("microbatches", [1, 2, 4])
def test_megatron_valid_token_scale_matches_unsplit_gradient(
    monkeypatch, tmp_path, role, dp_size, microbatches
):
    pytest.importorskip("megatron.training")
    from megatron.core import parallel_state
    from omegaconf import OmegaConf

    from rlinf.scheduler import Worker
    from rlinf.utils.utils import masked_mean
    from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor
    from rlinf.workers.critic.ma_megatron_critic_worker import MAMegatronCritic

    class TokenModel(torch.nn.Module):
        pre_process = True
        post_process = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.01))

        def forward(self, input_ids, attention_mask, position_ids, packed_seq_params):
            output = (input_ids.float() * self.weight).unsqueeze(-1)
            if role == "actor":
                output = output * torch.arange(8, dtype=torch.float32)
            return output

    monkeypatch.setattr(Worker, "torch_device_type", "cpu")
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    torch.distributed.init_process_group(
        "gloo", init_method=(tmp_path / "loss-group").as_uri(), rank=0, world_size=1
    )
    try:
        for name in (
            "_TENSOR_MODEL_PARALLEL_GROUP",
            "_CONTEXT_PARALLEL_GROUP",
            "_DATA_PARALLEL_GROUP",
        ):
            monkeypatch.setattr(parallel_state, name, torch.distributed.group.WORLD)
        worker_cls = MAMegatronActor if role == "actor" else MAMegatronCritic
        worker = object.__new__(worker_cls)
        worker.cfg = OmegaConf.create(
            {
                "runner": {"task_type": "reasoning"},
                "algorithm": {
                    "loss_type": "actor",
                    "use_valid_token_scale": False,
                    "sampling_params": {"temperature": 1.0},
                    "early_stop_imp_ratio": None,
                },
            }
        )
        worker.transformer_config = SimpleNamespace(sequence_parallel=False)
        worker.return_loss = True
        worker.loss_agg_func = masked_mean
        worker.calculate_entropy = False
        worker.kl_beta = 0
        worker.clip_ratio_c = None
        worker.clip_ratio_low = worker.clip_ratio_high = 0.2
        worker.value_clip = 0.5
        worker.num_microbatches = 1
        input_ids = torch.arange(128).reshape(32, 4) % 7 + 1
        positions = torch.arange(4).expand_as(input_ids).clone()
        mask = (positions > 0) & (positions <= (torch.arange(32) % 3 + 1)[:, None])
        mask[-1] = False
        batch = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(mask),
            "position_ids": positions,
            "response_mask": mask,
            "loss_scales": mask.float(),
            "advantages": input_ids.float() / 7 - 0.5,
            "recomputed_logprobs": torch.zeros_like(mask, dtype=torch.float32),
            "values": torch.zeros_like(mask, dtype=torch.float32),
            "returns": torch.full_like(mask, 0.7, dtype=torch.float32),
        }
        expected_model = TokenModel()
        forward = worker.get_forward_step_func()
        output, loss_func = forward(iter([copy.deepcopy(batch)]), expected_model)
        expected_loss, _ = loss_func(output)
        expected_loss.backward()

        worker.cfg.algorithm.use_valid_token_scale = True
        worker._setup_valid_token_scale(batch)
        assert worker.global_valid_token.item() == mask.sum().item()
        worker.num_microbatches = microbatches
        monkeypatch.setattr(
            parallel_state, "get_data_parallel_world_size", lambda: dp_size
        )
        model = TokenModel()
        for rank in range(dp_size):
            indices = torch.arange(rank, len(input_ids), dp_size)
            for rows in torch.tensor_split(indices, microbatches):
                microbatch = {key: value[rows] for key, value in batch.items()}
                output, loss_func = forward(iter([microbatch]), model)
                loss, metrics = loss_func(output)
                assert loss.requires_grad and torch.isfinite(loss)
                metric = (
                    "actor/final_loss" if role == "actor" else "critic/final_value_loss"
                )
                torch.testing.assert_close(metrics[metric].squeeze(), loss)
                # Megatron averages over microbatches, then DP gradients.
                (loss / microbatches / dp_size).backward()
        torch.testing.assert_close(model.weight.grad, expected_model.weight.grad)
    finally:
        torch.distributed.destroy_process_group()


def make_result():
    return DynamicRolloutResult(
        num_sequence=3,
        group_size=2,
        idx_to_traj=[0, 0, 1],
        input_ids=[[10, 11, 12], [10, 11, 12, 50, 51, 13], [20, 21, 22]],
        prompt_lengths=[1, 5, 1],
        response_lengths=[2, 1, 2],
        is_end=[False, True, False],
        rewards=[1.0, 1.0, 0.0],
        values=torch.tensor(
            [[0.0, 0.2, 0.3, 0, 0, 0], [0, 0, 0, 0, 0, 0.4], [0, 0.1, 0.2, 0, 0, 0]]
        ),
    )


def test_ppo_targets_fan_out_complete_batches_without_reordering():
    input_channel, actor_channel, critic_channel = (
        _local_channel(),
        _local_channel(),
        _local_channel(),
    )
    results = [make_result() for _ in range(3)]
    for index, result in enumerate(results):
        result.input_ids[0][0] -= index
    for order in (results, list(reversed(results))):
        for result in order:
            input_channel.put(result)
        prepare_agent_ppo_targets(
            input_channel, [actor_channel, critic_channel], 3, 1.0, 1.0, True
        )
        for channel in (actor_channel, critic_channel):
            for result in order:
                received = channel.get()
                assert received is result
                assert received.advantages is not None and received.returns is not None
            assert channel.empty()
        assert input_channel.empty()


@pytest.mark.parametrize("duplicate_turns", [False, True])
def test_ppo_targets_join_parallel_predictions_and_preserve_fixed_values(
    duplicate_turns,
):
    results = DynamicRolloutResult.split_results(make_result(), [2, 1])
    if duplicate_turns:
        results[1] = copy.deepcopy(results[0])
    for result in results:
        result.group_size = 1
        result.idx_to_traj = [0] * result.num_sequence
    input_channel, output_channel, logprob_channel = (
        _local_channel(),
        _local_channel(),
        _local_channel(),
    )
    old_values = [result.values.clone() for result in results]
    for result in reversed(results):
        input_channel.put(result)
    for index, result in enumerate(results):
        prediction = copy.deepcopy(result)
        prediction.recomputed_logprobs = torch.full_like(result.values, -index - 1.0)
        logprob_channel.put(prediction)
    prepare_agent_ppo_targets(
        input_channel, output_channel, 2, 1.0, 1.0, False, logprob_channel
    )
    for index in range(2):
        received = output_channel.get()
        torch.testing.assert_close(received.values, old_values[1 - index])
        prediction_index = index if duplicate_turns else 1 - index
        torch.testing.assert_close(
            received.recomputed_logprobs,
            torch.full_like(received.values, -prediction_index - 1.0),
        )
        assert received.advantages is not None
        assert received.returns is not None
    assert input_channel.empty() and logprob_channel.empty() and output_channel.empty()


def test_ppo_targets_reject_mismatched_parallel_predictions():
    input_channel, output_channel, logprob_channel = (
        _local_channel(),
        _local_channel(),
        _local_channel(),
    )
    result = make_result()
    prediction = copy.deepcopy(result)
    prediction.input_ids[0][0] += 1
    input_channel.put(result)
    logprob_channel.put(prediction)
    with pytest.raises(ValueError, match="same agent turns"):
        prepare_agent_ppo_targets(
            input_channel, output_channel, 1, 1.0, 1.0, False, logprob_channel
        )
    assert output_channel.empty()


@pytest.mark.parametrize("synchronous", [False, True])
@pytest.mark.parametrize("role", ["actor", "critic"])
def test_megatron_batch_input_uses_default_queue(monkeypatch, role, synchronous):
    pytest.importorskip("megatron.training")
    from omegaconf import OmegaConf

    from rlinf.scheduler import Channel, Worker
    from rlinf.scheduler.channel.channel_worker import LocalChannel
    from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor
    from rlinf.workers.critic.ma_megatron_critic_worker import MAMegatronCritic

    monkeypatch.setattr(Worker, "torch_device_type", "cpu")
    worker = object.__new__(MAMegatronActor if role == "actor" else MAMegatronCritic)
    worker.cfg = OmegaConf.create(
        {
            "runner": {"synchronous": synchronous},
            "actor": {"model": {"encoder_seq_length": 6}},
        }
    )
    worker.role = role
    worker.role_cfg = worker.cfg.actor
    worker.is_dynamic_rollout_batch = True
    worker.tokenizer = SimpleNamespace(eos_token_id=0)
    channel = Channel()
    channel._initialize("batch-input", None, None, None, local_channel=LocalChannel())
    result = make_result()
    channel.put(result)
    # A wrong key must fail immediately instead of hanging this regression test.
    for key in (None, f"{role}:0", f"{role}:1"):
        channel.put("unexpected queue", key=key)
    batch, received = worker.get_batch(channel)
    assert received is result
    assert batch["input_ids"].shape == (result.num_sequence, 6)


def test_multiturn_critic_preserves_values_loss_and_update_with_packing(
    monkeypatch, tmp_path
):
    pytest.importorskip("megatron.training")
    from megatron.core import parallel_state
    from omegaconf import OmegaConf

    from rlinf.scheduler import Worker
    from rlinf.utils.utils import masked_mean
    from rlinf.workers.critic.ma_megatron_critic_worker import MAMegatronCritic

    class TokenValueModel(torch.nn.Module):
        pre_process = True
        post_process = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.01))

        def forward(self, input_ids, attention_mask, position_ids, packed_seq_params):
            return (input_ids.float() * self.weight).unsqueeze(-1)

    monkeypatch.setattr(Worker, "torch_device_type", "cpu")
    for name in (
        "get_tensor_model_parallel_world_size",
        "get_context_parallel_world_size",
        "get_data_parallel_world_size",
    ):
        monkeypatch.setattr(parallel_state, name, lambda: 1)
    monkeypatch.setattr(parallel_state, "get_context_parallel_rank", lambda: 0)
    torch.distributed.init_process_group(
        "gloo", init_method=(tmp_path / "critic-group").as_uri(), rank=0, world_size=1
    )
    monkeypatch.setattr(
        parallel_state, "get_data_parallel_group", lambda: torch.distributed.group.WORLD
    )
    try:
        worker = object.__new__(MAMegatronCritic)
        worker.cfg = OmegaConf.create(
            {
                "critic": {"model": {"encoder_seq_length": 6}},
                "algorithm": {"use_valid_token_scale": True},
            }
        )
        worker.role_cfg = worker.cfg.critic
        worker.tokenizer = SimpleNamespace(eos_token_id=0)
        worker.transformer_config = SimpleNamespace(sequence_parallel=False)
        worker.num_microbatches = 1
        worker.value_clip = 0.5
        worker.loss_agg_func = masked_mean

        result = make_result()
        _prepare_targets([result], 1.0, 1.0, False)
        channel = _local_channel()
        channel.put(result)
        batch, received = worker.get_batch(channel)
        assert received is result and channel.empty()
        frozen = copy.deepcopy(batch)
        outcomes = []
        for pack_traj in (False, True):
            minibatch = copy.deepcopy(batch)
            if pack_traj:
                minibatch = DynamicRolloutResult.pack_traj_batch({}, minibatch)
            minibatch.pop("idx_to_traj")
            assert minibatch["input_ids"].shape[0] == (2 if pack_traj else 3)
            model = TokenValueModel()
            worker._setup_valid_token_scale(minibatch)
            assert worker.global_valid_token.item() == 5
            forward = worker.get_forward_step_func()

            worker.return_loss = False
            values, _ = forward(iter([minibatch]), model)
            mask = minibatch["response_mask"]
            expected = torch.nn.functional.pad(
                minibatch["input_ids"].float()[:, :-1] * model.weight, (1, 0)
            )
            torch.testing.assert_close(values, expected * mask)

            worker.return_loss = True
            output, loss_func = forward(iter([minibatch]), model)
            loss, metrics = loss_func(output)
            reference_loss, _ = compute_ppo_critic_loss(
                values=expected,
                returns=minibatch["returns"],
                prev_values=minibatch["values"],
                value_clip=worker.value_clip,
                huber_delta=10000,
                loss_mask=mask,
            )
            torch.testing.assert_close(loss, reference_loss)
            torch.testing.assert_close(
                metrics["critic/final_value_loss"].squeeze(), loss
            )
            loss.backward()
            assert torch.isfinite(model.weight.grad) and model.weight.grad != 0
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            optimizer.step()
            assert model.weight.item() != pytest.approx(0.01)
            outcomes.append(
                (values[mask].detach(), loss.detach(), model.weight.detach())
            )

        for unpacked, packed in zip(*outcomes):
            torch.testing.assert_close(unpacked, packed)
        for key in ("values", "returns", "advantages"):
            torch.testing.assert_close(batch[key], frozen[key])
    finally:
        torch.distributed.destroy_process_group()


def test_agent_targets_reward_once_across_turns_and_stay_frozen():
    result = make_result()
    old_values = result.values.clone()
    _prepare_targets([result], 1.0, 1.0, False)
    torch.testing.assert_close(result.returns[0, 1:3], torch.ones(2))
    assert result.returns[1, 5] == 1.0
    torch.testing.assert_close(result.returns[2, 1:3], torch.zeros(2))
    assert result.advantages[1, :5].count_nonzero() == 0
    torch.testing.assert_close(result.values, old_values)
    snapshot = copy.deepcopy(result)
    parts = DynamicRolloutResult.split_results(result, [2, 1])
    restored = DynamicRolloutResult.merge_result_list(parts)
    torch.testing.assert_close(restored.values, snapshot.values)
    torch.testing.assert_close(restored.returns, snapshot.returns)
    torch.testing.assert_close(restored.advantages, snapshot.advantages)


def test_budget_episode_does_not_earn_value_beyond_its_final_reward():
    result = make_result()
    _prepare_targets([result], 1.0, 1.0, False)
    torch.testing.assert_close(result.returns[2, 1:3], torch.zeros(2))
    torch.testing.assert_close(result.advantages[2, 1:3], torch.tensor([-0.1, -0.2]))
    torch.testing.assert_close(result.returns[0, 1:3], torch.ones(2))


def test_agent_advantage_normalization_is_global_over_actions():
    result = make_result()
    _prepare_targets([result], 1.0, 1.0, True)
    valid = torch.cat(
        [
            result.advantages[i, p : p + n]
            for i, (p, n) in enumerate(
                zip(result.prompt_lengths, result.response_lengths, strict=True)
            )
        ]
    )
    torch.testing.assert_close(valid.mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(valid.var(unbiased=False), torch.tensor(1.0))


def test_agent_targets_reject_an_episode_without_generated_turns():
    result = make_result()
    result.group_size = 3
    with pytest.raises(ValueError, match="nonempty generated turns"):
        _prepare_targets([result], 1.0, 1.0, True)


@pytest.mark.parametrize("role", ["actor", "critic"])
def test_trajectory_packing_preserves_ppo_targets_loss_and_gradients(monkeypatch, role):
    from rlinf.scheduler import Worker

    monkeypatch.setattr(Worker, "torch_device_type", "cpu")
    result = make_result()
    _prepare_targets([result], 0.99, 0.95, True)
    torch.manual_seed(7)
    model = torch.nn.Sequential(
        torch.nn.Embedding(64, 4), torch.nn.Linear(4, 64 if role == "actor" else 1)
    )

    def predict(model, batch):
        hidden = model[0](batch["input_ids"]).cumsum(dim=1)
        raw = model[1](hidden) * 0.05
        predictions = torch.zeros_like(batch["input_ids"], dtype=torch.float32)
        if role == "actor":
            next_logprobs = (
                raw[:, :-1]
                .log_softmax(-1)
                .gather(-1, batch["input_ids"][:, 1:].unsqueeze(-1))
                .squeeze(-1)
            )
            predictions[:, 1:] = next_logprobs
        else:
            predictions[:, 1:] = raw[:, :-1, 0]
        return predictions

    batch = result.to_actor_batch(6, 0)
    batch["recomputed_logprobs"] = predict(model, batch).detach() - 0.02
    original = copy.deepcopy(batch)
    packed = DynamicRolloutResult.pack_traj_batch({}, batch)
    assert packed["idx_to_traj"] == [0, 1]
    assert packed["input_ids"].shape[0] == 2
    assert packed["attention_mask"].sum() < original["attention_mask"].sum()
    assert packed["response_mask"].sum() == original["response_mask"].sum()
    for key in ("recomputed_logprobs", "advantages", "values", "returns"):
        torch.testing.assert_close(
            packed[key][packed["response_mask"]],
            original[key][original["response_mask"]],
        )
    for key in batch:
        if torch.is_tensor(batch[key]):
            torch.testing.assert_close(batch[key], original[key])

    packed_model = copy.deepcopy(model)

    def loss(model, batch):
        predictions = predict(model, batch)
        if role == "actor":
            return compute_ppo_actor_loss(
                logprobs=predictions,
                old_logprobs=batch["recomputed_logprobs"],
                advantages=batch["advantages"],
                clip_ratio_low=0.2,
                clip_ratio_high=0.2,
                loss_mask=batch["response_mask"],
            )[0]
        return compute_ppo_critic_loss(
            values=predictions,
            prev_values=batch["values"],
            returns=batch["returns"],
            value_clip=0.5,
            huber_delta=10000,
            loss_mask=batch["response_mask"],
        )[0]

    original_loss, packed_loss = loss(model, original), loss(packed_model, packed)
    torch.testing.assert_close(packed_loss, original_loss)
    original_loss.backward()
    packed_loss.backward()
    for before, after in zip(
        model.parameters(), packed_model.parameters(), strict=True
    ):
        torch.testing.assert_close(after.grad, before.grad, rtol=1e-5, atol=1e-7)


def test_trajectory_packing_keeps_one_token_overlap_as_separate_rows(monkeypatch):
    from rlinf.scheduler import Worker

    monkeypatch.setattr(Worker, "torch_device_type", "cpu")
    result = make_result()
    result.input_ids[1] = result.input_ids[0]
    result.prompt_lengths[1] = 2
    result.response_lengths[1] = 1
    result.values = None
    batch = result.to_actor_batch(6, 0)
    packed = DynamicRolloutResult.pack_traj_batch({}, batch)
    assert packed["idx_to_traj"] == batch["idx_to_traj"]
    torch.testing.assert_close(packed["response_mask"], batch["response_mask"])


@pytest.mark.parametrize("role", ["actor", "critic"])
@pytest.mark.parametrize("pack_traj", [False, True])
def test_ppo_batch_balance_before_minibatches_preserves_targets_and_loss(
    monkeypatch, tmp_path, role, pack_traj
):
    from rlinf.scheduler import Worker
    from rlinf.utils.data_iter_utils import (
        get_iterator_k_split,
        get_seqlen_balanced_partitions,
    )
    from rlinf.utils.distributed import RolloutDataBalance

    monkeypatch.setattr(Worker, "torch_device_type", "cpu")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    result = make_result()
    _prepare_targets([result], 0.99, 0.95, True)
    multi = DynamicRolloutResult.split_results(result, [2, 1])[0]
    multi.group_size = 1
    multi.idx_to_traj = [0, 0]
    single = DynamicRolloutResult.split_results(result, [2, 1])[1]
    single.group_size = 1
    single.idx_to_traj = [0]
    groups = [multi, single, single, single]
    snapshots = [copy.deepcopy(group) for group in groups]
    original = DynamicRolloutResult.merge_batches(
        [group.to_actor_batch(6, 0) for group in groups], 1
    )
    batch = copy.deepcopy(original)
    if pack_traj:
        batch = DynamicRolloutResult.pack_traj_batch({}, batch)
    batch.pop("idx_to_traj")
    batch["loss_scales"] = batch["response_mask"].float()

    torch.distributed.init_process_group(
        "gloo", init_method=(tmp_path / "balance-group").as_uri(), rank=0, world_size=1
    )
    try:
        batch = RolloutDataBalance.from_rollout_batches_dynamic(
            batch,
            dp_world_size=1,
            dp_rank=0,
            dp_group=torch.distributed.group.WORLD,
            rollout_batch_pad=DynamicRolloutResult.get_batch_pad(6, batch.keys()),
            split_fix_chunk=2,
            partitioning_tool=get_seqlen_balanced_partitions,
        )
    finally:
        torch.distributed.destroy_process_group()

    assert batch["input_ids"].shape[0] == (4 if pack_traj else 6)
    minibatches = list(get_iterator_k_split(batch, num_splits=2))
    assert all(
        part["input_ids"].shape[0] == (2 if pack_traj else 3) for part in minibatches
    )
    assert sum(int(part["response_mask"].sum()) for part in minibatches) == 9
    for field in ("input_ids", "values", "advantages", "returns"):
        expected = original[field][original["response_mask"]].sort().values
        actual = batch[field][batch["response_mask"]].sort().values
        torch.testing.assert_close(actual, expected)

    def loss(parameter, data):
        mask = data["response_mask"]
        prediction = parameter * data["input_ids"].float()
        if role == "actor":
            return compute_ppo_actor_loss(
                logprobs=prediction,
                old_logprobs=torch.zeros_like(prediction),
                advantages=data["advantages"],
                clip_ratio_low=0.2,
                clip_ratio_high=0.2,
                loss_mask=mask,
            )[0]
        return compute_ppo_critic_loss(
            values=prediction,
            returns=data["returns"],
            prev_values=data["values"],
            value_clip=0.5,
            huber_delta=10000,
            loss_mask=mask,
        )[0]

    expected_parameter = torch.tensor(0.01, requires_grad=True)
    expected_loss = loss(expected_parameter, original)
    expected_loss.backward()
    parameter = torch.tensor(0.01, requires_grad=True)
    total_loss = (
        sum(loss(parameter, part) * part["response_mask"].sum() for part in minibatches)
        / original["response_mask"].sum()
    )
    total_loss.backward()
    torch.testing.assert_close(total_loss, expected_loss)
    torch.testing.assert_close(parameter.grad, expected_parameter.grad)
    for group, snapshot in zip(groups, snapshots, strict=True):
        for field in ("values", "advantages", "returns"):
            torch.testing.assert_close(getattr(group, field), getattr(snapshot, field))


@pytest.mark.parametrize(
    "option,value",
    [
        ("recompute_logprobs", False),
        ("group_size", 4),
        ("adv_type", "gae"),
        ("kl_beta", 0.01),
        ("loss_agg_func", "seq-mean-token-mean"),
        ("use_valid_token_scale", False),
        ("loss_scales", ["group_level"]),
    ],
)
def test_searchr1_ppo_rejects_incompatible_algorithm_options(option, value):
    from omegaconf import OmegaConf

    from rlinf.config import validate_searchr1_cfg

    cfg = OmegaConf.create(
        {
            "reward": {"reward_type": "searchr1"},
            "agentloop": {"is_dynamic_rollout_batch": True},
            "runner": {"synchronous": True},
            "rollout": {"rollout_backend": "sglang"},
            "actor": {
                "enable_dp_load_balance": True,
                "training_backend": "megatron",
                "model": {
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                },
            },
            "critic": {
                "use_critic_model": True,
                "training_backend": "megatron",
                "model": {
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                },
            },
            "algorithm": {
                "adv_type": "actor_gae",
                "group_size": 1,
                "kl_beta": 0,
                "recompute_logprobs": True,
                "use_valid_token_scale": True,
            },
        }
    )
    validate_searchr1_cfg(cfg)
    cfg.algorithm[option] = value
    with pytest.raises(ValueError, match="SearchR1 PPO requires"):
        validate_searchr1_cfg(cfg)


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
def test_rollout_config_needs_only_the_selected_backend(backend):
    from omegaconf import OmegaConf

    from rlinf.config import validate_rollout_cfg

    cfg = OmegaConf.create(
        {
            "model": {"model_type": "qwen2.5", "model_path": "/local/model"},
            "rollout_backend": backend,
            "return_logprobs": False,
            backend: {},
        }
    )
    validated = validate_rollout_cfg(cfg, OmegaConf.create({}))
    assert validated[backend].attention_backend
    unused_backend = "vllm" if backend == "sglang" else "sglang"
    assert unused_backend not in validated
