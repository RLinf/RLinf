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
import json
from types import SimpleNamespace

import pytest
import torch

from rlinf.algorithms.losses import (
    compute_ppo_actor_loss,
    compute_ppo_critic_loss,
)
from rlinf.algorithms.registry import get_adv_and_returns
from rlinf.data.schema.agent_trajectory import prepare_agent_ppo_targets
from rlinf.data.schema.reasoning_results import DynamicRolloutResult
from rlinf.scheduler import Channel
from rlinf.scheduler.channel.channel_worker import LocalChannel


def _local_channel():
    channel = Channel()
    channel._initialize("ppo-targets", None, None, None, local_channel=LocalChannel())
    return channel


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

    rows = result.to_actor_batch()
    dense_batch = DynamicRolloutResult.materialize_actor_batch(rows, 6, 0)
    scores = predict(model, dense_batch).detach() - 0.02
    rows["recomputed_logprobs"] = [
        score[: len(ids)] for score, ids in zip(scores, rows["input_ids"], strict=True)
    ]
    original = DynamicRolloutResult.materialize_actor_batch(rows, 6, 0)
    packed = DynamicRolloutResult.materialize_actor_batch(
        DynamicRolloutResult.pack_traj_batch({}, rows),
        6,
        0,
    )
    batch = dense_batch
    batch["recomputed_logprobs"] = original["recomputed_logprobs"]
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


# A Qwen3.5 shape small enough to test, with Gated DeltaNet sizes that differ
# from Megatron's defaults.
QWEN3_5_SHAPE = {
    "hidden_size": 8,
    "num_attention_heads": 4,
    "num_query_groups": 2,
    "kv_channels": 3,
    "attention_output_gate": True,
    "linear_key_head_dim": 2,
    "linear_num_key_heads": 2,
    "linear_value_head_dim": 3,
    "linear_num_value_heads": 4,
}


@pytest.mark.parametrize("presence_penalty", [0.0, 1.5])
def test_tis_corrects_penalized_behavior_without_differentiating_the_weights(
    presence_penalty,
):
    """Enumerating the vocabulary gives an exact expected policy gradient."""
    from rlinf.algorithms.registry import get_policy_loss
    from rlinf.algorithms.utils import apply_importance_sampling_correction

    temperature, clip = 0.7, 2.0
    logits = torch.tensor([[0.4, -0.2, 0.8, 0.1]], requires_grad=True)
    seen = torch.tensor([[True, False, True, False]])
    target = (logits / temperature).log_softmax(-1)
    behavior = ((logits.detach() - presence_penalty * seen) / temperature).softmax(-1)
    mask = torch.ones_like(seen)
    weights, _, _ = apply_importance_sampling_correction(
        torch.ones_like(target), target, behavior.log(), clip
    )
    assert not weights.requires_grad
    torch.testing.assert_close(
        weights, (target.detach().exp() / behavior).clamp(max=clip)
    )

    advantages = torch.tensor([[1.0, -0.5, -1.0, 0.25]])
    weighted_advantages = advantages * weights
    loss, _ = get_policy_loss("actor")(
        logprobs=target,
        old_logprobs=target.detach(),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        advantages=weighted_advantages,
        loss_mask=mask,
        loss_agg_func=lambda values, loss_mask, *args: (
            values * behavior * loss_mask
        ).sum(),
    )
    gradient = torch.autograd.grad(loss, logits)[0]
    probabilities = target.detach().exp()
    mass = torch.minimum(probabilities, clip * behavior) * advantages
    expected = -(mass - probabilities * mass.sum()) / temperature
    torch.testing.assert_close(gradient, expected, atol=1e-6, rtol=1e-6)


def test_rollout_failure_cancels_and_drains_sibling_episodes():
    from rlinf.workers.agent.agent_loop import MultiAgentLoopWorker

    cleaned = []

    async def sibling():
        try:
            await asyncio.sleep(10)
        finally:
            await asyncio.sleep(0)
            cleaned.append("sibling")

    async def failing():
        await asyncio.sleep(0)
        raise RuntimeError("episode failed")

    async def main():
        tasks = [asyncio.create_task(sibling()), asyncio.create_task(failing())]
        with pytest.raises(RuntimeError, match="episode failed"):
            await MultiAgentLoopWorker._gather_rollout_tasks(tasks)
        # The sibling finished its cleanup before the failure reached the caller.
        assert cleaned == ["sibling"] and tasks[0].cancelled()

    asyncio.run(asyncio.wait_for(main(), timeout=5))


def test_qwen35_convertor_preserves_bf16_gdn_norm_offset():
    from types import SimpleNamespace

    from rlinf.utils.convertor.utils import Qwen35Convertor

    # All values are exact in BF16, but adding one in BF16 loses their offsets.
    centered = torch.tensor([1 / 256, -1 / 512, 1 / 1024], dtype=torch.bfloat16)
    original = centered.clone()
    converted = Qwen35Convertor(
        SimpleNamespace(
            reshard_tp_size=1,
            rollout_vision_tp_size=1,
            model_config=SimpleNamespace(
                num_attention_heads=2,
                num_query_groups=1,
                linear_num_key_heads=2,
                linear_num_value_heads=2,
                ffn_hidden_size=8,
            ),
            vision_config=SimpleNamespace(
                num_heads=2, intermediate_size=8, hidden_size=4, spatial_merge_size=2
            ),
        ),
        strict=True,
    ).convert(
        {"language_model.decoder.layers.0.self_attention.out_norm.weight": centered}
    )
    expected = torch.tensor([257 / 256, 511 / 512, 1025 / 1024])
    torch.testing.assert_close(
        converted["model.language_model.layers.0.linear_attn.norm.weight"],
        expected,
        rtol=0,
        atol=0,
    )
    assert torch.equal(centered, original)


def test_checkpoint_loader_patch_is_released_after_a_failed_load(monkeypatch):
    training = pytest.importorskip("megatron.training.training")
    from rlinf.hybrid_engines.megatron.megatron_model_manager import (
        patch_load_checkpoint_to_be_non_strict,
    )

    def loader(*, strict=True):
        return strict

    monkeypatch.setattr(training, "load_checkpoint", loader)
    with pytest.raises(OSError, match="checkpoint unavailable"):
        with patch_load_checkpoint_to_be_non_strict(True):
            assert training.load_checkpoint() is False
            raise OSError("checkpoint unavailable")
    assert training.load_checkpoint() is True


def test_batch_merge_pads_rows_for_a_text_only_batch():
    """A batch without images must still contribute one slot per row."""

    def batch(payloads):
        data = {
            "input_ids": torch.zeros(3, 6, dtype=torch.long),
            "idx_to_traj": [0, 1, 2],
            "prompt_lengths": torch.ones(3, dtype=torch.int32),
            "response_lengths": torch.ones(3, dtype=torch.int32),
            "is_end": torch.zeros(3, dtype=torch.bool),
        }
        if payloads is not None:
            data["multi_modal_inputs"] = payloads
        return data

    with_images = batch([None, {"tag": torch.tensor([1.0])}, None])
    for order in ([batch(None), with_images], [with_images, batch(None)]):
        merged = DynamicRolloutResult.merge_batches(
            [dict(item) for item in order], group_size=3, adjust_traj_indices=True
        )
        payloads = merged["multi_modal_inputs"]
        assert len(payloads) == merged["input_ids"].shape[0] == 6
        assert sum(entry is not None for entry in payloads) == 1


@pytest.mark.parametrize(
    "settings,token_space,error",
    [
        ({}, False, None),
        ({"cache_bytes": 1024}, False, None),
        ({"prefetch_batches": 2, "prefetch_bytes": 4096}, True, None),
        ({"cache_bytes": -1}, False, "cache_bytes"),
        ({"prefetch_batches": True}, True, "prefetch_batches"),
        ({"prefetch_bytes": 1.5}, True, "prefetch_bytes"),
        ({"prefetch_batches": 1}, True, "positive prefetch_bytes"),
        (
            {"prefetch_batches": 1, "prefetch_bytes": 4096},
            False,
            "enable_shared_multi_modal_processor",
        ),
    ],
)
def test_multimodal_learner_budgets_without_rpent(settings, token_space, error):
    from omegaconf import OmegaConf

    from rlinf.config import validate_multimodal_learner_cfg

    cfg = OmegaConf.create(
        {
            "actor": {"multimodal": settings},
            "rollout": {"sglang": {"enable_shared_multi_modal_processor": token_space}},
        }
    )
    if error:
        with pytest.raises(ValueError, match=error):
            validate_multimodal_learner_cfg(cfg)
    else:
        snapshot = OmegaConf.to_container(cfg)
        validate_multimodal_learner_cfg(cfg)
        assert OmegaConf.to_container(cfg) == snapshot


def test_group_candidates_spread_320_members_over_32_workers():
    from rlinf.data.schema.reasoning_requests import build_rollout_requests_from_batch

    batch = {
        "prompt": torch.arange(40).reshape(40, 1),
        "length": torch.ones(40, dtype=torch.long),
        "answer": [[str(group)] for group in range(40)],
        "image_data": [None] * 40,
        "multi_modal_inputs": [None] * 40,
    }
    requests = build_rollout_requests_from_batch(
        batch, group_size=8, split_size=32, split_trajectories=True
    )
    assert len(requests) == 32
    assert all(len(request.input_ids) == 10 and request.n == 1 for request in requests)
    answers = [answer for request in requests for answer in request.answers]
    assert answers == [[str(group)] for group in range(40) for _ in range(8)]


def test_grouped_episodes_keep_trajectory_rewards_images_and_action_masks():
    trajectories = []
    for member, turns in enumerate([1, 3, 2, 4, 1, 2, 3, 1]):
        trajectories.append(
            DynamicRolloutResult(
                num_sequence=turns,
                group_size=1,
                idx_to_traj=[0] * turns,
                input_ids=[[10, 11, 12]] * turns,
                prompt_lengths=[1] * turns,
                response_lengths=[2] * turns,
                response_mask=[[1, 0]] * turns,
                rollout_logprobs=[[-0.3, -0.4]] * turns,
                is_end=[True] * turns,
                rewards=[float(member % 2)] * turns,
                extra_fields_traj={"episode_id": [str(member)]},
                multi_modal_inputs=[None] * turns,
                version_start=[2] * turns,
                version_end=[2] * turns,
            )
        )
    result = DynamicRolloutResult.from_trajectory_group(trajectories, group_id=6)
    assert result.group_size == 8
    assert result.group_id == 6
    assert result.idx_to_traj == [
        member
        for member, trajectory in enumerate(trajectories)
        for _ in trajectory.idx_to_traj
    ]
    assert result.group_metadata().rewards == [0.0, 1.0] * 4
    assert result.extra_fields_traj["episode_id"] == list(map(str, range(8)))
    assert result.response_mask == [[1, 0]] * 17
    assert result.rollout_logprobs == [[-0.3, -0.4]] * 17


def test_grpo_hierarchical_scales_keep_action_masks_and_weight_variable_turns():
    from rlinf.algorithms.registry import get_loss_scales

    mask = torch.tensor([[True, False, False], [True, True, True], [True, True, False]])
    batch = {
        "idx_to_traj": [0, 0, 1],
        "extra:idx_to_sub_traj": torch.tensor([0, 0, 0]),
        "advantages": mask.float(),
        "response_mask": mask,
        "loss_scales": mask.float(),
    }
    names = ["group_level", "agent_level", "turn_level"]
    context = {
        "folding_scale": [],
        "actor_global_batch_size": 32 * 8,
        "data_parallel_world_size": 16,
    }
    for scale in get_loss_scales(names):
        batch = scale(context, batch)
    assert context["folding_scale"] == names
    batch["advantages"] = torch.stack(tuple(batch["advantages"]))
    batch["loss_scales"] = torch.stack(tuple(batch["loss_scales"]))
    torch.testing.assert_close(batch["advantages"], mask.float() * 3 * 16 / 256)
    torch.testing.assert_close(
        batch["loss_scales"],
        torch.tensor([[0.25, 0.0, 0.0], [0.75, 0.75, 0.75], [1.0, 1.0, 0.0]]),
    )
    assert not batch["loss_scales"][~mask].any()


@pytest.mark.parametrize("failure", [None, "candidate", "cleanup"])
@pytest.mark.parametrize("group_size,mixed", [(1, False), (8, False), (2, True)])
def test_sync_runner_selects_first_complete_candidates_and_admits_only_after_cleanup(
    tmp_path, monkeypatch, failure, group_size, mixed
):
    from omegaconf import OmegaConf

    from rlinf.agents.rpent.simulators import Simulator
    from rlinf.runners.agent_ppo_runner import AgentPPORunner

    monkeypatch.setattr(Channel, "create", lambda *_args, **_kwargs: _local_channel())
    order = []
    selected_ids = []
    completed_groups = (2, 0, 3, 4) if mixed else (2, 0, 4)
    chosen_groups = (2, 3, 4) if mixed else completed_groups
    records = [
        {
            "candidate_id": i,
            "status": "complete"
            if i // group_size in completed_groups
            else "cancelled",
            "cleanup_complete": True,
        }
        for i in range(5 * group_size)
    ]
    if failure == "candidate":
        records[group_size]["status"] = "error"
        records[group_size]["error"] = "transport failed after retry"
    runner = object.__new__(AgentPPORunner)
    runner.cfg = OmegaConf.create(
        {
            "data": {"rollout_batch_size": 3, "sampling_batch_size": 5},
            "runner": {
                "task_type": "reasoning",
                "synchronous": True,
                "logger": {"log_path": str(tmp_path)},
            },
            "algorithm": {"group_size": group_size},
            "agentloop": {"is_dynamic_rollout_batch": True},
            "rollout": {"rollout_backend": "sglang", "replica_affinity": True},
        }
    )
    runner.environment_cohort = None
    if mixed:
        from rlinf.agents.rpent import cohort

        monkeypatch.setattr(cohort, "rpc", lambda *args: {"ready": 0, "reserved": 0})
        simulators = {
            name: Simulator("http://backend", name, 1, candidates)
            for name, candidates in (("libero", 2), ("robocasa", 2), ("robodojo", 1))
        }
        runner.cfg.tools = {"rpent": {}}
        runner.environment_cohort = cohort.SimulatorCohort(runner.cfg, simulators)
        runner.environment_cohort.groups = [
            "libero",
            "robocasa",
            "libero",
            "robocasa",
            "robodojo",
        ]
    runner.global_steps = 2
    runner._sampling_cohort_id = "cpu-cohort"
    runner.candidate_channel = _local_channel()
    runner.candidate_result_channel = _local_channel()
    runner.dataloader_channel = _local_channel()
    learner = _local_channel()
    runner.rollout_output_channels = [learner]
    values = {}

    def handle(value=None):
        return SimpleNamespace(wait=lambda: value)

    def launch(input_channel, output_channel):
        order.append("launch")
        # Fast members of unfinished groups must not be admitted individually.
        for candidate_id in (
            group * group_size + member
            for member in range(group_size)
            for group in completed_groups
        ):
            result = DynamicRolloutResult(
                num_sequence=1,
                group_size=1,
                group_id=candidate_id,
                idx_to_traj=[0],
                prompt_lengths=[1],
                response_lengths=[1],
                input_ids=[[1, 2]],
                is_end=[True],
                rewards=[1.0 if candidate_id == 0 else 0.0],
                version_start=[2],
                version_end=[2],
            )
            values[candidate_id] = result
            output_channel.put(
                {
                    "cohort_id": "cpu-cohort",
                    "candidate_id": candidate_id,
                    "status": "complete",
                }
            )
        return handle([{}])

    def cancel(cohort_id, selected):
        assert cohort_id == "cpu-cohort"
        selected_ids.extend(selected)
        order.append("cancel_requested")

        def wait():
            assert order[-1] == "engine_fence"
            assert learner.empty()
            if failure == "cleanup":
                raise RuntimeError("remote lease cleanup failed")
            order.append("agent_cleanup")
            return [{"cohort_id": cohort_id, "candidates": records}]

        return SimpleNamespace(wait=wait)

    def finish(cohort_id):
        assert cohort_id == "cpu-cohort"
        order.append("engine_fence")
        return handle([{"cohort_id": cohort_id, "inflight": 0, "queue_fenced": True}])

    def admit(cohort_id, channels):
        assert cohort_id == "cpu-cohort"
        if channels is None:
            return handle([{"admitted": 0, "admitted_ids": []}])
        assert order[-1] == "engine_fence"
        order.append("admit")
        for candidate_id in selected_ids:
            for channel in channels:
                channel.put(values[candidate_id])
        return handle(
            [{"admitted": len(selected_ids), "admitted_ids": list(selected_ids)}]
        )

    runner.agent_loop = SimpleNamespace(
        run_agentloop_rollout=launch,
        cancel_rollout_candidates=cancel,
        admit_rollout_candidates=admit,
    )
    runner.rollout = SimpleNamespace(finish_generation_cohort=finish)
    if failure is None:
        runner._start_rollout()
        assert selected_ids == [
            group * group_size + member
            for group in chosen_groups
            for member in range(group_size)
        ]
        admitted = [learner.get() for _ in range(3)]
        assert [result.group_size for result in admitted] == [group_size] * 3
        assert [result.idx_to_traj for result in admitted] == [
            list(range(group_size))
        ] * 3
        assert admitted[0].rewards == [0.0] * group_size
        assert learner.empty()
        assert order == [
            "launch",
            "cancel_requested",
            "engine_fence",
            "agent_cleanup",
            "engine_fence",
            "admit",
        ]
    else:
        with pytest.raises(RuntimeError, match="infrastructure|cleanup failed"):
            runner._start_rollout()
        assert learner.empty()
    audit = json.loads((tmp_path / "sampling/cpu-cohort.json").read_text())
    assert audit["sampling_batch_size"] == 5
    assert audit["effective_batch_size"] == 3
    assert audit["status"] == ("passed" if failure is None else "failed")
    assert audit["selected_ids"] == [
        group * group_size + member
        for group in chosen_groups
        for member in range(group_size)
    ]
    assert audit["selected_group_ids"] == list(chosen_groups)
    if mixed:
        assert audit["selected_groups_by_simulator"] == {
            "libero": 1,
            "robocasa": 1,
            "robodojo": 1,
        }
