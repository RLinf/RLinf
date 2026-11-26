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

"""Multi-turn Megatron batches, packing, and weight-version bookkeeping.

The actor is built only in Megatron environments, so every test skips when
`megatron.core` is absent. No Ray or GPUs are launched.
"""

import copy
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from rlinf.data.schema.reasoning_results import DynamicRolloutResult


class TestExtractVersionedTrajectoryRecords:
    """The actor's record extraction from DynamicRolloutResult objects."""

    @staticmethod
    def _make_actor():
        pytest.importorskip("megatron.core")
        # __new__ bypasses Megatron init; only the private extraction
        # method is exercised.
        from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor

        return MAMegatronActor.__new__(MAMegatronActor)

    @staticmethod
    def _dynamic_result(
        num_sequence: int = 2,
        group_size: int = 1,
        version_start: list[int] | None = None,
        version_end: list[int] | None = None,
    ):
        return DynamicRolloutResult(
            num_sequence=num_sequence,
            group_size=group_size,
            idx_to_traj=list(range(num_sequence)),
            input_ids=[[1, 2, 3] for _ in range(num_sequence)],
            prompt_lengths=[2] * num_sequence,
            response_lengths=[1] * num_sequence,
            is_end=[True] * num_sequence,
            rewards=[0.0] * num_sequence,
            version_start=version_start,
            version_end=version_end,
        )

    def test_extracts_per_trajectory_records(self):
        actor = self._make_actor()
        # One result = one group (result_index 0); its trajectories are
        # trajectory 0 (two turns) and trajectory 1 (one turn).
        result = DynamicRolloutResult(
            num_sequence=3,
            group_size=2,
            idx_to_traj=[0, 0, 1],
            input_ids=[[1] * 4, [1] * 4, [1] * 4],
            prompt_lengths=[1, 1, 1],
            response_lengths=[10, 30, 20],
            is_end=[True, True, True],
            rewards=[0.0, 1.0, 2.0],
            version_start=[5, 6, 7],
            version_end=[6, 6, 7],
        )
        records = actor._extract_versioned_trajectory_records([result])
        assert len(records) == 2
        first, second = records
        # Every trajectory of one result shares the result's group index.
        assert first.group_index == 0 and second.group_index == 0
        assert first.earliest_version == 5
        assert first.completion_version == 6
        assert first.generated_tokens == 40
        assert first.call_count == 2
        assert first.cross_version_calls == 1
        assert first.reward == 0.5
        assert first.token_weighted_version_start_sum == 5 * 10 + 6 * 30
        assert second.earliest_version == 7
        assert second.generated_tokens == 20
        assert second.reward == 2.0

    def test_returns_none_without_versions(self):
        actor = self._make_actor()
        result = self._dynamic_result()
        assert actor._extract_versioned_trajectory_records([result]) is None

    def test_returns_none_if_any_result_lacks_versions(self):
        actor = self._make_actor()
        versioned = self._dynamic_result(version_start=[1, 2], version_end=[1, 2])
        assert (
            actor._extract_versioned_trajectory_records(
                [versioned, self._dynamic_result()]
            )
            is None
        )


class _LocalDeliveryChannel:
    """A local input channel that returns one delivered group."""

    is_local = True

    def __init__(self, delivery):
        self._delivery = delivery
        self.get_count = 0

    def get(self):
        self.get_count += 1
        return self._delivery


def test_get_step_batch_assembles_delivered_groups(monkeypatch):
    """get_step_batch assembles a group delivered by the session server.

    A delivered group arrives as a GroupDelivery whose data is in a payload
    channel, and only MAMegatronActor._receive_rollout_result turns it into
    training data. If get_step_batch received the group with a plain
    channel.get(), the GroupDelivery would reach the batch code and the
    first training step would fail.
    """
    pytest.importorskip("megatron.core")
    from megatron.core import parallel_state

    from rlinf.data.schema.reasoning_results import GroupDelivery, PayloadRef
    from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor

    actor = MAMegatronActor.__new__(MAMegatronActor)
    actor.cfg = OmegaConf.create(
        {"data": {"rollout_batch_size": 1}, "algorithm": {"group_size": 1}}
    )
    delivery = GroupDelivery(
        group_id=7,
        task_name="searchr1",
        payload_ref=PayloadRef(locator="channel://Payload", key=7),
    )
    assembled = DynamicRolloutResult(
        num_sequence=1,
        group_size=1,
        idx_to_traj=[0],
        input_ids=[[1, 2, 3]],
        prompt_lengths=[2],
        response_lengths=[1],
        is_end=[True],
        rewards=[0.0],
    )
    assembled_calls = []
    monkeypatch.setattr(
        actor,
        "_assemble_delivery",
        lambda delivery: assembled_calls.append(delivery) or assembled,
    )
    monkeypatch.setattr(parallel_state, "get_data_parallel_world_size", lambda: 1)
    channel = _LocalDeliveryChannel(delivery)

    _batch, results = actor.get_step_batch(channel)

    assert assembled_calls == [delivery]
    assert results == [assembled]
    assert channel.get_count == 1


def _versioned_group(identity):
    lengths = [3 + identity, 5 + identity]
    return DynamicRolloutResult(
        num_sequence=2,
        group_size=2,
        idx_to_traj=[0, 1],
        input_ids=[[identity] * n for n in lengths],
        prompt_lengths=[1, 1],
        response_lengths=[n - 1 for n in lengths],
        is_end=[True, True],
        rewards=[0.0, 1.0],
        group_id=identity,
        task_name="searchr1",
        version_start=[5 + identity, 5 + identity],
        version_end=[5 + identity, 6 + identity],
    )


def _distributed_staleness_process(rank, counts, balance, missing, init_file):
    from rlinf.utils.distributed import balance_rollout_groups
    from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=len(counts),
        timeout=timedelta(seconds=90),
    )
    try:
        first = sum(counts[:rank])
        results = [_versioned_group(i) for i in range(first, first + counts[rank])]
        if missing and rank == 1:
            results[0].version_start = results[0].version_end = None
        if balance:
            results = balance_rollout_groups(results, dist.group.WORLD)
        actor = object.__new__(MAMegatronActor)
        metrics = actor._compute_staleness_metrics(results, 10, dist.group.WORLD)
        assert actor._compute_staleness_metrics(results, None, dist.group.WORLD) is None
        if missing:
            assert metrics is None
            return
        assert metrics["data_staleness/mean"] == 3.5
        assert metrics["data_staleness/max"] == 5
        assert metrics["queue_steps/mean"] == 2.5
        weighted_versions = sum((5 + i) * (6 + 2 * i) for i in range(4))
        tokens = sum(6 + 2 * i for i in range(4))
        assert metrics["data_staleness/token_mean"] == pytest.approx(
            10 - weighted_versions / tokens
        )
        for i in range(4):
            assert metrics[f"data_staleness_{5 - i}/ratio"] == 0.25
        for result in results:
            assert result.task_name == "searchr1"
            assert result.version_start == [5 + result.group_id] * 2
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "counts,balance,missing",
    [
        ((1, 3), False, False),
        ((1, 3), True, False),
        ((0, 4), False, False),
        ((0, 4), True, False),
        ((1, 3), True, True),
    ],
)
def test_staleness_survives_ragged_dp_redistribution(
    tmp_path, counts, balance, missing
):
    pytest.importorskip("megatron.core")
    mp.spawn(
        _distributed_staleness_process,
        args=(counts, balance, missing, str(tmp_path / "versions")),
        nprocs=len(counts),
        join=True,
    )


def _packing_process(rank, tp_size, cp_size, init_file):
    from megatron.core import parallel_state

    from rlinf.data.schema.reasoning_results import DynamicRolloutResult
    from rlinf.hybrid_engines.megatron.utils import (
        postprocess_packed_seqs,
        postprocess_ragged_seqs,
        preprocess_packed_seqs,
        preprocess_ragged_seqs,
    )

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=tp_size * cp_size,
        timeout=timedelta(seconds=60),
    )
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
    )
    try:
        lengths = [101, 201, 301]
        rows = [torch.arange(n).float() for n in lengths]
        dense = torch.stack(
            [torch.nn.functional.pad(row, (0, 8192 - len(row))) for row in rows]
        )
        mask = torch.arange(8192)[None, :] < torch.tensor(lengths)[:, None]
        packed, params = preprocess_ragged_seqs(rows)
        reference, reference_params = preprocess_packed_seqs(dense, mask)
        torch.testing.assert_close(packed, reference)
        torch.testing.assert_close(params.cu_seqlens_q, reference_params.cu_seqlens_q)
        labels = dense.roll(-1, dims=1)
        label_mask = mask.roll(-1, dims=1)
        label_mask[:, -1] = False
        reference_labels, _ = preprocess_packed_seqs(labels, mask)
        reference_label_mask, _ = preprocess_packed_seqs(label_mask, mask)
        new_labels, _ = preprocess_ragged_seqs(
            [torch.cat((row[1:], row.new_zeros(1))) for row in rows]
        )
        new_label_mask, _ = preprocess_ragged_seqs(
            [torch.arange(n) < n - 1 for n in lengths]
        )
        torch.testing.assert_close(new_label_mask, reference_label_mask)
        torch.testing.assert_close(
            new_labels[new_label_mask], reference_labels[reference_label_mask]
        )
        current = packed.clone().requires_grad_()
        restored = postprocess_ragged_seqs(current, params, lengths)
        output = DynamicRolloutResult.materialize_actor_batch(
            {"log_probs": restored}, 8192, 0, current.device
        )["log_probs"]
        torch.testing.assert_close(output, dense)
        output.square().sum().backward()
        previous = reference.clone().requires_grad_()
        padded = postprocess_packed_seqs(previous, reference_params, mask, 3, 8192)
        padded.square().sum().backward()
        torch.testing.assert_close(current.grad, previous.grad)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.parametrize(("tp_size", "cp_size"), [(1, 1), (4, 1), (1, 2)])
def test_ragged_packing_matches_dense_and_gradients(tmp_path, tp_size, cp_size):
    pytest.importorskip("megatron.core")
    mp.spawn(
        _packing_process,
        args=(tp_size, cp_size, str(tmp_path / "rendezvous")),
        nprocs=tp_size * cp_size,
        join=True,
    )


@pytest.mark.parametrize(
    ("count", "micro_batch_size"),
    [(1, 4), (4, 2), (5, 2), (5, 1), (10, 4), (14, 4)],
)
def test_ragged_fixed_inference_microbatches(count, micro_batch_size):
    pytest.importorskip("megatron.core")
    from rlinf.utils.data_iter_utils import get_iterator_k_split
    from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor

    batch = {
        "input_ids": [torch.arange(101 + i * 100) for i in range(count)],
        "prompt_lengths": torch.arange(count),
    }
    actor = object.__new__(MAMegatronActor)
    actor.logprob_forward_micro_batch_size = micro_batch_size
    num_microbatches = actor._get_num_microbatches(batch, True)
    chunks = list(
        get_iterator_k_split(batch, num_microbatches, enforce_divisible_batch=False)
    )
    assert num_microbatches == (count + micro_batch_size - 1) // micro_batch_size
    assert all(0 < len(chunk["input_ids"]) <= micro_batch_size for chunk in chunks)
    torch.testing.assert_close(
        torch.cat([chunk["prompt_lengths"] for chunk in chunks]),
        batch["prompt_lengths"],
    )
    for actual, expected in zip(
        [row for chunk in chunks for row in chunk["input_ids"]],
        batch["input_ids"],
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("layout", ["prepacked", "packed", "padded"])
@pytest.mark.parametrize("last_stage", [False, True])
def test_pipeline_temperature_only_scales_logits(monkeypatch, layout, last_stage):
    parallel_state = pytest.importorskip("megatron.core.parallel_state")
    from megatron.core.packed_seq_params import PackedSeqParams

    from rlinf.hybrid_engines.megatron.megatron_model_manager import (
        MegatronModelManager,
    )

    monkeypatch.setattr(
        parallel_state, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_context_parallel_rank", lambda: 0)

    class Stage(torch.nn.Module):
        pre_process = not last_stage
        post_process = last_stage

        def forward(self, input_ids, **kwargs):
            return input_ids.float().unsqueeze(-1)

    ids = torch.tensor([[1, 2]])
    output = MegatronModelManager.custom_forward(
        Stage(),
        input_ids=ids,
        attention_mask=torch.ones_like(ids, dtype=torch.bool),
        position_ids=None,
        sequence_parallel=False,
        pack_seqs=layout != "padded",
        packed_seq_params=PackedSeqParams() if layout == "prepacked" else None,
        temperature=0.5,
    )
    expected = ids.float().unsqueeze(-1)
    torch.testing.assert_close(output, expected / 0.5 if last_stage else expected)


def _pipeline_probabilities_process(rank, pp_size, dp_size, init_file):
    from megatron.core import parallel_state

    from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=pp_size * dp_size,
        timeout=timedelta(seconds=90),
    )
    parallel_state.initialize_model_parallel(pipeline_model_parallel_size=pp_size)
    try:
        actor = object.__new__(MAMegatronActor)
        actor.dbs_indices = [[2, 0], [1]]
        dp_rank = parallel_state.get_data_parallel_rank()
        for dtype in (torch.float32, torch.float64):
            rows = [
                torch.linspace(-0.5, -0.1, n, dtype=dtype) - dp_rank
                for n in (101, 201, 301)
            ]
            for dynamic in (False, True):
                actor.enable_dynamic_batch_size = dynamic
                outputs = (
                    [[rows[2], rows[0]], [rows[1]]] if dynamic else [rows[:2], rows[2:]]
                )
                if not parallel_state.is_pipeline_last_stage():
                    outputs = []
                restored = actor._process_fwd_bwd_outputs(outputs, True)
                assert len(restored) == len(rows)
                for actual, expected in zip(restored, rows, strict=True):
                    assert actual.device.type == "cpu"
                    torch.testing.assert_close(actual, expected)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.parametrize(("pp_size", "dp_size"), [(1, 1), (2, 1), (2, 2)])
def test_ragged_probabilities_reach_all_pipeline_stages(tmp_path, pp_size, dp_size):
    pytest.importorskip("megatron.core")
    mp.spawn(
        _pipeline_probabilities_process,
        args=(pp_size, dp_size, str(tmp_path / "rendezvous")),
        nprocs=pp_size * dp_size,
        join=True,
    )


def _training_rows_process(rank, counts, micro_batch_size, target, init_file):
    from megatron.core import parallel_state

    from rlinf.utils.data_iter_utils import get_iterator_k_split
    from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    parallel_state.initialize_model_parallel()
    try:
        actor = object.__new__(MAMegatronActor)
        actor.torch_platform = torch.cpu
        actor.num_train_steps = 4
        actor.cfg = OmegaConf.create({"actor": {"micro_batch_size": micro_batch_size}})
        actor.role_cfg = actor.cfg.actor
        count = counts[rank]
        rows = [torch.arange(i + 1) + rank * 1000 for i in range(count)]
        batch = {
            "input_ids": rows,
            "attention_mask": [torch.ones_like(row, dtype=torch.bool) for row in rows],
            "response_mask": [torch.ones_like(row, dtype=torch.bool) for row in rows],
            "recomputed_logprobs": [row.float() for row in rows],
            "advantages": [torch.ones_like(row, dtype=torch.float32) for row in rows],
            "rewards": torch.arange(count).float(),
            "idx_to_traj": list(range(count)),
        }
        original = copy.deepcopy(batch)
        padded = actor._pad_ragged_rows(batch)
        assert all(len(column) == count for column in batch.values())
        assert all(len(column) == target for column in padded.values())
        for key, value in original.items():
            if isinstance(value, list):
                for actual, expected in zip(padded[key][:count], value, strict=True):
                    torch.testing.assert_close(actual, expected)
            else:
                torch.testing.assert_close(padded[key][:count], value)
        for row in padded["attention_mask"][count:]:
            assert row.tolist() == [True]
        for key in ("response_mask", "advantages", "recomputed_logprobs"):
            assert all(not row.any() for row in padded[key][count:])
        chunks = list(get_iterator_k_split(padded, actor.num_train_steps))
        assert len(chunks) == actor.num_train_steps
        for chunk in chunks:
            assert all(len(column) == target // 4 for column in chunk.values())
            assert len(chunk["input_ids"]) % micro_batch_size == 0
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("counts", "micro_batch_size", "target"),
    [((5, 10), 1, 12), ((10, 14), 1, 16), ((10, 12), 2, 16)],
)
def test_ragged_training_rows_match_schedule(
    tmp_path, counts, micro_batch_size, target
):
    pytest.importorskip("megatron.core")
    mp.spawn(
        _training_rows_process,
        args=(counts, micro_batch_size, target, str(tmp_path / "rendezvous")),
        nprocs=2,
        join=True,
    )


class _DecoderStage(torch.nn.Module):
    pre_process = post_process = True

    def __init__(self):
        super().__init__()
        self.output_layer = torch.nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.output_layer.weight.copy_(torch.diag(torch.tensor([1.0, 2.0, 3.0])))

    def forward(self, input_ids, attention_mask, position_ids, packed_seq_params):
        hidden = torch.nn.functional.one_hot(input_ids.T, 3).to(
            self.output_layer.weight
        )
        return self.output_layer(hidden).transpose(0, 1).contiguous()


@pytest.fixture
def scoring_actor(tmp_path, monkeypatch):
    from types import SimpleNamespace

    pytest.importorskip("megatron.training")
    import torch.distributed as dist
    from megatron.core import parallel_state
    from omegaconf import OmegaConf

    from rlinf.utils.utils import masked_mean
    from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor

    actor = MAMegatronActor.__new__(MAMegatronActor)
    actor.cfg = OmegaConf.create(
        {
            "runner": {"task_type": "reasoning"},
            "actor": {"model": {"encoder_seq_length": 4}},
            "algorithm": {
                "sampling_params": {"temperature": 0.5},
                "loss_type": "actor",
                "use_valid_token_scale": False,
                "early_stop_imp_ratio": None,
            },
        }
    )
    actor.transformer_config = SimpleNamespace(sequence_parallel=False)
    actor.trains_mtp = False
    actor.tokenizer = SimpleNamespace(eos_token_id=0)
    actor.clip_ratio_c, actor.clip_ratio_low, actor.clip_ratio_high = 3.0, 0.2, 0.2
    actor.kl_beta = 0.0
    actor.loss_agg_func = masked_mean
    actor.torch_platform = SimpleNamespace(current_device=lambda: "cpu")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    ids = torch.tensor([[0, 1, 2, 1]])
    batch = {
        "input_ids": ids,
        "response_mask": torch.tensor([[False, True, False, True]]),
        "advantages": torch.ones_like(ids, dtype=torch.float),
        "loss_scales": torch.ones_like(ids, dtype=torch.float),
        "rollout_logprobs": torch.zeros_like(ids, dtype=torch.float),
    }
    batch = {key: list(value.unbind()) for key, value in batch.items()}
    dist.init_process_group(
        "gloo", init_method=(tmp_path / "pg").as_uri(), rank=0, world_size=1
    )
    try:
        for name in (
            "_TENSOR_MODEL_PARALLEL_GROUP",
            "_DATA_PARALLEL_GROUP",
            "_CONTEXT_PARALLEL_GROUP",
        ):
            monkeypatch.setattr(parallel_state, name, dist.group.WORLD)
        yield actor, batch
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("entropy_bonus", [0.0, 0.1])
def test_actor_entropy_uses_response_predictors(scoring_actor, entropy_bonus):
    actor, batch = scoring_actor
    actor.cfg.algorithm.entropy_bonus = entropy_bonus
    actor.calculate_entropy = actor.return_loss = True
    actor.calculate_entropy_loss = entropy_bonus > 0
    model = _DecoderStage()
    # Responses at 1 and 3 are predicted from tokens 0 and 2.
    logits = (
        model.output_layer.weight.T[[0, 2]]
        / actor.cfg.algorithm.sampling_params.temperature
    )
    logp = logits.log_softmax(-1)
    selected = logp[:, 1]
    old = selected.detach()
    batch["rollout_logprobs"][0][[1, 3]] = old
    scores, loss_fn = actor.get_forward_step_func()(iter([batch]), model)
    loss, metrics = loss_fn(scores)
    expected_entropy = -(logp.exp() * logp).sum(-1).mean()
    expected_loss = (
        -(selected - old).exp().mean()
        - actor.cfg.algorithm.entropy_bonus * expected_entropy
    )
    torch.testing.assert_close(
        metrics["actor/entropy_loss"].squeeze(), expected_entropy
    )
    torch.testing.assert_close(loss, expected_loss)
    weight = model.output_layer.weight
    torch.testing.assert_close(
        torch.autograd.grad(loss, weight)[0],
        torch.autograd.grad(expected_loss, weight)[0],
    )


def _check_tp_entropy_gradient(rank, init_file):
    from megatron.core import parallel_state

    from rlinf.utils.distributed import vocab_parallel_entropy_and_log_probs

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=2)
        torch.manual_seed(101)
        logits = (torch.randn(2, 5, 8) + 37).requires_grad_()
        labels = torch.arange(10).reshape(2, 5) % 8
        reference = logits.log_softmax(-1)
        expected_lp = reference.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        expected_h = -(reference.exp() * reference).sum(-1)
        shard = logits.detach().chunk(2, dim=-1)[rank].contiguous().requires_grad_()
        entropy, log_probs = vocab_parallel_entropy_and_log_probs(
            shard, labels, calculate_entropy_loss=True
        )
        torch.testing.assert_close(entropy, expected_h)
        torch.testing.assert_close(log_probs, expected_lp)
        (grad,) = torch.autograd.grad((log_probs + 0.1 * entropy).sum(), shard)
        (ref_grad,) = torch.autograd.grad(
            (expected_lp + 0.1 * expected_h).sum(), logits
        )
        torch.testing.assert_close(grad, ref_grad.chunk(2, dim=-1)[rank])
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def test_tp_entropy_and_log_probs_match_dense_gradient(tmp_path):
    pytest.importorskip("megatron.core")
    mp.spawn(_check_tp_entropy_gradient, args=(str(tmp_path / "tp"),), nprocs=2)


def _parallel_groups(monkeypatch, tp, cp, rank):
    pytest.importorskip("megatron.core")
    from megatron.core import parallel_state

    monkeypatch.setattr(
        parallel_state, "get_tensor_model_parallel_world_size", lambda: tp
    )
    monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: cp)
    monkeypatch.setattr(parallel_state, "get_context_parallel_rank", lambda: rank)
    return SimpleNamespace(
        tp=SimpleNamespace(size=lambda: tp),
        cp=SimpleNamespace(size=lambda: cp, rank=lambda: rank),
    )


def _batch():
    ids = torch.arange(1, 49).reshape(3, 16)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[0] = True
    mask[1, 3:12] = True
    mask[2, :5] = True
    return ids, mask


def _expected_shard(ids, mask, tp, cp, rank):
    parts = []
    alignment = tp * cp * 2 if cp > 1 else tp
    for row, valid in zip(ids, mask):
        values = row[valid].tolist()
        values += [0] * (-len(values) % alignment)
        if cp > 1:
            size = len(values) // (2 * cp)
            values = (
                values[rank * size : (rank + 1) * size]
                + values[(2 * cp - rank - 1) * size : (2 * cp - rank) * size]
            )
        parts.extend(values)
    return torch.tensor([parts], dtype=ids.dtype)


@pytest.mark.parametrize("tp,cp", [(1, 1), (2, 1), (1, 2), (2, 2), (2, 4)])
@pytest.mark.parametrize("padding", [None, 16, 32])
def test_model_owned_packing_matches_local_labels(monkeypatch, tp, cp, padding):
    pytest.importorskip("megatron.bridge")
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import (
        preprocess_packed_seqs as bridge_pack,
    )

    from rlinf.hybrid_engines.megatron.utils import (
        prepare_full_model_inputs,
        preprocess_packed_seqs,
    )

    ids, mask = _batch()
    for rank in range(cp):
        groups = _parallel_groups(monkeypatch, tp, cp, rank)
        tokens, model_mask, params = prepare_full_model_inputs(ids, mask, padding)
        for i, length in enumerate(mask.sum(-1)):
            assert torch.equal(tokens[i, :length], ids[i, mask[i]])
            assert not tokens[i, length:].any()
        actual, internal_params = bridge_pack(tokens, model_mask, pg_collection=groups)
        expected = _expected_shard(ids, mask, tp, cp, rank)
        if padding is not None:
            expected = torch.nn.functional.pad(
                expected, (0, padding * 3 // cp - expected.shape[1])
            )
        assert torch.equal(actual, expected)
        assert torch.equal(params.cu_seqlens_q, internal_params.cu_seqlens_q)
        assert actual.shape[1] * cp == params.cu_seqlens_q[-1]
        labels, _ = preprocess_packed_seqs(ids, mask, padding_seqlen=padding)
        assert torch.equal(actual, labels)


@pytest.mark.parametrize("internal", [False, True])
@pytest.mark.parametrize("padding", [None, 16, 32])
@pytest.mark.parametrize("pre_process", [False, True])
def test_custom_forward_preserves_tokens_labels_and_gradients(
    monkeypatch, internal, padding, pre_process
):
    pytest.importorskip("megatron.bridge")
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import (
        preprocess_packed_seqs as bridge_pack,
    )

    from rlinf.hybrid_engines.megatron.megatron_model_manager import (
        MegatronModelManager,
    )

    groups = _parallel_groups(monkeypatch, 2, 1, 0)
    ids, mask = _batch()
    # A CPU backend implements the vendor input/output contract; RLinf's
    # preparation, label packing and output restoration all run unchanged.

    class Backend(Qwen3VLModel if internal else torch.nn.Module):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.pre_process = pre_process if internal else True
            self.post_process = True
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def forward(
            self, input_ids, attention_mask, position_ids, packed_seq_params, **kwargs
        ):
            if internal:
                for i, length in enumerate(mask.sum(-1)):
                    assert torch.equal(input_ids[i, :length], ids[i, mask[i]])
                assert kwargs["image_grid_thw"] is grid
                input_ids, _ = bridge_pack(
                    input_ids, attention_mask, pg_collection=groups
                )
            else:
                assert attention_mask is None
            return input_ids.float().unsqueeze(-1) * self.weight

    grid = torch.tensor([[1, 2, 2]])
    model = Backend()
    manager = MegatronModelManager.__new__(MegatronModelManager)

    def process(logits, labels):
        assert torch.equal(logits[..., 0].detach(), labels.float() * 2)
        return {"scores": logits[..., 0]}

    result = manager.custom_forward(
        model,
        ids,
        mask,
        torch.zeros_like(ids),
        sequence_parallel=False,
        logits_processor=process,
        logits_processor_args={"labels": ids},
        padding_seqlen=padding,
        image_grid_thw=grid,
    )["scores"]
    assert torch.equal(result[mask], ids[mask].float() * 2)
    assert torch.equal(result[~mask], torch.zeros_like(result[~mask]))
    result.sum().backward()
    assert model.weight.grad.item() == ids[mask].sum().item()


@pytest.mark.parametrize("cp", [1, 2, 4])
@pytest.mark.parametrize("padding", [None, 16, 32])
@pytest.mark.parametrize("prepacked", [False, True])
def test_bridge_multimodal_forward_splits_once_on_cpu(
    monkeypatch, cp, padding, prepacked
):
    pytest.importorskip("megatron.bridge")
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel

    from rlinf.hybrid_engines.megatron.utils import prepare_full_model_inputs

    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda *args: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)
    ids, mask = _batch()

    class LanguageBackend(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
            self.rotary_pos_emb = SimpleNamespace(is_thd_format=False)

        def embedding(self, input_ids, position_ids):
            return input_ids.float().T.unsqueeze(-1) * self.weight

        def forward(self, decoder_input, **kwargs):
            self.positions = kwargs["position_ids"]
            return decoder_input.transpose(0, 1)

    for rank in range(cp):
        groups = _parallel_groups(monkeypatch, 2, cp, rank)
        model = Qwen3VLModel.__new__(Qwen3VLModel)
        torch.nn.Module.__init__(model)
        model.pg_collection = groups
        model.config = SimpleNamespace(sequence_parallel=False, spatial_merge_size=2)
        model.pre_process = True
        model.use_dist_train = False
        model.image_token_id = 1000
        model.video_token_id = 1001
        model.vision_start_token_id = 1002
        model.square_merge_size = 4
        model.language_model = LanguageBackend()
        tokens, model_mask, params = prepare_full_model_inputs(ids, mask, padding)
        if prepacked:
            from rlinf.hybrid_engines.megatron.megatron_model_manager import (
                MegatronModelManager,
            )

            model.post_process = True
            output = MegatronModelManager.custom_forward(
                model,
                tokens,
                model_mask,
                torch.zeros_like(tokens),
                sequence_parallel=False,
                packed_seq_params=params,
            )
        else:
            output = model(
                input_ids=tokens, attention_mask=model_mask, packed_seq_params=params
            )
        expected = _expected_shard(ids, mask, 2, cp, rank)
        if padding is not None:
            expected = torch.nn.functional.pad(
                expected, (0, padding * 3 // cp - expected.shape[1])
            )
        positions = torch.arange(tokens.shape[1]).expand_as(tokens)
        expected_positions = _expected_shard(positions, model_mask, 2, cp, rank)
        assert torch.equal(model.language_model.positions[0], expected_positions)
        assert torch.equal(output[..., 0], expected.float() * 2)
        output.sum().backward()
        assert model.language_model.weight.grad.item() == expected.sum().item()


@pytest.mark.parametrize("cp,padding", [(1, None), (1, 32), (2, None), (2, 32)])
@pytest.mark.parametrize("post_process", [False, True])
@pytest.mark.parametrize("send_labels", [False, True])
def test_custom_forward_mtp_labels(monkeypatch, cp, padding, post_process, send_labels):
    pytest.importorskip("megatron.bridge")
    from rlinf.hybrid_engines.megatron.megatron_model_manager import (
        MegatronModelManager,
    )
    from rlinf.hybrid_engines.megatron.utils import preprocess_packed_seqs

    ids, mask = _batch()
    response = mask.clone()
    response[:, :8] = False
    for rank in range(cp):
        _parallel_groups(monkeypatch, 1, cp, rank)
        expected, _ = preprocess_packed_seqs(ids, mask, padding_seqlen=padding)
        expected_mask, _ = preprocess_packed_seqs(
            response, mask, padding_seqlen=padding
        )

        class Backend(torch.nn.Module):
            pre_process = True

            def __init__(self):
                super().__init__()
                self.post_process = post_process
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, input_ids, **kwargs):
                if post_process and send_labels:
                    assert torch.equal(kwargs["labels"], expected)
                    assert torch.equal(kwargs["loss_mask"], expected_mask.float())
                    assert kwargs["loss_mask"].dtype == torch.float32
                else:
                    assert "labels" not in kwargs
                    assert "loss_mask" not in kwargs
                return input_ids.float().unsqueeze(-1) * self.weight

        # Output restoration gathers CP ranks; stop at the vendor forward contract.
        class Checked(Exception):
            pass

        class CheckingBackend(Backend):
            def forward(self, *args, **kwargs):
                super().forward(*args, **kwargs)
                raise Checked

        with pytest.raises(Checked):
            MegatronModelManager.custom_forward(
                CheckingBackend(),
                ids,
                mask,
                torch.zeros_like(ids),
                False,
                padding_seqlen=padding,
                mtp_labels=ids if send_labels else None,
                mtp_loss_mask=response if send_labels else None,
            )


@pytest.mark.parametrize("send_labels", [False, True])
def test_unpacked_forward_mtp_labels(monkeypatch, send_labels):
    pytest.importorskip("megatron.bridge")
    from rlinf.hybrid_engines.megatron.megatron_model_manager import (
        MegatronModelManager,
    )
    from rlinf.hybrid_engines.megatron.utils import tensor_rm_left_padding

    _parallel_groups(monkeypatch, 1, 1, 0)
    ids, mask = _batch()

    class Backend(torch.nn.Module):
        pre_process = post_process = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, input_ids, **kwargs):
            if send_labels:
                assert torch.equal(kwargs["labels"], tensor_rm_left_padding(ids, mask))
                assert torch.equal(
                    kwargs["loss_mask"], tensor_rm_left_padding(mask, mask).float()
                )
            else:
                assert "labels" not in kwargs
            return input_ids.float().unsqueeze(-1) * self.weight

    output = MegatronModelManager.custom_forward(
        Backend(),
        ids,
        mask,
        torch.zeros_like(ids),
        False,
        pack_seqs=False,
        mtp_labels=ids if send_labels else None,
        mtp_loss_mask=mask if send_labels else None,
    )
    assert torch.equal(output[..., 0][mask], ids[mask].float())


@pytest.mark.parametrize("internal", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_model_input_ownership_follows_unwrapped_class(internal, wrapped):
    pytest.importorskip("megatron.bridge")
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
    from megatron.core.transformer.module import Float16Module

    from rlinf.hybrid_engines.megatron.utils import model_packs_inputs

    if internal:
        model = Qwen3VLModel.__new__(Qwen3VLModel)
        torch.nn.Module.__init__(model)
    else:
        model = torch.nn.Linear(2, 2)
    if wrapped:
        model = Float16Module(
            SimpleNamespace(
                fp16=False, bf16=True, virtual_pipeline_model_parallel_size=None
            ),
            model,
        )
    assert model_packs_inputs(model) is internal


@pytest.mark.parametrize("wrapped_vlm", [False, True])
@pytest.mark.parametrize("loss_scale", [0.0, 0.1, 0.3])
@pytest.mark.parametrize("send_labels", [False, True])
def test_mtp_postprocess_returns_logits_and_auxiliary_gradients(
    monkeypatch, loss_scale, send_labels, wrapped_vlm
):
    pytest.importorskip("megatron.bridge")
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
    from megatron.core import parallel_state
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.multi_token_prediction import (
        MTPLossAutoScaler,
        MTPLossLoggingHelper,
    )

    from rlinf.hybrid_engines.megatron.megatron_model_manager import (
        _install_mtp_logits_processor,
    )

    monkeypatch.setattr(
        parallel_state, "get_data_parallel_group", lambda **kwargs: None
    )
    monkeypatch.setattr(MTPLossLoggingHelper, "tracker", {"values": torch.zeros(1)})
    monkeypatch.setattr(
        MTPLossAutoScaler, "main_loss_backward_scale", torch.tensor(1.0)
    )

    class Head(torch.nn.Linear):
        def forward(self, states, **kwargs):
            return super().forward(states), None

    class LanguageBackend(GPTModel):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.mtp_process = self.post_process = True
            self.share_embeddings_and_output_weights = False
            self.config = SimpleNamespace(
                mtp_num_layers=1,
                mtp_loss_scaling_factor=loss_scale,
                calculate_per_token_loss=False,
                use_mup=False,
            )
            self.pg_collection = SimpleNamespace(cp=None)
            self.output_layer = Head(2, 3, bias=False)

        def compute_language_model_loss(self, labels, logits):
            return torch.nn.functional.cross_entropy(
                logits.transpose(0, 1).transpose(1, 2), labels, reduction="none"
            )

    model = Qwen3VLModel.__new__(Qwen3VLModel)
    torch.nn.Module.__init__(model)
    model.language_model = LanguageBackend()
    _install_mtp_logits_processor([model if wrapped_vlm else model.language_model])
    language = model.language_model
    torch.nn.init.constant_(language.output_layer.weight, 0.2)
    with torch.no_grad():
        language.output_layer.weight[0] = torch.tensor([0.1, 0.5])
    backbone = torch.tensor(
        [[[0.2, 0.3]], [[0.4, 0.1]], [[0.5, 0.8]], [[0.2, 0.6]]], requires_grad=True
    )
    mtp = torch.nn.Linear(2, 2, bias=False)
    torch.nn.init.eye_(mtp.weight)
    hidden = torch.cat([backbone, mtp(backbone)], dim=0)
    labels = torch.tensor([[0, 1, 2, 1]])
    loss_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    output = language._postprocess(
        hidden,
        None,
        None,
        labels if send_labels else None,
        None,
        None,
        None,
        loss_mask=loss_mask,
    )
    expected = language.output_layer(backbone)[0].transpose(0, 1).contiguous()
    assert torch.equal(output, expected)
    # Isolate the native MTP auxiliary gradient from the main policy loss.
    (output.sum() * 0).backward()
    has_gradient = send_labels and loss_scale > 0
    assert bool(mtp.weight.grad.abs().sum()) is has_gradient
    assert bool(backbone.grad.abs().sum()) is has_gradient
    if send_labels:
        shifted = torch.tensor([[1, 2, 1, 0]])
        shifted_mask = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
        expected_loss = (
            language.compute_language_model_loss(
                shifted, language.output_layer(mtp(backbone))[0]
            )
            * shifted_mask
        ).sum() / 2
        assert torch.allclose(MTPLossLoggingHelper.tracker["values"][0], expected_loss)
        expected_mtp_grad, expected_backbone_grad = torch.autograd.grad(
            expected_loss * loss_scale, (mtp.weight, backbone)
        )
        assert torch.allclose(mtp.weight.grad, expected_mtp_grad)
        assert torch.allclose(backbone.grad, expected_backbone_grad)


@pytest.mark.parametrize("internal", [False, True])
@pytest.mark.parametrize("cp", [1, 2])
@pytest.mark.parametrize("multi_turn", [False, True])
@pytest.mark.parametrize("training,mtp_layers", [(False, 1), (True, 0), (True, 1)])
def test_actor_passes_response_only_mtp_targets(
    monkeypatch, internal, cp, multi_turn, training, mtp_layers
):
    pytest.importorskip("megatron.bridge")
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
    from omegaconf import OmegaConf

    from rlinf.hybrid_engines.megatron.utils import preprocess_packed_seqs
    from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor
    from rlinf.workers.actor.megatron_actor_worker import MegatronActor

    _parallel_groups(monkeypatch, 1, cp, cp - 1)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor)
    ids, mask = _batch()
    response = torch.zeros_like(mask)
    response[0, 10:16] = True
    response[1, 10:12] = True
    actor = (MAMegatronActor if multi_turn else MegatronActor).__new__(
        MAMegatronActor if multi_turn else MegatronActor
    )
    actor.torch_platform = SimpleNamespace(current_device=lambda: torch.device("cpu"))
    actor.return_loss = training
    actor.mbridge = True
    actor.provider = SimpleNamespace(mtp_num_layers=mtp_layers)
    actor.trains_mtp = bool(mtp_layers)
    actor.variable_seq_lengths = True
    actor.response_len = 6
    actor.transformer_config = SimpleNamespace(sequence_parallel=False)
    actor.cfg = OmegaConf.create(
        {"algorithm": {"sampling_params": {"temperature": 1.0}}}
    )
    expected_mask = torch.zeros_like(mask)
    expected_mask[:, :-1] = response[:, 1:]
    expected_mask, _ = preprocess_packed_seqs(expected_mask, mask)
    expected_labels = torch.zeros_like(ids)
    expected_labels[:, :-1] = ids[:, 1:]
    expected_labels, _ = preprocess_packed_seqs(expected_labels, mask)

    class Checked(Exception):
        pass

    class Backend(Qwen3VLModel if internal else torch.nn.Module):
        pre_process = post_process = True

        def __init__(self):
            torch.nn.Module.__init__(self)
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, **kwargs):
            if internal:
                for i, (row, valid) in enumerate(zip(ids, mask)):
                    assert torch.equal(
                        kwargs["input_ids"][i, : valid.sum()], row[valid]
                    )
            else:
                expected_ids, _ = preprocess_packed_seqs(ids, mask)
                assert torch.equal(kwargs["input_ids"], expected_ids)
            if training and mtp_layers:
                assert torch.equal(kwargs["loss_mask"], expected_mask.float())
                valid = expected_mask.bool()
                assert torch.equal(kwargs["labels"][valid], expected_labels[valid])
            else:
                assert "labels" not in kwargs
                assert "loss_mask" not in kwargs
            raise Checked

    batch = {
        "input_ids": ids,
        "attention_mask": mask,
        "response_mask": response,
        "position_ids": torch.zeros_like(ids),
    }
    if multi_turn:
        from rlinf.data.schema.actor_batch import DynamicActorBatch

        batch = DynamicActorBatch(
            input_ids=[row[valid] for row, valid in zip(ids, mask)],
            response_mask=[row[valid] for row, valid in zip(response, mask)],
        )
    with pytest.raises(Checked):
        actor.get_forward_step_func()(iter([batch]), Backend())


def _mtp_metric_pipeline_rank(rank, world_size, init_file):
    from megatron.core import parallel_state
    from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

    from rlinf.hybrid_engines.megatron.utils import collect_mtp_loss

    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    # Only the last pipeline stage owns MTP losses; every stage reports the metric.
    parallel_state._PIPELINE_MODEL_PARALLEL_GROUP = torch.distributed.group.WORLD
    try:
        for scale in (1, 3):
            MTPLossLoggingHelper.tracker = (
                {"values": torch.tensor([4.0, 8.0]) * scale}
                if rank == world_size - 1
                else {}
            )
            loss = collect_mtp_loss(2, 2, torch.device("cpu"))
            assert loss.item() == 3 * scale
            if rank == world_size - 1:
                assert not MTPLossLoggingHelper.tracker["values"].any()
            assert collect_mtp_loss(2, 2, torch.device("cpu")).item() == 0
    finally:
        parallel_state._PIPELINE_MODEL_PARALLEL_GROUP = None
        MTPLossLoggingHelper.tracker = {}
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("pp", [1, 2])
def test_mtp_loss_metric_reaches_all_stages_and_clears_tracker(tmp_path, pp):
    pytest.importorskip("megatron.core")
    import torch.multiprocessing as mp

    mp.spawn(
        _mtp_metric_pipeline_rank,
        args=(pp, str(tmp_path / "gloo")),
        nprocs=pp,
        join=True,
    )


@pytest.mark.parametrize("encoder_only", [False, True])
def test_mtp_processor_preserves_chunks_without_mtp(encoder_only):
    pytest.importorskip("megatron.bridge")
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
    from megatron.core.models.gpt.gpt_model import GPTModel

    from rlinf.hybrid_engines.megatron.megatron_model_manager import (
        _install_mtp_logits_processor,
    )

    model = Qwen3VLModel.__new__(Qwen3VLModel)
    torch.nn.Module.__init__(model)
    if encoder_only:
        model.language_model = None
    else:
        model.language_model = GPTModel.__new__(GPTModel)
        torch.nn.Module.__init__(model.language_model)
        model.language_model.mtp_process = False
    _install_mtp_logits_processor([model])
    if not encoder_only:
        assert model.language_model._postprocess.__func__ is GPTModel._postprocess


@pytest.fixture(
    params=[
        ("qwen3_5", False, 0.1),
        ("qwen3_5", True, 0.1),
        ("qwen3_5", False, 0.25),
        ("qwen3_5", True, 0.25),
        ("glm4_moe_lite", False, 0.1),
        ("glm4_moe_lite", True, 0.1),
        ("deepseek_v3", False, 0.1),
        ("deepseek_v3", True, 0.1),
    ]
)
def mtp_configured_actor(request, tmp_path, monkeypatch):
    pytest.importorskip("megatron.bridge")
    from megatron.bridge.models.model_provider import ModelProviderMixin
    from omegaconf import OmegaConf
    from transformers import DeepseekV3Config, Glm4MoeLiteConfig, Qwen3_5Config

    from rlinf.workers.actor.megatron_actor_worker import MegatronActor

    model_type, load_mtp, loss_scale = request.param
    if model_type == "qwen3_5":
        config = Qwen3_5Config(architectures=["Qwen3_5ForConditionalGeneration"])
        config.text_config.mtp_num_hidden_layers = 1
        config.text_config.mtp_use_dedicated_embeddings = False
    elif model_type == "glm4_moe_lite":
        config = Glm4MoeLiteConfig(
            architectures=["Glm4MoeLiteForCausalLM"], num_nextn_predict_layers=1
        )
    else:
        config = DeepseekV3Config(
            architectures=["DeepseekV3ForCausalLM"], num_nextn_predict_layers=1
        )
    config.save_pretrained(tmp_path)
    actor = MegatronActor.__new__(MegatronActor)
    actor._cfg = OmegaConf.create(
        {
            "megatron": {"load": str(tmp_path)},
            "model": {
                "model_type": model_type,
                "load_mtp": load_mtp,
                "mtp_loss_scaling_factor": loss_scale,
            },
            "optim": {"use_distributed_optimizer": False},
        }
    )
    actor.transformer_config = SimpleNamespace(
        sequence_parallel=False,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
    )
    actor._rank = 1
    actor.mbridge = True

    class ProviderReady(Exception):
        pass

    def stop_before_allocating_model(provider, **kwargs):
        raise ProviderReady

    monkeypatch.setattr(
        ModelProviderMixin, "provide_distributed_model", stop_before_allocating_model
    )
    with pytest.raises(ProviderReady):
        actor.setup_mbridge_model_and_optimizer()
    expected = model_type == "qwen3_5" and load_mtp
    assert actor.trains_mtp is expected
    assert actor.provider.mtp_num_layers == (
        int(load_mtp) if model_type == "qwen3_5" else 1
    )
    assert actor.provider.pipeline_model_parallel_size == 2
    assert actor.provider.virtual_pipeline_model_parallel_size == 2
    if expected:
        assert actor.provider.mtp_loss_scaling_factor == loss_scale
    actor.expected_mtp_training = expected
    return actor


@pytest.mark.parametrize("multi_turn", [False, True])
def test_actor_mtp_opt_in_is_independent_of_checkpoint_layers(
    mtp_configured_actor, monkeypatch, multi_turn
):
    from omegaconf import OmegaConf

    from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor
    from rlinf.workers.actor.megatron_actor_worker import MegatronActor

    actor = mtp_configured_actor
    _parallel_groups(monkeypatch, 1, 1, 0)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor)
    actor.torch_platform = SimpleNamespace(current_device=lambda: torch.device("cpu"))
    actor.return_loss = True
    actor.variable_seq_lengths = True
    actor.response_len = 6
    actor.cfg = OmegaConf.create(
        {"algorithm": {"sampling_params": {"temperature": 1.0}}}
    )

    class Checked(Exception):
        pass

    class Backend(torch.nn.Module):
        pre_process = post_process = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, **kwargs):
            assert ("labels" in kwargs) is actor.expected_mtp_training
            assert ("loss_mask" in kwargs) is actor.expected_mtp_training
            raise Checked

    ids, mask = _batch()
    batch = {
        "input_ids": ids,
        "attention_mask": mask,
        "response_mask": mask,
        "position_ids": torch.zeros_like(ids),
    }
    if multi_turn:
        from rlinf.data.schema.actor_batch import DynamicActorBatch

        batch = DynamicActorBatch(
            input_ids=[row[valid] for row, valid in zip(ids, mask)],
            response_mask=[row[valid] for row, valid in zip(mask, mask)],
        )
    actor_cls = MAMegatronActor if multi_turn else MegatronActor
    with pytest.raises(Checked):
        actor_cls.get_forward_step_func(actor)(iter([batch]), Backend())


def test_training_step_mtp_metric_requires_opt_in(
    mtp_configured_actor, monkeypatch, tmp_path
):
    from megatron.core import parallel_state
    from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
    from omegaconf import OmegaConf

    import rlinf.workers.megatron_worker as worker_module
    from rlinf.scheduler import Worker
    from rlinf.utils.logging import get_logger

    actor = mtp_configured_actor
    actor.role = "actor"
    actor.cfg = OmegaConf.create({"algorithm": {"n_minibatches": 1}})
    actor.enable_dynamic_batch_size = False
    actor.use_profiler = False
    actor.total_batch_size_per_dp = 1
    actor._logger = get_logger()
    actor._stacklevel = 1
    actor.lr_scheduler = SimpleNamespace(step=lambda **kwargs: None)

    class Backend(torch.nn.Linear):
        def __init__(self):
            super().__init__(2, 2)
            self.config = SimpleNamespace()

        def zero_grad_buffer(self):
            self.zero_grad()

    class Optimizer(torch.optim.SGD):
        def scale_loss(self, loss):
            return loss

        def step(self):
            super().step()
            return True, 0.0, 0

    actor.model = [Backend()]
    actor.optimizer = Optimizer(actor.model[0].parameters(), lr=0.01)
    monkeypatch.setattr(
        Worker,
        "torch_platform",
        SimpleNamespace(
            synchronize=lambda: None, ipc_collect=lambda: None, empty_cache=lambda: None
        ),
    )
    monkeypatch.setattr(worker_module, "get_num_microbatches", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_data_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        parallel_state, "get_pipeline_model_parallel_world_size", lambda: 1
    )
    # Replace the vendor GPU schedule, retaining RLinf's training-step and metric paths.
    monkeypatch.setattr(
        worker_module,
        "get_forward_backward_func",
        lambda: lambda **kwargs: [{"policy_loss": torch.tensor(0.0)}],
    )
    monkeypatch.setattr(
        MTPLossLoggingHelper, "tracker", {"values": torch.tensor([2.0])}
    )
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'metrics'}", rank=0, world_size=1
    )
    try:
        metrics = actor.training_step({"input_ids": torch.ones(1, 4, dtype=torch.long)})
        assert ("actor/mtp_loss" in metrics) is actor.expected_mtp_training
        if actor.expected_mtp_training:
            assert metrics["actor/mtp_loss"] == 2.0
            assert not MTPLossLoggingHelper.tracker["values"].any()
        else:
            assert MTPLossLoggingHelper.tracker["values"].item() == 2.0
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("packed", [False, True])
def test_bf16_model_mtp_mask_counts_response_tokens_in_fp32(monkeypatch, packed):
    pytest.importorskip("megatron.core")
    from megatron.core.transformer.multi_token_prediction import roll_tensor

    from rlinf.hybrid_engines.megatron.megatron_model_manager import (
        MegatronModelManager,
    )

    _parallel_groups(monkeypatch, 1, 1, 0)
    ids = torch.arange(1003).reshape(1, -1)
    mask = torch.ones_like(ids, dtype=torch.bool)
    response = mask.clone()
    response[:, -1] = False

    class Backend(torch.nn.Module):
        pre_process = post_process = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.bfloat16))

        def forward(self, input_ids, loss_mask, **kwargs):
            assert loss_mask.dtype == torch.float32
            _, count = roll_tensor(
                loss_mask, packed_seq_params=kwargs.get("packed_seq_params")
            )
            assert count.item() == 1001
            return input_ids.float().unsqueeze(-1) * self.weight

    result = MegatronModelManager.custom_forward(
        Backend(),
        ids,
        mask,
        torch.zeros_like(ids),
        False,
        pack_seqs=packed,
        mtp_labels=ids,
        mtp_loss_mask=response,
    )
    assert torch.equal(result[..., 0], ids.float())


def _shared_embeddings_gloo_worker(rank, rendezvous):
    import itertools
    from datetime import timedelta

    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge

    from rlinf.hybrid_engines.megatron.megatron_model_manager import (
        _bridge_shared_embeddings,
    )

    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=90),
    )
    try:
        end_group = torch.distributed.new_group([0, 2], backend="gloo")
        # Bridge's model-specific policy is independent of the broadcast mechanism.
        bridge = SimpleNamespace(
            _share_embeddings_and_output_weights=lambda config: (
                config.share_embeddings_and_output_weights
            )
        )
        for vp, wrapped, mtp, dtype in itertools.product(
            (1, 2, 3), (False, True), (False, True), (torch.float32, torch.bfloat16)
        ):
            group = torch.distributed.group.WORLD if mtp else end_group
            config = SimpleNamespace(
                pipeline_model_parallel_size=3,
                share_embeddings_and_output_weights=True,
            )
            chunks = []
            copies = []
            expected = torch.arange(12, dtype=dtype).reshape(4, 3)
            for index in range(vp):
                chunk = torch.nn.Module()
                chunk.config = config
                chunk.pg_collection = SimpleNamespace(embd=group)
                owns_input = rank == 0 and index == 0
                owns_output = rank == 2 and index == vp - 1
                owns_mtp = mtp and rank in (1, 2) and index == vp - 1
                if owns_input or owns_mtp:
                    chunk.embedding = torch.nn.Module()
                    chunk.embedding.word_embeddings = torch.nn.Embedding(
                        4, 3, dtype=dtype
                    )
                    weight = chunk.embedding.word_embeddings.weight
                    with torch.no_grad():
                        weight.copy_(
                            expected
                            if owns_input
                            else torch.full_like(expected, float("nan"))
                        )
                    copies.append(weight)
                if owns_output:
                    chunk.output_layer = torch.nn.Linear(3, 4, bias=False, dtype=dtype)
                    with torch.no_grad():
                        chunk.output_layer.weight.fill_(float("nan"))
                    copies.append(chunk.output_layer.weight)
                if wrapped:
                    wrapper = torch.nn.Module()
                    wrapper.config = config
                    wrapper.language_model = chunk
                    chunk = wrapper
                chunks.append(chunk)
            pointers = [p.data_ptr() for p in copies]
            with _bridge_shared_embeddings():
                MegatronModelBridge._broadcast_shared_embeddings(bridge, chunks)
            for weight in copies:
                assert torch.equal(
                    weight.detach().view(torch.uint8), expected.view(torch.uint8)
                )
            assert [p.data_ptr() for p in copies] == pointers
            torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


def test_bridge_shared_embeddings_across_virtual_stages(tmp_path):
    """CPU collectives cover later output chunks, MTP copies and nonmember ranks."""
    pytest.importorskip("megatron.bridge")
    torch.multiprocessing.spawn(
        _shared_embeddings_gloo_worker,
        args=(f"file://{tmp_path / 'shared-embeddings'}",),
        nprocs=3,
        join=True,
    )


@pytest.mark.parametrize("tied,pp", [(False, 1), (False, 2), (True, 1)])
def test_bridge_shared_embeddings_no_broadcast_when_unneeded(tied, pp):
    pytest.importorskip("megatron.bridge")
    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge

    from rlinf.hybrid_engines.megatron.megatron_model_manager import (
        _bridge_shared_embeddings,
    )

    chunk = torch.nn.Linear(3, 4, bias=False)
    chunk.config = SimpleNamespace(
        pipeline_model_parallel_size=pp, share_embeddings_and_output_weights=tied
    )
    before = chunk.weight.detach().clone()
    bridge = SimpleNamespace(
        _share_embeddings_and_output_weights=lambda config: (
            config.share_embeddings_and_output_weights
        )
    )
    with _bridge_shared_embeddings():
        MegatronModelBridge._broadcast_shared_embeddings(bridge, [chunk])
    assert torch.equal(chunk.weight, before)


def test_bridge_shared_embeddings_restores_loader_after_error():
    pytest.importorskip("megatron.bridge")
    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge

    from rlinf.hybrid_engines.megatron.megatron_model_manager import (
        _bridge_shared_embeddings,
    )

    original = MegatronModelBridge._broadcast_shared_embeddings
    with pytest.raises(RuntimeError, match="load failed"):
        with _bridge_shared_embeddings():
            raise RuntimeError("load failed")
    assert MegatronModelBridge._broadcast_shared_embeddings is original


@pytest.mark.parametrize("internal", [False, True])
@pytest.mark.parametrize("tp,cp", [(1, 1), (2, 1), (1, 2), (2, 2), (2, 4)])
@pytest.mark.parametrize(
    "pre_process,post_process",
    [(True, True), (True, False), (False, True), (False, False)],
)
@pytest.mark.parametrize("mtp", [False, True])
def test_ragged_forward_preserves_packing_and_mtp_contract(
    monkeypatch, internal, tp, cp, pre_process, post_process, mtp
):
    pytest.importorskip("megatron.bridge")
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import (
        preprocess_packed_seqs as bridge_pack,
    )

    from rlinf.hybrid_engines.megatron.megatron_model_manager import (
        MegatronModelManager,
    )
    from rlinf.hybrid_engines.megatron.utils import (
        prepare_ragged_model_inputs,
        preprocess_ragged_seqs,
    )

    rows = [torch.tensor([11]), torch.arange(21, 28), torch.arange(31, 44)]
    labels = [torch.cat((row[1:], row.new_zeros(1))) for row in rows]
    masks = [
        torch.zeros(1),
        torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
        torch.zeros(13),
    ]
    for rank in range(cp):
        groups = _parallel_groups(monkeypatch, tp, cp, rank)
        local, params = preprocess_ragged_seqs(rows)
        local_labels, _ = preprocess_ragged_seqs(labels)
        local_mask, _ = preprocess_ragged_seqs(masks)

        class Backend(Qwen3VLModel if internal else torch.nn.Module):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.pre_process = pre_process
                self.post_process = post_process
                self.weight = torch.nn.Parameter(torch.tensor(2.0))

            def forward(
                self,
                input_ids,
                attention_mask,
                packed_seq_params,
                position_ids,
                **kwargs,
            ):
                assert packed_seq_params is prepared_params
                if internal:
                    assert position_ids is None
                else:
                    assert position_ids.shape == input_ids.shape
                if internal:
                    for i, row in enumerate(rows):
                        assert torch.equal(input_ids[i, : len(row)], row)
                    input_ids, _ = bridge_pack(
                        input_ids, attention_mask, pg_collection=groups
                    )
                else:
                    assert attention_mask is None
                assert torch.equal(input_ids, local)
                assert torch.equal(
                    packed_seq_params.cu_seqlens_q_padded, params.cu_seqlens_q_padded
                )
                if post_process and mtp:
                    assert torch.equal(kwargs["labels"], local_labels)
                    assert torch.equal(kwargs["loss_mask"], local_mask)
                    assert kwargs["loss_mask"].dtype == torch.float32
                else:
                    assert "labels" not in kwargs and "loss_mask" not in kwargs
                return input_ids.float().unsqueeze(-1) * self.weight

        model = Backend()
        tokens, mask, positions, prepared_params = prepare_ragged_model_inputs(
            rows, model
        )
        output = MegatronModelManager.custom_forward(
            model,
            tokens,
            mask,
            positions,
            sequence_parallel=False,
            packed_seq_params=prepared_params,
            mtp_labels=local_labels if mtp else None,
            mtp_loss_mask=local_mask.to(torch.bfloat16) if mtp else None,
        )
        assert torch.equal(output[..., 0], local.float() * 2)
        output.sum().backward()
        assert model.weight.grad.item() == local.sum().item()


def test_mtp_postprocess_installation_requires_training(
    mtp_configured_actor, monkeypatch
):
    from functools import partial

    import megatron.training.global_vars as global_vars
    import megatron.training.training as training
    from megatron.bridge.models.model_provider import ModelProviderMixin
    from megatron.core.models.gpt.gpt_model import GPTModel

    actor = mtp_configured_actor
    model = GPTModel.__new__(GPTModel)
    torch.nn.Module.__init__(model)
    model.mtp_process = bool(actor.provider.mtp_num_layers)
    original = model._postprocess
    monkeypatch.setattr(
        ModelProviderMixin, "provide_distributed_model", lambda *a, **kw: [model]
    )
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", SimpleNamespace())

    class ModelReady(Exception):
        pass

    def stop_before_optimizer(args):
        raise ModelReady

    monkeypatch.setattr(
        training, "get_megatron_optimizer_config", stop_before_optimizer
    )
    with pytest.raises(ModelReady):
        actor.setup_mbridge_model_and_optimizer()
    if actor.expected_mtp_training:
        assert isinstance(model._postprocess, partial)
        assert model._postprocess.func == original
    else:
        assert model._postprocess == original


@pytest.mark.parametrize("role", ["actor", "critic"])
@pytest.mark.parametrize("internal", [False, True])
@pytest.mark.parametrize("tp,cp", [(1, 1), (2, 1), (1, 2), (2, 4)])
def test_ragged_actor_and_critic_model_input_contract(
    monkeypatch, role, internal, tp, cp
):
    pytest.importorskip("megatron.bridge")
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import (
        preprocess_packed_seqs as bridge_pack,
    )

    from rlinf.data.schema.actor_batch import DynamicActorBatch
    from rlinf.hybrid_engines.megatron.utils import preprocess_ragged_seqs
    from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor
    from rlinf.workers.critic.ma_megatron_critic_worker import MAMegatronCritic

    groups = _parallel_groups(monkeypatch, tp, cp, cp - 1)
    rows = [torch.tensor([11]), torch.arange(21, 28), torch.arange(31, 44)]
    expected, expected_params = preprocess_ragged_seqs(rows)
    cls = MAMegatronActor if role == "actor" else MAMegatronCritic
    worker = cls.__new__(cls)
    worker.torch_platform = SimpleNamespace(current_device=lambda: torch.device("cpu"))
    worker.torch_device_type = "cpu"
    worker.return_loss = False
    worker.trains_mtp = False
    worker.transformer_config = SimpleNamespace(sequence_parallel=False)
    worker.cfg = OmegaConf.create(
        {"algorithm": {"sampling_params": {"temperature": 1.0}}}
    )

    class Backend(Qwen3VLModel if internal else torch.nn.Module):
        pre_process = True
        post_process = False

        def __init__(self):
            torch.nn.Module.__init__(self)
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def forward(
            self, input_ids, attention_mask, position_ids, packed_seq_params, **kwargs
        ):
            if internal:
                assert attention_mask.shape == input_ids.shape
                assert position_ids is None
                for i, row in enumerate(rows):
                    assert torch.equal(input_ids[i, : len(row)], row)
                tokens, internal_params = bridge_pack(
                    input_ids, attention_mask, pg_collection=groups
                )
                assert torch.equal(
                    internal_params.cu_seqlens_q_padded,
                    packed_seq_params.cu_seqlens_q_padded,
                )
            else:
                assert attention_mask is None
                assert position_ids.shape == input_ids.shape
                tokens = input_ids
            assert torch.equal(tokens, expected)
            assert torch.equal(
                packed_seq_params.cu_seqlens_q_padded,
                expected_params.cu_seqlens_q_padded,
            )
            return tokens.float().unsqueeze(-1) * self.weight

    model = Backend()
    batch = DynamicActorBatch(
        input_ids=rows,
        response_mask=[torch.ones_like(row, dtype=torch.bool) for row in rows],
    )
    output, collect = worker.get_forward_step_func()(iter([batch]), model)
    assert torch.equal(collect(output)[..., 0], expected.float() * 2)
    output.sum().backward()
    assert model.weight.grad.item() == expected.sum().item()
