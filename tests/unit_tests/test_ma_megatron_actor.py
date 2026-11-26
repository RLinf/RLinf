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

"""Multi-turn Megatron batches, packing, chunked LM head, and weight-version bookkeeping.

Megatron-dependent cases skip when `megatron.core` is absent, while pure-PyTorch
chunked LM head and config-validation cases always run. No Ray or GPUs are launched.
"""

import copy
from datetime import timedelta

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


def test_chunked_actor_scores_ragged_rows_without_crossing_boundaries(monkeypatch):
    from types import SimpleNamespace

    pytest.importorskip("megatron.training")
    from rlinf.hybrid_engines.megatron import utils
    from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor

    for name, value in (
        ("get_tensor_model_parallel_world_size", 2),
        ("get_context_parallel_world_size", 1),
        ("get_context_parallel_rank", 0),
    ):
        monkeypatch.setattr(utils.parallel_state, name, lambda value=value: value)

    def score(model, input_ids, position_ids, packed_seq_params, label, label_mask):
        assert packed_seq_params.cu_seqlens_q.tolist() == [0, 4, 6, 8]
        assert label.tolist() == [[2, 3, 0, 0, 0, 0, 6, 0]]
        assert label_mask.nonzero(as_tuple=True)[1].tolist() == [0, 1, 6]
        return {"log_probs": label.float().masked_fill(~label_mask, 0)}

    actor = MAMegatronActor.__new__(MAMegatronActor)
    actor.torch_platform = SimpleNamespace(current_device=lambda: "cpu")
    actor.cfg = SimpleNamespace(
        algorithm=SimpleNamespace(sampling_params=SimpleNamespace(temperature=1.0))
    )
    actor._score_packed_tokens = score
    actor.return_loss = False
    model = torch.nn.Module()
    model.post_process = True
    batch = {
        "input_ids": [torch.tensor(ids) for ids in ([1, 2, 3], [4], [5, 6])],
    }
    packed, collect = MAMegatronActor.get_forward_step_func(actor)(iter([batch]), model)
    assert [row.tolist() for row in collect(packed)] == [[0, 2, 3], [0], [0, 6]]


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

    def __init__(self, chunked=False):
        super().__init__()
        from megatron.core import ModelParallelConfig
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear

        self.output_layer = ColumnParallelLinear(
            3,
            3,
            config=ModelParallelConfig(use_cpu_initialization=True),
            init_method=torch.nn.init.zeros_,
            bias=False,
        )
        with torch.no_grad():
            self.output_layer.weight.copy_(torch.diag(torch.tensor([1.0, 2.0, 3.0])))
        if chunked:
            from rlinf.hybrid_engines.megatron.output_projection import (
                install_chunked_projection,
            )

            install_chunked_projection(self)

    def forward(self, input_ids, attention_mask, position_ids, packed_seq_params):
        hidden = torch.nn.functional.one_hot(input_ids.T, 3).to(
            self.output_layer.weight
        )
        output, _ = self.output_layer(hidden)
        return output.transpose(0, 1).contiguous()


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
    actor.tokenizer = SimpleNamespace(eos_token_id=0)
    actor.clip_ratio_c, actor.clip_ratio_low, actor.clip_ratio_high = 3.0, 0.2, 0.2
    actor.lm_head_chunk_size = 0
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


@pytest.mark.parametrize(
    ("chunk_size", "entropy_bonus"), [(0, 0.0), (0, 0.1), (1, 0.0)]
)
def test_actor_entropy_uses_response_predictors(
    scoring_actor, chunk_size, entropy_bonus
):
    actor, batch = scoring_actor
    actor.lm_head_chunk_size = chunk_size
    actor.cfg.algorithm.entropy_bonus = entropy_bonus
    actor.calculate_entropy = actor.return_loss = True
    actor.calculate_entropy_loss = entropy_bonus > 0
    model = _DecoderStage(chunked=chunk_size > 0)
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


def test_run_inference_uses_packed_scoring_for_ref_logprobs(scoring_actor, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from megatron.core.tensor_parallel.layers import ColumnParallelLinear

    from rlinf.data.schema.reasoning_results import DynamicRolloutResult
    from rlinf.scheduler import Worker

    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(
        Worker, "torch_platform", SimpleNamespace(synchronize=lambda: None)
    )

    actor, _ = scoring_actor
    actor.lm_head_chunk_size = 1
    actor.calculate_entropy = False
    actor.cfg.actor.model.encoder_seq_length = 6

    model = _DecoderStage(chunked=True)
    actor_weight = model.output_layer.weight.detach().clone()
    ref_weight = torch.diag(torch.tensor([3.0, 1.0, 2.0]))
    actor.model = [model]
    actor.ref_policy_state_dict = {"output_layer.weight": ref_weight.clone()}

    result = DynamicRolloutResult(
        num_sequence=2,
        group_size=1,
        idx_to_traj=[0, 0],
        input_ids=[[0, 1, 2, 1], [0, 1, 2, 1, 0, 2]],
        prompt_lengths=[2, 4],
        response_lengths=[2, 2],
        is_end=[False, True],
        rewards=[1.0, 1.0],
    )
    step_batch = result.to_actor_batch()
    emitted = []

    def run_single_inference(batch):
        actor.return_loss = False
        packed, collect = actor.get_forward_step_func()(iter([batch]), actor.model[0])
        return collect(packed)

    def forbid_dense(*args, **kwargs):
        raise AssertionError("Dense ColumnParallelLinear.forward should not be called")

    monkeypatch.setattr(ColumnParallelLinear, "forward", forbid_dense)
    actor.get_step_batch = lambda _channel: (step_batch, [result])
    actor._load_weight_and_optimizer = lambda: None
    actor.worker_timer = nullcontext
    actor.inference_step = run_single_inference
    actor.put_result = lambda item, out: out.append(item)
    actor.scheduler_offload_sync = lambda: None

    actor.run_inference(None, emitted, compute_ref_logprobs=True)

    assert len(emitted) == 1
    torch.testing.assert_close(model.output_layer.weight, actor_weight)

    temperature = actor.cfg.algorithm.sampling_params.temperature

    def expected_row(weight, token_ids):
        ids = torch.tensor(token_ids)
        logits = weight.T[ids[:-1]] / temperature
        logp = logits.log_softmax(-1).gather(-1, ids[1:, None]).squeeze(-1)
        return torch.cat((logp.new_zeros(1), logp))

    for idx, ids in enumerate(result.input_ids):
        torch.testing.assert_close(
            emitted[0].recomputed_logprobs[idx], expected_row(actor_weight, ids)
        )
        torch.testing.assert_close(
            emitted[0].ref_logprobs[idx], expected_row(ref_weight, ids)
        )

    packed_batch = DynamicRolloutResult.pack_traj_batch({}, emitted[0].to_actor_batch())
    combined_mask = torch.tensor([False, False, True, True, True, True])
    torch.testing.assert_close(packed_batch["response_mask"][0], combined_mask)
    torch.testing.assert_close(
        packed_batch["recomputed_logprobs"][0],
        expected_row(actor_weight, result.input_ids[1]).masked_fill(
            ~combined_mask, 0.0
        ),
    )
    torch.testing.assert_close(
        packed_batch["ref_logprobs"][0],
        expected_row(ref_weight, result.input_ids[1]).masked_fill(~combined_mask, 0.0),
    )


def test_pipeline_output_projection(tmp_path):
    pytest.importorskip("megatron.core")
    torch.multiprocessing.spawn(
        _pipeline_output_projection, args=((tmp_path / "pg").as_uri(),), nprocs=2
    )


def _pipeline_output_projection(rank, rendezvous):
    import torch.distributed as dist
    from megatron.core import ModelParallelConfig, parallel_state
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear

    from rlinf.hybrid_engines.megatron.output_projection import (
        chunked_scoring,
        install_chunked_projection,
    )
    from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor
    from rlinf.workers.actor.megatron_actor_worker import MegatronActor

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        groups = [dist.new_group([i], backend="gloo") for i in range(2)]
        with pytest.MonkeyPatch.context() as mp:
            for name in ("_TENSOR_MODEL_PARALLEL_GROUP", "_DATA_PARALLEL_GROUP"):
                mp.setattr(parallel_state, name, groups[rank])
            model = torch.nn.Module()
            model.post_process = rank == 1
            if rank == 0:
                install_chunked_projection(model)
                hidden = torch.randn(5, 1, 4)
                dist.send(hidden, 1)
                grad = torch.empty_like(hidden)
                dist.recv(grad, 1)
                assert torch.isfinite(grad).all() and grad.abs().sum() > 0
                return
            model.output_layer = ColumnParallelLinear(
                4,
                8,
                config=ModelParallelConfig(use_cpu_initialization=True),
                init_method=torch.nn.init.xavier_uniform_,
                bias=True,
                skip_bias_add=True,
            )
            with pytest.raises(ValueError, match="separate bias"):
                install_chunked_projection(model)
            model.output_layer.skip_bias_add = False
            model.output_layer.gather_output = True
            with pytest.raises(ValueError, match="parallel_output"):
                install_chunked_projection(model)
            model.output_layer.gather_output = False
            model.output_layer.config.logit_softcapping = 30.0
            with pytest.raises(ValueError, match="logit_softcapping"):
                install_chunked_projection(model)
            model.output_layer.config.logit_softcapping = None
            with torch.no_grad():
                model.output_layer.bias.copy_(torch.arange(8))
            hidden = torch.empty(5, 1, 4)
            dist.recv(hidden, 0)
            hidden.requires_grad_()
            labels = torch.arange(5).unsqueeze(0)
            dense, _ = model.output_layer(hidden)
            keys = list(model.state_dict())
            params = {k: id(v) for k, v in model.named_parameters()}
            layer = model.output_layer
            hooks = (dict(layer._forward_pre_hooks), dict(layer._forward_hooks))
            actor = MAMegatronActor.__new__(MAMegatronActor)
            mp.setattr(MegatronActor, "model_provider_func", lambda *args: model)
            for size in (0, 2):
                actor.lm_head_chunk_size = size
                assert actor.model_provider_func(False, True) is model
                assert (type(layer) is ColumnParallelLinear) == (size == 0)
                assert hooks == (layer._forward_pre_hooks, layer._forward_hooks)
                assert list(model.state_dict()) == keys
                assert {k: id(v) for k, v in model.named_parameters()} == params

            all_reduce_calls = 0
            orig_all_reduce = dist.all_reduce

            def _count_all_reduce(*args, **kwargs):
                nonlocal all_reduce_calls
                all_reduce_calls += 1
                return orig_all_reduce(*args, **kwargs)

            with (
                pytest.MonkeyPatch.context() as count_mp,
                chunked_scoring(
                    model,
                    labels=labels,
                    label_mask=torch.ones_like(labels, dtype=torch.bool),
                    chunk_size=2,
                    calculate_entropy=True,
                ) as captured,
            ):
                count_mp.setattr(dist, "all_reduce", _count_all_reduce)
                with pytest.raises(ValueError, match="runtime_gather_output"):
                    model.output_layer(hidden, runtime_gather_output=True)
                scores, _ = model.output_layer(hidden)
            assert all_reduce_calls == 0
            assert scores.shape == (5, 1, 1)
            dense_lp = dense.transpose(0, 1).log_softmax(-1)
            expected = dense_lp.gather(-1, labels[..., None])
            expected_entropy = -(dense_lp.exp() * dense_lp).sum(-1)
            torch.testing.assert_close(scores.transpose(0, 1), expected)
            torch.testing.assert_close(captured["entropy"], expected_entropy)
            assert not captured["entropy"].requires_grad
            grads = torch.autograd.grad(
                scores.sum(), (hidden, model.output_layer.weight)
            )
            reference = torch.autograd.grad(
                expected.sum(), (hidden, model.output_layer.weight)
            )
            torch.testing.assert_close(grads, reference)
            dist.send(grads[0].contiguous(), 0)
            assert torch.equal(model.output_layer(hidden)[0], dense)
    finally:
        dist.destroy_process_group()


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


def _check_tp_cp_chunked_actor(rank, init_file):
    from types import SimpleNamespace

    from megatron.core import ModelParallelConfig, parallel_state
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear

    from rlinf.hybrid_engines.megatron.output_projection import (
        install_chunked_projection,
    )
    from rlinf.utils.utils import masked_mean
    from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=60),
    )
    try:
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=2, context_parallel_size=2
        )
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        torch.manual_seed(31)
        full_embed = torch.randn(12, 6)
        full_head = torch.randn(12, 6)
        full_bias = torch.randn(12)
        rows = [
            torch.tensor([1, 4, 7, 2, 9, 0, 5, 11, 3, 8, 6]),
            torch.tensor([2, 10, 5, 1, 8, 3, 9, 4, 11, 0, 7, 6, 2, 5, 1, 9, 4, 3, 8]),
        ]
        masks = [
            torch.tensor([0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1], dtype=torch.bool),
            torch.tensor(
                [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                dtype=torch.bool,
            ),
        ]

        class _TPCPStage(torch.nn.Module):
            pre_process = post_process = True

            def __init__(self, chunked: bool, sequence_parallel: bool):
                super().__init__()
                self.embed = torch.nn.Parameter(full_embed.clone())
                self.sequence_parallel = sequence_parallel
                cfg = ModelParallelConfig(
                    use_cpu_initialization=True,
                    tensor_model_parallel_size=2,
                    context_parallel_size=2,
                    sequence_parallel=sequence_parallel,
                )
                self.output_layer = ColumnParallelLinear(
                    6,
                    12,
                    config=cfg,
                    init_method=torch.nn.init.zeros_,
                    bias=True,
                    gather_output=False,
                )
                with torch.no_grad():
                    self.output_layer.weight.copy_(full_head.chunk(2, dim=0)[tp_rank])
                    self.output_layer.bias.copy_(full_bias.chunk(2, dim=0)[tp_rank])
                if chunked:
                    install_chunked_projection(self)

            def forward(
                self, input_ids, attention_mask, position_ids, packed_seq_params
            ):
                hidden = self.embed[input_ids.T]
                if self.sequence_parallel:
                    hidden = hidden.chunk(2, dim=0)[tp_rank].contiguous()
                out, _ = self.output_layer(hidden)
                return out.transpose(0, 1).contiguous()

        def run_actor_pass(chunk_size: int, sequence_parallel: bool):
            actor = MAMegatronActor.__new__(MAMegatronActor)
            actor.cfg = OmegaConf.create(
                {
                    "runner": {"task_type": "reasoning"},
                    "actor": {"model": {"encoder_seq_length": 24}},
                    "algorithm": {
                        "sampling_params": {"temperature": 0.8},
                        "loss_type": "actor",
                        "entropy_bonus": 0.0,
                        "use_valid_token_scale": False,
                        "early_stop_imp_ratio": None,
                    },
                }
            )
            actor.transformer_config = SimpleNamespace(
                sequence_parallel=sequence_parallel
            )
            actor.tokenizer = SimpleNamespace(eos_token_id=0)
            actor.clip_ratio_c, actor.clip_ratio_low, actor.clip_ratio_high = (
                None,
                0.2,
                0.2,
            )
            actor.lm_head_chunk_size = chunk_size
            actor.calculate_entropy = True
            actor.calculate_entropy_loss = False
            actor.kl_beta = 0.05
            actor.kl_penalty_type = "low_var_kl"
            actor.loss_agg_func = masked_mean
            actor.torch_platform = SimpleNamespace(current_device=lambda: "cpu")

            model = _TPCPStage(
                chunked=chunk_size > 0, sequence_parallel=sequence_parallel
            )
            actor.return_loss = False
            infer_batch = {"input_ids": [r.clone() for r in rows]}
            with torch.no_grad():
                packed_lp, collect = actor.get_forward_step_func()(
                    iter([infer_batch]), model
                )
                recomputed = collect(packed_lp)

            actor.return_loss = True
            train_batch = {
                "input_ids": [r.clone() for r in rows],
                "response_mask": [m.clone() for m in masks],
                "advantages": [m.float() * 0.7 for m in masks],
                "loss_scales": [m.float() for m in masks],
                "recomputed_logprobs": [lp - 0.03 for lp in recomputed],
                "ref_logprobs": [lp + 0.02 for lp in recomputed],
            }
            scores, loss_fn = actor.get_forward_step_func()(iter([train_batch]), model)
            loss, metrics = loss_fn(scores)
            loss.backward()
            assert model.embed.grad is not None and model.embed.grad.abs().sum() > 0
            return (
                recomputed,
                loss.detach(),
                metrics["actor/entropy_loss"].detach(),
                metrics["actor/kl_loss"].detach(),
                model.embed.grad.detach().clone(),
                model.output_layer.weight.grad.detach().clone(),
                model.output_layer.bias.grad.detach().clone(),
            )

        with pytest.MonkeyPatch.context() as mpatch:
            mpatch.setattr(torch.cuda, "current_device", lambda: "cpu")
            for sequence_parallel in (True, False):
                dense_res = run_actor_pass(0, sequence_parallel)
                chunk_res = run_actor_pass(3, sequence_parallel)
                for got_lp, want_lp in zip(chunk_res[0], dense_res[0], strict=True):
                    torch.testing.assert_close(got_lp, want_lp, atol=2e-6, rtol=2e-6)
                for got, want in zip(chunk_res[1:], dense_res[1:], strict=True):
                    torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-6)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def test_chunked_actor_scores_and_backpropagates_under_tp_and_cp(tmp_path):
    pytest.importorskip("megatron.training")
    mp.spawn(_check_tp_cp_chunked_actor, args=(str(tmp_path / "tpcp"),), nprocs=4)


@pytest.mark.parametrize(
    "backend,dynamic,chunk_size,entropy_bonus,message",
    [
        ("fsdp", True, 16, 0.0, "dynamic Megatron"),
        ("megatron", False, 16, 0.0, "dynamic Megatron"),
        ("megatron", True, -1, 0.0, "nonnegative integer"),
        ("megatron", True, True, 0.0, "nonnegative integer"),
        ("megatron", True, False, 0.0, "nonnegative integer"),
        ("megatron", True, 16, 0.01, "logging only"),
    ],
)
def test_chunked_lm_head_rejects_unsupported_actor(
    backend, dynamic, chunk_size, entropy_bonus, message
):
    from rlinf.config import validate_reasoning_cfg

    cfg = OmegaConf.create(
        {
            "actor": {
                "training_backend": backend,
                "model": {
                    "context_parallel_size": 2,
                    "lm_head_chunk_size": chunk_size,
                },
            },
            "algorithm": {"entropy_bonus": entropy_bonus},
            "agentloop": {"is_dynamic_rollout_batch": dynamic},
        }
    )
    with pytest.raises(ValueError, match=message):
        validate_reasoning_cfg(cfg)


@pytest.mark.parametrize(
    "chunk_size,temperature,dtype,entropy",
    [
        (1, 0.7, torch.float32, False),
        (4, 1.3, torch.float32, True),
        (3, 1.0, torch.bfloat16, False),
        (32, 1.0, torch.float16, False),
    ],
)
def test_chunked_lm_head_matches_dense_values_and_gradients(
    chunk_size, temperature, dtype, entropy, monkeypatch
):
    from rlinf.hybrid_engines.megatron.lm_head import chunked_vocab_parallel_log_probs

    torch.manual_seed(81)
    hidden = torch.randn(2, 7, 5, dtype=dtype, requires_grad=True)
    weight = torch.randn(19, 5, dtype=dtype, requires_grad=True)
    bias = torch.randn(19, dtype=dtype, requires_grad=True)
    labels = torch.randint(19, (2, 7))
    mask = torch.rand(2, 7) > 0.3
    # Out-of-bounds labels at masked positions must be ignored without nonzero sync.
    masked_labels = labels.masked_fill(~mask, -100)
    scale = torch.randn(2, 7)
    dense = torch.nn.functional.linear(hidden, weight, bias).float() / temperature
    dense_log_probs = dense.log_softmax(-1)
    expected = (
        dense_log_probs.gather(-1, labels.unsqueeze(-1))
        .squeeze(-1)
        .masked_fill(~mask, 0)
    )
    reference_grads = torch.autograd.grad(
        (expected * scale).sum(), (hidden, weight, bias)
    )

    def _forbid_nonzero(*args, **kwargs):
        raise AssertionError("chunked_vocab_parallel_log_probs must not call nonzero()")

    monkeypatch.setattr(torch.Tensor, "nonzero", _forbid_nonzero)
    actual = chunked_vocab_parallel_log_probs(
        hidden,
        weight,
        masked_labels,
        mask,
        chunk_size=chunk_size,
        temperature=temperature,
        bias=bias,
        calculate_entropy=entropy,
    )
    torch.testing.assert_close(actual["log_probs"], expected, atol=2e-6, rtol=2e-6)
    actual_grads = torch.autograd.grad(
        (actual["log_probs"] * scale).sum(), (hidden, weight, bias)
    )
    tolerance = 0.02 if dtype in (torch.bfloat16, torch.float16) else 2e-6
    for got, want in zip(actual_grads, reference_grads):
        torch.testing.assert_close(got, want, atol=tolerance, rtol=tolerance)
    if entropy:
        assert not actual["entropy"].requires_grad
        torch.testing.assert_close(
            actual["entropy"], -(dense_log_probs.exp() * dense_log_probs).sum(-1)
        )


@pytest.mark.parametrize(
    "fusion,need_dummy_grad",
    [(True, True), (False, False), (False, True)],
)
def test_chunked_lm_head_accumulates_main_grad_without_double_add(
    fusion, need_dummy_grad
):
    from rlinf.hybrid_engines.megatron.lm_head import chunked_vocab_parallel_log_probs

    torch.manual_seed(93)
    weight = torch.nn.Parameter(torch.randn(17, 5, dtype=torch.bfloat16))
    weight.main_grad = torch.ones(17, 5, dtype=torch.float32)
    weight.grad_added_to_main_grad = False
    expected = weight.main_grad.clone()
    for _ in range(2):
        hidden = torch.randn(7, 5, dtype=torch.bfloat16, requires_grad=True)
        labels = torch.randint(17, (7,))
        dense = torch.nn.functional.linear(hidden, weight).float().log_softmax(-1)
        expected += torch.autograd.grad(dense.gather(1, labels[:, None]).sum(), weight)[
            0
        ].float()
        result = chunked_vocab_parallel_log_probs(
            hidden,
            weight,
            labels,
            torch.ones(7, dtype=torch.bool),
            chunk_size=3,
            gradient_accumulation_fusion=fusion,
            need_dummy_grad=need_dummy_grad,
        )
        result["log_probs"].sum().backward()
    torch.testing.assert_close(weight.main_grad, expected, atol=0.02, rtol=0.02)
    assert weight.grad_added_to_main_grad
    if need_dummy_grad:
        torch.testing.assert_close(weight.grad, torch.zeros_like(weight))
    else:
        assert weight.grad is None


def test_chunked_lm_head_skips_fsdp_and_empty_main_grad():
    from rlinf.hybrid_engines.megatron.lm_head import chunked_vocab_parallel_log_probs

    for make_unusable in ("fsdp", "empty"):
        torch.manual_seed(94)
        weight = torch.nn.Parameter(torch.randn(11, 4, dtype=torch.bfloat16))
        weight.main_grad = torch.zeros(11, 4, dtype=torch.float32)
        if make_unusable == "fsdp":
            weight.__fsdp_param__ = True
        else:
            weight.main_grad.untyped_storage().resize_(0)
        hidden = torch.randn(5, 4, dtype=torch.bfloat16, requires_grad=True)
        labels = torch.randint(11, (5,))
        dense = torch.nn.functional.linear(hidden, weight).float().log_softmax(-1)
        (expected_grad,) = torch.autograd.grad(
            dense.gather(1, labels[:, None]).sum(), weight
        )
        result = chunked_vocab_parallel_log_probs(
            hidden,
            weight,
            labels,
            torch.ones(5, dtype=torch.bool),
            chunk_size=2,
            need_dummy_grad=False,
        )
        result["log_probs"].sum().backward()
        assert weight.grad is not None and weight.grad.dtype == torch.bfloat16
        torch.testing.assert_close(weight.grad, expected_grad, atol=0.02, rtol=0.02)
        if make_unusable == "fsdp":
            torch.testing.assert_close(
                weight.main_grad, torch.zeros_like(weight.main_grad)
            )


def _chunked_lm_head_tp_reference(rank, rendezvous):
    from rlinf.hybrid_engines.megatron.lm_head import chunked_vocab_parallel_log_probs

    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.manual_seed(17)
        hidden = torch.randn(9, 5, requires_grad=True)
        full_weight = torch.randn(22, 5, requires_grad=True)
        labels = torch.tensor([0, 10, 11, 21, 3, 16, 7, 18, 20])
        mask = torch.tensor([True, False, True, True, True, True, False, True, True])
        scales = torch.randn(9)
        reference = (hidden @ full_weight.t() / 0.7).log_softmax(-1)
        reference = (
            reference.gather(1, labels[:, None]).squeeze(1).masked_fill(~mask, 0)
        )
        dh, dw = torch.autograd.grad((reference * scales).sum(), (hidden, full_weight))
        shard = full_weight.detach().chunk(2)[rank].clone().requires_grad_()
        actual = chunked_vocab_parallel_log_probs(
            hidden,
            shard,
            labels,
            mask,
            chunk_size=3,
            temperature=0.7,
            tp_group=dist.group.WORLD,
        )
        (actual["log_probs"] * scales).sum().backward()
        dist.all_reduce(hidden.grad)
        torch.testing.assert_close(actual["log_probs"], reference, atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(hidden.grad, dh, atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(shard.grad, dw.chunk(2)[rank], atol=3e-6, rtol=3e-6)
    finally:
        dist.destroy_process_group()


def test_chunked_lm_head_two_rank_vocab_shards(tmp_path):
    if not dist.is_gloo_available():
        pytest.skip("The TP reference test requires Gloo")
    mp.spawn(
        _chunked_lm_head_tp_reference, args=((tmp_path / "tp-init").as_uri(),), nprocs=2
    )


def test_chunked_lm_head_accumulates_bias_gradient_across_chunks():
    from rlinf.hybrid_engines.megatron.lm_head import chunked_vocab_parallel_log_probs

    # Equal logits give a bias gradient of (+0.5, -0.5) for each target zero.
    hidden = torch.zeros(1024, 1, dtype=torch.bfloat16)
    weight = hidden.new_zeros(2, 1)
    bias = hidden.new_zeros(2, requires_grad=True)
    labels = torch.zeros(1024, dtype=torch.long)
    scores = chunked_vocab_parallel_log_probs(
        hidden, weight, labels, labels == 0, bias=bias, chunk_size=2
    )
    scores["log_probs"].sum().backward()
    torch.testing.assert_close(bias.grad, bias.new_tensor([512, -512]), rtol=0, atol=0)


def test_chunked_lm_head_accumulates_bf16_weight_gradient_across_chunks():
    from rlinf.hybrid_engines.megatron.lm_head import chunked_vocab_parallel_log_probs

    # Constant hidden=1 with equal logits gives (+0.5, -0.5) per token, which
    # saturates at 256 if accumulated in BF16 across 2-token chunks.
    hidden = torch.ones(4096, 1, dtype=torch.bfloat16)
    weight = hidden.new_zeros(2, 1, requires_grad=True)
    labels = torch.zeros(4096, dtype=torch.long)
    scores = chunked_vocab_parallel_log_probs(
        hidden, weight, labels, labels == 0, chunk_size=2
    )
    scores["log_probs"].sum().backward()
    torch.testing.assert_close(
        weight.grad, weight.new_tensor([[2048.0], [-2048.0]]), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "dtype,tokens,chunk_size,need_dummy_grad",
    [
        (torch.bfloat16, 131_072, 256, False),
        (torch.float16, 8_192, 2, True),
    ],
)
def test_chunked_lm_head_accumulates_low_precision_main_grad_in_fp32(
    dtype, tokens, chunk_size, need_dummy_grad
):
    from rlinf.hybrid_engines.megatron.lm_head import chunked_vocab_parallel_log_probs

    # Accumulating each chunk directly into a BF16/FP16 main_grad saturates at
    # +/-0.25 once per-chunk mean updates fall below half a ULP; FP32 tiled
    # accumulation flushed once per microbatch matches dense backward (+/-0.5).
    hidden = torch.ones(tokens, 1, dtype=dtype)
    labels = torch.zeros(tokens, dtype=torch.long)
    mask = torch.ones(tokens, dtype=torch.bool)
    weight = torch.nn.Parameter(torch.zeros(2, 1, dtype=dtype))
    weight.main_grad = torch.zeros_like(weight)

    for step in (1, 2):
        weight.grad = None
        weight.grad_added_to_main_grad = False
        scores = chunked_vocab_parallel_log_probs(
            hidden,
            weight,
            labels,
            mask,
            chunk_size=chunk_size,
            gradient_accumulation_fusion=True,
            need_dummy_grad=need_dummy_grad,
        )
        scores["log_probs"].mean().backward()
        torch.testing.assert_close(
            weight.main_grad,
            weight.new_tensor([[0.5 * step], [-0.5 * step]]),
            rtol=0,
            atol=0,
        )
        assert weight.grad_added_to_main_grad
        if need_dummy_grad:
            torch.testing.assert_close(weight.grad, torch.zeros_like(weight))
        else:
            assert weight.grad is None
