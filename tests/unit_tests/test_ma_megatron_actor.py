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
