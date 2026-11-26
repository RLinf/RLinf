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

"""Multi-turn actor helpers, packing, and weight-version bookkeeping.

Megatron-specific tests skip when `megatron.core` is absent. No Ray or GPUs
are launched.
"""

import copy
from contextlib import contextmanager
from datetime import timedelta
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from rlinf.data.schema.reasoning_results import (
    DynamicRolloutResult,
    GroupDelivery,
    PayloadRef,
)
from rlinf.utils.distributed import receive_balanced_groups
from rlinf.utils.metric_utils import (
    compute_dp_staleness_metrics,
    extract_versioned_trajectory_records,
)
from rlinf.workers.rollout.session_server.assembly import receive_rollout_result
from rlinf.workers.rollout.session_server.payload import GroupPayload, PayloadPart


class TestExtractVersionedTrajectoryRecords:
    """Record extraction from DynamicRolloutResult objects."""

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
        records = extract_versioned_trajectory_records([result])
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
        result = self._dynamic_result()
        assert extract_versioned_trajectory_records([result]) is None

    def test_returns_none_if_any_result_lacks_versions(self):
        versioned = self._dynamic_result(version_start=[1, 2], version_end=[1, 2])
        assert (
            extract_versioned_trajectory_records([versioned, self._dynamic_result()])
            is None
        )


class _LocalDeliveryChannel:
    """A local input channel that returns one item per get."""

    is_local = True

    def __init__(self, item):
        self._item = item
        self.get_count = 0

    def get(self):
        self.get_count += 1
        return self._item


class _StubPayloadChannel:
    """In-memory stand-in for one payload channel shard."""

    def __init__(self, name: str, items: dict):
        self.name = name
        self._items = dict(items)
        self.removed_keys: list[tuple] = []

    def get(self, key):
        return self._items[key]

    def remove_key(self, key, async_op: bool = False):
        self.removed_keys.append((key, async_op))
        self._items.pop(key, None)


def _sample_group_payload(group_id: int = 7) -> GroupPayload:
    turn = PayloadPart(
        group_id=group_id,
        traj_index=0,
        turn_index=0,
        input_ids=[1, 2],
        output_ids=[3, 4],
        logprobs=[-0.1, -0.2],
        version_start=2,
        weight_version=2,
        finish_reason={"type": "stop"},
    )
    finalize = PayloadPart(
        group_id=group_id,
        traj_index=0,
        turn_index=0,
        input_ids=[],
        output_ids=[],
        logprobs=None,
        version_start=None,
        weight_version=None,
        finish_reason={},
        last=True,
        verdicts={
            0: {
                "kept_lens": {0: 2},
                "continues_prev": {0: False},
                "is_end": {0: True},
                "not_training": {},
            }
        },
        rewards=[1.0],
    )
    return GroupPayload(group_id=group_id, parts=[turn, finalize])


def test_receive_rollout_result_passes_through_and_assembles_delivery():
    """receive_rollout_result returns inline results untouched, assembles
    GroupDelivery payloads and removes their keys, and asserts when payload
    channels are missing."""
    cfg = OmegaConf.create({"algorithm": {"recompute_logprobs": False}})
    inline = _versioned_group(0)
    assert receive_rollout_result(_LocalDeliveryChannel(inline), None, cfg) is inline

    delivery = GroupDelivery(
        group_id=7,
        task_name="searchr1",
        payload_ref=PayloadRef(locator="channel://Payload", key=7),
    )
    with pytest.raises(AssertionError, match="payload channels"):
        receive_rollout_result(_LocalDeliveryChannel(delivery), None, cfg)

    payload_channel = _StubPayloadChannel("Payload", {7: _sample_group_payload(7)})
    assembled = receive_rollout_result(
        _LocalDeliveryChannel(delivery), [payload_channel], cfg
    )
    assert isinstance(assembled, DynamicRolloutResult)
    assert assembled.group_id == 7
    assert assembled.task_name == "searchr1"
    assert assembled.input_ids == [[1, 2, 3, 4]]
    assert assembled.rollout_logprobs == [[-0.1, -0.2]]
    assert payload_channel.removed_keys == [(7, True)]


def test_receive_balanced_groups_tags_and_times_balance(monkeypatch):
    """receive_balanced_groups skips tagging/balancing for local channels and
    tags source_dp + wraps balance_rollout_groups in group_dp_balance otherwise."""
    import rlinf.utils.distributed as dist_mod

    local_group = _versioned_group(0)
    local_results = receive_balanced_groups(
        lambda: local_group,
        1,
        is_local=True,
        source_rank=None,
        dp_group=None,
        balance=True,
    )
    assert local_results == [local_group]
    assert "source_dp" not in local_group.extra_fields_train

    remote_group = _versioned_group(1)
    timers = []
    balance_calls = []

    @contextmanager
    def _timer(name):
        timers.append(name)
        yield

    monkeypatch.setattr(
        dist_mod,
        "balance_rollout_groups",
        lambda results, dp_group: balance_calls.append((results, dp_group)) or results,
    )
    sentinel_group = object()
    remote_results = receive_balanced_groups(
        lambda: remote_group,
        1,
        is_local=False,
        source_rank=3,
        dp_group=sentinel_group,
        balance=True,
        timer_ctx=_timer,
    )
    assert remote_results == [remote_group]
    assert remote_group.extra_fields_train["source_dp"] == [3, 3]
    assert timers == ["group_dp_balance"]
    assert balance_calls == [([remote_group], sentinel_group)]


def test_get_step_batch_assembles_delivered_groups(monkeypatch):
    """MAMegatronActor.get_step_batch assembles a GroupDelivery through
    receive_rollout_result."""
    pytest.importorskip("megatron.core")
    from megatron.core import parallel_state

    from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor

    actor = MAMegatronActor.__new__(MAMegatronActor)
    actor.cfg = OmegaConf.create(
        {
            "data": {"rollout_batch_size": 1},
            "algorithm": {"group_size": 1, "recompute_logprobs": False},
        }
    )
    actor.enable_dp_load_balance = False
    payload_channel = _StubPayloadChannel("Payload", {7: _sample_group_payload(7)})
    actor._payload_channels = [payload_channel]
    delivery = GroupDelivery(
        group_id=7,
        task_name="searchr1",
        payload_ref=PayloadRef(locator="channel://Payload", key=7),
    )
    monkeypatch.setattr(parallel_state, "get_data_parallel_world_size", lambda: 1)
    channel = _LocalDeliveryChannel(delivery)

    _batch, results = actor.get_step_batch(channel)

    assert len(results) == 1
    assert results[0].group_id == 7
    assert results[0].task_name == "searchr1"
    assert payload_channel.removed_keys == [(7, True)]
    assert channel.get_count == 1


def test_ma_fsdp_actor_get_step_batch_assembles_delivered_groups(monkeypatch):
    """MAFSDPActor.get_step_batch assembles a GroupDelivery through the shared
    read-side helpers."""
    from rlinf.workers.actor.ma_fsdp_actor_worker import MAFSDPActor

    actor = MAFSDPActor.__new__(MAFSDPActor)
    actor.cfg = OmegaConf.create(
        {
            "data": {"rollout_batch_size": 1},
            "algorithm": {"group_size": 1, "recompute_logprobs": False},
        }
    )
    actor.enable_dp_load_balance = False
    payload_channel = _StubPayloadChannel("Payload", {7: _sample_group_payload(7)})
    actor._payload_channels = [payload_channel]
    delivery = GroupDelivery(
        group_id=7,
        task_name="searchr1",
        payload_ref=PayloadRef(locator="channel://Payload", key=7),
    )
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    channel = _LocalDeliveryChannel(delivery)

    _batch, results = actor.get_step_batch(channel)

    assert len(results) == 1
    assert results[0].group_id == 7
    assert results[0].task_name == "searchr1"
    assert payload_channel.removed_keys == [(7, True)]
    assert channel.get_count == 1


def test_fsdp_pipeline_feature_validation_differs_for_base_and_ma_actor():
    """Base FSDPActor rejects DP load balance and dynamic batching in pipeline
    mode, while MAFSDPActor allows both under disaggregated placement."""
    from rlinf.workers.actor.fsdp_actor_worker import FSDPActor
    from rlinf.workers.actor.ma_fsdp_actor_worker import MAFSDPActor

    base = FSDPActor.__new__(FSDPActor)
    base.is_pipeline = True
    base.enable_dp_load_balance = True
    base.enable_dynamic_batch_size = False
    with pytest.raises(AssertionError, match="DP load balance"):
        base._validate_pipeline_features()

    base.enable_dp_load_balance = False
    base.enable_dynamic_batch_size = True
    with pytest.raises(AssertionError, match="Dynamic batch size"):
        base._validate_pipeline_features()

    ma_actor = MAFSDPActor.__new__(MAFSDPActor)
    ma_actor.is_pipeline = True
    ma_actor.enable_dp_load_balance = True
    ma_actor.enable_dynamic_batch_size = True
    ma_actor._validate_pipeline_features()


def test_fsdp_actor_init_skips_inference_block_without_dedicated_inference(monkeypatch):
    """Disaggregated FSDPActor without a dedicated inference group does not
    require cfg.inference to exist."""
    from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
    from rlinf.scheduler import Worker
    from rlinf.utils.placement import PlacementMode
    from rlinf.workers.actor.ma_fsdp_actor_worker import MAFSDPActor

    monkeypatch.setattr(Worker, "__init__", lambda self: None)
    monkeypatch.setattr(FSDPModelManager, "__init__", lambda self, *a, **kw: None)

    cfg = OmegaConf.create(
        {
            "runner": {
                "task_type": "reasoning",
                "enable_dynamic_batch_size": True,
                "max_tokens_per_mbs": 4096,
            },
            "data": {"max_prompt_length": 512, "rollout_batch_size": 4},
            "algorithm": {
                "group_size": 2,
                "n_minibatches": 1,
                "calculate_entropy": False,
                "entropy_bonus": 0.0,
                "kl_beta": 0.0,
                "kl_penalty_type": "low_var_kl",
                "loss_agg_func": "token-mean",
                "loss_scales": [],
            },
            "agentloop": {"is_dynamic_rollout_batch": True},
            "rollout": {"group_name": "RolloutGroup"},
            "actor": {
                "micro_batch_size": 1,
                "enable_dp_load_balance": True,
                "enable_offload": False,
                "model": {"encoder_seq_length": 1024},
            },
        }
    )
    placement = MagicMock()
    placement.is_disaggregated = True
    placement.has_dedicated_actor_inference = False
    placement.placement_mode = PlacementMode.DISAGGREGATED

    actor = MAFSDPActor.__new__(MAFSDPActor)
    actor._world_size = 4
    actor._rank = 0
    MAFSDPActor.__init__(actor, cfg, placement)

    assert actor.is_pipeline is True
    assert actor._inference_group_name is None
    assert actor._inference_world_size == 0
    assert actor._payload_channels is None


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
        metrics = compute_dp_staleness_metrics(results, 10, dist.group.WORLD)
        assert compute_dp_staleness_metrics(results, None, dist.group.WORLD) is None
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
