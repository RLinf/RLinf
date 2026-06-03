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
import contextlib
import importlib.util
import sys
from collections import defaultdict, deque
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


class _FakeChannel:
    def __init__(self, items):
        self._items = deque(items)

    def get_nowait(self):
        if not self._items:
            raise asyncio.QueueEmpty
        return self._items.popleft()


@pytest.fixture
def pipeline_actor_module(monkeypatch):
    actor_dir = Path(__file__).resolve().parents[2] / "rlinf/workers/actor"
    fake_fsdp_module = ModuleType("rlinf.workers.actor.fsdp_actor_worker")

    class FakeEmbodiedFSDPActor:
        pass

    fake_fsdp_module.EmbodiedFSDPActor = FakeEmbodiedFSDPActor
    monkeypatch.setitem(
        sys.modules,
        "rlinf.workers.actor.fsdp_actor_worker",
        fake_fsdp_module,
    )

    module_name = "fsdp_actor_worker_pipeline_under_test"
    spec = importlib.util.spec_from_file_location(
        module_name,
        actor_dir / "fsdp_actor_worker_pipeline.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _make_micro_batch(batch_id: int) -> dict[str, torch.Tensor]:
    value = torch.tensor([float(batch_id)], dtype=torch.float32)
    return {
        "mb_id": torch.tensor([batch_id], dtype=torch.int64),
        "rewards": value.clone(),
        "advantages": value.clone(),
        "returns": value.clone(),
    }


def _build_actor(pipeline_actor_module):
    actor = object.__new__(pipeline_actor_module.PipelineEmbodiedFSDPActor)
    actor._accelerator_type = pipeline_actor_module.Worker.accelerator_type
    actor._timer_metrics = {}
    actor.worker_timer = lambda tag: contextlib.nullcontext()
    actor.is_weight_offloaded = False
    actor.is_optimizer_offloaded = False
    actor.model = SimpleNamespace(train=MagicMock())
    actor.lr_scheduler = SimpleNamespace(step=MagicMock())
    actor.micro_batches_per_step = 4
    actor.gradient_accumulation = 2
    actor.global_batches_per_step = 2
    actor.update_epoch = 2
    return actor


def test_select_global_batch_never_reuses_done_bucket(pipeline_actor_module):
    actor = object.__new__(pipeline_actor_module.PipelineEmbodiedFSDPActor)
    actor.update_epoch = 2

    ready_batch = pipeline_actor_module.GlobalBatchState(
        micro_batches=[_make_micro_batch(0)], train_count=1
    )
    done_batch = pipeline_actor_module.GlobalBatchState(
        micro_batches=[_make_micro_batch(1)], train_count=2
    )
    global_batches = defaultdict(deque)
    global_batches[1].append(ready_batch)
    global_batches[2].append(done_batch)

    selected = actor.select_global_batch(global_batches)

    assert selected is ready_batch
    assert list(global_batches[2]) == [done_batch]

    selected = actor.select_global_batch(global_batches)
    assert selected is None


def test_compute_micro_batches_requires_exact_divisibility(pipeline_actor_module):
    actor = object.__new__(pipeline_actor_module.PipelineEmbodiedFSDPActor)

    assert (
        actor.compute_micro_batches(
            total_num_envs=80,
            actor_world_size=2,
            micro_batch_size=32,
            rollout_epoch=1,
            n_train_chunk_steps=16,
        )
        == 20
    )

    with pytest.raises(
        AssertionError,
        match="Total flattened rollout samples",
    ):
        actor.compute_micro_batches(
            total_num_envs=81,
            actor_world_size=2,
            micro_batch_size=32,
            rollout_epoch=1,
            n_train_chunk_steps=16,
        )


def test_run_training_replays_each_global_batch_exactly_update_epoch_times(
    pipeline_actor_module, monkeypatch
):
    actor = _build_actor(pipeline_actor_module)
    channel = _FakeChannel([_make_micro_batch(i) for i in range(4)])

    train_order = []
    finish_count = 0

    def train_micro_batch(*, micro_batch, metrics, is_last):
        train_order.append((int(micro_batch["mb_id"].item()), is_last))
        metrics.setdefault("actor/loss", []).append(float(micro_batch["mb_id"].item()))

    def finish_global_batch(metrics):
        nonlocal finish_count
        finish_count += 1

    actor.train_micro_batch = train_micro_batch
    actor.finish_global_batch = finish_global_batch

    monkeypatch.setattr(
        pipeline_actor_module,
        "compute_rollout_metrics",
        lambda batch: {"received_rows": int(batch["rewards"].shape[0])},
    )
    monkeypatch.setattr(
        pipeline_actor_module,
        "all_reduce_dict",
        lambda metric_dict, op=None: metric_dict,
    )
    result = actor.run_training(channel)

    assert train_order == [
        (0, False),
        (1, True),
        (2, False),
        (3, True),
        (0, False),
        (1, True),
        (2, False),
        (3, True),
    ]
    assert finish_count == 4
    actor.model.train.assert_called_once_with()
    actor.lr_scheduler.step.assert_called_once_with()
    assert result["rollout_metrics"] == {"received_rows": 4}
    assert result["training_metrics"] == {"actor/loss": 1.5}
