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

"""Worker-level tests for StreamingVLA-only SFT behavior."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import torch

from rlinf.models.embodiment.streamingvla.training import (
    StreamingVLAStepInputBuffer,
    sample_streamingvla_step_inputs,
)
from rlinf.workers.sft.fsdp_vla_sft_worker import (
    FSDPVlaSftWorker,
    _get_streamingvla_seed_rank,
)


class _RecordingModel:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def __call__(self, **kwargs: Any) -> torch.Tensor:
        time = kwargs["time"]
        noise = kwargs["noise"]
        self.calls.append((time.clone(), noise.clone()))
        return time.mean() + noise.mean() * 0.0


class _UnchangedModel:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def __call__(self, **kwargs: Any) -> torch.Tensor:
        self.kwargs = kwargs
        return torch.tensor(1.0)


def test_streamingvla_seed_uses_global_rank_across_nodes(monkeypatch):
    """Global rank prevents duplicated RNG streams on different nodes."""
    monkeypatch.setenv("RANK", "9")
    monkeypatch.setenv("LOCAL_RANK", "1")

    assert _get_streamingvla_seed_rank() == 9


def test_streamingvla_seed_falls_back_to_local_rank(monkeypatch):
    """Local rank remains a fallback outside a distributed launcher."""
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setenv("LOCAL_RANK", "3")

    assert _get_streamingvla_seed_rank() == 3


def test_streamingvla_worker_lazily_slices_one_rank_local_random_batch():
    """Two micro-batches consume one reproducible local random batch."""
    worker = object.__new__(FSDPVlaSftWorker)
    worker._is_streamingvla = True
    worker._streamingvla_step_inputs = StreamingVLAStepInputBuffer(
        seed=42,
        rank=0,
        local_batch_size=8,
        action_dim=32,
        device=torch.device("cpu"),
    )
    worker.amp_context = nullcontext()
    worker.model = _RecordingModel()

    batch = (object(), torch.zeros(4, 10, 32))
    worker.get_train_model_output(batch)
    assert worker._streamingvla_step_inputs.step == 0
    worker.get_train_model_output(batch)

    expected_time, expected_noise = sample_streamingvla_step_inputs(
        seed=42,
        rank=0,
        global_step=0,
        local_batch_size=8,
        action_dim=32,
        device=torch.device("cpu"),
    )
    actual_time = torch.cat([call[0] for call in worker.model.calls])
    actual_noise = torch.cat([call[1] for call in worker.model.calls])
    torch.testing.assert_close(actual_time, expected_time, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_noise, expected_noise, rtol=0.0, atol=0.0)
    assert worker._streamingvla_step_inputs.step == 1


def test_streamingvla_step_buffer_accepts_rlinf_integer_device():
    """RLinf's integer accelerator index is normalized before SFP sampling."""
    buffer = StreamingVLAStepInputBuffer(
        seed=42,
        rank=0,
        local_batch_size=8,
        action_dim=32,
        device=0,
    )

    assert buffer._device == torch.device("cuda", 0)


def test_streamingvla_worker_uses_world_group_for_fsdp2_grad_norm():
    """StreamingVLA supplies the missing FSDP-only gradient reduction group."""
    worker = object.__new__(FSDPVlaSftWorker)
    worker._is_streamingvla = True
    worker._strategy = SimpleNamespace(_dp_group=None)
    world_group = object()

    worker._configure_streamingvla_grad_norm_group(world_group)

    assert worker._strategy._dp_group is world_group


def test_existing_fsdp_grad_norm_group_is_preserved():
    """Hybrid meshes keep their existing data-parallel reduction group."""
    worker = object.__new__(FSDPVlaSftWorker)
    worker._is_streamingvla = True
    existing_group = object()
    worker._strategy = SimpleNamespace(_dp_group=existing_group)

    worker._configure_streamingvla_grad_norm_group(object())

    assert worker._strategy._dp_group is existing_group


def test_non_streamingvla_worker_keeps_the_existing_forward_contract():
    """Other VLA models receive only the original SFT forward arguments."""
    worker = object.__new__(FSDPVlaSftWorker)
    worker._is_streamingvla = False
    worker._streamingvla_step_inputs = None
    worker.amp_context = nullcontext()
    worker.model = _UnchangedModel()

    batch = {"actions": torch.zeros(2, 10, 7)}
    loss, metrics = worker.get_train_model_output(batch)

    assert loss.item() == 1.0
    assert metrics == {"loss": 1.0}
    assert worker.model.kwargs is not None
    assert set(worker.model.kwargs) == {"forward_type", "data"}
    assert worker.model.kwargs["data"] is batch
