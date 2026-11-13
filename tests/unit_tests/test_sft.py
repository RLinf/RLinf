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

"""SFT evaluation metrics, sample coverage, and distributed forward parity."""

from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from rlinf.scheduler import Worker
from rlinf.workers.sft.fsdp_value_sft_worker import FSDPValueSftWorker


class _ValueDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {
            "observation": {"images": {"camera": torch.tensor([float(index)])}},
            "target_values": torch.tensor(float(index % 3)),
        }


class _ValueModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([0.5]))
        self.batch_sizes = []

    def forward(self, observation, target_values):
        inputs = observation["images"]["camera"].flatten()
        self.batch_sizes.append(len(inputs))
        predictions = inputs * self.scale
        errors = predictions - target_values
        return SimpleNamespace(
            loss=errors.square().mean(),
            predicted_values=predictions,
            cat_acc_best=(errors.abs() < 0.25).float().mean(),
            cat_acc_neighbor=(errors.abs() < 0.75).float().mean(),
            mae=errors.abs().mean(),
        )


def _eval_worker(datasets, batch_size, rank=0, world_size=1, fsdp=False):
    # Supply a small model and real loaders without starting a Ray cluster or
    # downloading pretrained assets. The evaluation path itself is unchanged.
    worker = object.__new__(FSDPValueSftWorker)
    worker.cfg = OmegaConf.create({"actor": {"enable_offload": False}})
    worker.device = torch.device("cpu")
    worker._rank = rank
    worker._world_size = world_size
    worker._timer_metrics = {}
    worker.amp_context = nullcontext()
    model = _ValueModel()
    worker.model = model
    if fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel

        worker.model = FullyShardedDataParallel(model, device_id=torch.device("cpu"))
    worker.eval_data_loaders = []
    for name, size in datasets:
        dataset = _ValueDataset(size)
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        worker.eval_data_loaders.append(
            (name, DataLoader(dataset, batch_size=batch_size, sampler=sampler))
        )
    return worker, model


def _reference(size):
    inputs = np.arange(size, dtype=np.float64)
    predictions = inputs * 0.5
    targets = inputs % 3
    errors = predictions - targets
    return {
        "loss": np.mean(errors**2),
        "mae": np.mean(np.abs(errors)),
        "cat_acc_best": np.mean(np.abs(errors) < 0.25),
        "cat_acc_neighbor": np.mean(np.abs(errors) < 0.75),
        "predicted_value_mean": predictions.mean(),
        "predicted_value_std": predictions.std(ddof=1) if size > 1 else 0.0,
        "target_value_mean": targets.mean(),
        "target_value_std": targets.std(ddof=1) if size > 1 else 0.0,
        "value_spearman": (
            float(spearmanr(predictions, targets).statistic)
            if size > 1 and np.ptp(targets) > 0
            else 0.0
        ),
    }


def _assert_metrics(metrics, size, name="validation"):
    reference = _reference(size)
    assert set(metrics) == set(reference) | {f"{name}/{key}" for key in reference}
    for key, expected in reference.items():
        assert metrics[key] == pytest.approx(expected, rel=1e-6, abs=1e-7), key
        assert metrics[f"{name}/{key}"] == pytest.approx(
            expected, rel=1e-6, abs=1e-7
        ), key


@pytest.mark.parametrize(
    "permutation", [list(range(6)), [1, 0, 2, 3, 4, 5], list(reversed(range(6)))]
)
def test_value_sft_spearman_ties_are_order_invariant(permutation):
    predictions = np.array([1, 1, 2, 2, 3, 3])[permutation]
    targets = np.array([1, 2, 1, 3, 2, 3])[permutation]
    actual = FSDPValueSftWorker._compute_spearman(predictions, targets)
    assert actual == pytest.approx(spearmanr(predictions, targets).statistic)


@pytest.mark.parametrize(
    "predictions,targets",
    [([1, 1, 1], [1, 2, 3]), ([1, 2, 3], [0, 0, 0]), ([1], [2]), ([], [])],
)
def test_value_sft_degenerate_spearman_is_zero(predictions, targets):
    assert (
        FSDPValueSftWorker._compute_spearman(np.array(predictions), np.array(targets))
        == 0.0
    )


def test_value_sft_untied_spearman():
    assert FSDPValueSftWorker._compute_spearman(
        np.arange(4), np.arange(4)[::-1]
    ) == pytest.approx(-1.0)


@pytest.fixture
def single_rank_gloo(tmp_path, monkeypatch):
    assert not dist.is_initialized()
    monkeypatch.setattr(Worker, "torch_platform", torch.cpu)
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("size,batch_size", [(9, 4), (8, 4), (1, 4), (3, 1)])
def test_value_sft_metrics_cover_each_sample(single_rank_gloo, size, batch_size):
    worker, _ = _eval_worker([("validation", size)], batch_size)
    _assert_metrics(worker.run_eval(), size)


def test_value_sft_dataset_macro_average(single_rank_gloo):
    worker, _ = _eval_worker([("small", 1), ("large", 9)], 4)
    metrics = worker.run_eval()
    for key in _reference(1):
        assert metrics[key] == pytest.approx(
            (_reference(1)[key] + _reference(9)[key]) / 2, abs=1e-7
        )


def test_value_sft_empty_validation(single_rank_gloo):
    worker, _ = _eval_worker([("empty", 0)], 4)
    assert worker.run_eval() == {}
    worker.eval_data_loaders = []
    assert worker.run_eval() == {}


def test_value_sft_without_process_group():
    assert not dist.is_initialized()
    worker, _ = _eval_worker([("validation", 9)], 4)
    worker.eval_data_loaders = [
        ("validation", DataLoader(_ValueDataset(9), batch_size=4))
    ]
    _assert_metrics(worker.run_eval(), 9)


def test_value_sft_trims_real_collator_output(single_rank_gloo):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from rlinf.models.embodiment.value_model.recap.data_collator import (
        ValueDataCollator,
    )
    from rlinf.models.embodiment.value_model.recap.processing import (
        ValueImageProcessor,
        ValueProcessor,
    )

    processor = ValueProcessor(
        image_processor=ValueImageProcessor(
            image_size=(4, 4), image_keys=("camera", "wrist"), do_augment=False
        ),
        tokenizer=PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
            unk_token="[UNK]",
        ),
    )
    dataset = [
        {
            "images": {"camera": torch.full((4, 4, 3), index, dtype=torch.uint8)},
            "prompt": "move object",
            "target_values": float(index),
        }
        for index in range(3)
    ]
    sampler = DistributedSampler(dataset, num_replicas=2, rank=1, shuffle=False)
    loader = DataLoader(
        dataset,
        batch_size=2,
        sampler=sampler,
        collate_fn=ValueDataCollator(processor=processor, max_length=8, train=False),
    )

    class CollatedValueModel(_ValueModel):
        def forward(self, observation, target_values):
            size = len(target_values)
            for image in observation["images"].values():
                assert image.shape == (size, 3, 4, 4)
            for mask in observation["image_masks"].values():
                assert mask.shape == (size,)
            assert observation["tokenized_prompt"].shape == (size, 8)
            assert observation["tokenized_prompt_mask"].shape == (size, 8)
            assert observation["action_mask"].shape == (size,)
            return super().forward(
                {"images": {"camera": observation["images"]["camera"][:, 0, 0, 0]}},
                target_values,
            )

    worker, _ = _eval_worker([], 2, rank=1, world_size=2)
    worker.eval_data_loaders = [("validation", loader)]
    worker.model = CollatedValueModel()
    metrics = worker.run_eval()
    assert worker.model.batch_sizes == [1]
    assert metrics["target_value_mean"] == 1.0
    assert metrics["target_value_std"] == 0.0


def _distributed_eval(rank, rendezvous, output_dir, size):
    Worker.torch_platform = torch.cpu
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        worker, model = _eval_worker(
            [("validation", size)], 2, rank=rank, world_size=2, fsdp=True
        )
        metrics = worker.run_eval()
        torch.save(
            {"metrics": metrics, "batch_sizes": model.batch_sizes},
            f"{output_dir}/{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("size", [1, 3, 9, 10, 0])
def test_value_sft_distributed_padding(tmp_path, size):
    mp.spawn(
        _distributed_eval,
        args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path), size),
        nprocs=2,
        join=True,
    )
    ranks = [
        torch.load(tmp_path / f"{rank}.pt", weights_only=True) for rank in range(2)
    ]
    assert ranks[0]["metrics"] == ranks[1]["metrics"]
    for rank in ranks:
        if size:
            _assert_metrics(rank["metrics"], size)
        else:
            assert rank["metrics"] == {}
        # Every rank forwards all padded batches, including padding-only ranks.
        assert len(rank["batch_sizes"]) == (size + 3) // 4
