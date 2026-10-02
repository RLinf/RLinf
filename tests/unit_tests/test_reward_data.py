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

"""Reward model data: split preprocessing and dataloader checks."""

import importlib.util
import pickle
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from rlinf.workers.reward.reward_worker import check_reward_dataloaders

_PREPROCESS_SCRIPT = (
    Path(__file__).resolve().parents[2] / "examples/reward/preprocess_reward_dataset.py"
)


@pytest.fixture(scope="module")
def preprocess():
    spec = importlib.util.spec_from_file_location(
        "preprocess_reward_dataset_under_test", _PREPROCESS_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_episodes(raw_dir: Path, successes: list[bool], num_frames: int = 4):
    """Write episode pickles in the layout produced by ``CollectEpisode``.

    The first entry is the reset frame with an empty info dict. A successful
    episode reports ``success`` on its last frame only.
    """
    raw_dir.mkdir()
    for ep_idx, success in enumerate(successes):
        observations = [
            {"main_images": torch.full((3, 8, 8), ep_idx, dtype=torch.uint8)}
            for _ in range(num_frames)
        ]
        infos = [{}] + [
            {"success": success and step == num_frames - 1}
            for step in range(1, num_frames)
        ]
        label = "success" if success else "fail"
        with open(raw_dir / f"episode_{ep_idx}_{label}.pkl", "wb") as f:
            pickle.dump({"observations": observations, "infos": infos}, f)


@pytest.mark.parametrize(
    ("successes", "seed", "empty_split"),
    [
        # No successful episode: both splits are empty, train is reported first.
        ([False] * 5, 0, "train"),
        # A single episode always goes to val, leaving train with nothing.
        ([True], 0, "train"),
        # The only successful episode lands in train under this seed.
        ([True] + [False] * 9, 0, "val"),
    ],
)
def test_preprocess_rejects_split_without_success_frames(
    preprocess, tmp_path, successes, seed, empty_split
):
    raw_dir = tmp_path / "raw"
    _write_episodes(raw_dir, successes)
    train_path = tmp_path / "out" / "train.pt"
    val_path = tmp_path / "out" / "val.pt"

    with pytest.raises(ValueError, match=f"The {empty_split} split is empty"):
        preprocess.preprocess_and_save_reward_datasets(
            raw_data_path=str(raw_dir),
            train_output_path=str(train_path),
            val_output_path=str(val_path),
            random_seed=seed,
        )

    assert not train_path.exists()
    assert not val_path.exists()


def test_preprocess_writes_splits_when_both_have_success_frames(preprocess, tmp_path):
    raw_dir = tmp_path / "raw"
    _write_episodes(raw_dir, [True] * 5)
    train_path = tmp_path / "out" / "train.pt"
    val_path = tmp_path / "out" / "val.pt"

    metadata = preprocess.preprocess_and_save_reward_datasets(
        raw_data_path=str(raw_dir),
        train_output_path=str(train_path),
        val_output_path=str(val_path),
        random_seed=0,
    )

    assert metadata["num_train_samples"] > 0
    assert metadata["num_val_samples"] > 0
    assert train_path.exists()
    assert val_path.exists()


def _loader(num_samples: int, batch_size: int, world_size: int, drop_last: bool):
    """Build a loader the way ``FSDPRewardWorker.build_dataloader`` does."""
    dataset = TensorDataset(torch.zeros(num_samples, 3), torch.zeros(num_samples))
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=0, shuffle=False
    )
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, drop_last=drop_last
    )


@pytest.mark.parametrize("num_train", [0, 6])
def test_reward_dataloader_check_rejects_train_split_without_full_batch(num_train):
    # 6 samples over 2 ranks leave 3 per rank, fewer than a micro batch of 4.
    train_loader = _loader(num_train, batch_size=4, world_size=2, drop_last=True)
    val_loader = _loader(8, batch_size=4, world_size=2, drop_last=False)
    assert len(train_loader) == 0

    with pytest.raises(ValueError, match=f"has {num_train} samples"):
        check_reward_dataloaders(
            train_loader, val_loader, world_size=2, validation_enabled=True
        )


def test_reward_dataloader_check_rejects_empty_val_split():
    train_loader = _loader(8, batch_size=4, world_size=2, drop_last=True)
    val_loader = _loader(0, batch_size=4, world_size=2, drop_last=False)

    with pytest.raises(ValueError, match="validation split is empty"):
        check_reward_dataloaders(
            train_loader, val_loader, world_size=2, validation_enabled=True
        )


def test_reward_dataloader_check_accepts_one_batch_per_rank():
    train_loader = _loader(8, batch_size=4, world_size=2, drop_last=True)
    val_loader = _loader(1, batch_size=4, world_size=2, drop_last=False)

    check_reward_dataloaders(
        train_loader, val_loader, world_size=2, validation_enabled=True
    )


def test_reward_dataloader_check_allows_empty_val_split_without_validation():
    # runner.val_check_interval <= 0 disables validation, so val data is unused.
    train_loader = _loader(8, batch_size=4, world_size=2, drop_last=True)
    val_loader = _loader(0, batch_size=4, world_size=2, drop_last=False)

    check_reward_dataloaders(
        train_loader, val_loader, world_size=2, validation_enabled=False
    )
