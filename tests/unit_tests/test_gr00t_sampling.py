# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Legal GR00T windows, distributed sampling, masks and consumed-cursor resume."""

import itertools
import json
from types import SimpleNamespace

import pytest
import torch

from rlinf.data.datasets.gr00t_n1d7.window_dataset import (
    Gr00tSftDataLoader,
    WindowDataset,
    data_metadata,
    validate_action_supervision,
)
from rlinf.data.datasets.gr00t_n1d7.window_sampling import WindowIndex, WindowSampler


@pytest.mark.parametrize("horizon", [1, 16, 40])
def test_window_index_includes_every_legal_start(horizon):
    index = WindowIndex([horizon, horizon + 2, horizon + 1], horizon)
    assert len(index) == 6
    assert [index.locate(i) for i in range(6)] == [
        (0, 0),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
    ]
    for invalid in (-1, 6):
        with pytest.raises(IndexError):
            index.locate(invalid)
    with pytest.raises(ValueError):
        WindowIndex([horizon - 1], horizon)


def test_window_rank_stream_covers_nondivisible_epochs():
    streams = [iter(WindowSampler(7, 42, rank, 4)) for rank in range(4)]
    merged = [next(streams[i % 4]) for i in range(28)]
    for epoch in range(4):
        assert sorted(i for e, i in merged if e == epoch) == list(range(7))


def test_sampler_restores_cursor_across_epochs():
    sampler = WindowSampler(7, 42, 1, 2)
    original = iter(sampler)
    list(itertools.islice(original, 5))
    state = original.state_dict()
    expected = list(itertools.islice(original, 20))
    resumed = iter(WindowSampler(7, 42, 1, 2))
    resumed.load_state_dict(state)
    assert list(itertools.islice(resumed, 20)) == expected
    with pytest.raises(ValueError):
        resumed.load_state_dict({"cursor": -1})
    with pytest.raises(ValueError):
        WindowSampler(7, 43, 1, 2).load_state_dict(sampler.state_dict())


@pytest.mark.parametrize("horizon", [1, 16, 40])
def test_supervision_mask_rejects_padded_targets(horizon):
    action = torch.zeros(40, 132)
    mask = torch.zeros_like(action)
    mask[:horizon, :12] = 1
    validate_action_supervision({"action": action, "action_mask": mask}, horizon)
    mask[0, 12] = 1
    with pytest.raises(ValueError, match="exactly"):
        validate_action_supervision({"action": action, "action_mask": mask}, horizon)
    mask[0, 12] = 0
    mask[0, 0] = 0
    with pytest.raises(ValueError, match="exactly"):
        validate_action_supervision({"action": action, "action_mask": mask}, horizon)


def test_h16_mask_rejects_supervised_future_padding():
    action = torch.zeros(40, 132)
    mask = torch.zeros_like(action)
    mask[:17, :12] = 1
    with pytest.raises(ValueError, match="exactly"):
        validate_action_supervision({"action": action, "action_mask": mask}, 16)
    with pytest.raises(ValueError, match="padded"):
        validate_action_supervision({"action": action, "action_mask": mask[:16]}, 16)


@pytest.mark.parametrize("horizon", [16, 40])
def test_dataset_metadata_records_supervised_horizon(tmp_path, horizon):
    root = tmp_path / "meta"
    root.mkdir()
    for name in (
        "info.json",
        "tasks.jsonl",
        "stats.json",
        "modality.json",
        "embodiment.json",
    ):
        (root / name).write_text("{}")
    (root / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 80}) + "\n"
    )
    metadata = data_metadata(
        {"data": {"path": str(tmp_path), "action_horizon": horizon}}
    )
    assert metadata["modality"]["action"]["delta_indices"] == list(range(horizon))
    assert len(metadata["sha256"]) == 6


@pytest.mark.parametrize("horizon", [1, 16, 40])
def test_random_window_dataset_uses_configured_horizon(horizon):
    loader = SimpleNamespace(episode_lengths=[40, 50])
    dataset = WindowDataset({"data": {"action_horizon": horizon}}, None, loader)
    assert len(dataset) == (40 - horizon + 1) + (50 - horizon + 1)


def test_window_dataset_returns_one_processed_sample(monkeypatch):
    action = torch.ones(40, 132)
    mask = torch.zeros_like(action)
    mask[:16, :12] = 1
    content = object()

    class Processor:
        train_image_transform = None

        def __call__(self, contents):
            assert contents == [{"content": content}]
            return {"action": action, "action_mask": mask}

    dataset = WindowDataset(
        {"data": {"action_horizon": 16}},
        Processor(),
        SimpleNamespace(episode_lengths=[40]),
    )
    monkeypatch.setattr(
        dataset,
        "raw_content",
        lambda index: (content, {"augmentation_seed": 42}),
    )
    sample = dataset[(0, 0)]
    assert set(sample) == {"action", "action_mask"}
    torch.testing.assert_close(sample["action"], action)
    torch.testing.assert_close(sample["action_mask"], mask)


class _IdentityDataset(torch.utils.data.Dataset):
    def __init__(self, size=8):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return torch.tensor(index)


@pytest.mark.parametrize(
    "size,global_batch_size,world_size,micro_batch_size,expected_batches",
    [
        (1, 16, 4, 2, 2),
        (7, 12, 2, 3, 2),
        (13, 12, 2, 3, 2),
        (24, 12, 2, 3, 4),
        (25, 12, 2, 3, 4),
        (129, 128, 8, 16, 1),
    ],
)
def test_loader_nominal_length_does_not_bound_sample_stream(
    size, global_batch_size, world_size, micro_batch_size, expected_batches
):
    loader = Gr00tSftDataLoader(
        _IdentityDataset(size),
        global_batch_size=global_batch_size,
        world_size=world_size,
        batch_size=micro_batch_size,
        sampler=WindowSampler(size, 42, 1, world_size),
        num_workers=0,
    )
    assert len(loader) == expected_batches
    expected = iter(WindowSampler(size, 42, 1, world_size))
    actual = iter(loader)
    for _ in range(2 * len(loader) + 1):
        batch = next(actual)
        assert batch.shape == (micro_batch_size, 2)
        torch.testing.assert_close(
            batch,
            torch.tensor([next(expected) for _ in range(micro_batch_size)]),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("workers", [0, 2])
def test_stateful_loader_restores_consumed_not_prefetched_samples(workers):
    def make():
        return Gr00tSftDataLoader(
            _IdentityDataset(7),
            global_batch_size=12,
            world_size=2,
            batch_size=3,
            sampler=WindowSampler(7, 42, 1, 2),
            num_workers=workers,
            **({"prefetch_factor": 2} if workers else {}),
        )

    loader = make()
    iterator = iter(loader)
    for _ in range(len(loader) + 1):
        assert next(iterator).shape == (3, 2)
    saved = loader.state_dict()
    expected = [next(iterator) for _ in range(4)]
    restored = make()
    restored.load_state_dict(saved)
    resumed = iter(restored)
    for want in expected:
        torch.testing.assert_close(next(resumed), want, rtol=0, atol=0)
