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

import argparse
import importlib.util
import json
import pickle
import random
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from rlinf.data.datasets.vlm import VLMTrendRewardSFTDataset
from rlinf.data.datasets.vlm_trend_success import build_rows
from rlinf.utils.state_success_value import StateSuccessValue


def _write_episode(path, *, observation_count, success=False, complete=True):
    observations = [
        {
            "main_images": np.full((8, 8, 3), step, dtype=np.uint8),
            "extra_view_images": np.full((1, 8, 8, 3), step, dtype=np.uint8),
        }
        for step in range(observation_count)
    ]
    payload = {
        "observations": observations,
        "actions": [np.zeros(1)] * (observation_count - 1),
        "success": success,
        "terminated": [False] * (observation_count - 2) + [complete],
        "truncated": [False] * (observation_count - 1),
        "infos": [
            {"success": success and step == observation_count - 1}
            for step in range(observation_count)
        ],
        "task": "test task",
    }
    with path.open("wb") as stream:
        pickle.dump(payload, stream)


def _args(root):
    return argparse.Namespace(
        raw_data_path=[str(root)],
        window_size=5,
        val_split=0.0,
        max_positive=8000,
        negative_positive_ratio=3.0,
        hard_negatives_per_episode=3,
        success_exclusion_steps=8,
        near_terminal_positives_per_episode=1,
        success_positive_lead_steps=4,
        workers=2,
        seed=42,
    )


def test_success_dataset_reproduces_terminal_and_hard_negative_rules(tmp_path):
    _write_episode(tmp_path / "success.pkl", observation_count=23, success=True)
    _write_episode(tmp_path / "failure.pkl", observation_count=51)
    _write_episode(tmp_path / "partial.pkl", observation_count=20, complete=False)

    rows_by_split, stats = build_rows(_args(tmp_path))
    rows = rows_by_split["train"]

    positive = [row for row in rows if row["answer"] == "1"]
    assert len(positive) == 2
    terminal = next(
        row
        for row in positive
        if row["segment_metadata"]["target_type"] == "success_observed"
    )
    assert (
        terminal["segment_metadata"]["start_step"],
        terminal["segment_metadata"]["end_step"],
    ) == (18, 22)
    near_terminal = next(row for row in positive if row is not terminal)
    assert near_terminal["segment_metadata"]["target_type"] == "success_near_observed"
    assert 18 <= near_terminal["segment_metadata"]["end_step"] < 22
    assert (
        sum(
            row["segment_metadata"]["target_type"] == "failure_terminal" for row in rows
        )
        == 1
    )
    assert (
        sum(
            row["segment_metadata"]["target_type"] == "nonterminal_hard_negative"
            for row in rows
        )
        == 5
    )
    assert stats["complete_episodes"] == 2
    assert stats["partial_episodes"] == 1
    assert stats["splits"]["train"]["positive"] == 2


def test_global_seed_makes_manifest_rows_deterministic(tmp_path):
    for index in range(8):
        _write_episode(
            tmp_path / f"episode_{index}.pkl",
            observation_count=51,
            success=index % 2 == 0,
        )

    first, _ = build_rows(_args(tmp_path))
    second, _ = build_rows(_args(tmp_path))

    def keys(rows):
        return [
            (
                row["source_episode_path"],
                row["segment_metadata"]["start_step"],
                row["segment_metadata"]["end_step"],
                row["answer"],
            )
            for row in rows
        ]

    assert keys(first["train"]) == keys(second["train"])


def test_vlm_trend_loader_slices_raw_episode_window(tmp_path):
    path = tmp_path / "episode.pkl"
    _write_episode(path, observation_count=20, success=True)

    question, answer, videos, image_data = VLMTrendRewardSFTDataset._parse_raw_record(
        {
            "question": "potential",
            "answer": "1",
            "pkl_path": str(path),
            "segment_metadata": {"start_step": 10, "end_step": 14},
        },
        idx=0,
        data_root=None,
    )

    assert question == "potential"
    assert answer == "1"
    assert len(videos[0]) == len(videos[1]) == 5
    assert int(np.asarray(videos[0][0])[0, 0, 0]) == 10
    assert int(np.asarray(videos[1][-1])[0, 0, 0]) == 14
    assert image_data == [str(path), str(path)]


def test_online_interval_matches_inference_window_ends(tmp_path):
    _write_episode(tmp_path / "success.pkl", observation_count=21, success=True)
    _write_episode(tmp_path / "failure.pkl", observation_count=21)
    args = _args(tmp_path)
    args.online_interval = 5

    rows_by_split, stats = build_rows(args)
    rows = rows_by_split["train"]

    assert sorted(row["segment_metadata"]["end_step"] for row in rows) == [
        5,
        5,
        10,
        10,
        15,
        15,
        20,
        20,
    ]
    assert sum(row["answer"] == "1" for row in rows) == 1
    assert stats["splits"]["train"] == {
        "positive": 1,
        "negative": 7,
        "online_interval": 5,
    }


def _load_preprocess_success_module(monkeypatch) -> ModuleType:
    script = (
        Path(__file__).resolve().parents[3]
        / "examples/reward/preprocess_vlm_trend_success_dataset.py"
    )
    module_name = "preprocess_vlm_trend_success_dataset_under_test"
    spec = importlib.util.spec_from_file_location(module_name, script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _write_potential_episode(
    path: Path, *, observation_count: int = 20, success: bool = False
) -> None:
    observations = [
        {
            "main_images": np.full((8, 8, 3), step, dtype=np.uint8),
            "extra_view_images": np.full((1, 8, 8, 3), step, dtype=np.uint8),
            "states": np.full(4, float(step), dtype=np.float32),
        }
        for step in range(observation_count)
    ]
    payload = {
        "observations": observations,
        "actions": [np.zeros(1)] * (observation_count - 1),
        "success": success,
        "task": "test task",
    }
    with path.open("wb") as stream:
        pickle.dump(payload, stream)


def _write_teacher_checkpoint(
    path: Path, *, state_dim: int = 4, history_size: int = 1
) -> None:
    model = StateSuccessValue(
        input_dim=state_dim * history_size, hidden_dim=8, num_layers=2
    )
    mean = np.zeros(state_dim * history_size, dtype=np.float32)
    std = np.ones(state_dim * history_size, dtype=np.float32)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "state_dim": state_dim,
                "history_size": history_size,
                "hidden_dim": 8,
                "num_layers": 2,
                "dropout": 0.0,
                "mean": mean.tolist(),
                "std": std.tolist(),
            },
        },
        path,
    )


def test_potential_preprocess_writes_manifests(tmp_path, monkeypatch):
    module = _load_preprocess_success_module(monkeypatch)
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    _write_potential_episode(raw_root / "success.pkl", success=True)
    _write_potential_episode(raw_root / "failure.pkl", success=False)

    ckpt = tmp_path / "teacher.pt"
    _write_teacher_checkpoint(ckpt)
    output_dir = tmp_path / "out"

    args = argparse.Namespace(
        raw_data_path=[str(raw_root)],
        output_dir=str(output_dir),
        value_checkpoint=str(ckpt),
        window_size=5,
        stride=5,
        num_bins=5,
        progress_deadband=0.03,
        progress_gap_steps=[5],
        temporal_smoothing_window=1,
        val_split=0.0,
        only_split=None,
        seed=0,
        max_episodes=None,
        score_batch_size=64,
        potential_per_bucket_train=8,
        potential_per_bucket_eval=2,
        progress_per_bucket_train=8,
        progress_per_bucket_eval=2,
        task_description=None,
        device="cpu",
    )
    metadata = module.run_potential(args)

    train_manifest = output_dir / "train" / "segments.jsonl"
    assert train_manifest.is_file()
    assert (output_dir / "dataset_info.json").is_file()
    rows = [
        json.loads(line)
        for line in train_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    assert metadata["splits"]["train"]["num_samples"] == len(rows)
    assert metadata["splits"]["eval"]["num_samples"] == 0
    sample_types = {row["segment_metadata"]["sample_type"] for row in rows}
    assert "potential" in sample_types
    assert "progress" in sample_types


def test_potential_bin_and_progress_label_helpers(monkeypatch):
    module = _load_preprocess_success_module(monkeypatch)
    assert module.potential_bin(0.0, 10) == 0
    assert module.potential_bin(0.99, 10) == 9
    assert module.progress_label(0.1, 0.03) == "up"
    assert module.progress_label(-0.1, 0.03) == "down"
    assert module.progress_label(0.01, 0.03) == "same"


def test_only_split_eval_with_zero_val_split_raises(tmp_path, monkeypatch):
    module = _load_preprocess_success_module(monkeypatch)
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    _write_potential_episode(raw_root / "ep.pkl")
    args = argparse.Namespace(
        raw_data_path=[str(raw_root)],
        val_split=0.0,
        only_split="eval",
        max_episodes=None,
    )
    with pytest.raises(ValueError, match="only-split.*eval.*val_split"):
        module._collect_pkl_files_and_splits(args, random.Random(0))


def test_validate_preprocess_args_rejects_bad_val_split(monkeypatch):
    module = _load_preprocess_success_module(monkeypatch)
    args = argparse.Namespace(
        num_bins=5,
        temporal_smoothing_window=1,
        progress_gap_steps=[5],
        val_split=1.5,
    )
    with pytest.raises(ValueError, match="val_split must be in"):
        module._validate_preprocess_args(args)
