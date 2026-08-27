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

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from examples.reward.extract_vlm_trend_potential_features import (
    extract_features,
    feature_rows,
)
from examples.reward.extract_vlm_trend_potential_features import (
    parse_args as parse_extract_args,
)
from examples.reward.preprocess_vlm_trend_potential import parse_args, run_potential
from examples.reward.preprocess_vlm_trend_reward_dataset import main as trend_main
from examples.reward.vlm_trend_data import load_dual_view_sample
from rlinf.models.embodiment.modules.utils import make_mlp

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_episode(path: Path, n_obs: int = 12) -> None:
    observations = [
        {
            "states": torch.full((2,), float(index) / n_obs),
            "main_images": np.zeros((2, 2, 3), dtype=np.uint8),
            "extra_view_images": np.zeros((1, 2, 2, 3), dtype=np.uint8),
        }
        for index in range(n_obs)
    ]
    episode = {
        "observations": observations,
        "actions": [torch.zeros(1) for _ in range(n_obs - 1)],
        "infos": [{"success": False} for _ in range(n_obs)],
        "success": False,
        "task": "pick cube",
    }
    with path.open("wb") as stream:
        pickle.dump(episode, stream)


def _write_teacher(path: Path) -> None:
    state_dim = 2
    history_size = 1
    hidden_dim = 4
    num_layers = 1
    model = nn.Sequential(
        *make_mlp(
            state_dim * history_size,
            [hidden_dim] * num_layers + [1],
            act_builder=nn.SiLU,
            last_act=False,
            use_layer_norm=True,
        )
    )
    torch.save(
        {
            "config": {
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "state_dim": state_dim,
                "history_size": history_size,
                "dropout": 0.0,
                "mean": np.zeros(state_dim * history_size, dtype=np.float32),
                "std": np.ones(state_dim * history_size, dtype=np.float32),
            },
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def _dual_view_pkl(path: Path, frames: int) -> None:
    with path.open("wb") as stream:
        pickle.dump(
            {
                "main_frames": [
                    np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(frames)
                ],
                "extra_view_frames": [
                    np.ones((2, 2, 3), dtype=np.uint8) for _ in range(frames)
                ],
            },
            stream,
        )


def _manifest_row(
    pkl_path: Path,
    sample_type: str,
    *,
    window_size: int,
    answer: str,
    teacher_value: float,
    teacher_delta: float = 0.0,
    source_name: str = "episode.pkl",
) -> dict[str, Any]:
    return {
        "task": "pick cube",
        "answer": answer,
        "pkl_path": str(pkl_path),
        "source_episode_path": str(pkl_path.parent / source_name),
        "source_run": "run0",
        "segment_metadata": {
            "sample_type": sample_type,
            "window_size": window_size,
            "end_step": 8,
            "success": True,
        },
        "supervision": {
            "teacher_value": teacher_value,
            "teacher_delta": teacher_delta,
        },
    }


@pytest.mark.parametrize(
    ("mode", "match"),
    [
        ("terminal_success", "moved to"),
        ("potential", "moved to"),
        ("features", "moved to"),
        ("trend", "Omit --mode"),
        ("unknown", "Unknown --mode"),
    ],
)
def test_trend_preprocess_rejects_moved_modes(mode: str, match: str) -> None:
    with pytest.raises(SystemExit, match=match):
        trend_main(["--mode", mode, "--raw-data-path", "unused"])


def test_potential_preprocess_writes_train_manifest(tmp_path) -> None:
    data_dir = tmp_path / "collected"
    data_dir.mkdir()
    _write_episode(data_dir / "episode.pkl")
    checkpoint = tmp_path / "teacher.pt"
    _write_teacher(checkpoint)
    output_dir = tmp_path / "potential_data"

    metadata = run_potential(
        parse_args(
            [
                "--raw-data-path",
                str(data_dir),
                "--output-dir",
                str(output_dir),
                "--value-checkpoint",
                str(checkpoint),
                "--window-size",
                "5",
                "--stride",
                "5",
                "--val-split",
                "0",
                "--device",
                "cpu",
                "--potential-samples-train",
                "16",
                "--progress-samples-train",
                "16",
            ]
        )
    )

    train = metadata["splits"]["train"]
    assert train["num_samples"] >= 2
    assert train["sample_type_counts"]["potential"] >= 1
    assert train["sample_type_counts"]["progress"] >= 1
    manifest = Path(train["manifest"])
    assert manifest.is_file()
    row = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    main_frames, extra_frames = load_dual_view_sample(row, 0)
    assert row["answer"]
    assert len(main_frames) >= 5
    assert len(extra_frames) == len(main_frames)


def test_load_dual_view_sample_from_manifest_pkl(tmp_path) -> None:
    pkl_path = tmp_path / "sample.pkl"
    _dual_view_pkl(pkl_path, frames=5)
    row = _manifest_row(
        pkl_path, "potential", window_size=5, answer="3", teacher_value=0.4
    )

    main_frames, extra_frames = load_dual_view_sample(row, 0)

    assert len(main_frames) == 5
    assert len(extra_frames) == 5


def test_load_dual_view_sample_requires_pkl_path() -> None:
    with pytest.raises(ValueError, match="missing pkl_path"):
        load_dual_view_sample({"answer": "1"}, 3)


def test_feature_rows_shards_by_source_hash(tmp_path) -> None:
    pkl_path = tmp_path / "sample.pkl"
    _dual_view_pkl(pkl_path, frames=5)
    rows = [
        _manifest_row(
            pkl_path,
            "potential",
            window_size=5,
            answer="1",
            teacher_value=0.1,
            source_name=f"episode_{index}.pkl",
        )
        for index in range(4)
    ]
    manifest = tmp_path / "segments.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    common = [
        "--model-path",
        "unused",
        "--checkpoint",
        str(tmp_path / "actor" / "model_state_dict" / "full_weights.pt"),
        "--manifest",
        str(manifest),
        "--output",
        str(tmp_path / "out.pt"),
        "--sample-type",
        "potential",
        "--world-size",
        "2",
    ]

    rank0 = feature_rows(parse_extract_args([*common, "--rank", "0"]))
    rank1 = feature_rows(parse_extract_args([*common, "--rank", "1"]))

    assert rank0 and rank1
    assert {row["source_episode_path"] for row in rank0}.isdisjoint(
        {row["source_episode_path"] for row in rank1}
    )
    assert len(rank0) + len(rank1) == 4


def test_extract_potential_features_without_vlm(tmp_path, monkeypatch) -> None:
    pkl_path = tmp_path / "sample.pkl"
    _dual_view_pkl(pkl_path, frames=5)
    row = _manifest_row(
        pkl_path, "potential", window_size=5, answer="3", teacher_value=0.4
    )
    manifest = tmp_path / "segments.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

    def fake_encode(model, prompts, videos, video_fps):
        assert len(prompts) == 1
        assert len(videos[0][0]) == 5
        assert video_fps == 24.0
        return torch.ones(1, 3)

    monkeypatch.setattr(
        "examples.reward.extract_vlm_trend_potential_features.encode_feature_batch",
        fake_encode,
    )
    args = parse_extract_args(
        [
            "--model-path",
            "unused",
            "--checkpoint",
            str(tmp_path / "actor" / "model_state_dict" / "full_weights.pt"),
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "out.pt"),
            "--sample-type",
            "potential",
        ]
    )
    payload = extract_features(object(), feature_rows(args), args)

    assert payload["features"].shape == (1, 3)
    assert payload["targets"].tolist() == pytest.approx([0.4])


def test_extract_progress_features_without_vlm(tmp_path, monkeypatch) -> None:
    pkl_path = tmp_path / "sample.pkl"
    _dual_view_pkl(pkl_path, frames=10)
    row = _manifest_row(
        pkl_path,
        "progress",
        window_size=5,
        answer="up",
        teacher_value=0.6,
        teacher_delta=0.2,
    )
    manifest = tmp_path / "segments.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

    def fake_encode(model, prompts, videos, video_fps):
        assert len(prompts) == 2
        assert len(videos) == 2
        return torch.ones(2, 4)

    monkeypatch.setattr(
        "examples.reward.extract_vlm_trend_potential_features.encode_feature_batch",
        fake_encode,
    )
    args = parse_extract_args(
        [
            "--model-path",
            "unused",
            "--checkpoint",
            str(tmp_path / "full_weights.pt"),
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "out.pt"),
            "--sample-type",
            "progress",
        ]
    )
    payload = extract_features(object(), feature_rows(args), args)

    assert payload["features"].shape == (1, 2, 4)
    assert payload["labels"] == ["up"]
    assert payload["teacher_deltas"].tolist() == pytest.approx([0.2])


def test_recipe_docs_use_sft_full_weights_path() -> None:
    files = [
        REPO_ROOT / "docs/source-en/rst_source/extending/reward_model.rst",
        REPO_ROOT / "docs/source-zh/rst_source/extending/reward_model.rst",
        REPO_ROOT / "examples/reward/extract_vlm_trend_potential_features.py",
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "actor/model_state_dict/full_weights.pt" in text
        assert "/actor/lora_adapter" not in text

    yaml_text = (
        REPO_ROOT
        / "examples/embodiment/config/maniskill_ppo_mlp_vlm_trend_success_potential.yaml"
    ).read_text(encoding="utf-8")
    assert "full_weights.pt" in yaml_text
    assert "lora_adapter" not in yaml_text
