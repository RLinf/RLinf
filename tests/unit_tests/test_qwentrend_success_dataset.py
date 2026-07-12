import argparse
import pickle

import numpy as np

from examples.reward.preprocess_qwentrend_terminal_success_dataset import build_rows
from rlinf.data.datasets.vlm import QwenTrendProgressSFTDataset


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
        workers=2,
        seed=42,
    )


def test_success_dataset_reproduces_terminal_and_hard_negative_rules(tmp_path):
    _write_episode(tmp_path / "success.pkl", observation_count=23, success=True)
    _write_episode(tmp_path / "failure.pkl", observation_count=51)
    _write_episode(tmp_path / "partial.pkl", observation_count=20, complete=False)

    rows_by_split, stats = build_rows(_args(tmp_path))
    rows = rows_by_split["train"]

    positive = next(row for row in rows if row["answer"] == "1")
    assert positive["segment_metadata"]["target_type"] == "success_terminal"
    assert (
        positive["segment_metadata"]["start_step"],
        positive["segment_metadata"]["end_step"],
    ) == (17, 21)
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
        == 2
    )
    assert stats["complete_episodes"] == 2
    assert stats["partial_episodes"] == 1


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


def test_qwentrend_loader_slices_raw_episode_window(tmp_path):
    path = tmp_path / "episode.pkl"
    _write_episode(path, observation_count=20, success=True)

    question, answer, videos, image_data = (
        QwenTrendProgressSFTDataset._parse_raw_record(
            {
                "question": "potential",
                "answer": "1",
                "pkl_path": str(path),
                "segment_metadata": {"start_step": 10, "end_step": 14},
            },
            idx=0,
            data_root=None,
        )
    )

    assert question == "potential"
    assert answer == "1"
    assert len(videos[0]) == len(videos[1]) == 5
    assert int(np.asarray(videos[0][0])[0, 0, 0]) == 10
    assert int(np.asarray(videos[1][-1])[0, 0, 0]) == 14
    assert image_data == [str(path), str(path)]
