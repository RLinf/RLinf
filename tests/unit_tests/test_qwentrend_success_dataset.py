import pickle

import numpy as np

from examples.reward.preprocess_qwentrend_success_dataset import build_samples


def _write_episode(path, *, steps, success):
    observations = [
        {
            "main_images": np.zeros((8, 8, 3), dtype=np.uint8),
            "extra_view_images": np.zeros((1, 8, 8, 3), dtype=np.uint8),
        }
        for _ in range(steps)
    ]
    with path.open("wb") as stream:
        pickle.dump({"observations": observations, "success": success}, stream)


def test_success_dataset_uses_terminal_and_hard_negative_windows(tmp_path):
    success_path = tmp_path / "success.pkl"
    failure_path = tmp_path / "failure.pkl"
    _write_episode(success_path, steps=50, success=True)
    _write_episode(failure_path, steps=50, success=False)

    samples = build_samples(
        [success_path, failure_path],
        window_size=5,
        min_failure_steps=50,
        hard_negative_margin=5,
        task_description="test task",
    )

    assert [sample["answer"] for sample in samples].count("1") == 1
    assert [sample["answer"] for sample in samples].count("0") == 3
    assert {sample["target_type"] for sample in samples} == {
        "success_terminal",
        "failure_terminal",
        "nonterminal_hard_negative",
    }
    assert all(len(sample["main_frames"]) == 5 for sample in samples)


def test_success_dataset_drops_short_failure_terminal(tmp_path):
    path = tmp_path / "short_failure.pkl"
    _write_episode(path, steps=20, success=False)

    samples = build_samples(
        [path],
        window_size=5,
        min_failure_steps=50,
        hard_negative_margin=5,
        task_description="test task",
    )

    assert samples == []
