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

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf

import examples.offline_rl.advantage_labeling.recap.process.compute_returns as returns_module
from examples.offline_rl.advantage_labeling.recap.process.compute_returns import (
    _process_single_parquet,
    compute_hitl_aware_returns_for_episode,
    compute_returns_for_episode,
)


def test_successful_episode_split_at_first_teleop_frame() -> None:
    returns, rewards = compute_hitl_aware_returns_for_episode(
        episode_length=6,
        is_success=True,
        teleop_mask=np.asarray([0, 0, 0, 1, 1, 0], dtype=np.int64),
        gamma=1.0,
        failure_reward=-300.0,
    )

    np.testing.assert_array_equal(
        rewards,
        np.asarray([-1.0, -1.0, -300.0, -1.0, -1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        returns,
        np.asarray([-302.0, -301.0, -300.0, -2.0, -1.0, 0.0], dtype=np.float32),
    )


def test_successful_episode_without_teleop_uses_standard_returns() -> None:
    hitl_returns, hitl_rewards = compute_hitl_aware_returns_for_episode(
        episode_length=4,
        is_success=True,
        teleop_mask=np.zeros(4, dtype=np.int64),
        gamma=1.0,
        failure_reward=-300.0,
    )
    normal_returns, normal_rewards = compute_returns_for_episode(
        episode_length=4,
        is_success=True,
        gamma=1.0,
        failure_reward=-300.0,
    )

    np.testing.assert_array_equal(hitl_returns, normal_returns)
    np.testing.assert_array_equal(hitl_rewards, normal_rewards)


def test_successful_episode_split_at_zero_uses_standard_returns() -> None:
    hitl_returns, hitl_rewards = compute_hitl_aware_returns_for_episode(
        episode_length=3,
        is_success=True,
        teleop_mask=np.asarray([1, 1, 0], dtype=np.int64),
        gamma=1.0,
        failure_reward=-300.0,
    )
    normal_returns, normal_rewards = compute_returns_for_episode(
        episode_length=3,
        is_success=True,
        gamma=1.0,
        failure_reward=-300.0,
    )

    np.testing.assert_array_equal(hitl_returns, normal_returns)
    np.testing.assert_array_equal(hitl_rewards, normal_rewards)


def test_failed_episode_with_teleop_remains_failed() -> None:
    hitl_returns, hitl_rewards = compute_hitl_aware_returns_for_episode(
        episode_length=4,
        is_success=False,
        teleop_mask=np.asarray([0, 1, 1, 0], dtype=np.int64),
        gamma=1.0,
        failure_reward=-300.0,
    )
    normal_returns, normal_rewards = compute_returns_for_episode(
        episode_length=4,
        is_success=False,
        gamma=1.0,
        failure_reward=-300.0,
    )

    np.testing.assert_array_equal(hitl_returns, normal_returns)
    np.testing.assert_array_equal(hitl_rewards, normal_rewards)


def test_teleop_mask_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="does not match episode length"):
        compute_hitl_aware_returns_for_episode(
            episode_length=4,
            is_success=True,
            teleop_mask=np.asarray([0, 1], dtype=np.int64),
            gamma=1.0,
            failure_reward=-300.0,
        )


def test_invalid_teleop_mask_shape_raises() -> None:
    with pytest.raises(ValueError, match=r"1-D or shape \(N, 1\)"):
        compute_hitl_aware_returns_for_episode(
            episode_length=2,
            is_success=True,
            teleop_mask=np.ones((2, 2), dtype=np.int64),
            gamma=1.0,
            failure_reward=-300.0,
        )


def test_process_single_parquet_accepts_lerobot_column_teleop_mask(tmp_path) -> None:
    pq_file = tmp_path / "episode.parquet"
    table = pa.table(
        {
            "episode_index": pa.array([0, 0, 0, 0, 0, 0], type=pa.int64()),
            "frame_index": pa.array([0, 1, 2, 3, 4, 5], type=pa.int64()),
            "is_success": pa.array([True, True, True, True, True, True]),
            "teleop_mask": pa.array([[0], [0], [0], [1], [1], [0]]),
            "task_index": pa.array([0, 0, 0, 0, 0, 0], type=pa.int64()),
            "task": pa.array(["pick"] * 6),
        }
    )
    pq.write_table(table, pq_file)

    result = _process_single_parquet(
        str(pq_file),
        dataset_type="rollout",
        gamma=1.0,
        failure_reward=-300.0,
        tasks={},
        hitl_aware_returns=True,
    )

    np.testing.assert_array_equal(
        result.column("reward").to_numpy(),
        np.asarray([-1.0, -1.0, -300.0, -1.0, -1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        result.column("return").to_numpy(),
        np.asarray([-302.0, -301.0, -300.0, -2.0, -1.0, 0.0], dtype=np.float32),
    )
    assert result.column("done").to_pylist() == [
        False,
        False,
        True,
        False,
        False,
        True,
    ]


def test_process_single_parquet_normalizes_singleton_list_is_success(
    tmp_path,
) -> None:
    pq_file = tmp_path / "episode.parquet"
    table = pa.table(
        {
            "episode_index": pa.array([0, 0, 0], type=pa.int64()),
            "frame_index": pa.array([0, 1, 2], type=pa.int64()),
            "is_success": pa.array([[False], [False], [False]]),
            "task_index": pa.array([0, 0, 0], type=pa.int64()),
            "task": pa.array(["pick"] * 3),
        }
    )
    pq.write_table(table, pq_file)

    result = _process_single_parquet(
        str(pq_file),
        dataset_type="rollout",
        gamma=1.0,
        failure_reward=-300.0,
        tasks={},
        hitl_aware_returns=False,
    )

    np.testing.assert_array_equal(
        result.column("reward").to_numpy(),
        np.asarray([-1.0, -1.0, -300.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        result.column("return").to_numpy(),
        np.asarray([-302.0, -301.0, -300.0], dtype=np.float32),
    )
    assert result.column("done").to_pylist() == [False, False, True]


def test_process_single_parquet_missing_teleop_mask_warns_and_falls_back(
    tmp_path, caplog
) -> None:
    pq_file = tmp_path / "episode.parquet"
    table = pa.table(
        {
            "episode_index": pa.array([0, 0, 0, 0], type=pa.int64()),
            "frame_index": pa.array([0, 1, 2, 3], type=pa.int64()),
            "is_success": pa.array([True, True, True, True]),
            "task_index": pa.array([0, 0, 0, 0], type=pa.int64()),
            "task": pa.array(["pick"] * 4),
        }
    )
    pq.write_table(table, pq_file)

    with caplog.at_level("WARNING"):
        result = _process_single_parquet(
            str(pq_file),
            dataset_type="rollout",
            gamma=1.0,
            failure_reward=-300.0,
            tasks={},
            hitl_aware_returns=True,
        )

    assert "teleop_mask is missing" in caplog.text
    np.testing.assert_array_equal(
        result.column("reward").to_numpy(),
        np.asarray([-1.0, -1.0, -1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        result.column("return").to_numpy(),
        np.asarray([-3.0, -2.0, -1.0, 0.0], dtype=np.float32),
    )


def test_compute_returns_uses_per_entry_hitl_aware_override(
    tmp_path, monkeypatch
) -> None:
    first_dataset = tmp_path / "first"
    second_dataset = tmp_path / "second"
    first_dataset.mkdir()
    second_dataset.mkdir()
    captured_hitl_values = []

    def fake_process_dataset(**kwargs):
        captured_hitl_values.append(kwargs["hitl_aware_returns"])
        return {"return": {"min": 0.0, "max": 0.0}, "reward": {}}

    monkeypatch.setattr(returns_module, "process_dataset", fake_process_dataset)
    cfg = OmegaConf.create(
        {
            "data": {
                "data_root": None,
                "train_data_paths": [
                    {
                        "dataset_path": str(first_dataset),
                        "type": "rollout",
                        "hitl_aware_returns": True,
                    },
                    {
                        "dataset_path": str(second_dataset),
                        "type": "rollout",
                    },
                ],
                "dataset_type": "rollout",
                "gamma": 1.0,
                "failure_reward": -300.0,
                "hitl_aware_returns": False,
                "num_workers": 1,
                "tag": None,
            }
        }
    )

    returns_module.compute_returns(cfg)

    assert captured_hitl_values == [True, False]
