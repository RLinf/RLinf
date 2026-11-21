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
import pytest

from rlinf.data.datasets.world_model import (
    NpyTrajectoryDatasetWrapper,
    get_libero_task_descriptions,
)

OBJECT_TASKS = ["push the plate", "open the middle drawer", "close the basket"]
SPATIAL_TASKS = [
    "push the bowl on the stove",
    "put the wine on the rack",
    "open the top drawer of the cabinet",
]


def _write_trajectory(path, instruction):
    frames = []
    for _ in range(4):
        frames.append(
            {
                "image": np.zeros((8, 8, 3), dtype=np.uint8),
                "instruction": instruction,
                "delta_action": np.zeros(7, dtype=np.float32),
            }
        )
    np.save(path, np.array(frames, dtype=object), allow_pickle=True)


@pytest.fixture()
def object_dataset_dir(tmp_path):
    for i, task in enumerate(OBJECT_TASKS):
        _write_trajectory(tmp_path / f"seed_0_traj_{i}.npy", task)
    return str(tmp_path)


def test_dataset_accepts_matching_expected_tasks(object_dataset_dir):
    dataset = NpyTrajectoryDatasetWrapper(
        object_dataset_dir, expected_tasks=OBJECT_TASKS
    )
    assert len(dataset) == len(OBJECT_TASKS)
    assert dataset[0]["task"] == OBJECT_TASKS[0]


def test_dataset_accepts_partial_overlap(object_dataset_dir):
    # Suites may share some tasks; only a total mismatch is an error.
    dataset = NpyTrajectoryDatasetWrapper(
        object_dataset_dir, expected_tasks=OBJECT_TASKS + SPATIAL_TASKS
    )
    assert len(dataset) == len(OBJECT_TASKS)


def test_dataset_rejects_wrong_suite(object_dataset_dir):
    # Regression test for issue #1655: initial trajectories shipped for the
    # wrong LIBERO suite must fail fast instead of silently skewing rollouts.
    with pytest.raises(ValueError, match="do not match the expected task suite"):
        NpyTrajectoryDatasetWrapper(object_dataset_dir, expected_tasks=SPATIAL_TASKS)


def test_dataset_skips_validation_without_expected_tasks(object_dataset_dir):
    dataset = NpyTrajectoryDatasetWrapper(object_dataset_dir, expected_tasks=None)
    assert len(dataset) == len(OBJECT_TASKS)


def test_get_libero_task_descriptions_returns_none_without_libero():
    # On machines without LIBERO installed the helper must degrade to None
    # so that validation is skipped rather than breaking env construction.
    result = get_libero_task_descriptions("libero_object")
    if result is not None:
        assert all(isinstance(t, str) and t for t in result)
