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

"""LeRobot archive schema and frame ownership contracts."""

import copy
import json

import numpy as np
import pytest
import torch

from rlinf.data.datasets.dagger.dataset import RollingLeRobotDataset
from rlinf.data.storage.lerobot.writer import LeRobotDatasetWriter


@pytest.mark.parametrize("with_images", [False, True])
def test_episode_roundtrip_preserves_scalar_schema_and_shared_frames(
    tmp_path, with_images
):
    pytest.importorskip("lerobot")
    parquet = pytest.importorskip("pyarrow.parquet")
    root = tmp_path / "archive"
    writer = LeRobotDatasetWriter()
    writer.create(
        repo_id=str(root),
        has_image=with_images,
        image_shape=(32, 32, 3),
        wrist_image_keys={"wrist_image": (32, 32, 3)} if with_images else None,
        has_segment_id=True,
        image_writer_processes=0,
        image_writer_threads=0,
    )
    frames = []
    for step in range(3):
        frame = {
            "state": np.arange(8, dtype=np.float32) + step,
            "actions": np.arange(7, dtype=np.float32) + step,
            "done": np.array([step == 2], dtype=bool),
            "is_success": np.array([True], dtype=bool),
            "intervene_flag": np.array([step == 1], dtype=bool),
            "segment_id": np.array([step], dtype=np.uint8),
            "task": "pick up the cube",
        }
        if with_images:
            frame["image"] = np.full((32, 32, 3), step * 32, dtype=np.uint8)
            frame["wrist_image"] = np.full((32, 32, 3), step * 16, dtype=np.uint8)
        frames.append(frame)
    before = copy.deepcopy(frames)
    try:
        writer.add_episode(frames)
        writer.add_episode(frames)
        assert writer.dataset.meta.total_episodes == 2
        assert writer.dataset.meta.total_frames == 6
    finally:
        writer.finalize()

    files = sorted(root.rglob("*.parquet"))
    assert files
    table = parquet.read_table(files)
    assert table.num_rows == 6
    assert str(table.schema.field("segment_id").type) == "uint8"
    assert table.column("segment_id").to_pylist() == [0, 1, 2] * 2
    assert table.column("done").to_pylist() == [False, False, True] * 2
    assert table.column("intervene_flag").to_pylist() == [False, True, False] * 2
    assert (
        table.column("actions").to_pylist()
        == [frame["actions"].tolist() for frame in before] * 2
    )
    assert (
        table.column("state").to_pylist()
        == [frame["state"].tolist() for frame in before] * 2
    )
    if with_images:
        assert {"image", "wrist_image"}.issubset(table.column_names)
        assert all(row["bytes"] for row in table.column("image").to_pylist())
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2
    assert info["features"]["segment_id"]["shape"] == [1]
    for actual, expected in zip(frames, before, strict=True):
        assert actual.keys() == expected.keys()
        for key in actual:
            if isinstance(expected[key], np.ndarray):
                np.testing.assert_array_equal(actual[key], expected[key])
                assert actual[key].shape == expected[key].shape
                assert actual[key].dtype == expected[key].dtype
            else:
                assert actual[key] == expected[key]

    if with_images:
        opts = {
            "root_dir": tmp_path,
            "chunk_size": 2,
            "vision_horizon": 2,
            "window_size": 5,
            "in_memory_mode": True,
        }
        legacy = RollingLeRobotDataset(**opts)
        live = RollingLeRobotDataset(**opts, keep_uint8_images=True)
        resumed = RollingLeRobotDataset(**opts, keep_uint8_images=True)
        for dataset in (legacy, live):
            dataset.append_episode_to_memory(root, frames)
            dataset.append_episode_to_memory(root, frames)
        resumed.publish_staged_resume_shards(
            resumed.load_archived_shards_staged([root])
        )

        image_keys = ("image", "wrist_image", "vision_images", "vision_wrist_images")
        for dataset in (live, resumed):
            assert len(dataset) == len(legacy) == 5
            for index in range(len(dataset)):
                sample, expected = dataset[index], legacy[index]
                for key in image_keys:
                    assert sample[key].dtype == torch.uint8
                    assert torch.equal(
                        sample[key], (expected[key] * 255).to(torch.uint8)
                    )
                for key in ("actions", "actions_is_pad"):
                    assert torch.equal(sample[key], expected[key])
                assert sample["vision_valid"] == expected["vision_valid"]
                assert sample["task"] == expected["task"]

        sampled = live[2]
        assert sampled["vision_valid"]
        sampled["vision_images"].fill_(255)
        assert torch.equal(live[2]["vision_images"], resumed[2]["vision_images"])


@pytest.mark.parametrize("horizon", [2, 16])
@pytest.mark.parametrize("require_intervene", [False, True])
def test_full_training_windows_match_after_append_resume_and_eviction(
    tmp_path, horizon, require_intervene
):
    pytest.importorskip("lerobot")
    root = tmp_path / "archive"
    writer = LeRobotDatasetWriter()
    writer.create(
        repo_id=str(root),
        has_image=True,
        image_shape=(32, 32, 3),
        image_writer_processes=0,
        image_writer_threads=0,
    )
    opts = {
        "root_dir": tmp_path,
        "chunk_size": horizon,
        "vision_horizon": horizon,
        "require_full_window": True,
        "require_all_intervene": require_intervene,
        "window_size": 5,
        "in_memory_mode": True,
        "keep_uint8_images": True,
    }
    live = RollingLeRobotDataset(**opts)
    expected_starts = []
    try:
        for episode, length in enumerate([horizon, horizon + 4, horizon + 4, horizon]):
            frames = [
                {
                    "state": np.zeros(8, dtype=np.float32),
                    "actions": np.full(7, episode * 100 + step, dtype=np.float32),
                    "image": np.full((32, 32, 3), step, dtype=np.uint8),
                    "done": np.array([step == length - 1], dtype=bool),
                    "is_success": np.array([True], dtype=bool),
                    "intervene_flag": np.array([True], dtype=bool),
                    "task": "pick up the cube",
                }
                for step in range(length)
            ]
            writer.add_episode(frames)
            live.append_episode_to_memory(root, frames)
            expected_starts.extend(
                episode * 100 + start for start in range(max(0, length - horizon))
            )
    finally:
        writer.finalize()

    resumed = RollingLeRobotDataset(**opts)
    resumed.publish_staged_resume_shards(resumed.load_archived_shards_staged([root]))
    assert len(live) == len(resumed) == 5
    for index, expected_start in enumerate(expected_starts[-5:]):
        current, restored = live[index], resumed[index]
        assert float(current["actions"][0, 0]) == expected_start
        assert current["vision_images"].shape[:2] == (horizon + 1, 3)
        assert current["vision_valid"] and restored["vision_valid"]
        assert not current["actions_is_pad"].any()
        for key in ("actions", "actions_is_pad", "vision_images"):
            assert torch.equal(current[key], restored[key])


def test_full_window_filter_combines_with_expert_label_filter(tmp_path):
    pytest.importorskip("datasets")
    frames = [
        {
            "state": np.zeros(8, dtype=np.float32),
            "actions": np.full(7, step, dtype=np.float32),
            "image": np.full((32, 32, 3), step, dtype=np.uint8),
            "done": np.array([step == 5], dtype=bool),
            "is_success": np.array([True], dtype=bool),
            "intervene_flag": np.array([step != 2], dtype=bool),
            "task": "pick up the cube",
        }
        for step in range(6)
    ]
    dataset = RollingLeRobotDataset(
        tmp_path,
        chunk_size=2,
        vision_horizon=2,
        require_full_window=True,
        require_all_intervene=True,
        in_memory_mode=True,
    )
    dataset.append_episode_to_memory(tmp_path / "live", frames)
    # Starts 1 and 2 contain an unlabelled action; starts 4 and 5 lack a future frame.
    assert len(dataset) == 2
    assert [float(dataset[i]["actions"][0, 0]) for i in range(2)] == [0.0, 3.0]
