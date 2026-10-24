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

import io

import numpy as np
import pytest
from PIL import Image

from toolkits.lerobot.convert_bench2dex_to_lerobot import (
    active_joint_indices,
    decode_rgb,
    valid_segments,
    validate_timing,
)


@pytest.fixture
def replay(tmp_path):
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "episode.hdf5", "w") as source:
        source["meta/robot_key"] = "test_robot"
        source["robot/joint_names"] = np.array(
            ["right", "passive", "left"], dtype=h5py.string_dtype()
        )
        source["robot/qpos"] = np.arange(30, dtype=np.float32).reshape(10, 3)
        source["action/commanded"] = source["robot/qpos"][:] + 0.25
        source["frame_valid"] = np.ones(10, dtype=bool)
        source["action/action_valid"] = np.ones(10, dtype=bool)
        source["time/sim_step"] = np.arange(10) * 3
        source["meta/physics_dt"] = 1 / 60
        source["meta/effective_fps"] = 20.0
        source["meta/homing_start_sim_step"] = -1
        yield source


def test_active_joints_preserve_runtime_order(replay):
    maps = {
        "robots": {
            "test_robot": {
                "full_joint_names": ["left", "right", "passive"],
                "active_joint_names": ["left", "right"],
                "active_dof": 2,
            }
        }
    }
    robot, names, indices = active_joint_indices(replay, maps)
    assert robot == "test_robot"
    assert [names[i] for i in indices] == ["right", "left"]
    np.testing.assert_array_equal(replay["robot/qpos"][0][indices], [0, 2])
    del replay["action/commanded"]
    replay["action/commanded"] = np.zeros((10, 2))
    with pytest.raises(ValueError, match="one column per runtime joint"):
        active_joint_indices(replay, maps)


def test_filter_splits_invalid_gaps_and_excludes_homing(replay):
    replay["frame_valid"][2] = False
    replay["action/action_valid"][4] = False
    replay["robot/qpos"][6, 0] = np.nan
    replay["meta/homing_start_sim_step"][()] = 24
    segments = valid_segments(replay)
    assert [frames.tolist() for frames in segments] == [[0, 1], [3], [5], [7]]
    validate_timing(replay, segments, 20)


def test_no_homing_marker_value_retains_full_episode(replay):
    assert [frames.tolist() for frames in valid_segments(replay)] == [list(range(10))]
    del replay["meta/homing_start_sim_step"]
    with pytest.raises(ValueError, match="homing_start_sim_step"):
        valid_segments(replay)


def test_invalid_arrays_are_rejected_and_empty_segments_are_supported(replay):
    replay["frame_valid"][:] = False
    assert valid_segments(replay) == []
    del replay["frame_valid"]
    replay["frame_valid"] = np.ones(9)
    with pytest.raises(ValueError, match="validity flags"):
        valid_segments(replay)


def test_timing_preserves_recorded_20_hz_and_rejects_resampling(replay):
    segments = valid_segments(replay)
    validate_timing(replay, segments, 20)
    with pytest.raises(ValueError, match="effective_fps"):
        validate_timing(replay, segments, 30)
    replay["time/sim_step"][4] += 1
    with pytest.raises(ValueError, match="resampling is required"):
        validate_timing(replay, segments, 20)


def test_decode_rgb_keeps_pixels_and_resolution(replay):
    import h5py

    pixels = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    replay["raw"] = pixels[None]
    np.testing.assert_array_equal(decode_rgb(replay["raw"], 0), pixels)
    encoded = io.BytesIO()
    Image.fromarray(pixels).save(encoded, format="JPEG")
    jpeg = encoded.getvalue()
    dataset = replay.create_dataset("jpeg", (1,), dtype=h5py.vlen_dtype(np.uint8))
    dataset[0] = np.frombuffer(jpeg, dtype=np.uint8)
    with Image.open(io.BytesIO(jpeg)) as expected:
        np.testing.assert_array_equal(decode_rgb(dataset, 0), np.asarray(expected))
