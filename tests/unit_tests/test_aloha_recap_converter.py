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

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from examples.offline_rl.data.convert_aloha_hdf5_to_lerobot_v21 import (
    _aloha_features,
    _build_hil_segments_entry,
    _episode_success_array,
    _load_episode_payload,
    _read_images,
    _read_scalar_reward,
    _teleop_mask,
)


def _write_episode(
    path: Path,
    reward: float | np.ndarray = 1.0,
    width: int = 14,
) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "action",
            data=np.arange(4 * width, dtype=np.float32).reshape(4, width),
        )
        f.create_dataset("reward", data=np.asarray(reward, dtype=np.float32))
        f.create_dataset(
            "teleop_segments",
            data=np.asarray([[1, 3]], dtype=np.int64),
        )

        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=np.ones((4, width), dtype=np.float32))
        obs.create_dataset("qvel", data=np.full((4, width), 2.0, dtype=np.float32))
        obs.create_dataset("effort", data=np.full((4, width), 3.0, dtype=np.float32))

        images = obs.create_group("images")
        for name in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            images.create_dataset(
                name,
                data=np.zeros((4, 8, 8, 3), dtype=np.uint8),
            )


def test_teleop_mask_maps_half_open_segments() -> None:
    segments = np.asarray([[1, 3], [4, 5]], dtype=np.int64)
    mask = _teleop_mask(segments, num_frames=6, episode_name="episode_0.hdf5")

    np.testing.assert_array_equal(mask, np.asarray([0, 1, 1, 0, 1, 0], dtype=np.int64))


def test_teleop_mask_rejects_invalid_segment() -> None:
    with pytest.raises(ValueError, match="invalid teleop segment"):
        _teleop_mask(
            np.asarray([[2, 7]], dtype=np.int64),
            num_frames=6,
            episode_name="episode_bad.hdf5",
        )


def test_episode_success_array_uses_scalar_reward() -> None:
    np.testing.assert_array_equal(
        _episode_success_array(1.0, num_frames=3),
        np.asarray([True, True, True], dtype=bool),
    )
    np.testing.assert_array_equal(
        _episode_success_array(0.0, num_frames=3),
        np.asarray([False, False, False], dtype=bool),
    )


def test_aloha_features_include_required_recap_columns() -> None:
    features = _aloha_features(
        image_shape=(8, 8, 3),
        include_velocity=True,
        include_effort=True,
    )

    assert features["observation.images.cam_high"]["dtype"] == "image"
    assert features["observation.images.cam_left_wrist"]["dtype"] == "image"
    assert features["observation.images.cam_right_wrist"]["dtype"] == "image"
    assert features["observation.state"]["shape"] == (14,)
    assert features["observation.state"]["names"] == ["state"]
    assert features["observation.velocity"]["shape"] == (14,)
    assert features["observation.velocity"]["names"] == ["velocity"]
    assert features["observation.effort"]["shape"] == (14,)
    assert features["observation.effort"]["names"] == ["effort"]
    assert features["action"]["shape"] == (14,)
    assert features["action"]["names"] == ["action"]
    assert features["is_success"]["dtype"] == "bool"
    assert features["is_success"]["shape"] == (1,)
    assert features["is_success"]["names"] == ["is_success"]
    assert features["teleop_mask"]["dtype"] == "int64"
    assert features["teleop_mask"]["shape"] == (1,)
    assert features["teleop_mask"]["names"] == ["teleop_mask"]
    assert features["observation.images.cam_high"]["shape"] == (8, 8, 3)
    assert features["observation.images.cam_high"]["names"] == [
        "height",
        "width",
        "channel",
    ]
    assert "task" not in features


def test_read_scalar_reward_accepts_scalar_and_singleton(tmp_path: Path) -> None:
    scalar_path = tmp_path / "scalar_reward.hdf5"
    singleton_path = tmp_path / "singleton_reward.hdf5"
    _write_episode(scalar_path, reward=1.0)
    _write_episode(singleton_path, reward=np.asarray([0.5], dtype=np.float32))

    with h5py.File(scalar_path, "r") as ep:
        assert _read_scalar_reward(ep, "scalar_reward.hdf5") == 1.0
    with h5py.File(singleton_path, "r") as ep:
        assert _read_scalar_reward(ep, "singleton_reward.hdf5") == pytest.approx(0.5)


def test_read_scalar_reward_rejects_non_scalar(tmp_path: Path) -> None:
    episode_path = tmp_path / "bad_reward.hdf5"
    _write_episode(episode_path, reward=np.asarray([0.0, 1.0], dtype=np.float32))

    with h5py.File(episode_path, "r") as ep:
        with pytest.raises(ValueError, match="reward must be scalar"):
            _read_scalar_reward(ep, "bad_reward.hdf5")


def test_read_images_rejects_missing_images_group(tmp_path: Path) -> None:
    episode_path = tmp_path / "missing_images.hdf5"
    _write_episode(episode_path)

    with h5py.File(episode_path, "a") as ep:
        del ep["observations"]["images"]

    with h5py.File(episode_path, "r") as ep:
        with pytest.raises(ValueError, match="missing observations/images group"):
            _read_images(ep, "missing_images.hdf5")


def test_read_images_rejects_missing_camera(tmp_path: Path) -> None:
    episode_path = tmp_path / "missing_camera.hdf5"
    _write_episode(episode_path)

    with h5py.File(episode_path, "a") as ep:
        del ep["observations"]["images"]["cam_left_wrist"]

    with h5py.File(episode_path, "r") as ep:
        with pytest.raises(ValueError, match="missing camera cam_left_wrist"):
            _read_images(ep, "missing_camera.hdf5")


def test_read_images_rejects_invalid_camera_shape(tmp_path: Path) -> None:
    episode_path = tmp_path / "bad_camera_shape.hdf5"
    _write_episode(episode_path)

    with h5py.File(episode_path, "a") as ep:
        del ep["observations"]["images"]["cam_high"]
        ep["observations"]["images"].create_dataset(
            "cam_high",
            data=np.zeros((4, 8, 8, 1), dtype=np.uint8),
        )

    with h5py.File(episode_path, "r") as ep:
        with pytest.raises(ValueError, match=r"shape \(T, H, W, 3\)"):
            _read_images(ep, "bad_camera_shape.hdf5")


def test_load_episode_payload_reads_sandwich_hdf5(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode_0.hdf5"
    _write_episode(episode_path, reward=1.0)

    payload = _load_episode_payload(episode_path, task="Assemble a sandwich.")

    assert payload.episode_name == "episode_0.hdf5"
    assert payload.raw_reward == 1.0
    assert payload.num_frames == 4
    assert payload.frames[0]["task"] == "Assemble a sandwich."
    assert payload.frames[0]["observation.state"].shape == (14,)
    assert payload.frames[0]["action"].shape == (14,)
    np.testing.assert_array_equal(
        payload.frames[1]["teleop_mask"],
        np.asarray([1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        payload.frames[3]["teleop_mask"],
        np.asarray([0], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        payload.frames[0]["is_success"],
        np.asarray([True], dtype=bool),
    )


def test_load_episode_payload_scalar_fields_match_feature_schema(
    tmp_path: Path,
) -> None:
    episode_path = tmp_path / "episode_0.hdf5"
    _write_episode(episode_path, reward=1.0)
    features = _aloha_features(
        image_shape=(8, 8, 3),
        include_velocity=True,
        include_effort=True,
    )

    payload = _load_episode_payload(episode_path, task="Assemble a sandwich.")

    for field, expected in (
        ("is_success", np.asarray([True], dtype=bool)),
        ("teleop_mask", np.asarray([0], dtype=np.int64)),
    ):
        value = payload.frames[0][field]
        assert value.shape == features[field]["shape"]
        assert value.dtype == np.dtype(features[field]["dtype"])
        np.testing.assert_array_equal(value, expected)


def test_load_episode_payload_casts_numeric_arrays_to_float32(
    tmp_path: Path,
) -> None:
    episode_path = tmp_path / "episode_float64.hdf5"
    _write_episode(episode_path)

    with h5py.File(episode_path, "a") as ep:
        del ep["action"]
        del ep["observations"]["qpos"]
        del ep["observations"]["qvel"]
        del ep["observations"]["effort"]
        ep.create_dataset(
            "action",
            data=np.arange(56, dtype=np.float64).reshape(4, 14),
        )
        ep["observations"].create_dataset(
            "qpos",
            data=np.ones((4, 14), dtype=np.float64),
        )
        ep["observations"].create_dataset(
            "qvel",
            data=np.full((4, 14), 2.0, dtype=np.float64),
        )
        ep["observations"].create_dataset(
            "effort",
            data=np.full((4, 14), 3.0, dtype=np.float64),
        )

    payload = _load_episode_payload(episode_path, task="Assemble a sandwich.")

    assert payload.frames[0]["observation.state"].dtype == np.float32
    assert payload.frames[0]["action"].dtype == np.float32
    assert payload.frames[0]["observation.velocity"].dtype == np.float32
    assert payload.frames[0]["observation.effort"].dtype == np.float32


def test_load_episode_payload_rejects_non_aloha_state_width(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode_bad_width.hdf5"
    _write_episode(episode_path, width=13)

    with pytest.raises(ValueError, match=r"\(T, 14\)"):
        _load_episode_payload(episode_path, task="Assemble a sandwich.")


def test_build_hil_segments_entry_is_json_serializable(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode_0.hdf5"
    _write_episode(episode_path, reward=0.0)

    payload = _load_episode_payload(episode_path, task="Assemble a sandwich.")
    entry = _build_hil_segments_entry(payload)

    assert entry == {
        "episode_index": 0,
        "episode_name": "episode_0.hdf5",
        "num_frames": 4,
        "raw_reward": 0.0,
        "is_success": False,
        "teleop_segments": [[1, 3]],
    }
    json.dumps(entry)
