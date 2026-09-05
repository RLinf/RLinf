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

import pickle

import numpy as np

from examples.reward.preprocess_vlm_trend_reward_dataset import (
    load_episodes_with_labels,
)


def _observation(tcp_x: float) -> dict:
    states = np.zeros(19, dtype=np.float32)
    states[4] = tcp_x
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    return {
        "states": states,
        "main_images": image,
        "third_view_images": image,
    }


def _load_episode(data_path, target_ee_pose):
    return load_episodes_with_labels(
        str(data_path),
        window_size=2,
        tail_unclear_ratio=0.0,
        load_workers=1,
        target_ee_pose=target_ee_pose,
    )[0]


def test_tcp_distance_is_used_when_gae_is_missing(tmp_path):
    episode_path = tmp_path / "episode.pkl"
    episode = {
        "observations": [
            _observation(0.0),
            _observation(0.5),
            _observation(1.0),
        ],
        "rewards": [0.0, -1.0, -2.0],
    }
    with open(episode_path, "wb") as file:
        pickle.dump(episode, file)

    loaded = _load_episode(tmp_path, [1.0, 0.0, 0.0])
    assert {sample["score_source"] for sample in loaded["samples"]} == {"tcp_distance"}
    assert all(sample["label"] == "positive" for sample in loaded["samples"])

    loaded = _load_episode(tmp_path, None)
    assert {sample["score_source"] for sample in loaded["samples"]} == {"rewards"}
    assert all(sample["label"] == "negative" for sample in loaded["samples"])

    episode["gae"] = [0.0, 0.0, 0.0]
    with open(episode_path, "wb") as file:
        pickle.dump(episode, file)

    loaded = _load_episode(tmp_path, [1.0, 0.0, 0.0])
    assert {sample["score_source"] for sample in loaded["samples"]} == {"gae"}
    assert all(sample["label"] == "unclear" for sample in loaded["samples"])
