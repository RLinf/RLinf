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

import gymnasium as gym
import numpy as np

from rlinf.envs.wrappers.collect_episode import CollectEpisode


class OneStepCollectionEnv(gym.Env):
    def __init__(
        self,
        intervened=False,
        success=True,
        executed_action=None,
        intervene_action=None,
        raw_intervene_action=None,
    ):
        self.intervened = intervened
        self.success = success
        self.executed_action = executed_action
        self.intervene_action = intervene_action
        self.raw_intervene_action = raw_intervene_action
        self.action_space = gym.spaces.Box(-10.0, 10.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {"state": gym.spaces.Box(-10.0, 10.0, shape=(1,), dtype=np.float32)}
        )

    def reset(self, *, seed=None, options=None):
        return {"state": np.zeros((1, 1), dtype=np.float32)}, {}

    def step(self, action):
        info = {}
        if self.intervened:
            info["intervene_flag"] = np.asarray([True], dtype=bool)
        if self.executed_action is not None:
            info["executed_action"] = np.asarray(
                self.executed_action, dtype=np.float32
            )
        if self.intervene_action is not None:
            info["intervene_action"] = np.asarray(
                self.intervene_action, dtype=np.float32
            )
        if self.raw_intervene_action is not None:
            info["raw_intervene_action"] = np.asarray(
                self.raw_intervene_action, dtype=np.float32
            )
            info["intervene_rejected"] = True
        info["success"] = self.success
        return {"state": np.ones((1, 1), dtype=np.float32)}, 0.0, True, False, info


class ChunkCollectionEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(-10.0, 10.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {"state": gym.spaces.Box(-10.0, 10.0, shape=(1,), dtype=np.float32)}
        )

    def reset(self, *, seed=None, options=None):
        return {"state": np.zeros((1, 1), dtype=np.float32)}, {}

    def chunk_step(self, chunk_actions):
        obs_list = [
            {"state": np.ones((1, 1), dtype=np.float32)},
            {"state": np.ones((1, 1), dtype=np.float32) * 2},
        ]
        rewards = np.zeros((1, 2), dtype=np.float32)
        terminations = np.asarray([[False, True]], dtype=bool)
        truncations = np.asarray([[False, False]], dtype=bool)
        infos_list = [
            {"intervene_flag": np.asarray([False], dtype=bool)},
            {
                "intervene_flag": np.asarray([[False, True]], dtype=bool),
                "intervene_action": np.asarray([[9.0, 9.0, 8.0, 8.0]], dtype=np.float32),
            },
        ]
        return obs_list, rewards, terminations, truncations, infos_list


def test_collect_episode_records_full_intervened_episode_with_executed_action(tmp_path):
    wrapped = CollectEpisode(
        OneStepCollectionEnv(intervened=True, executed_action=[7.0]),
        save_dir=str(tmp_path),
        record_executed_action=True,
    )
    wrapped.reset()
    wrapped.step(np.asarray([[1.0]], dtype=np.float32))
    wrapped.close()

    [path] = list(tmp_path.glob("*.pkl"))
    with path.open("rb") as f:
        episode = pickle.load(f)

    assert episode["intervened"]
    np.testing.assert_array_equal(episode["actions"][0], np.asarray([7.0]))


def test_collect_episode_default_records_input_action_even_if_executed_action_exists(
    tmp_path,
):
    wrapped = CollectEpisode(
        OneStepCollectionEnv(intervened=True, executed_action=[7.0]),
        save_dir=str(tmp_path),
    )
    wrapped.reset()
    wrapped.step(np.asarray([[1.0]], dtype=np.float32))
    wrapped.close()

    [path] = list(tmp_path.glob("*.pkl"))
    with path.open("rb") as f:
        episode = pickle.load(f)

    np.testing.assert_array_equal(episode["actions"][0], np.asarray([1.0]))


def test_collect_episode_record_executed_action_uses_accepted_intervene_action(
    tmp_path,
):
    wrapped = CollectEpisode(
        OneStepCollectionEnv(intervened=True, intervene_action=[8.0]),
        save_dir=str(tmp_path),
        record_executed_action=True,
    )
    wrapped.reset()
    wrapped.step(np.asarray([[1.0]], dtype=np.float32))
    wrapped.close()

    [path] = list(tmp_path.glob("*.pkl"))
    with path.open("rb") as f:
        episode = pickle.load(f)

    np.testing.assert_array_equal(episode["actions"][0], np.asarray([8.0]))


def test_collect_episode_rejected_takeover_records_executed_not_raw_candidate(
    tmp_path,
):
    wrapped = CollectEpisode(
        OneStepCollectionEnv(
            intervened=False,
            executed_action=[6.0],
            raw_intervene_action=[9.0],
        ),
        save_dir=str(tmp_path),
        record_executed_action=True,
    )
    wrapped.reset()
    wrapped.step(np.asarray([[1.0]], dtype=np.float32))
    wrapped.close()

    [path] = list(tmp_path.glob("*.pkl"))
    with path.open("rb") as f:
        episode = pickle.load(f)

    np.testing.assert_array_equal(episode["actions"][0], np.asarray([6.0]))
    np.testing.assert_array_equal(
        episode["infos"][1]["raw_intervene_action"], np.asarray([9.0])
    )
    assert episode["infos"][1]["intervene_rejected"] is True


def test_lerobot_legacy_intervention_override_stays_opt_in_to_lerobot(tmp_path):
    wrapped = CollectEpisode(OneStepCollectionEnv(), save_dir=str(tmp_path))
    try:
        np.testing.assert_array_equal(
            wrapped._lerobot_action_with_legacy_intervention(
                np.asarray([1.0], dtype=np.float32),
                {
                    "intervene_flag": np.asarray([True], dtype=bool),
                    "intervene_action": np.asarray([8.0], dtype=np.float32),
                },
            ),
            np.asarray([8.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            wrapped._lerobot_action_with_legacy_intervention(
                np.asarray([1.0], dtype=np.float32),
                {
                    "intervene_flag": np.asarray([False], dtype=bool),
                    "intervene_action": np.asarray([8.0], dtype=np.float32),
                },
            ),
            np.asarray([1.0], dtype=np.float32),
        )
    finally:
        wrapped.close()


def test_collect_episode_success_filter_records_success_without_intervention(tmp_path):
    success_dir = tmp_path / "success"
    success_wrapped = CollectEpisode(
        OneStepCollectionEnv(intervened=False, success=True),
        save_dir=str(success_dir),
        only_success=True,
    )
    success_wrapped.reset()
    success_wrapped.step(np.asarray([[1.0]], dtype=np.float32))
    success_wrapped.close()

    [success_path] = list(success_dir.glob("*.pkl"))
    with success_path.open("rb") as f:
        success_episode = pickle.load(f)
    assert success_episode["success"] is True
    assert success_episode["intervened"] is False

    failed_dir = tmp_path / "failed"
    failed_wrapped = CollectEpisode(
        OneStepCollectionEnv(intervened=True, success=False),
        save_dir=str(failed_dir),
        only_success=True,
    )
    failed_wrapped.reset()
    failed_wrapped.step(np.asarray([[1.0]], dtype=np.float32))
    failed_wrapped.close()

    assert list(failed_dir.glob("*.pkl")) == []


def test_collect_episode_does_not_record_chunk_wide_intervene_action(tmp_path):
    wrapped = CollectEpisode(ChunkCollectionEnv(), save_dir=str(tmp_path))
    wrapped.reset()
    wrapped.chunk_step(np.asarray([[[1.0, 1.0], [2.0, 2.0]]], dtype=np.float32))
    wrapped.close()

    [path] = list(tmp_path.glob("*.pkl"))
    with path.open("rb") as f:
        episode = pickle.load(f)

    np.testing.assert_array_equal(episode["actions"][0], np.asarray([1.0, 1.0]))
    np.testing.assert_array_equal(episode["actions"][1], np.asarray([2.0, 2.0]))
