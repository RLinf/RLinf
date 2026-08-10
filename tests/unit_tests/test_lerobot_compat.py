# Copyright 2025 The RLinf Authors.
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

"""Unit tests for the LeRobot compatibility shims across lerobot versions."""

import pytest
import torch

from rlinf.data.storage.lerobot import add_frame_to_dataset, episode_boundaries
from rlinf.data.storage.lerobot.writer import LeRobotDatasetWriter


class _LegacyDataset:
    """Mimics lerobot < 0.2: the task lives inside the frame dict."""

    def __init__(self):
        self.frames = []
        self.saved_episodes = 0

    def add_frame(self, frame):
        if "task" not in frame:
            raise ValueError("Missing features: {'task'}")
        self.frames.append(frame)

    def save_episode(self):
        self.saved_episodes += 1


class _PostRevertDataset(_LegacyDataset):
    """Mimics lerobot >= 0.4: back to ``add_frame(frame)``, but it pops the task.

    0.4 reverted the 0.3.x signature, so a version-number check would dispatch
    this one wrongly. It also mutates the caller's dict.
    """

    def add_frame(self, frame):
        if "task" not in frame:
            raise ValueError("Missing features: {'task'}")
        self.frames.append({**frame, "task": frame.pop("task")})


class _CurrentDataset:
    """Mimics lerobot >= 0.3: the task is a separate argument.

    Like the real implementation, a ``task`` key inside *frame* is rejected
    because it is not part of the feature schema.
    """

    def __init__(self):
        self.frames = []
        self.tasks = []
        self.saved_episodes = 0

    def add_frame(self, frame, task, timestamp=None):
        if "task" in frame:
            raise ValueError("Extra features: {'task'}")
        self.frames.append(frame)
        self.tasks.append(task)

    def save_episode(self):
        self.saved_episodes += 1


def _make_writer(dataset):
    # ``create()`` needs a real lerobot install, so attach the dataset the way
    # ``create()`` would.
    writer = LeRobotDatasetWriter()
    writer.dataset = dataset
    return writer


def _episode(n=2):
    return [{"state": i, "actions": i, "task": "pick up the cube"} for i in range(n)]


def test_legacy_dataset_keeps_task_in_frame():
    dataset = _LegacyDataset()
    _make_writer(dataset).add_episode(_episode())

    assert [f["task"] for f in dataset.frames] == ["pick up the cube"] * 2
    assert dataset.saved_episodes == 1


def test_current_dataset_gets_task_as_argument():
    dataset = _CurrentDataset()
    _make_writer(dataset).add_episode(_episode())

    assert dataset.tasks == ["pick up the cube"] * 2
    assert all("task" not in f for f in dataset.frames)
    assert dataset.frames[0]["state"] == 0
    assert dataset.saved_episodes == 1


ALL_SHAPES = [_LegacyDataset, _CurrentDataset, _PostRevertDataset]


def test_post_revert_dataset_keeps_task_in_frame():
    # lerobot >= 0.4 took the 0.3.x signature back out again.
    dataset = _PostRevertDataset()
    _make_writer(dataset).add_episode(_episode())

    assert [f["task"] for f in dataset.frames] == ["pick up the cube"] * 2
    assert dataset.saved_episodes == 1


@pytest.mark.parametrize("dataset_cls", ALL_SHAPES)
def test_caller_frames_are_not_mutated(dataset_cls):
    # The DAgger worker shares these dicts with the in-memory training store,
    # and lerobot >= 0.4 pops "task" out of whatever frame it is handed.
    episode = _episode()
    before = [dict(f) for f in episode]
    _make_writer(dataset_cls()).add_episode(episode)

    assert episode == before


@pytest.mark.parametrize("dataset_cls", ALL_SHAPES)
def test_frame_without_task_is_rejected(dataset_cls):
    with pytest.raises(ValueError, match="missing the required 'task' field"):
        add_frame_to_dataset(dataset_cls(), {"state": 0, "actions": 0})


@pytest.mark.parametrize("dataset_cls", ALL_SHAPES)
def test_add_frame_to_dataset_is_usable_standalone(dataset_cls):
    # The toolkit collectors drive LeRobotDataset directly, without the writer.
    dataset = dataset_cls()
    add_frame_to_dataset(dataset, {"state": 0, "task": "wipe the table"})

    assert len(dataset.frames) == 1


def test_empty_episode_is_skipped():
    dataset = _CurrentDataset()
    _make_writer(dataset).add_episode([])

    assert dataset.frames == []
    assert dataset.saved_episodes == 0


# --------------------------------------------------------------------------
# episode_boundaries: dataset format v2.1 vs v3.0
# --------------------------------------------------------------------------


class _V21Dataset:
    """Dataset format v2.1: a dict of two tensors on the dataset itself."""

    def __init__(self, starts, ends):
        self.episode_data_index = {
            "from": torch.tensor(starts),
            "to": torch.tensor(ends),
        }


class _V30Meta:
    def __init__(self, starts, ends):
        self.episodes = {"dataset_from_index": starts, "dataset_to_index": ends}


class _V30Dataset:
    """Dataset format v3.0 (lerobot >= 0.4): columns on ``meta.episodes``."""

    def __init__(self, starts, ends):
        self.episode_data_index = None
        self.meta = _V30Meta(starts, ends)


@pytest.mark.parametrize("dataset_cls", [_V21Dataset, _V30Dataset])
def test_episode_boundaries_agree_across_formats(dataset_cls):
    starts, ends = episode_boundaries(dataset_cls([0, 3, 7], [3, 7, 9]))

    assert starts == [0, 3, 7]
    assert ends == [3, 7, 9]
    assert all(isinstance(x, int) for x in starts + ends)


def test_episode_boundaries_reports_an_unknown_layout():
    class _Alien:
        episode_data_index = None
        meta = None

    with pytest.raises(RuntimeError, match="Cannot determine episode boundaries"):
        episode_boundaries(_Alien())


def test_episode_boundaries_rejects_v30_meta_without_the_columns():
    class _Partial:
        episode_data_index = None
        meta = _V30Meta([0], [1])

    _Partial.meta.episodes = {"length": [1]}

    with pytest.raises(RuntimeError, match="Cannot determine episode boundaries"):
        episode_boundaries(_Partial())
