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

"""The reward's threshold semantics and its per-slot frame history, VLM forward stubbed."""

import numpy as np
import pytest
import torch

from rlinf.models.embodiment.reward.topreward_model import TOPRewardModel

CHUNK = 4
WINDOW = 8
NUM_ENVS = 3
IMAGE = (3, 8, 8)


class _StubbedScorer(TOPRewardModel):
    """Scripted scores, recording the window each call was given."""

    def __init__(self, logps, threshold=np.log(0.5), window_frames=WINDOW):
        torch.nn.Module.__init__(self)
        self.chunk = CHUNK
        self.threshold = float(np.exp(threshold))
        self.window_frames = window_frames
        self.fps = 2.0
        self._history = {}
        self._logps = list(logps)
        self.calls = []

    def _score(self, frames_chw, instruction):
        self.calls.append((frames_chw.copy(), instruction))
        return self._logps.pop(0)


def _obs(value: float) -> torch.Tensor:
    return torch.full((NUM_ENVS * CHUNK, *IMAGE), value)


def _instructions() -> list[str]:
    return [f"task-{env}" for env in range(NUM_ENVS) for _ in range(CHUNK)]


def test_the_label_is_the_thresholded_probability():
    scorer = _StubbedScorer([np.log(0.9), np.log(0.5), np.log(0.1)])

    rewards = scorer.predict_rew(_obs(0.0), _instructions())

    assert rewards.shape == (NUM_ENVS * CHUNK,)
    # The middle env sits exactly on the threshold, which counts as success.
    assert rewards.reshape(NUM_ENVS, CHUNK)[:, 0].tolist() == [1.0, 1.0, 0.0]
    for env in range(NUM_ENVS):
        chunk = rewards.reshape(NUM_ENVS, CHUNK)[env]
        assert chunk.unique().numel() == 1


def test_each_env_is_scored_against_its_own_instruction():
    scorer = _StubbedScorer([np.log(0.9)] * NUM_ENVS)

    scorer.predict_rew(_obs(0.0), _instructions())

    assert [instruction for _, instruction in scorer.calls] == [
        "task-0",
        "task-1",
        "task-2",
    ]


def test_the_window_grows_into_the_previous_chunk():
    scorer = _StubbedScorer([np.log(0.1)] * (2 * NUM_ENVS))

    scorer.predict_rew(_obs(-1.0), _instructions())
    first = [frames for frames, _ in scorer.calls[:NUM_ENVS]]
    scorer.predict_rew(_obs(1.0), _instructions())
    second = [frames for frames, _ in scorer.calls[NUM_ENVS:]]

    assert all(len(frames) == CHUNK for frames in first)
    assert all(len(frames) == WINDOW for frames in second)
    # -1.0 and 1.0 land on 0 and 255, so the window runs oldest to newest.
    assert second[0][0].max() == 0
    assert second[0][-1].max() == 255


@pytest.mark.parametrize("slots", [None, [0, 2]])
def test_reset_history_stops_a_window_spanning_episodes(slots):
    scorer = _StubbedScorer([np.log(0.1)] * (2 * NUM_ENVS))
    scorer.predict_rew(_obs(-1.0), _instructions())

    scorer.reset_history(slots)
    scorer.predict_rew(_obs(1.0), _instructions())

    lengths = [len(frames) for frames, _ in scorer.calls[NUM_ENVS:]]
    cleared = set(range(NUM_ENVS)) if slots is None else set(slots)
    assert lengths == [CHUNK if env in cleared else WINDOW for env in range(NUM_ENVS)]
