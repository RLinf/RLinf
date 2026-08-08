# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import queue

from rlinf.envs.realworld.common.video_player import video_player


class _EmptyThenStopQueue:
    def __init__(self):
        self.calls = 0

    def get(self, timeout=None):
        self.calls += 1
        if self.calls == 1:
            raise queue.Empty
        return None


def test_video_player_processes_events_while_waiting(monkeypatch):
    player = object.__new__(video_player.VideoPlayer)
    player.queue = _EmptyThenStopQueue()
    wait_key_calls = []

    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(video_player.cv2, "waitKey", wait_key_calls.append)

    player._play()

    assert wait_key_calls == [1]
