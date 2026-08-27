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

from types import SimpleNamespace
from unittest.mock import Mock

from rlinf.runners.embodied_runner import EmbodiedRunner


def _worker() -> Mock:
    worker = Mock()
    worker.start_profile.return_value.wait.return_value = None
    worker.stop_profile.return_value.wait.return_value = None
    return worker


class _GroupConfig(SimpleNamespace):
    def get(self, name: str, default=None):
        return getattr(self, name, default)


def test_profile_window_only_touches_selected_worker_groups() -> None:
    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.actor = _worker()
    runner.rollout = _worker()
    runner.env = _worker()
    runner.reward = None
    runner.critic = None
    runner.logger = Mock()
    runner.cfg = SimpleNamespace(
        actor=_GroupConfig(group_name="ActorGroup"),
        rollout=_GroupConfig(group_name="RolloutGroup"),
        env=_GroupConfig(group_name="EnvGroup"),
        reward=_GroupConfig(),
        critic=_GroupConfig(),
    )
    runner._profile_worker_groups = {"actorgroup", "rolloutgroup"}

    runner._open_profiling_window(step_idx=3)
    runner._close_profiling_window(step_idx=3)

    runner.actor.start_profile.assert_called_once_with(3)
    runner.actor.stop_profile.assert_called_once_with()
    runner.rollout.start_profile.assert_called_once_with(3)
    runner.rollout.stop_profile.assert_called_once_with()
    runner.env.start_profile.assert_not_called()
    runner.env.stop_profile.assert_not_called()
