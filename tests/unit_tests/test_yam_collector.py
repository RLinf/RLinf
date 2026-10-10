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

"""Regression tests for YAM's generic real-world data collector."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import psutil
import torch


def _load_collector(monkeypatch):
    # The installed i2rt SDK also ships an "examples" package. Load this
    # repository's entrypoint by its exact path to avoid namespace shadowing.
    monkeypatch.setattr(psutil, "process_iter", lambda: ())
    path = (
        Path(__file__).resolve().parents[2] / "examples/embodiment/collect_real_data.py"
    )
    spec = importlib.util.spec_from_file_location("_yam_collection_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_replay_collector_skips_start_preview_and_aborted_transitions(
    monkeypatch, tmp_path
):
    module = _load_collector(monkeypatch)

    class Accumulator:
        def __init__(self, **kwargs):
            self.samples = []

        def append(self, step):
            self.samples.append(
                (step.curr_obs["states"].item(), step.next_obs["states"].item())
            )

        def to_trajectory(self):
            return SimpleNamespace(
                intervene_flags=torch.zeros(1), samples=self.samples.copy()
            )

    class Env:
        index = 0

        def reset(self, **kwargs):
            return {"states": torch.tensor([[float(self.index)]])}, {}

        def step(self, action):
            self.index += 1
            start = self.index in (2, 6)
            abort = self.index == 4
            invalid_done = self.index == 8
            success = self.index == 10
            done = invalid_done or success
            pre = self.index in (1, 4, 5)
            info = {
                "keyboard_event": ["start" if start else "abort" if abort else None],
                "keyboard_phase": ["pre" if pre else "rec"],
                "record_reset": [start or abort],
                "pre_record": [pre],
                "manual_done": [done],
                "recording_invalid": [invalid_done],
            }
            return (
                {"states": torch.tensor([[float(self.index)]])},
                torch.tensor([float(done)]),
                torch.tensor([done]),
                torch.tensor([False]),
                info,
            )

        def close(self):
            pass

    monkeypatch.setattr(module, "TrajectoryAccumulator", Accumulator)
    collector = object.__new__(module.DataCollector)
    collector.cfg = SimpleNamespace(
        runner=SimpleNamespace(
            record_task_description=True, logger=SimpleNamespace(log_path=str(tmp_path))
        ),
        env=SimpleNamespace(eval=SimpleNamespace(max_episode_steps=100)),
    )
    saved = []
    collector.replay_buffer = SimpleNamespace(
        add_trajectories=saved.extend, close=lambda: None
    )
    env = Env()
    collector.env = env
    collector.num_data_episodes = 1
    collector._preexisting_success = 0
    collector._target_step_period = None
    collector.action_dim = 14
    collector.total_cnt = 0
    collector.manual_episode_control_only = True
    logs = []
    collector.log_info = logs.append
    collector.run()
    assert env.index == 10
    assert collector.total_cnt == 2
    assert len(saved) == 1
    assert saved[0].samples == [(8.0, 9.0), (9.0, 10.0)]
    assert any("Recording overflow" in message for message in logs)
    assert sum("Total: 1/1" in message for message in logs) == 1


def test_lerobot_only_collector_does_not_build_replay_trajectories(monkeypatch):
    module = _load_collector(monkeypatch)

    class Accumulator:
        def __init__(self, **kwargs):
            raise AssertionError("replay accumulator should stay unused")

    class Env:
        resets = 0

        def reset(self, **kwargs):
            self.resets += 1
            return {"states": torch.tensor([[0.0]])}, {}

        def step(self, action):
            return (
                {"states": torch.tensor([[1.0]])},
                torch.tensor([1.0]),
                torch.tensor([True]),
                torch.tensor([False]),
                {"manual_done": [True]},
            )

        def close(self):
            pass

    monkeypatch.setattr(module, "TrajectoryAccumulator", Accumulator)
    collector = object.__new__(module.DataCollector)
    collector.replay_buffer = None
    collector.env = Env()
    collector.num_data_episodes = 1
    collector._preexisting_success = 0
    collector._target_step_period = None
    collector.action_dim = 14
    collector.total_cnt = 0
    collector.manual_episode_control_only = True
    logs = []
    collector.log_info = logs.append
    collector.run()
    assert collector.total_cnt == 1
    assert all("Replay buffer saved" not in message for message in logs)
