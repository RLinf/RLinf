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
import pytest
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


def test_yam_collector_accepts_recorded_task_descriptions(monkeypatch):
    """The YAM recipe keeps string task metadata alongside tensor observations."""
    # Importing RealWorldEnv has an existing process-cleanup side effect. Keep
    # this unit test isolated from host ROS processes.
    monkeypatch.setattr(psutil, "process_iter", lambda: ())

    DataCollector = _load_collector(monkeypatch).DataCollector

    collector = object.__new__(DataCollector)
    collector.cfg = SimpleNamespace(
        runner=SimpleNamespace(record_task_description=True)
    )
    states = torch.arange(14, dtype=torch.float32).reshape(1, 14)
    descriptions = ["pick_block"]

    processed = collector._process_obs(
        {"states": states, "task_descriptions": descriptions}
    )

    assert processed["task_descriptions"] == descriptions
    assert torch.equal(processed["states"], states)


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
    collector.buffer = SimpleNamespace(
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


def test_realworld_close_releases_vector_env_once(monkeypatch):
    """Closing the collector-facing env must close nested robot resources."""
    monkeypatch.setattr(psutil, "process_iter", lambda: ())
    from rlinf.envs.real import RealWorldEnv

    calls = []
    outer = RealWorldEnv.__new__(RealWorldEnv)
    outer.env = SimpleNamespace(close=lambda: calls.append("vector closed"))
    outer.close()
    outer.close()
    assert calls == ["vector closed"]

    retry_calls = []

    def close_after_retry():
        retry_calls.append("attempt")
        if len(retry_calls) == 1:
            raise RuntimeError("temporary close failure")

    retry = RealWorldEnv.__new__(RealWorldEnv)
    retry.env = SimpleNamespace(close=close_after_retry)
    with pytest.raises(RuntimeError, match="temporary close failure"):
        retry.close()
    retry.close()
    retry.close()
    assert retry_calls == ["attempt", "attempt"]


def test_collection_closes_hardware_before_dataset_finalization():
    """Slow metadata export must not leave a real robot powered."""
    from rlinf.envs.wrappers import CollectEpisode

    events = []
    collector = CollectEpisode.__new__(CollectEpisode)
    collector._closed = False
    collector.streaming = False
    collector.env = SimpleNamespace(close=lambda: events.append("robot closed"))
    collector._finalize_lerobot = lambda: events.append("dataset finalized")
    collector._wait_futures = lambda: None
    collector._wait_save_futures = lambda: None
    collector._executor = None
    collector._save_executor = None

    collector.close()
    assert events == ["robot closed", "dataset finalized"]

    retry_events = []

    def close_after_retry():
        retry_events.append("robot close attempted")
        if retry_events.count("robot close attempted") == 1:
            raise RuntimeError("robot close failed")

    retry = CollectEpisode.__new__(CollectEpisode)
    retry._closed = False
    retry.streaming = False
    retry.env = SimpleNamespace(close=close_after_retry)
    retry._finalize_lerobot = lambda: retry_events.append("dataset finalized")
    retry._wait_futures = lambda: None
    retry._wait_save_futures = lambda: None
    retry._executor = None
    retry._save_executor = None
    with pytest.raises(RuntimeError, match="robot close failed"):
        retry.close()
    assert not retry._closed
    retry.close()
    assert retry._closed
    assert retry_events.count("robot close attempted") == 2


@pytest.mark.parametrize("save_demos", [False, True])
def test_collection_step_failure_closes_robot_and_optional_replay_buffer(
    monkeypatch, save_demos
):
    module = _load_collector(monkeypatch)
    events = []

    class Env:
        def reset(self):
            return {"states": torch.zeros((1, 14))}, {}

        def step(self, action):
            del action
            raise RuntimeError("camera frame failed")

        def close(self):
            events.append("robot closed")

    monkeypatch.setattr(module, "TrajectoryAccumulator", lambda **kwargs: None)
    collector = object.__new__(module.DataCollector)
    collector.cfg = SimpleNamespace(
        runner=SimpleNamespace(record_task_description=True),
        env=SimpleNamespace(eval=SimpleNamespace(max_episode_steps=10)),
    )
    collector.env = Env()
    collector.save_demos = save_demos
    collector.buffer = (
        SimpleNamespace(close=lambda: events.append("buffer closed"))
        if save_demos
        else None
    )
    collector._preexisting_success = 0
    collector.num_data_episodes = 1
    collector.action_dim = 14
    collector.log_info = lambda message: None

    with pytest.raises(RuntimeError, match="camera frame failed"):
        collector.run()
    assert events == (
        ["robot closed", "buffer closed"] if save_demos else ["robot closed"]
    )


def test_streaming_resume_does_not_count_old_failures_as_current_successes(monkeypatch):
    module = _load_collector(monkeypatch)
    import rlinf.envs.wrappers as wrappers

    class Config(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as error:
                raise AttributeError(name) from error

    class Env:
        preexisting_episode_count = 5
        action_space = SimpleNamespace(shape=(14,))

        def __init__(self, *args, **kwargs):
            pass

        def reset(self):
            return {"states": torch.zeros((1, 14))}, {}

        def step(self, action):
            raise RuntimeError("entered collection loop")

        def close(self):
            pass

    monkeypatch.setattr(
        module.Worker, "__init__", lambda self: setattr(self, "_worker_info", None)
    )
    monkeypatch.setattr(module.DataCollector, "log_info", lambda self, message: None)
    monkeypatch.setattr(module, "RealWorldEnv", Env)
    monkeypatch.setattr(wrappers, "CollectEpisode", Env)
    monkeypatch.setattr(module, "tqdm", lambda **kwargs: None)
    cfg = Config(
        runner=Config(num_data_episodes=2, save_demos=False),
        env=Config(
            eval=Config(
                override_cfg={},
                data_collection=Config(
                    enabled=True,
                    save_dir="unused",
                    export_format="lerobot",
                    only_success=False,
                    streaming=True,
                    resume=True,
                ),
            )
        ),
    )
    collector = module.DataCollector(cfg)

    assert collector._preexisting_success == 0
    with pytest.raises(RuntimeError, match="entered collection loop"):
        collector.run()
