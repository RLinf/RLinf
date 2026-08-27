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

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


def _load_backend_module(monkeypatch):
    """Load the Wan backend with diffsynth stubbed out, so it runs without the Wan deps."""
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "rlinf" / "envs" / "world_model" / "wan_backend.py"

    fake_wan_video = types.ModuleType("diffsynth.pipelines.wan_video_new")
    fake_wan_video.ModelConfig = object
    fake_wan_video.WanVideoPipeline = object
    fake_pipelines = types.ModuleType("diffsynth.pipelines")
    fake_pipelines.wan_video_new = fake_wan_video
    fake_diffsynth = types.ModuleType("diffsynth")
    fake_diffsynth.pipelines = fake_pipelines

    monkeypatch.setitem(sys.modules, "diffsynth", fake_diffsynth)
    monkeypatch.setitem(sys.modules, "diffsynth.pipelines", fake_pipelines)
    monkeypatch.setitem(
        sys.modules, "diffsynth.pipelines.wan_video_new", fake_wan_video
    )

    spec = importlib.util.spec_from_file_location(
        "rlinf.envs.world_model.wan_backend", module_path
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _make_backend(module, num_frames=9, num_inference_steps=5):
    backend = object.__new__(module.WanBackend)
    backend.device = torch.device("cpu")
    backend.num_frames = num_frames
    backend.num_inference_steps = num_inference_steps
    backend._sessions = {}
    return backend


def test_batch_seed_requires_an_open_session(monkeypatch):
    module = _load_backend_module(monkeypatch)
    backend = _make_backend(module)

    backend.open_session(
        env_ids=[0, 1], init_frames=[[], []], task_ids=[7, 7], seeds=[0, 0]
    )
    assert backend._batch_seed([0, 1]) == 0

    backend.close_session([1])
    with pytest.raises(RuntimeError):
        backend._batch_seed([0, 1])


def test_batch_seed_rejects_mixed_seeds(monkeypatch):
    module = _load_backend_module(monkeypatch)
    backend = _make_backend(module)

    backend.open_session(
        env_ids=[0, 1], init_frames=[[], []], task_ids=[7, 8], seeds=[0, 1]
    )
    with pytest.raises(NotImplementedError):
        backend._batch_seed([0, 1])


def test_generate_rejects_a_batch_that_does_not_line_up(monkeypatch):
    module = _load_backend_module(monkeypatch)
    backend = _make_backend(module)
    backend.open_session(
        env_ids=[0, 1], init_frames=[[], []], task_ids=[7, 7], seeds=[0, 0]
    )

    condition = [[torch.zeros(3, 1, 256, 256)]]
    with pytest.raises(ValueError):
        backend.generate(
            env_ids=[0, 1], actions=torch.zeros(2, 4, 7), condition=condition
        )


def test_pipe_kwargs_takes_the_last_four_condition_frames(monkeypatch):
    module = _load_backend_module(monkeypatch)
    backend = _make_backend(module)
    backend.open_session(env_ids=[0], init_frames=[[]], task_ids=[7], seeds=[3])

    condition = [[torch.zeros(3, 1, 256, 256) for _ in range(5)]]
    kwargs = backend._pipe_kwargs(
        env_ids=[0], actions=torch.zeros(1, 4, 7), condition=condition
    )

    assert kwargs["seed"] == 3
    assert kwargs["batch_size"] == 1
    assert len(kwargs["input_image4"][0]) == 4
    assert kwargs["input_image"][0].size == (256, 256)
