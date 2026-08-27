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

import numpy as np
import pytest
import torch
from PIL import Image

WINDOW = 5
CHUNK = 8


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


def _make_backend(module, retain_action=True):
    backend = object.__new__(module.WanBackend)
    backend.device = torch.device("cpu")
    backend.num_frames = WINDOW + CHUNK
    backend.num_inference_steps = 5
    backend.condition_frame_length = WINDOW
    backend.retain_action = retain_action
    backend._sessions = {}
    return backend


def _open(backend, env_ids, seeds, size=8):
    frames = [[torch.zeros(3, 1, size, size) for _ in range(WINDOW)] for _ in env_ids]
    backend.open_session(
        env_ids=env_ids,
        init_frames=frames,
        init_actions=torch.zeros(len(env_ids), WINDOW, 7),
        task_ids=[7] * len(env_ids),
        seeds=seeds,
    )


def _gray_frames(values, size=8):
    return [
        Image.fromarray(np.full((size, size, 3), v, dtype=np.uint8)) for v in values
    ]


def test_batch_seed_requires_an_open_session(monkeypatch):
    module = _load_backend_module(monkeypatch)
    backend = _make_backend(module)
    _open(backend, [0, 1], seeds=[0, 0])

    assert backend._batch_seed([0, 1]) == 0

    backend.close_session([1])
    with pytest.raises(RuntimeError):
        backend._batch_seed([0, 1])


def test_batch_seed_rejects_mixed_seeds(monkeypatch):
    module = _load_backend_module(monkeypatch)
    backend = _make_backend(module)
    _open(backend, [0, 1], seeds=[0, 1])

    with pytest.raises(NotImplementedError):
        backend._batch_seed([0, 1])


def test_generate_rejects_a_batch_that_does_not_line_up(monkeypatch):
    module = _load_backend_module(monkeypatch)
    backend = _make_backend(module)
    _open(backend, [0, 1], seeds=[0, 0])

    with pytest.raises(ValueError):
        backend.generate(env_ids=[0, 1], actions=torch.zeros(1, CHUNK, 7))


def test_pipe_kwargs_conditions_on_the_session_window(monkeypatch):
    module = _load_backend_module(monkeypatch)
    backend = _make_backend(module)
    _open(backend, [0], seeds=[3])
    backend._sessions[0]["frames"] = _gray_frames([10, 20, 30, 40, 50])

    actions = torch.arange(CHUNK * 7, dtype=torch.float32).reshape(1, CHUNK, 7)
    kwargs = backend._pipe_kwargs(env_ids=[0], actions=actions)

    assert kwargs["seed"] == 3
    assert kwargs["batch_size"] == 1
    assert np.asarray(kwargs["input_image"][0]).max() == 10
    assert [np.asarray(f).max() for f in kwargs["input_image4"][0]] == [20, 30, 40, 50]
    # retain_action prepends the window's actions, and the window then keeps the last ones sent
    assert kwargs["action"].shape == (1, WINDOW + CHUNK, 7)
    assert torch.equal(
        backend._sessions[0]["actions"][1:], kwargs["action"][0, -(WINDOW - 1) :]
    )
    assert torch.all(backend._sessions[0]["actions"][0] == 0)


def test_generate_keeps_the_reference_frame_and_rolls_the_window(monkeypatch):
    module = _load_backend_module(monkeypatch)
    backend = _make_backend(module)
    _open(backend, [0], seeds=[0])
    backend._sessions[0]["frames"] = _gray_frames([10, 20, 30, 40, 50])

    generated = _gray_frames(range(100, 100 + WINDOW + CHUNK))
    backend._pipe = lambda **kwargs: [generated]

    videos = backend.generate(env_ids=[0], actions=torch.zeros(1, CHUNK, 7))

    assert videos.shape == (1, 3, WINDOW + CHUNK, 8, 8)
    window = backend._sessions[0]["frames"]
    assert np.asarray(window[0]).max() == 10
    # the pipeline's own frames go into the window untouched, not through a [-1, 1] round trip
    assert [np.asarray(f).max() for f in window[1:]] == [109, 110, 111, 112]
    assert all(a is b for a, b in zip(window[1:], generated[-(WINDOW - 1) :]))
