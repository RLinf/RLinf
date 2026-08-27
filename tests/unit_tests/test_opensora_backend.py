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
from collections import deque
from pathlib import Path

import numpy as np
import pytest
import torch

WINDOW = 4
CHUNK = 8
LATENT_CHANNELS = 3
LATENT_HW = 4


def _load_backend_module(monkeypatch):
    """Load the OpenSora backend with opensora stubbed out, so it runs without the OpenSora deps."""
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "rlinf" / "envs" / "world_model" / "opensora_backend.py"

    registry = types.ModuleType("opensora.registry")
    registry.MODELS = object()
    registry.SCHEDULERS = object()
    registry.build_module = lambda *args, **kwargs: None
    inference_utils = types.ModuleType("opensora.utils.inference_utils")
    inference_utils.prepare_multi_resolution_info = lambda *args, **kwargs: {}
    misc = types.ModuleType("opensora.utils.misc")
    misc.to_torch_dtype = lambda dtype: torch.float32
    utils = types.ModuleType("opensora.utils")
    utils.inference_utils = inference_utils
    utils.misc = misc
    opensora = types.ModuleType("opensora")
    opensora.registry = registry
    opensora.utils = utils

    for name, module in {
        "opensora": opensora,
        "opensora.registry": registry,
        "opensora.utils": utils,
        "opensora.utils.inference_utils": inference_utils,
        "opensora.utils.misc": misc,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "rlinf.envs.world_model.opensora_backend", module_path
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class _FakeVae:
    """Stands in for the OpenSora VAE: encode and decode are the identity on the frame axis."""

    out_channels = LATENT_CHANNELS

    def get_latent_size(self, input_size):
        return (input_size[0], LATENT_HW, LATENT_HW)

    def encode(self, frames):
        return frames

    def decode(self, latents, num_frames=None):
        return latents


class _FakeScheduler:
    """Returns fresh latents for the masked tail and records what it was asked to sample."""

    def __init__(self):
        self.calls = []

    def sample(self, model, z, y, device, additional_args, progress, mask):
        self.calls.append({"z": z, "y": y, "mask": mask})
        samples = z.clone()
        tail = z.shape[2] - WINDOW
        for offset in range(tail):
            samples[:, :, WINDOW + offset] = 1000 + offset
        return samples


def _make_backend(module):
    backend = object.__new__(module.OpenSoraBackend)
    backend.device = torch.device("cpu")
    backend.inference_dtype = torch.float32
    backend.chunk = CHUNK
    backend.condition_frame_length = WINDOW
    backend.num_frames = WINDOW + CHUNK
    backend.image_size = (LATENT_HW, LATENT_HW)
    backend.vae = _FakeVae()
    backend.model = object()
    backend.scheduler = _FakeScheduler()
    backend.model_args = {}
    backend.is_vae_v1_2 = False
    backend.z_mask_frame_num = CHUNK
    backend.z_condition_frame_length = WINDOW
    backend.action_stats = {
        "q01": np.full(7, -2.0, dtype=np.float32),
        "q99": np.full(7, 2.0, dtype=np.float32),
    }
    backend._sessions = {}
    return backend


def _init_frames(env_values):
    """One window per env slot; frame ``t`` of env ``e`` is filled with ``env_values[e] + t``."""
    return [
        [
            torch.full((LATENT_CHANNELS, 1, LATENT_HW, LATENT_HW), float(base + t_idx))
            for t_idx in range(WINDOW)
        ]
        for base in env_values
    ]


def _open(backend, env_ids, env_values):
    backend.open_session(
        env_ids=env_ids,
        init_frames=_init_frames(env_values),
        init_actions=torch.zeros(len(env_ids), WINDOW, 7),
        task_ids=[7] * len(env_ids),
        seeds=[0] * len(env_ids),
    )


def _window_values(backend, env_id):
    return [
        latent.flatten()[0].item() for latent in backend._sessions[env_id]["latents"]
    ]


def test_open_session_encodes_the_window_in_order(monkeypatch):
    module = _load_backend_module(monkeypatch)
    backend = _make_backend(module)
    _open(backend, [0, 1], env_values=[10, 20])

    assert isinstance(backend._sessions[0]["latents"], deque)
    assert _window_values(backend, 0) == [10, 11, 12, 13]
    assert _window_values(backend, 1) == [20, 21, 22, 23]
    for latent in backend._sessions[0]["latents"]:
        assert latent.shape == (1, LATENT_CHANNELS, 1, LATENT_HW, LATENT_HW)


def test_generate_requires_an_open_session(monkeypatch):
    module = _load_backend_module(monkeypatch)
    backend = _make_backend(module)
    _open(backend, [0, 1], env_values=[10, 20])
    backend.close_session([1])

    with pytest.raises(RuntimeError):
        backend.generate(env_ids=[0, 1], actions=torch.zeros(2, CHUNK, 7))


def test_generate_rejects_a_batch_that_does_not_line_up(monkeypatch):
    module = _load_backend_module(monkeypatch)
    backend = _make_backend(module)
    _open(backend, [0, 1], env_values=[10, 20])

    with pytest.raises(ValueError):
        backend.generate(env_ids=[0, 1], actions=torch.zeros(1, CHUNK, 7))


def test_generate_conditions_on_the_window_and_rolls_it(monkeypatch):
    module = _load_backend_module(monkeypatch)
    backend = _make_backend(module)
    _open(backend, [0], env_values=[10])

    frames = backend.generate(env_ids=[0], actions=torch.zeros(1, CHUNK, 7))

    call = backend.scheduler.calls[0]
    assert call["z"].shape == (1, LATENT_CHANNELS, WINDOW + CHUNK, LATENT_HW, LATENT_HW)
    assert [call["z"][0, 0, t].flatten()[0].item() for t in range(WINDOW)] == [
        10,
        11,
        12,
        13,
    ]
    assert call["mask"].shape == (1, WINDOW + CHUNK)
    assert call["mask"][0].tolist() == [0] * WINDOW + [1] * CHUNK

    # the window keeps the newest latents only, and never mixes decoded pixels back in
    assert _window_values(backend, 0) == [1004, 1005, 1006, 1007]
    assert frames.shape == (1, LATENT_CHANNELS, CHUNK, LATENT_HW, LATENT_HW)


def test_actions_are_normalized_with_the_dataset_stats(monkeypatch):
    module = _load_backend_module(monkeypatch)
    backend = _make_backend(module)
    _open(backend, [0], env_values=[10])

    backend.generate(env_ids=[0], actions=torch.ones(1, CHUNK, 7))

    # q01 = -2, q99 = 2, so an action of 1 lands halfway up [-1, 1]
    actions = backend.scheduler.calls[0]["y"]
    assert actions.shape == (1, CHUNK, 7)
    assert actions.flatten()[0].item() == pytest.approx(0.5)
