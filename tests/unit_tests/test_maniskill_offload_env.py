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

import importlib.util
import io
import sys
import types
from pathlib import Path

import torch


class _FakeCudaTensor:
    def torch(self):
        return torch.tensor([1.0])


class _FakePx:
    def __init__(self):
        self.cuda_articulation_link_data = _FakeCudaTensor()
        self.cuda_articulation_qacc = _FakeCudaTensor()
        self.cuda_articulation_qf = _FakeCudaTensor()
        self.cuda_articulation_qpos = _FakeCudaTensor()
        self.cuda_articulation_qvel = _FakeCudaTensor()
        self.cuda_articulation_target_qpos = _FakeCudaTensor()
        self.cuda_articulation_target_qvel = _FakeCudaTensor()
        self.cuda_rigid_body_data = _FakeCudaTensor()
        self.cuda_rigid_dynamic_data = _FakeCudaTensor()

    def gpu_update_articulation_kinematics(self):
        return None


class _FakeScene:
    def __init__(self):
        self.px = _FakePx()
        self.timestep = None

    def get_timestep(self):
        return 0.02

    def set_timestep(self, timestep):
        self.timestep = timestep

    def _gpu_apply_all(self):
        return None

    def _gpu_fetch_all(self):
        return None


class _FakeBatchedRng:
    rngs = ["rng"]


class _FakeController:
    def __init__(self):
        self.loaded_state = None

    def get_state(self):
        return {"controller": torch.tensor([1.0])}

    def set_state(self, state):
        self.loaded_state = state


class _FakeAgent:
    def __init__(self):
        self.controller = _FakeController()


class _FakeEnv:
    def __init__(self):
        self.unwrapped = self
        self.device = "cpu"
        self.scene = _FakeScene()
        self._main_rng = "main_rng"
        self._batched_main_rng = _FakeBatchedRng()
        self._main_seed = 123
        self._episode_rng = "episode_rng"
        self._batched_episode_rng = _FakeBatchedRng()
        self._episode_seed = 456
        self.action_space = "action_space"
        self.single_action_space = "single_action_space"
        self._orig_single_action_space = "orig_single_action_space"
        self._elapsed_steps = torch.tensor([3])
        self._init_raw_obs = {"obs": torch.tensor([1.0])}
        self.agent = _FakeAgent()
        self.task_reset_states = {}
        self.task_metric_states = {}
        self.reset_seed = None
        self.reset_options = None
        self.loaded_state = None

    def get_state(self):
        return {"sim": torch.tensor([2.0])}

    def reset(self, seed=None, options=None):
        self.reset_seed = seed
        self.reset_options = options

    def set_state(self, state):
        self.loaded_state = state


class _FakeManiskillEnv:
    pass


def _load_maniskill_offload_module(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = (
        repo_root / "rlinf" / "envs" / "maniskill" / "maniskill_offload_env.py"
    )

    fake_package = types.ModuleType("rlinf.envs.maniskill")
    fake_package.__path__ = [str(module_path.parent)]
    fake_env_module = types.ModuleType("rlinf.envs.maniskill.maniskill_env")
    fake_env_module.ManiskillEnv = _FakeManiskillEnv

    monkeypatch.setitem(sys.modules, "rlinf.envs.maniskill", fake_package)
    monkeypatch.setitem(
        sys.modules, "rlinf.envs.maniskill.maniskill_env", fake_env_module
    )
    monkeypatch.delitem(
        sys.modules, "rlinf.envs.maniskill.maniskill_offload_env", raising=False
    )

    spec = importlib.util.spec_from_file_location(
        "rlinf.envs.maniskill.maniskill_offload_env", module_path
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _make_core(module):
    core = object.__new__(module._ManiskillEnvCore)
    core.env = _FakeEnv()
    core.seed = 10
    core.device = "cpu"
    core.prev_step_reward = torch.tensor([0.5])
    core.reset_state_ids = torch.tensor([1])
    core._generator = torch.Generator()
    core._generator.manual_seed(0)
    core.is_start = True
    core.record_metrics = False
    return core


def test_maniskill_offload_state_does_not_require_record_video_counter(monkeypatch):
    module = _load_maniskill_offload_module(monkeypatch)
    core = _make_core(module)

    state_buffer = core.get_state()
    state = torch.load(io.BytesIO(state_buffer), map_location="cpu", weights_only=False)

    assert "video_cnt" not in state


def test_maniskill_offload_load_state_accepts_state_without_video_counter(
    monkeypatch,
):
    module = _load_maniskill_offload_module(monkeypatch)
    monkeypatch.setattr(
        module, "set_batch_rng_state", lambda rng_state: _FakeBatchedRng()
    )
    core = _make_core(module)

    source_state = torch.load(
        io.BytesIO(core.get_state()),
        map_location="cpu",
        weights_only=False,
    )
    assert "video_cnt" not in source_state

    core.load_state(_serialize_state(source_state))

    assert core.env.reset_seed == core.seed
    assert core.env.reset_options == {"reconfigure": False}
    torch.testing.assert_close(core.env.loaded_state["sim"], torch.tensor([2.0]))
    assert core.env.scene.timestep == 0.02
    assert not hasattr(core, "video_cnt")


def test_maniskill_offload_load_state_ignores_legacy_video_counter(monkeypatch):
    module = _load_maniskill_offload_module(monkeypatch)
    monkeypatch.setattr(
        module, "set_batch_rng_state", lambda rng_state: _FakeBatchedRng()
    )
    core = _make_core(module)

    source_state = torch.load(
        io.BytesIO(core.get_state()),
        map_location="cpu",
        weights_only=False,
    )
    source_state["video_cnt"] = 7

    core.load_state(_serialize_state(source_state))

    assert not hasattr(core, "video_cnt")


def _serialize_state(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()
