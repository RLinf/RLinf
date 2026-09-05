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

"""Auto-reset of a world-model env restarts only the slots whose episode ended.

The episode loop is shared by every backend, so one fake backend covers both models.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision.transforms as transforms

NUM_ENVS = 4
WINDOW = 5
CHUNK = 8
IMAGE_SIZE = (8, 8)


def _load_env_module(monkeypatch):
    """Load the shared env module; only the dataset wrapper needs stubbing out."""
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "rlinf" / "envs" / "world_model" / "world_model_env.py"

    # The dataset lives behind rlinf.data, whose package import pulls in the replay
    # buffer; these tests drive the env off a fake dataset instead.
    fake_dataset = types.ModuleType("rlinf.data.datasets.world_model")
    fake_dataset.NpyTrajectoryDatasetWrapper = object
    monkeypatch.setitem(sys.modules, "rlinf.data.datasets.world_model", fake_dataset)

    spec = importlib.util.spec_from_file_location(
        "rlinf.envs.world_model.world_model_env", module_path
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class _FakeDataset:
    """Episodes whose frames carry their own index, so slots can be traced back."""

    def __init__(self, num_episodes=32):
        self.num_episodes = num_episodes

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, episode_idx):
        value = float(episode_idx) / self.num_episodes
        frame = torch.full((3, *IMAGE_SIZE), value)
        return {
            "task": f"task-{episode_idx}",
            "start_items": [
                {
                    "image": frame.clone(),
                    "observation.state": torch.full((7,), value),
                }
            ],
            "target_items": [
                {
                    "image": frame.clone(),
                    "action": np.full(7, value, dtype=np.float32),
                }
                for _ in range(WINDOW - 1)
            ],
        }


class _FakeBackend:
    """Records the window every session opens with, and the slots each call names."""

    def __init__(self):
        self.chunk = CHUNK
        self.condition_frame_length = WINDOW
        self.image_size = IMAGE_SIZE
        self.sessions = {}
        self.closed = []

    def open_session(self, env_ids, init_frames, init_actions, task_ids, seeds):
        for row, env_id in enumerate(env_ids):
            self.sessions[int(env_id)] = {
                "frames": [frame.clone() for frame in init_frames[row]],
                "actions": init_actions[row].clone(),
                "task_id": task_ids[row],
            }

    def close_session(self, env_ids):
        slots = [int(env_id) for env_id in env_ids]
        self.closed.append(slots)
        for slot in slots:
            self.sessions.pop(slot, None)

    def generate(self, env_ids, actions):
        return torch.full((len(env_ids), 3, CHUNK, *IMAGE_SIZE), -1.0)

    def offload(self):
        pass

    def onload(self):
        pass


def _make_env(module, num_envs=NUM_ENVS):
    """Build the env without __init__, which would load a world and a reward model."""

    class _Env(module.WorldModelEnv):
        def _build_backend(self):
            raise AssertionError("the fixture installs a fake backend directly")

        def _load_reward_model(self):
            raise AssertionError("these tests do not score frames")

    env = object.__new__(_Env)
    env.cfg = types.SimpleNamespace(max_episode_steps=2 * CHUNK, action_dim=7)
    env.device = torch.device("cpu")
    env.num_envs = num_envs

    env.backend = _FakeBackend()
    env.chunk = env.backend.chunk
    env.condition_frame_length = env.backend.condition_frame_length
    env.image_size = env.backend.image_size

    env.trans_norm = transforms.Compose(
        [transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3, inplace=True)]
    )
    env.dataset = _FakeDataset()
    env.reset_gripper_open = True
    env.is_libero_env = True

    env.current_obs = None
    env.task_descriptions = [""] * num_envs
    env.init_ee_poses = [None] * num_envs

    env._elapsed_steps = torch.zeros(num_envs, dtype=torch.long)
    env.prev_step_reward = torch.zeros(num_envs)
    env.record_metrics = True
    env.success_once = torch.zeros(num_envs, dtype=torch.bool)
    env.returns = torch.zeros(num_envs)

    env._is_start = False
    env.use_fixed_reset_state_ids = False
    env._is_offloaded = False
    return env


def _grow_time_axis_to_full_window(env):
    """Mimic the state after a chunk_step: the time axis holds window + chunk frames."""
    generated = torch.full((env.num_envs, 3, 1, CHUNK, *IMAGE_SIZE), -1.0)
    env.current_obs = torch.cat([env.current_obs, generated], dim=3)
    env._elapsed_steps += CHUNK


def _slot_tail_frame(env, slot):
    return env.current_obs[slot, :, 0, -1].clone()


@pytest.fixture
def env(monkeypatch):
    module = _load_env_module(monkeypatch)
    env = _make_env(module)
    env.reset(episode_indices=np.arange(NUM_ENVS))
    return env


def test_elapsed_steps_is_tracked_per_slot(env):
    assert env.elapsed_steps.shape == (NUM_ENVS,)
    assert torch.equal(env.elapsed_steps, torch.zeros(NUM_ENVS, dtype=torch.long))

    _grow_time_axis_to_full_window(env)
    env.reset(env_idx=[1], episode_indices=[9])

    assert env.elapsed_steps.tolist() == [CHUNK, 0, CHUNK, CHUNK]


def test_full_reset_rebuilds_every_slot(env):
    assert env.current_obs.shape == (NUM_ENVS, 3, 1, WINDOW, *IMAGE_SIZE)
    assert env.task_descriptions == [f"task-{idx}" for idx in range(NUM_ENVS)]
    assert sorted(env.backend.sessions) == list(range(NUM_ENVS))

    env.reset(episode_indices=np.arange(NUM_ENVS, 2 * NUM_ENVS))

    assert env.task_descriptions == [
        f"task-{idx}" for idx in range(NUM_ENVS, 2 * NUM_ENVS)
    ]
    assert torch.equal(env.elapsed_steps, torch.zeros(NUM_ENVS, dtype=torch.long))


def test_subset_reset_leaves_the_other_slots_untouched(env):
    _grow_time_axis_to_full_window(env)
    env.returns += 1.0
    untouched = {slot: _slot_tail_frame(env, slot) for slot in (0, 2)}
    untouched_sessions = {slot: env.backend.sessions[slot] for slot in (0, 2)}

    env.reset(env_idx=[1, 3], episode_indices=[17, 19])

    for slot in (0, 2):
        assert torch.equal(_slot_tail_frame(env, slot), untouched[slot])
        assert env.backend.sessions[slot] is untouched_sessions[slot]
        assert env.task_descriptions[slot] == f"task-{slot}"
        assert env.returns[slot] == 1.0

    for slot, episode_idx in ((1, 17), (3, 19)):
        assert env.task_descriptions[slot] == f"task-{episode_idx}"
        assert env.returns[slot] == 0.0
        assert env.backend.sessions[slot]["task_id"] == episode_idx
        # The restarted condition window sits at the tail, where _wrap_obs reads.
        window = env.backend.sessions[slot]["frames"]
        assert len(window) == WINDOW
        assert torch.equal(_slot_tail_frame(env, slot), window[-1][:, 0])


def test_subset_reset_reopens_only_those_sessions(env):
    _grow_time_axis_to_full_window(env)
    env.backend.closed.clear()

    env.reset(env_idx=[2], episode_indices=[11])

    # A backend pools per-trajectory state, so it must be told which ones ended.
    assert env.backend.closed == [[2]]
    assert sorted(env.backend.sessions) == list(range(NUM_ENVS))


def test_subset_reset_keeps_the_shared_time_axis(env):
    _grow_time_axis_to_full_window(env)
    num_frames = env.current_obs.shape[3]

    env.reset(env_idx=[2], episode_indices=[11])

    assert env.current_obs.shape[3] == num_frames
    # Frames ahead of the condition window hold the reference frame of the new episode.
    reference = env.backend.sessions[2]["frames"][0][:, 0]
    for t_idx in range(num_frames - WINDOW):
        assert torch.equal(env.current_obs[2, :, 0, t_idx], reference)


def test_auto_reset_restarts_only_the_done_slots(env):
    _grow_time_axis_to_full_window(env)
    untouched = {slot: _slot_tail_frame(env, slot) for slot in (0, 2)}
    dones = torch.tensor([False, True, False, True])

    obs, infos = env._handle_auto_reset(dones, env._wrap_obs(), {})

    assert env.elapsed_steps.tolist() == [CHUNK, 0, CHUNK, 0]
    assert env.backend.closed[-1] == [1, 3]
    for slot in (0, 2):
        assert torch.equal(_slot_tail_frame(env, slot), untouched[slot])
        assert env.task_descriptions[slot] == f"task-{slot}"
    for slot in (1, 3):
        assert env.task_descriptions[slot] != f"task-{slot}"

    # The done flags now describe exactly the slots that restarted.
    assert torch.equal(infos["_final_observation"], dones)
    assert obs["task_descriptions"] == env.task_descriptions


def test_episode_len_survives_a_fresh_slot(env):
    _grow_time_axis_to_full_window(env)
    env.reset(env_idx=[0], episode_indices=[5])

    infos = env._record_metrics(
        torch.ones(NUM_ENVS), torch.zeros(NUM_ENVS, dtype=torch.bool), {}
    )

    episode = infos["episode"]
    assert episode["episode_len"].tolist() == [0.0, CHUNK, CHUNK, CHUNK]
    assert torch.isfinite(episode["reward"]).all()


def test_subset_reset_rejects_a_mismatched_episode_count(env):
    _grow_time_axis_to_full_window(env)

    with pytest.raises(ValueError, match="episode indices"):
        env.reset(env_idx=[1, 3], episode_indices=[17])
