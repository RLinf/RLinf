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

"""Dream SDK and environment contracts without network or hardware."""

import base64
import io
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from PIL import Image

from rlinf.envs.sim.robotwin.dream_env import DreamObservationBuffer
from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
from rlinf.workers.rollout.dream_worker import DreamActionClient

ROOT = Path(__file__).resolve().parents[2]


class FakeSDK:
    def __init__(self):
        self.calls = self.closes = self.resets = 0
        self.observations = []
        self.right = np.full((48, 7), 0.75)

    def policy(self, config):
        self.config = config

    def reset_episode(self):
        self.resets += 1

    def set_observation(self, frames, state, *, timestamp):
        self.observations.append((frames, state, timestamp))

    def get_action(self, *, commanded=None):
        self.calls += 1
        self.commanded = commanded
        return {"q_l_chunk": np.full((48, 7), 0.25), "q_r_chunk": self.right}

    def close(self):
        self.closes += 1


def config(monkeypatch, arm="aloha", task="click_bell"):
    monkeypatch.setenv("EMBODIED_PATH", str(ROOT / "examples/embodiment"))
    monkeypatch.setenv("REPO_PATH", str(ROOT))
    monkeypatch.setenv("DREAM_IK_ARM_FILE", "/path/to/arm.json")
    with initialize_config_dir(str(ROOT / "evaluations/robotwin"), version_base="1.1"):
        return compose(
            config_name=f"robotwin_dream_{arm}_click_bell_eval",
            overrides=[f"env.eval.task_config.task_name={task}"],
        )


@pytest.mark.parametrize(
    "arm,task",
    [
        ("aloha", "click_bell"),
        ("aloha", "move_playingcard_away"),
        ("piper", "click_bell"),
    ],
)
def test_examples(monkeypatch, arm, task):
    cfg = config(monkeypatch, arm, task)
    assert cfg.env.eval.rollout_epoch == 10
    assert not cfg.env.eval.use_fixed_reset_state_ids
    assert cfg.env.eval.task_config.task_name == task
    assert cfg.env.eval.max_episode_steps == 400
    assert cfg.env.eval.max_steps_per_rollout_epoch == 432
    assert cfg.env.eval.task_config.embodiment[0] == (
        "aloha-agilex" if arm == "aloha" else "piper"
    )
    assert cfg.rollout.dream.action_hz == (20 if arm == "aloha" else 30)


def test_images_and_sdk_lifecycle(monkeypatch):
    cfg = config(monkeypatch)
    buffer = DreamObservationBuffer()
    buffer.reset()
    buffer.append(
        {
            "main_images": torch.full((1, 8, 12, 3), 255, dtype=torch.uint8),
            "wrist_images": torch.full((1, 2, 8, 12, 3), 255, dtype=torch.uint8),
            "states": torch.arange(14, dtype=torch.float32)[None] / 14,
        },
        0.0,
    )
    samples = buffer.pack(["press the bell"], torch.ones((1, 14)), finished=False)
    sdk = FakeSDK()
    client = DreamActionClient(sdk, cfg.rollout)
    actions = client.predict(samples)
    np.testing.assert_allclose(actions[0, 0], [0.25] * 7 + [0.75] * 7)
    frames, state, timestamp = sdk.observations[0]
    assert timestamp == 0 and sdk.resets == 1
    assert set(frames) == {"video.overhead", "video.wrist_l", "video.wrist_r"}
    for encoded in frames.values():
        image = Image.open(io.BytesIO(base64.b64decode(encoded)))
        assert image.size == (320, 240)
        assert image.getpixel((160, 0)) == (0, 0, 0)
        assert image.getpixel((160, 120)) == (255, 255, 255)
    np.testing.assert_allclose(state["state.right_arm"], np.arange(7, 13) / 14)
    np.testing.assert_allclose(sdk.commanded["q_l_chunk"], 1)
    samples["finished"] = [True]
    hold = client.predict(samples)
    np.testing.assert_allclose(hold[:, 0], samples["states"][:, -1])
    assert sdk.calls == 1
    client.close()
    client.close()
    assert sdk.closes == 1


def test_nonfinite_active_arm_fails(monkeypatch):
    cfg = config(monkeypatch)
    sdk = FakeSDK()
    sdk.right[:] = np.nan
    client = DreamActionClient(sdk, cfg.rollout)
    sample = {
        "states": np.zeros((1, 1, 14)),
        "frames": [[{}]],
        "timestamps": [[0.0]],
        "finished": [False],
        "episode_id": [1],
        "task_descriptions": ["press"],
        "active_arms": [["left"]],
    }
    np.testing.assert_allclose(client.predict(sample)[..., 7:], 0)
    sample["active_arms"] = [["left", "right"]]
    with pytest.raises(ValueError, match="active right arm"):
        client.predict(sample)
    client.close()


def test_ten_distinct_native_seeds(monkeypatch):
    cfg = config(monkeypatch)
    env = object.__new__(RoboTwinEnv)
    env.cfg = cfg.env.eval
    env.task_name = "click_bell"
    env.seed = env.base_seed = env.seed_offset = 0
    env.total_num_processes = env.num_group = env.group_size = 1
    env.use_fixed_reset_state_ids = False
    env._init_reset_state_ids()
    seen = []
    for _ in range(10):
        seen.append(int(env.reset_state_ids[0]))
        env.update_reset_state_ids()
    assert sorted(seen) == list(range(100100001, 100100011))


def test_api_error_is_not_retried(monkeypatch):
    cfg = config(monkeypatch)
    sdk = FakeSDK()
    failure = RuntimeError("service unavailable")

    def fail(**kwargs):
        sdk.calls += 1
        raise failure

    sdk.get_action = fail
    client = DreamActionClient(sdk, cfg.rollout)
    sample = {
        "states": np.zeros((1, 1, 14)),
        "frames": [[{}]],
        "timestamps": [[0.0]],
        "finished": [False],
        "episode_id": [1],
        "task_descriptions": ["press"],
    }
    with pytest.raises(RuntimeError) as result:
        client.predict(sample)
    assert result.value is failure and sdk.calls == 1
    client.close()


@pytest.mark.parametrize("instruction_type", [None, "unseen"])
def test_native_instruction_type(monkeypatch, instruction_type):
    cfg = config(monkeypatch)
    if instruction_type is None:
        del cfg.env.eval.instruction_type
    module = ModuleType("robotwin.envs.vector_env")

    def vector_env(task_config, n_envs, env_seeds, instruction_type="seen"):
        return SimpleNamespace(instruction_type=instruction_type, seeds=env_seeds)

    module.VectorEnv = vector_env
    monkeypatch.setitem(sys.modules, module.__name__, module)
    env = object.__new__(RoboTwinEnv)
    env.cfg, env.num_envs = cfg.env.eval, 1
    env.reset_state_ids = torch.tensor([100100001])
    env._init_env()
    assert env.venv.instruction_type == (instruction_type or "seen")
    assert env.venv.seeds == [100100001]
