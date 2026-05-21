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

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

pytest.importorskip("mani_skill")

from rlinf.envs.maniskill.maniskill_env import ManiskillEnv


class _FakeUnwrapped:
    num_envs = 2


class _FakeEnv:
    unwrapped = _FakeUnwrapped()


def _make_env(default_task_description: str = "Pick up the cube.") -> ManiskillEnv:
    env = object.__new__(ManiskillEnv)
    env.env = _FakeEnv()
    env.cfg = OmegaConf.create({"default_task_description": default_task_description})
    return env


def test_instruction_uses_default_when_env_has_no_language_method():
    env = _make_env("Pick up the red cube.")

    assert env.instruction == ["Pick up the red cube.", "Pick up the red cube."]


def test_instruction_uses_default_when_env_language_method_returns_none():
    env = _make_env("Place the cube on the goal.")
    env.env.unwrapped = SimpleNamespace(
        num_envs=2,
        get_language_instruction=lambda: None,
    )

    assert env.instruction == [
        "Place the cube on the goal.",
        "Place the cube on the goal.",
    ]
