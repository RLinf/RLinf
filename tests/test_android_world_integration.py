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
import types
from typing import Any, Dict, List, Type

import pytest

from rlinf.algorithms.rewards.android import AndroidReward
from rlinf.data.datasets.android import AndroidWorldDataset
from omegaconf import DictConfig, OmegaConf


class _FakeResult:
    """Simple fake result object for testing AndroidReward."""

    def __init__(self, done: bool) -> None:
        self.done = done


class _FakeTask:
    """Simple fake Android World task for testing AndroidReward."""

    def __init__(self, score: float, raise_error: bool = False) -> None:
        self._score = score
        self._raise_error = raise_error
        self.initialized = False
        self.task_name = "FakeTask"
        self.params: Dict[str, Any] = {}
        self.class_name = "FakeTaskClass"

    def is_successful(self, env: Any) -> float:  # pylint: disable=unused-argument
        if self._raise_error:
            raise RuntimeError("is_successful failed")
        return self._score


def _build_reward_cfg() -> DictConfig:
    """Build a minimal DictConfig for AndroidReward."""
    cfg_dict = {
        "reward_scale": 2.0,
        "device_id": "localhost:5555",
        "grpc_port": 8554,
        "adb_path": "adb",
    }
    return OmegaConf.create(cfg_dict)


def test_android_reward_returns_zero_when_not_done() -> None:
    """AndroidReward should return 0.0 when result.done is False."""
    reward = AndroidReward(_build_reward_cfg())
    fake_env = object()
    fake_result = _FakeResult(done=False)
    fake_task = _FakeTask(score=1.0)

    value = reward.get_reward_new(fake_env, fake_result, fake_task)

    assert value == 0.0


def test_android_reward_scales_score_when_done() -> None:
    """AndroidReward should scale task score when result.done is True."""
    reward = AndroidReward(_build_reward_cfg())
    fake_env = object()
    fake_result = _FakeResult(done=True)
    fake_task = _FakeTask(score=1.5)

    value = reward.get_reward_new(fake_env, fake_result, fake_task)

    assert value == pytest.approx(3.0)  # 1.5 * reward_scale (2.0)


def test_android_reward_swallows_task_exception() -> None:
    """AndroidReward should not raise when task.is_successful fails."""
    reward = AndroidReward(_build_reward_cfg())
    fake_env = object()
    fake_result = _FakeResult(done=True)
    fake_task = _FakeTask(score=1.0, raise_error=True)

    value = reward.get_reward_new(fake_env, fake_result, fake_task)

    assert value == 0.0


def test_android_world_dataset_uses_android_world_parent_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """AndroidWorldDataset should load tasks via android_world_parent in config.

    This test injects a fake android_world.registry.TaskRegistry into sys.modules
    so that no real android_world installation is required.
    """

    # Build a fake Task class and registry.
    class _FakeTaskClass:
        complexity = 1

        @staticmethod
        def generate_random_params() -> Dict[str, Any]:
            return {"k": "v"}

        def __init__(self, params: Dict[str, Any]) -> None:  # pylint: disable=unused-argument
            self.goal = "do something"

    class _FakeTaskRegistry:
        def get_registry(
            self,
            family: str,  # pylint: disable=unused-argument
        ) -> Dict[str, Type[_FakeTaskClass]]:
            """Return a minimal registry mapping for the given family."""
            return {"FakeTask": _FakeTaskClass}

    # Inject fake android_world.registry into sys.modules.
    fake_registry_module = types.ModuleType("android_world.registry")
    setattr(fake_registry_module, "TaskRegistry", _FakeTaskRegistry)
    monkeypatch.setitem(
        __import__("sys").modules,
        "android_world.registry",
        fake_registry_module,
    )

    # Minimal config for AndroidWorldDataset.
    cfg = OmegaConf.create(
        {
            "data": {
                "max_prompt_length": 128,
                "task_family": "android_world",
                "n_instances_per_task": 1,
                "apply_chat_template": False,
                "android_world_parent": "/does/not/matter/for/test",
            }
        }
    )

    class _FakeTokenizer:
        """Minimal tokenizer stub for AndroidWorldDataset tests."""

        eos_token_id = 0

        def encode(self, text: str) -> List[int]:  # pylint: disable=unused-argument
            return [1, 2, 3]

        def apply_chat_template(
            self,
            messages: List[Dict[str, Any]],  # pylint: disable=unused-argument
            tokenize: bool = False,  # pylint: disable=unused-argument
            add_generation_prompt: bool = True,  # pylint: disable=unused-argument
        ) -> str:
            return "formatted"

    dataset = AndroidWorldDataset(config=cfg, tokenizer=_FakeTokenizer())

    assert len(dataset) == 1
