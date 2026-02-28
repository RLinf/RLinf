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

import json
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from rlinf.models.embodiment.starvla import get_model
from rlinf.models.embodiment.starvla.starvla_action_model import (
    StarVLAForRLActionPrediction,
)


class _DummyHFBackbone(nn.Module):
    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden_size, pad_token_id=0)


class _DummyVLMInterface(nn.Module):
    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.model = _DummyHFBackbone(hidden_size=hidden_size)
        self._ACTION_TOKEN_MIN = 1
        self._ACTION_TOKEN_MAX = 999


class _DummyStarVLAModel(nn.Module):
    def __init__(self, framework_name: str = "QwenFast"):
        super().__init__()
        self.framework_name = framework_name
        self.qwen_vl_interface = _DummyVLMInterface(hidden_size=16)
        self.proj = nn.Linear(4, 4)
        self.norm_stats = {}
        self.last_to_dtype = None

    def to(self, dtype=None, *args, **kwargs):  # noqa: ANN001
        self.last_to_dtype = dtype
        return self

    def get_action_stats(self, key):  # noqa: ANN001
        value = self.norm_stats.get(key)
        if isinstance(value, dict) and "action" in value:
            return value["action"]
        return value


def _install_fake_starvla_module(monkeypatch, model: _DummyStarVLAModel, calls: list[str]):
    def _from_pretrained(path: str):
        calls.append(path)
        return model

    baseframework_obj = types.SimpleNamespace(from_pretrained=_from_pretrained)

    starvla_pkg = types.ModuleType("starVLA")
    starvla_model_pkg = types.ModuleType("starVLA.model")
    starvla_framework_pkg = types.ModuleType("starVLA.model.framework")
    baseframework_mod = types.ModuleType("starVLA.model.framework.base_framework")
    baseframework_mod.baseframework = baseframework_obj

    monkeypatch.setitem(sys.modules, "starVLA", starvla_pkg)
    monkeypatch.setitem(sys.modules, "starVLA.model", starvla_model_pkg)
    monkeypatch.setitem(sys.modules, "starVLA.model.framework", starvla_framework_pkg)
    monkeypatch.setitem(
        sys.modules,
        "starVLA.model.framework.base_framework",
        baseframework_mod,
    )


def test_get_model_requires_model_path():
    cfg = OmegaConf.create(
        {
            "action_dim": 7,
            "num_action_chunks": 2,
        }
    )
    with pytest.raises(ValueError):
        get_model(cfg)


def test_get_model_loads_latest_checkpoint_and_builds_wrapper(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "run"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "step_0001.pt").write_bytes(b"a")
    (ckpt_dir / "step_0002.pt").write_bytes(b"b")

    dummy = _DummyStarVLAModel(framework_name="QwenFast")
    load_calls: list[str] = []
    _install_fake_starvla_module(monkeypatch, dummy, load_calls)

    cfg = OmegaConf.create(
        {
            "model_path": str(run_dir),
            "action_dim": 7,
            "num_action_chunks": 2,
            "add_value_head": True,
            "unnorm_key": "franka",
            "disable_action_unnormalization": False,
            "norm_stats": {
                "franka": {
                    "action": {
                        "q99": [1.0] * 7,
                        "q01": [-1.0] * 7,
                        "mask": [True] * 7,
                    }
                }
            },
        }
    )

    model = get_model(cfg, torch_dtype=torch.float16)

    assert isinstance(model, StarVLAForRLActionPrediction)
    assert load_calls and load_calls[0].endswith("step_0002.pt")
    assert dummy.last_to_dtype == torch.float16
    assert "franka" in dummy.norm_stats
    assert model._action_norm_stats is not None


def test_get_model_replace_norm_stats_from_file(tmp_path: Path, monkeypatch):
    ckpt_path = tmp_path / "checkpoint.pt"
    ckpt_path.write_bytes(b"x")

    dummy = _DummyStarVLAModel(framework_name="QwenAdapter")
    dummy.norm_stats = {
        "old": {"action": {"q99": [9.0] * 7, "q01": [-9.0] * 7, "mask": [True] * 7}}
    }
    load_calls: list[str] = []
    _install_fake_starvla_module(monkeypatch, dummy, load_calls)

    stats_path = tmp_path / "norm_stats.json"
    stats_payload = {
        "franka": {
            "action": {
                "q99": [2.0] * 7,
                "q01": [-2.0] * 7,
                "mask": [True] * 7,
            }
        }
    }
    stats_path.write_text(json.dumps(stats_payload), encoding="utf-8")

    cfg = OmegaConf.create(
        {
            "model_path": str(ckpt_path),
            "action_dim": 7,
            "num_action_chunks": 2,
            "unnorm_key": "franka",
            "norm_stats_path": str(stats_path),
            "replace_norm_stats": True,
        }
    )

    _ = get_model(cfg, torch_dtype=None)

    assert load_calls and load_calls[0].endswith("checkpoint.pt")
    assert dummy.norm_stats == stats_payload
