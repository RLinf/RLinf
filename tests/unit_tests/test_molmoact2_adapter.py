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

"""Tests for the MolmoAct2 evaluation adapter."""

import sys
import types

from omegaconf import OmegaConf

from rlinf.models.embodiment.molmoact2 import get_model


class _FakeMolmoAct2Config:
    def __init__(
        self,
        *,
        checkpoint_path,
        num_steps,
        inference_action_mode,
        discrete_action_tokenizer,
        enable_depth_reasoning,
        norm_tag,
    ):
        self.values = {
            "checkpoint_path": checkpoint_path,
            "num_steps": num_steps,
            "inference_action_mode": inference_action_mode,
            "discrete_action_tokenizer": discrete_action_tokenizer,
            "enable_depth_reasoning": enable_depth_reasoning,
            "norm_tag": norm_tag,
        }


class _FakeMolmoAct2Policy:
    def __init__(self, config):
        self.config = config

    def select_action(self, batch):
        del batch


def _install_fake_lerobot_modules(monkeypatch):
    module_names = [
        "lerobot",
        "lerobot.policies",
        "lerobot.policies.molmoact2",
    ]
    for module_name in module_names:
        monkeypatch.setitem(sys.modules, module_name, types.ModuleType(module_name))

    config_module_name = "lerobot.policies.molmoact2.configuration_molmoact2"
    config_module = types.ModuleType(config_module_name)
    config_module.MolmoAct2Config = _FakeMolmoAct2Config
    monkeypatch.setitem(sys.modules, config_module_name, config_module)

    modeling_module_name = "lerobot.policies.molmoact2.modeling_molmoact2"
    modeling_module = types.ModuleType(modeling_module_name)
    modeling_module.MolmoAct2Policy = _FakeMolmoAct2Policy
    monkeypatch.setitem(sys.modules, modeling_module_name, modeling_module)


def test_get_model_uses_supported_upstream_config_fields(monkeypatch):
    _install_fake_lerobot_modules(monkeypatch)
    cfg = OmegaConf.create(
        {
            "model_path": "/tmp/molmoact2-checkpoint",
            "num_steps": 10,
            "inference_action_mode": "continuous",
            "discrete_action_tokenizer": "allenai/MolmoAct2-FAST-Tokenizer",
            "enable_depth_reasoning": False,
            "norm_tag": "libero",
        }
    )

    model = get_model(cfg)

    assert model.config.values == {
        "checkpoint_path": "/tmp/molmoact2-checkpoint",
        "num_steps": 10,
        "inference_action_mode": "continuous",
        "discrete_action_tokenizer": "allenai/MolmoAct2-FAST-Tokenizer",
        "enable_depth_reasoning": False,
        "norm_tag": "libero",
    }
    assert callable(model.predict_action_batch)
