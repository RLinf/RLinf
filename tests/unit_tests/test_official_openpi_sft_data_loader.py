# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the official OpenPI SFT data-loader adapter."""

import dataclasses
import importlib

from omegaconf import OmegaConf


@dataclasses.dataclass(frozen=True)
class _OpenPiModelConfig:
    action_horizon: int = 50
    action_dim: int = 32


@dataclasses.dataclass(frozen=True)
class _OpenPiTrainConfig:
    model: _OpenPiModelConfig = dataclasses.field(default_factory=_OpenPiModelConfig)
    num_workers: int = 2
    seed: int = 0


class _DataLoader:
    def __init__(self, config):
        self.config = config

    def data_config(self):
        return "data-config"


def test_openpi_rlinf_data_loader_uses_configured_action_horizon(monkeypatch) -> None:
    """The OpenPI dataset should produce the same chunk length as the model."""
    adapter = importlib.import_module(
        "rlinf.data.datasets.openpi_rlinf.official_sft_data_loader"
    )
    dataconfig = importlib.import_module("rlinf.models.embodiment.openpi.dataconfig")
    openpi_data_loader = importlib.import_module("openpi.training.data_loader")
    official_config = _OpenPiTrainConfig()
    captured = {}

    monkeypatch.setattr(adapter, "resolve_lerobot_repo_id", lambda _: "test/repo")
    monkeypatch.setattr(
        dataconfig,
        "get_openpi_config",
        lambda *args, **kwargs: official_config,
    )

    def _create_data_loader(config, **kwargs):
        captured["config"] = config
        return _DataLoader(config)

    monkeypatch.setattr(openpi_data_loader, "create_data_loader", _create_data_loader)
    cfg = OmegaConf.create(
        {
            "data": {"num_workers": 4},
            "actor": {
                "micro_batch_size": 8,
                "seed": 7,
                "model": {
                    "model_type": "openpi_rlinf",
                    "model_path": "/tmp/model",
                    "num_action_chunks": 20,
                    "openpi": {
                        "config_name": "pi05_aloha_robotwin",
                        "model_action_dim": 32,
                    },
                },
            },
        }
    )

    loader, data_config = adapter.build_official_openpi_sft_dataloader(
        cfg, world_size=2, rank=0, data_paths="/tmp/data"
    )

    assert loader.config is captured["config"]
    assert data_config == "data-config"
    assert captured["config"].model.action_horizon == 20
    assert captured["config"].num_workers == 4
    assert captured["config"].seed == 7
    assert official_config.model.action_horizon == 50
