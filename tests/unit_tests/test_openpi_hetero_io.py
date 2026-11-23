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

"""LIBERO + RoboDojo + RoboLab share one Pi0.5 network and keep their own I/O."""

import os
from pathlib import Path

import numpy as np
import openpi.models.pi0_config as pi0_config
import openpi.transforms as transforms
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from rlinf.config import validate_embodied_cfg
from rlinf.models.embodiment.openpi.transforms.pipeline import build_openpi_transforms

_REPO = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO / "examples/embodiment/config"
_ENVS = (
    ("libero", "pi05_libero", 8, 7, None),
    ("robodojo", "pi05_aloha_robotwin", 14, 14, {"adapt_to_pi": False}),
    ("robolab", "pi05_droid_robolab", 8, 8, None),
)


@pytest.fixture
def hetero_cfg(monkeypatch):
    monkeypatch.setenv("EMBODIED_PATH", str(_CONFIG_DIR.parent))
    monkeypatch.setenv("REPO_PATH", str(_REPO))
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        return compose(config_name="libero_robodojo_robolab_ppo_openpi_pi05")


def test_hetero_pi05_config_is_valid(hetero_cfg):
    validate_embodied_cfg(hetero_cfg)
    assert hetero_cfg.actor.model.model_path == hetero_cfg.rollout.model.model_path
    assert list(hetero_cfg.env.train.envs) == ["libero", "robodojo", "robolab"]
    assert hetero_cfg.env.train.envs.libero.openpi_io == "libero"
    assert hetero_cfg.env.train.envs.robodojo.openpi_io == "robodojo"
    assert hetero_cfg.env.train.envs.robolab.openpi_io == "robolab"


def test_hetero_pi05_config_rejects_an_unknown_io(hetero_cfg):
    cfg = OmegaConf.create(OmegaConf.to_container(hetero_cfg, resolve=True))
    with open_dict(cfg):
        cfg.env.train.envs.libero.openpi_io = "missing"
    with pytest.raises(AssertionError, match="openpi_io must name"):
        validate_embodied_cfg(cfg)


def test_shared_network_slices_each_env_action():
    """Data pipelines differ; tokenization uses one Pi0.5 config."""
    os.environ.setdefault("OPENPI_DATA_HOME", "/opt/assets/.cache/openpi")
    shared = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=10,
        discrete_state_input=True,
    )
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    for _name, config_name, state_dim, action_dim, data_kwargs in _ENVS:
        inputs, outputs = build_openpi_transforms(
            "",
            config_name,
            data_kwargs=data_kwargs,
            model_config=shared,
            use_model_path_assets=False,
        )
        state = np.zeros(state_dim, dtype=np.float32)
        wrist = (
            np.zeros((2, 224, 224, 3), dtype=np.uint8)
            if config_name == "pi05_aloha_robotwin"
            else image
        )
        encoded = transforms.compose(inputs)(
            {
                "observation/image": image,
                "observation/wrist_image": wrist,
                "observation/state": state,
                "prompt": "pick up the object",
            }
        )
        assert "tokenized_prompt" in encoded
        decoded = transforms.compose(outputs)(
            {
                "actions": np.zeros((shared.action_horizon, shared.action_dim)),
                "state": state,
            }
        )
        assert decoded["actions"].shape == (shared.action_horizon, action_dim)
