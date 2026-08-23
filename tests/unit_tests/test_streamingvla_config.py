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

"""Static contract tests for the StreamingVLA SFT recipe."""

from pathlib import Path

import yaml

from rlinf.config import EMBODIED_MODEL, SupportedModel

_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENT = _ROOT / "examples/sft/config/libero_sft_streamingvla.yaml"
_MODEL = _ROOT / "examples/sft/config/model/streamingvla.yaml"


def test_streamingvla_is_registered_as_an_embodied_model():
    """The public model selector exposes exactly one StreamingVLA value."""
    assert SupportedModel.STREAMINGVLA.value == "streamingvla"
    assert SupportedModel.STREAMINGVLA in EMBODIED_MODEL


def test_streamingvla_recipe_has_canonical_training_values():
    """The checked-in recipe matches the validated 100k-step SFP setup."""
    experiment = yaml.safe_load(_EXPERIMENT.read_text(encoding="utf-8"))
    model = yaml.safe_load(_MODEL.read_text(encoding="utf-8"))
    streaming = model["streamingvla"]

    assert model["model_type"] == "streamingvla"
    assert streaming == {
        "config_name": "pi05_libero_sfp",
        "use_sfp": True,
        "use_action_states": True,
        "action_horizon": 10,
        "model_action_dim": 32,
        "sigma": 0.16,
        "noise_decay": 4.0,
        "data": {
            "repo_id": "/path/to/libero_lerobot_dataset",
            "assets": {
                "assets_dir": "/path/to/dataset-parent",
                "asset_id": "libero_lerobot_dataset",
            },
            "extra_delta_transform": False,
            "action_env_dim": 7,
        },
    }
    assert experiment["actor"]["micro_batch_size"] == 4
    assert experiment["actor"]["global_batch_size"] == 16
    assert experiment["actor"]["seed"] == 42
    assert experiment["runner"]["max_steps"] == 100_000
    assert experiment["runner"]["save_interval"] == 5_000
    assert experiment["actor"]["optim"]["lr_warmup_steps"] == 10_000
    assert experiment["runner"]["logger"]["logger_backends"] == ["tensorboard"]


def test_streamingvla_configs_contain_no_private_server_paths():
    """PR configs use placeholders and do not encode personal W&B settings."""
    text = _EXPERIMENT.read_text(encoding="utf-8") + _MODEL.read_text(encoding="utf-8")
    assert "/mnt" + "/public/" not in text
    assert "wandb_entity" not in text
    assert "/path/to/" in text
