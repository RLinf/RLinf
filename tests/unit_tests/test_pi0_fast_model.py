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

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.models import get_model
from rlinf.models.embodiment.pi0_fast.data_pipeline import (
    build_lerobot_batch_from_env_obs,
)


class _FakePI0FastPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(1, 1)

    @classmethod
    def from_pretrained(cls, model_path, *, config=None, **kwargs):
        policy = cls()
        policy.model_path = model_path
        policy.config = config
        policy.load_kwargs = kwargs
        return policy


class _FakePI0FastModule:
    PI0FastPolicy = _FakePI0FastPolicy


class _FakePI0FastConfig:
    text_tokenizer_name = "/path/to/paligemma-3b-pt-224"
    action_tokenizer_name = "/path/to/tokenizer-lib-mean"


def _pi0_fast_cfg():
    return OmegaConf.create(
        {
            "model_type": "pi0_fast",
            "precision": "bf16",
            "is_lora": False,
            "load_to_device": False,
            "model_path": "/path/to/pi0fast-libero",
            "action_dim": 7,
            "num_action_chunks": 10,
            "pi0_fast": {
                "text_tokenizer_name": "/path/to/paligemma-3b-pt-224",
                "action_tokenizer_name": "/path/to/tokenizer-lib-mean",
            },
        }
    )


def test_pi0_fast_get_model_builds_wrapper_without_recasting_checkpoint(monkeypatch):
    from rlinf.models.embodiment.pi0_fast import pi0_fast_action_model
    from rlinf.models.embodiment.pi0_fast.pi0_fast_action_model import (
        PI0FastForRLActionPrediction,
    )

    fake_config = _FakePI0FastConfig()
    monkeypatch.setattr(
        pi0_fast_action_model,
        "_load_lerobot_pi0_fast",
        lambda: _FakePI0FastModule,
    )
    monkeypatch.setattr(
        pi0_fast_action_model, "_load_optional_processor", lambda *args: None
    )
    monkeypatch.setattr(
        pi0_fast_action_model,
        "_load_policy_config",
        lambda pi0_fast_module, model_path, cfg: fake_config,
    )

    model = get_model(_pi0_fast_cfg())

    assert isinstance(model, PI0FastForRLActionPrediction)
    assert model.policy.model_path == "/path/to/pi0fast-libero"
    assert model.policy.config is fake_config
    assert model.policy.proj.weight.dtype == torch.float32
    assert (model.action_dim, model.num_action_chunks) == (7, 10)


def _env_obs(main_images):
    batch_size = main_images.shape[0]
    return {
        "main_images": main_images,
        "states": torch.zeros(batch_size, 8),
        "task_descriptions": ["pick up the object"] * batch_size,
    }


def test_build_lerobot_batch_maps_and_scales_libero_observations():
    main_images = torch.zeros(2, 4, 4, 3, dtype=torch.uint8)
    main_images[:, 0, 0, 0] = 1
    env_obs = _env_obs(main_images)
    env_obs["wrist_images"] = torch.full_like(main_images, 255)

    batch = build_lerobot_batch_from_env_obs(env_obs)

    assert batch["observation.images.image"].shape == (2, 3, 4, 4)
    assert torch.allclose(
        batch["observation.images.image"].max(), torch.tensor(1.0 / 255.0)
    )
    assert torch.allclose(batch["observation.images.image2"], torch.ones(2, 3, 4, 4))
    assert batch["observation.state"].shape == (2, 8)
    assert batch["task"] == ["pick up the object"] * 2


def test_pi0_fast_policy_config_loads_public_checkpoint_schema(tmp_path):
    module = pytest.importorskip("lerobot.policies.pi0_fast")
    from rlinf.models.embodiment.pi0_fast.pi0_fast_action_model import (
        _load_policy_config,
    )

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "type": "pi0_fast",
                "chunk_size": 10,
                "n_action_steps": 10,
                "action_tokenizer_name": "jadechoghari/fast-libero-tokenizer-mean-std",
                "device": "cuda",
                "gradient_checkpointing": True,
                "input_features": {
                    "observation.state": {"type": "STATE", "shape": [32]},
                },
                "output_features": {
                    "action": {"type": "ACTION", "shape": [7]},
                },
            }
        )
    )
    cfg = OmegaConf.create(
        {
            "load_to_device": False,
            "pi0_fast": {
                "text_tokenizer_name": "/tmp/local-paligemma-tokenizer",
                "action_tokenizer_name": "/tmp/local-fast-tokenizer",
                "gradient_checkpointing": False,
                "require_action_token_prefix": False,
            },
        }
    )

    policy_config = _load_policy_config(module, str(tmp_path), cfg)

    assert isinstance(policy_config, module.PI0FastConfig)
    assert policy_config.text_tokenizer_name == "/tmp/local-paligemma-tokenizer"
    assert policy_config.action_tokenizer_name == "/tmp/local-fast-tokenizer"
    assert policy_config.device == "cpu"
    assert policy_config.gradient_checkpointing is False
    assert policy_config.validate_action_token_prefix is False


def test_pi0_fast_postprocessor_receives_action_converters(monkeypatch):
    pytest.importorskip("lerobot.policies.pi0_fast")
    from lerobot.processor import PolicyProcessorPipeline

    from rlinf.models.embodiment.pi0_fast.pi0_fast_action_model import (
        _load_optional_processor,
    )

    captured = {}

    def fake_from_pretrained(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        PolicyProcessorPipeline,
        "from_pretrained",
        fake_from_pretrained,
    )

    _load_optional_processor(
        "/tmp/local-pi0-fast",
        "post",
        OmegaConf.create({}),
    )

    assert callable(captured["to_transition"])
    assert callable(captured["to_output"])
