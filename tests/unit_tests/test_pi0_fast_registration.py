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

import torch
from omegaconf import OmegaConf

from rlinf.config import EMBODIED_MODEL, SupportedModel
from rlinf.models import get_model


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


def _pi0_fast_cfg(**overrides):
    cfg = {
        "model_type": "pi0_fast",
        "precision": None,
        "is_lora": False,
        "lora_rank": 32,
        "lora_target_scope": "all_linear",
        "load_to_device": False,
        "model_path": "/path/to/pi0fast-libero",
        "action_dim": 7,
        "num_action_chunks": 10,
        "pi0_fast": {
            "text_tokenizer_name": "/path/to/paligemma-3b-pt-224",
            "action_tokenizer_name": "/path/to/tokenizer-lib-mean",
        },
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def test_pi0_fast_supported_model_is_embodied():
    model_type = SupportedModel("pi0_fast")
    assert model_type in EMBODIED_MODEL


def test_pi0_fast_get_model_uses_lazy_lerobot_loader(monkeypatch):
    import rlinf.models.embodiment.pi0_fast as pi0_fast
    from rlinf.models.embodiment.pi0_fast.pi0_fast_action_model import (
        PI0FastForRLActionPrediction,
    )

    fake_config = _FakePI0FastConfig()
    monkeypatch.setattr(pi0_fast, "_load_lerobot_pi0_fast", lambda: _FakePI0FastModule)
    monkeypatch.setattr(pi0_fast, "_load_optional_processor", lambda *args: None)
    monkeypatch.setattr(
        pi0_fast,
        "_load_policy_config",
        lambda pi0_fast_module, model_path, cfg: fake_config,
    )

    model = get_model(_pi0_fast_cfg())

    assert isinstance(model, PI0FastForRLActionPrediction)
    assert model.policy.model_path == "/path/to/pi0fast-libero"
    assert model.policy.config is fake_config
    assert model.action_dim == 7
    assert model.num_action_chunks == 10


def test_pi0_fast_get_model_preserves_policy_checkpoint_dtype(monkeypatch):
    import rlinf.models.embodiment.pi0_fast as pi0_fast

    fake_config = _FakePI0FastConfig()
    monkeypatch.setattr(pi0_fast, "_load_lerobot_pi0_fast", lambda: _FakePI0FastModule)
    monkeypatch.setattr(pi0_fast, "_load_optional_processor", lambda *args: None)
    monkeypatch.setattr(
        pi0_fast,
        "_load_policy_config",
        lambda pi0_fast_module, model_path, cfg: fake_config,
    )

    model = get_model(_pi0_fast_cfg(precision="bf16"))

    assert model.policy.proj.weight.dtype == torch.float32


def test_pi0_fast_get_model_import_error_is_actionable(monkeypatch):
    import rlinf.models.embodiment.pi0_fast as pi0_fast

    def _raise_import_error(name):
        if name == "lerobot.policies.pi0_fast":
            raise ImportError("missing lerobot")
        return __import__(name)

    monkeypatch.setattr(pi0_fast.importlib, "import_module", _raise_import_error)

    try:
        pi0_fast._load_lerobot_pi0_fast()
    except ImportError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected ImportError")

    assert "requirements/install.sh embodied" in message
    assert "--model pi0_fast --env libero" in message


def test_pi0_fast_get_model_applies_all_linear_lora(monkeypatch):
    import rlinf.models.embodiment.pi0_fast as pi0_fast

    fake_config = _FakePI0FastConfig()
    monkeypatch.setattr(pi0_fast, "_load_lerobot_pi0_fast", lambda: _FakePI0FastModule)
    monkeypatch.setattr(pi0_fast, "_load_optional_processor", lambda *args: None)
    monkeypatch.setattr(
        pi0_fast,
        "_load_policy_config",
        lambda pi0_fast_module, model_path, cfg: fake_config,
    )

    model = get_model(_pi0_fast_cfg(is_lora=True, lora_rank=16))

    trainable_names = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]
    assert trainable_names
    assert all("lora_" in name for name in trainable_names)
    assert hasattr(model.policy.proj, "lora_A")


def test_pi0_fast_rejects_unvalidated_lora_target_scope(monkeypatch):
    import rlinf.models.embodiment.pi0_fast as pi0_fast

    fake_config = _FakePI0FastConfig()
    monkeypatch.setattr(pi0_fast, "_load_lerobot_pi0_fast", lambda: _FakePI0FastModule)
    monkeypatch.setattr(pi0_fast, "_load_optional_processor", lambda *args: None)
    monkeypatch.setattr(
        pi0_fast,
        "_load_policy_config",
        lambda pi0_fast_module, model_path, cfg: fake_config,
    )

    try:
        get_model(
            _pi0_fast_cfg(
                is_lora=True,
                lora_target_scope="language",
            )
        )
    except ValueError as exc:
        assert "lora_target_scope" in str(exc)
    else:
        raise AssertionError("Expected language-only LoRA to be rejected")
