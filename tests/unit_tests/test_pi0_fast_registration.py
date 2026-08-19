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
    text_tokenizer_name = "google/paligemma-3b-pt-224"
    action_tokenizer_name = "physical-intelligence/fast"


def _pi0_fast_cfg(**overrides):
    cfg = {
        "model_type": "pi0_fast",
        "precision": None,
        "is_lora": False,
        "lora_rank": 32,
        "lora_target_scope": "all_linear",
        "load_to_device": False,
        "model_path": "lerobot/pi0fast-libero",
        "action_dim": 7,
        "num_action_chunks": 10,
        "pi0_fast": {
            "revision": "840f4b503f4c09110421c33c810a85b6684fd658",
            "text_tokenizer_name": "google/paligemma-3b-pt-224",
            "text_tokenizer_revision": ("35e4f46485b4d07967e7e9935bc3786aad50687c"),
            "action_tokenizer_name": "jadechoghari/tokenizer-lib-mean",
            "action_tokenizer_revision": ("79ae83e3cbd8786dcb84b628569f8d076ca8151e"),
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
    monkeypatch.setattr(pi0_fast, "_resolve_model_path", lambda path, cfg: path)
    monkeypatch.setattr(pi0_fast, "_load_optional_processor", lambda *args: None)
    monkeypatch.setattr(
        pi0_fast,
        "_load_policy_config",
        lambda pi0_fast_module, model_path, cfg: fake_config,
    )

    model = get_model(_pi0_fast_cfg())

    assert isinstance(model, PI0FastForRLActionPrediction)
    assert model.policy.model_path == "lerobot/pi0fast-libero"
    assert model.policy.config is fake_config
    assert model.action_dim == 7
    assert model.num_action_chunks == 10


def test_pi0_fast_get_model_resolves_pinned_hf_revision(monkeypatch):
    import rlinf.models.embodiment.pi0_fast as pi0_fast

    fake_config = _FakePI0FastConfig()
    seen = {}
    monkeypatch.setattr(pi0_fast, "_load_lerobot_pi0_fast", lambda: _FakePI0FastModule)
    monkeypatch.setattr(
        pi0_fast,
        "_resolve_hf_snapshot",
        lambda name_or_path, **kwargs: (
            "/hf-cache/snapshots/pi0fast-libero"
            if name_or_path == "lerobot/pi0fast-libero"
            else name_or_path
        ),
    )

    def _fake_load_policy_config(pi0_fast_module, model_path, cfg):
        seen["model_path"] = model_path
        return fake_config

    monkeypatch.setattr(pi0_fast, "_load_policy_config", _fake_load_policy_config)

    model = get_model(_pi0_fast_cfg())

    assert seen["model_path"] == "/hf-cache/snapshots/pi0fast-libero"
    assert model.policy.model_path == "/hf-cache/snapshots/pi0fast-libero"
    assert model.policy.load_kwargs == {}


def test_pi0_fast_get_model_preserves_policy_checkpoint_dtype(monkeypatch):
    import rlinf.models.embodiment.pi0_fast as pi0_fast

    fake_config = _FakePI0FastConfig()
    monkeypatch.setattr(pi0_fast, "_load_lerobot_pi0_fast", lambda: _FakePI0FastModule)
    monkeypatch.setattr(pi0_fast, "_resolve_model_path", lambda path, cfg: path)
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


def test_pi0_fast_get_model_applies_model_specific_lora(monkeypatch):
    import rlinf.models.embodiment.pi0_fast as pi0_fast

    fake_config = _FakePI0FastConfig()
    seen = {}
    monkeypatch.setattr(pi0_fast, "_load_lerobot_pi0_fast", lambda: _FakePI0FastModule)
    monkeypatch.setattr(pi0_fast, "_resolve_model_path", lambda path, cfg: path)
    monkeypatch.setattr(pi0_fast, "_load_optional_processor", lambda *args: None)
    monkeypatch.setattr(
        pi0_fast,
        "_load_policy_config",
        lambda pi0_fast_module, model_path, cfg: fake_config,
    )

    def _fake_apply_lora(policy, rank, target_scope):
        seen["policy"] = policy
        seen["rank"] = rank
        seen["target_scope"] = target_scope
        policy.lora_applied = True
        return policy

    monkeypatch.setattr(
        pi0_fast,
        "_apply_pi0_fast_lora",
        _fake_apply_lora,
        raising=False,
    )

    model = get_model(_pi0_fast_cfg(is_lora=True, lora_rank=16))

    assert model.policy is seen["policy"]
    assert model.policy.lora_applied is True
    assert seen["rank"] == 16
    assert seen["target_scope"] == "all_linear"


def test_pi0_fast_rejects_unvalidated_lora_target_scope():
    import rlinf.models.embodiment.pi0_fast as pi0_fast

    try:
        pi0_fast._resolve_pi0_fast_lora_target_modules("language")
    except ValueError as exc:
        assert "all_linear" in str(exc)
    else:
        raise AssertionError("Expected language-only LoRA to be rejected")


def test_pi0_fast_repo_id_is_distinguished_from_nested_local_path():
    import rlinf.models.embodiment.pi0_fast as pi0_fast

    assert pi0_fast._looks_like_hf_repo_id("lerobot/pi0fast-libero") is True
    assert pi0_fast._looks_like_hf_repo_id("google/paligemma-3b-pt-224") is True
    # A checkpoint directory resolved against an unexpected cwd must not be
    # mistaken for a hub repo, otherwise the failure surfaces as an HF 404.
    assert (
        pi0_fast._looks_like_hf_repo_id("outputs/run1/checkpoints/global_step_50")
        is False
    )


def test_pi0_fast_local_checkpoint_does_not_require_model_revision(
    monkeypatch, tmp_path
):
    import rlinf.models.embodiment.pi0_fast as pi0_fast

    fake_config = _FakePI0FastConfig()
    monkeypatch.setattr(pi0_fast, "_load_lerobot_pi0_fast", lambda: _FakePI0FastModule)
    monkeypatch.setattr(pi0_fast, "_load_optional_processor", lambda *args: None)
    monkeypatch.setattr(
        pi0_fast,
        "_load_policy_config",
        lambda pi0_fast_module, model_path, cfg: fake_config,
    )

    cfg = _pi0_fast_cfg(
        model_path=str(tmp_path),
        pi0_fast={
            "text_tokenizer_name": "google/paligemma-3b-pt-224",
            "text_tokenizer_revision": "35e4f46485b4d07967e7e9935bc3786aad50687c",
            "action_tokenizer_name": "jadechoghari/tokenizer-lib-mean",
            "action_tokenizer_revision": "79ae83e3cbd8786dcb84b628569f8d076ca8151e",
        },
    )

    model = get_model(cfg)

    assert model.policy.model_path == str(tmp_path)


def test_pi0_fast_local_checkpoint_still_requires_tokenizer_pins(monkeypatch, tmp_path):
    import rlinf.models.embodiment.pi0_fast as pi0_fast

    monkeypatch.setattr(pi0_fast, "_load_lerobot_pi0_fast", lambda: _FakePI0FastModule)
    cfg = _pi0_fast_cfg(model_path=str(tmp_path), pi0_fast={})

    try:
        get_model(cfg)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected missing tokenizer pins to fail")

    assert "revision" not in message.split("missing: ")[1].split(", ")
    assert "text_tokenizer_revision" in message
    assert "action_tokenizer_name" in message


def test_pi0_fast_requires_all_public_artifact_pins(monkeypatch):
    import rlinf.models.embodiment.pi0_fast as pi0_fast

    monkeypatch.setattr(pi0_fast, "_load_lerobot_pi0_fast", lambda: _FakePI0FastModule)
    cfg = _pi0_fast_cfg(pi0_fast={"revision": "model-only"})

    try:
        get_model(cfg)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected missing tokenizer revisions to fail")

    assert "text_tokenizer_name" in message
    assert "action_tokenizer_revision" in message
