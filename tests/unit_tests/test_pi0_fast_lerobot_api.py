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

"""Smoke tests executed only in the dedicated PI0-Fast runtime."""

import importlib
import importlib.metadata
import json

import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip(
    "lerobot.policies.pi0_fast",
    reason="PI0-Fast API checks require the dedicated LeRobot runtime.",
)


def test_lerobot_pi0_fast_api_is_available():
    module = importlib.import_module("lerobot.policies.pi0_fast")
    assert hasattr(module, "PI0FastPolicy")
    assert hasattr(module, "PI0FastConfig")

    constants = importlib.import_module("lerobot.utils.constants")
    assert constants.ACTION_TOKENS == "action.tokens"
    assert constants.ACTION_TOKEN_MASK == "action.token_mask"


def test_lerobot_install_uses_pinned_git_revision():
    distribution = importlib.metadata.distribution("lerobot")
    direct_url = json.loads(distribution.read_text("direct_url.json"))

    assert direct_url["vcs_info"]["commit_id"] == (
        "8a74e0ac6d01706d67fddfed682a09d694d9c8c0"
    )


def test_pi0_fast_policy_config_loads_public_checkpoint_schema(tmp_path):
    module = importlib.import_module("lerobot.policies.pi0_fast")
    from rlinf.models.embodiment.pi0_fast import _load_policy_config

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
                "action_tokenizer_name": "/tmp/local-fast-tokenizer",
                "gradient_checkpointing": False,
                "require_action_token_prefix": False,
            },
        }
    )

    policy_config = _load_policy_config(module, str(tmp_path), cfg)

    assert isinstance(policy_config, module.PI0FastConfig)
    assert policy_config.text_tokenizer_name == "google/paligemma-3b-pt-224"
    assert policy_config.action_tokenizer_name == "/tmp/local-fast-tokenizer"
    assert policy_config.device == "cpu"
    assert policy_config.gradient_checkpointing is False
    assert policy_config.validate_action_token_prefix is False


def test_pi0_fast_image_preprocess_does_not_require_policy_parameters():
    from rlinf.models.embodiment.pi0_fast.fast_replay import _preprocess_images

    class NoParamPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type(
                "Config",
                (),
                {
                    "image_features": {
                        "observation.images.base_0_rgb": object(),
                        "observation.images.left_wrist_0_rgb": object(),
                    },
                    "image_resolution": (4, 4),
                },
            )()

    batch = {
        "observation.images.base_0_rgb": torch.zeros(2, 3, 4, 4),
    }

    images, masks = _preprocess_images(NoParamPolicy(), batch)

    assert len(images) == 2
    assert len(masks) == 2
    assert images[0].shape == (2, 3, 4, 4)
    assert masks[0].tolist() == [True, True]
    assert masks[1].tolist() == [False, False]


def test_pi0_fast_all_linear_lora_preserves_policy_model_contract():
    from rlinf.models.embodiment.pi0_fast import _apply_pi0_fast_lora

    class SelfAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 4, bias=False)

    class LanguageLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = SelfAttention()

    class LanguageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([LanguageLayer()])

    class VisionTower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 4, bias=False)

    class PaliGemmaModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = LanguageModel()
            self.vision_tower = VisionTower()

    class PaliGemma(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = PaliGemmaModel()
            self.lm_head = torch.nn.Linear(4, 8, bias=False)

    class CoreModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.paligemma_with_expert = torch.nn.Module()
            self.paligemma_with_expert.paligemma = PaliGemma()

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = CoreModel()

    policy = Policy()
    core_model = policy.model
    language_q_proj = (
        core_model.paligemma_with_expert.paligemma.model.language_model.layers[
            0
        ].self_attn.q_proj
    )
    sample = torch.randn(2, 4)
    output_before = language_q_proj(sample).detach()

    policy = _apply_pi0_fast_lora(policy, rank=2, target_scope="all_linear")

    assert policy.model is core_model
    output_after = (
        policy.model.paligemma_with_expert.paligemma.model.language_model.layers[
            0
        ].self_attn.q_proj(sample)
    )
    assert torch.equal(output_before, output_after)

    trainable_names = [
        name for name, param in policy.named_parameters() if param.requires_grad
    ]
    assert len(trainable_names) == 6
    assert all("lora_" in name for name in trainable_names)
    assert any("vision_tower" in name for name in trainable_names)
    assert any("language_model" in name for name in trainable_names)
    assert any("lm_head" in name for name in trainable_names)


def test_pi0_fast_lora_rejects_unknown_target_scope():
    from rlinf.models.embodiment.pi0_fast import _apply_pi0_fast_lora

    with pytest.raises(ValueError, match="lora_target_scope"):
        _apply_pi0_fast_lora(
            torch.nn.Sequential(torch.nn.Linear(2, 2)),
            rank=2,
            target_scope="unknown",
        )
