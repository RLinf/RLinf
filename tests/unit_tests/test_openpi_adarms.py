# Copyright 2025 Shirui Chen.
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

import importlib
import sys

import torch
from transformers import (
    GemmaConfig,
    GemmaForCausalLM,
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
)

from rlinf.models.embodiment.openpi import (
    _ensure_openpi_transformers_overlay,
    _inject_missing_paligemma_embeddings,
    _retie_paligemma_embeddings,
    ensure_openpi_runtime_compat,
)
from rlinf.models.embodiment.openpi.adarms_expert import (
    AdaRMSGemmaRMSNorm,
    enable_openpi_adarms_expert,
    enable_openpi_transformers_compat,
)


class _FakePaliGemmaWithExpert:
    def __init__(self, gemma_model, paligemma=None):
        self.gemma_expert = type("_FakeGemmaExpert", (), {"model": gemma_model})()
        self.paligemma = (
            paligemma if paligemma is not None else _build_tiny_paligemma_model()
        )


def _build_tiny_gemma_model():
    config = GemmaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
        num_hidden_layers=1,
        vocab_size=32,
        use_adarms=True,
        adarms_cond_dim=16,
    )
    return GemmaForCausalLM(config).model


def _build_tiny_paligemma_model():
    config = PaliGemmaConfig()
    config.text_config.hidden_size = 16
    config.text_config.intermediate_size = 32
    config.text_config.num_hidden_layers = 1
    config.text_config.num_attention_heads = 2
    config.text_config.num_key_value_heads = 2
    config.text_config.head_dim = 8
    config.text_config.vocab_size = 32
    config.vision_config.hidden_size = 16
    config.vision_config.intermediate_size = 32
    config.vision_config.num_hidden_layers = 1
    config.vision_config.num_attention_heads = 2
    config.vision_config.image_size = 14
    config.vision_config.patch_size = 14
    config.hidden_size = 16
    config.projection_dim = 16
    config.vocab_size = 32
    return PaliGemmaForConditionalGeneration(config)


def test_ensure_openpi_transformers_overlay_installs_siglip_check_shim():
    import transformers.models.siglip as siglip_pkg

    original_module = sys.modules.pop("transformers.models.siglip.check", None)
    had_attr = hasattr(siglip_pkg, "check")
    original_attr = getattr(siglip_pkg, "check", None)
    if had_attr:
        delattr(siglip_pkg, "check")

    try:
        _ensure_openpi_transformers_overlay()
        check = importlib.import_module("transformers.models.siglip.check")
        assert check.check_whether_transformers_replace_is_installed_correctly()
    finally:
        sys.modules.pop("transformers.models.siglip.check", None)
        if original_module is not None:
            sys.modules["transformers.models.siglip.check"] = original_module
        if had_attr:
            siglip_pkg.check = original_attr


def test_ensure_openpi_runtime_compat_keeps_public_wrapper():
    import transformers.models.siglip as siglip_pkg

    original_module = sys.modules.pop("transformers.models.siglip.check", None)
    had_attr = hasattr(siglip_pkg, "check")
    original_attr = getattr(siglip_pkg, "check", None)
    if had_attr:
        delattr(siglip_pkg, "check")

    try:
        ensure_openpi_runtime_compat()
        check = importlib.import_module("transformers.models.siglip.check")
        assert check.check_whether_transformers_replace_is_installed_correctly()
    finally:
        sys.modules.pop("transformers.models.siglip.check", None)
        if original_module is not None:
            sys.modules["transformers.models.siglip.check"] = original_module
        if had_attr:
            siglip_pkg.check = original_attr


def test_enable_openpi_adarms_expert_exposes_dense_norm_weights():
    gemma_model = _build_tiny_gemma_model()
    wrapper = _FakePaliGemmaWithExpert(gemma_model)

    enable_openpi_adarms_expert(wrapper)

    state_dict = wrapper.gemma_expert.model.state_dict()
    assert "layers.0.input_layernorm.dense.weight" in state_dict
    assert "layers.0.post_attention_layernorm.dense.bias" in state_dict
    assert "norm.dense.weight" in state_dict
    assert isinstance(
        wrapper.gemma_expert.model.layers[0].input_layernorm,
        AdaRMSGemmaRMSNorm,
    )


def test_enable_openpi_adarms_expert_runs_conditional_forward():
    gemma_model = _build_tiny_gemma_model()
    wrapper = _FakePaliGemmaWithExpert(gemma_model)
    enable_openpi_adarms_expert(wrapper)

    inputs_embeds = torch.randn(2, 3, 16)
    attention_mask = torch.zeros(2, 1, 3, 3)
    position_ids = torch.arange(3).unsqueeze(0).expand(2, -1)
    adarms_cond = torch.randn(2, 16)

    outputs = wrapper.gemma_expert.model.forward(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        adarms_cond=adarms_cond,
        use_cache=False,
    )

    assert outputs.last_hidden_state.shape == (2, 3, 16)
    assert torch.isfinite(outputs.last_hidden_state).all()


def test_enable_openpi_transformers_compat_runs_vanilla_suffix_forward():
    gemma_model = _build_tiny_gemma_model()
    wrapper = _FakePaliGemmaWithExpert(
        gemma_model, paligemma=_build_tiny_paligemma_model()
    )
    enable_openpi_transformers_compat(wrapper)

    suffix_inputs = torch.randn(2, 3, 16)
    attention_mask = torch.zeros(2, 1, 3, 3)
    position_ids = torch.arange(3).unsqueeze(0).expand(2, -1)

    outputs, past_key_values = wrapper.forward(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[None, suffix_inputs],
        use_cache=False,
        adarms_cond=[None, None],
    )

    assert outputs[0] is None
    assert outputs[1].shape == (2, 3, 16)
    assert torch.isfinite(outputs[1]).all()
    assert past_key_values is None


def test_enable_openpi_transformers_compat_runs_vanilla_prefix_forward():
    gemma_model = _build_tiny_gemma_model()
    wrapper = _FakePaliGemmaWithExpert(
        gemma_model, paligemma=_build_tiny_paligemma_model()
    )
    enable_openpi_transformers_compat(wrapper)

    prefix_inputs = torch.randn(2, 3, 16)
    attention_mask = torch.zeros(2, 1, 3, 3)
    position_ids = torch.arange(3).unsqueeze(0).expand(2, -1)

    outputs, past_key_values = wrapper.forward(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_inputs, None],
        use_cache=True,
        adarms_cond=[None, None],
    )

    assert outputs[1] is None
    assert outputs[0].shape == (2, 3, 16)
    assert torch.isfinite(outputs[0]).all()
    assert past_key_values is not None


def test_paligemma_embedding_repair_backfills_and_reties():
    state_dict = {
        "paligemma_with_expert.paligemma.lm_head.weight": torch.randn(8, 4),
    }
    repaired = _inject_missing_paligemma_embeddings(state_dict)
    assert (
        "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        in repaired
    )
    assert torch.equal(
        repaired["paligemma_with_expert.paligemma.lm_head.weight"],
        repaired[
            "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        ],
    )

    paligemma = _build_tiny_paligemma_model()
    paligemma.model.language_model.embed_tokens = torch.nn.Embedding(32, 16)
    wrapper = type(
        "_FakeWrapper",
        (),
        {"paligemma_with_expert": type("_FakePali", (), {"paligemma": paligemma})()},
    )()

    _retie_paligemma_embeddings(wrapper)

    assert (
        wrapper.paligemma_with_expert.paligemma.lm_head.weight.data_ptr()
        == wrapper.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight.data_ptr()
    )
