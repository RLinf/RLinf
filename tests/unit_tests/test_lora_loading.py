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

import pytest
import torch
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from torch import nn

from rlinf.models import apply_lora, resolve_lora_target_modules
from rlinf.models.embodiment.reward.vlm_reward_model import _load_adapter


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)

    def forward(self, inputs):
        return self.q_proj(inputs)


def test_resolve_lora_target_modules() -> None:
    assert "proj" in resolve_lora_target_modules(OmegaConf.create({}))
    assert resolve_lora_target_modules(
        OmegaConf.create({"lora_target_modules": "q,k,v"})
    ) == ["q", "k", "v"]
    with pytest.raises(ValueError, match="empty list"):
        resolve_lora_target_modules(OmegaConf.create({"lora_target_modules": ""}))


def test_apply_lora_honours_explicit_target_modules() -> None:
    cfg = OmegaConf.create(
        {
            "is_lora": True,
            "model_type": "mlp_policy",
            "lora_rank": 2,
            "lora_path": None,
            "lora_target_modules": ["q_proj"],
        }
    )

    model = apply_lora(TinyModel(), cfg)

    assert set(next(iter(model.peft_config.values())).target_modules) == {"q_proj"}


def test_load_adapter_resolves_checkpoint_root(tmp_path) -> None:
    source = get_peft_model(
        TinyModel(), LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj"])
    )
    with torch.no_grad():
        source.base_model.model.q_proj.lora_A.default.weight.fill_(0.5)
    source.save_pretrained(tmp_path / "actor" / "lora_adapter")

    loaded = _load_adapter(TinyModel(), str(tmp_path))
    source.save_pretrained(tmp_path / "success")
    loaded = _load_adapter(loaded, str(tmp_path / "success"), "success")

    weight = loaded.base_model.model.q_proj.lora_A.default.weight
    assert torch.allclose(weight, torch.full_like(weight, 0.5))
    assert loaded.active_adapter == "default"


def test_load_adapter_normalizes_rlinf_default_keys(tmp_path) -> None:
    source = get_peft_model(
        TinyModel(), LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj"])
    )
    with torch.no_grad():
        source.base_model.model.q_proj.lora_B.default.weight.fill_(0.25)
    torch.save(
        {key: value for key, value in source.state_dict().items() if "lora_" in key},
        tmp_path / "full_weights.pt",
    )

    loaded = _load_adapter(TinyModel(), str(tmp_path))

    weight = loaded.base_model.model.q_proj.lora_B.default.weight
    assert torch.allclose(weight, torch.full_like(weight, 0.25))
