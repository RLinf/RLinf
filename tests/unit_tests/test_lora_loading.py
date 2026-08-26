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
from peft import LoraConfig, get_peft_model
from torch import nn

from rlinf.models.embodiment.reward.vlm_trend_success_potential_reward_model import (
    load_lora_adapter,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)

    def forward(self, inputs):
        return self.q_proj(inputs)


def test_load_adapter_from_peft_directory(tmp_path) -> None:
    source = get_peft_model(
        TinyModel(), LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj"])
    )
    with torch.no_grad():
        source.base_model.model.q_proj.lora_A.default.weight.fill_(0.5)
    adapter_dir = tmp_path / "actor" / "lora_adapter"
    source.save_pretrained(adapter_dir)

    loaded = load_lora_adapter(TinyModel(), str(adapter_dir))
    source.save_pretrained(tmp_path / "success")
    loaded = load_lora_adapter(loaded, str(tmp_path / "success"), "success")

    weight = loaded.base_model.model.q_proj.lora_A.default.weight
    assert torch.allclose(weight, torch.full_like(weight, 0.5))
    assert loaded.active_adapter == "default"


def test_load_adapter_rejects_checkpoint_root(tmp_path) -> None:
    source = get_peft_model(
        TinyModel(), LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj"])
    )
    source.save_pretrained(tmp_path / "actor" / "lora_adapter")

    with pytest.raises(FileNotFoundError, match="model_state_dict/full_weights.pt"):
        load_lora_adapter(TinyModel(), str(tmp_path))


def test_load_adapter_normalizes_rlinf_default_keys(tmp_path) -> None:
    source = get_peft_model(
        TinyModel(), LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj"])
    )
    with torch.no_grad():
        source.base_model.model.q_proj.lora_B.default.weight.fill_(0.25)
    weights_path = tmp_path / "full_weights.pt"
    torch.save(
        {key: value for key, value in source.state_dict().items() if "lora_" in key},
        weights_path,
    )

    loaded = load_lora_adapter(TinyModel(), str(weights_path))

    weight = loaded.base_model.model.q_proj.lora_B.default.weight
    assert torch.allclose(weight, torch.full_like(weight, 0.25))
