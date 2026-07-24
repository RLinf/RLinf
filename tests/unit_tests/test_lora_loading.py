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

"""Unit tests for QwenTrend LoRA checkpoint helpers."""

from __future__ import annotations

import torch

from rlinf.models.embodiment.reward.vlm_reward_utils.lora_loading import (
    extract_lora_state,
    full_weights_pt_path,
    lora_config_from_state,
)
from rlinf.workers.sft.lora_checkpoint import full_weights_path


def test_full_weights_paths() -> None:
    assert full_weights_pt_path("/tmp/ckpt").endswith(
        "actor/model_state_dict/full_weights.pt"
    )
    assert full_weights_path("/tmp/ckpt/actor").endswith(
        "model_state_dict/full_weights.pt"
    )


def test_extract_lora_state_strips_module_prefix() -> None:
    state = {
        "module.base.weight": torch.ones(2),
        "module.q_proj.lora_A.weight": torch.ones(4, 8),
        "v_proj.lora_B.weight": torch.ones(8, 4),
    }
    lora_state = extract_lora_state(state)
    assert set(lora_state) == {
        "q_proj.lora_A.weight",
        "v_proj.lora_B.weight",
    }


def test_lora_config_from_state_infers_rank_and_targets() -> None:
    lora_state = {
        "model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(8, 16),
        "model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(16, 8),
        "model.layers.0.self_attn.v_proj.lora_A.weight": torch.zeros(8, 16),
        "model.layers.0.self_attn.v_proj.lora_B.weight": torch.zeros(16, 8),
    }
    config = lora_config_from_state(lora_state)
    assert config.r == 8
    assert config.lora_alpha == 8
    assert set(config.target_modules) == {"q_proj", "v_proj"}
