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

"""CPU tests for checkpoint convertors."""

import numpy as np
import pytest
import torch

from rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf import (
    _normalize_state_dict_keys,
)
from rlinf.utils.ckpt_convertor.fsdp_convertor.utils import (
    copy_model_config_and_code,
)
from rlinf.utils.ckpt_convertor.openpi.jax_to_openpi import _unwrap_value_leaves


def test_jax_value_wrappers_are_unwrapped():
    kernel = np.zeros((4, 2), dtype=np.float32)
    bias = np.ones(2, dtype=np.float32)
    tree = {
        "PaliGemma": {
            "img": {"pos_embedding": {"value": kernel}},
            "llm": {"layers": {"Dense_0": {"kernel": {"value": kernel}}}},
        },
        "action_in_proj": {"kernel": {"value": kernel}, "bias": {"value": bias}},
    }

    unwrapped = _unwrap_value_leaves(tree)

    assert unwrapped["PaliGemma"]["img"]["pos_embedding"] is kernel
    assert unwrapped["PaliGemma"]["llm"]["layers"]["Dense_0"]["kernel"] is kernel
    assert unwrapped["action_in_proj"]["kernel"] is kernel
    assert unwrapped["action_in_proj"]["bias"] is bias


def test_fsdp_checkpoint_matches_model_weight_names():
    qwen = "q_proj.base_layer.weight"
    old = "_orig_mod.module.v_proj.base_layer.weight"
    checkpoint = {qwen: torch.ones(1), old: torch.zeros(1)}
    expected = {qwen: checkpoint[qwen], "v_proj.weight": checkpoint[old]}
    assert _normalize_state_dict_keys(checkpoint, expected.keys()) == expected


def test_fsdp_checkpoint_keeps_conflicting_alias_for_strict_load():
    model = torch.nn.Linear(1, 1, bias=False)
    checkpoint = {"weight": torch.ones(1, 1), "module.weight": torch.zeros(1, 1)}
    normalized = _normalize_state_dict_keys(checkpoint, model.state_dict().keys())

    with pytest.raises(RuntimeError, match="Unexpected key"):
        model.load_state_dict(normalized)
    torch.testing.assert_close(normalized["weight"], checkpoint["weight"])


def test_copy_model_config_and_code_preserves_chat_template_and_merges(tmp_path):
    src = tmp_path / "base_model"
    dst = tmp_path / "exported"
    src.mkdir()
    (src / "config.json").write_text("{}", encoding="utf-8")
    (src / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    (src / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    (src / "model.safetensors").write_bytes(b"weights")

    copy_model_config_and_code(str(src), str(dst))

    assert (dst / "config.json").read_text(encoding="utf-8") == "{}"
    assert (dst / "chat_template.jinja").read_text(encoding="utf-8") == "{{ messages }}"
    assert (dst / "merges.txt").read_text(encoding="utf-8") == "#version: 0.2\n"
    assert not (dst / "model.safetensors").exists()
