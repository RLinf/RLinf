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

from __future__ import annotations

import re
from pathlib import Path

import torch


def _map_vidar_transformer_key(key: str) -> str | None:
    if key.endswith(".te_attention._extra_state"):
        return None

    exact_prefixes = (
        ("text_embedding.0.", "condition_embedder.text_embedder.linear_1."),
        ("text_embedding.2.", "condition_embedder.text_embedder.linear_2."),
        ("time_embedding.0.", "condition_embedder.time_embedder.linear_1."),
        ("time_embedding.2.", "condition_embedder.time_embedder.linear_2."),
        ("time_projection.1.", "condition_embedder.time_proj."),
        ("head.head.", "proj_out."),
    )
    for source, target in exact_prefixes:
        if key.startswith(source):
            return key.replace(source, target, 1)

    if key == "head.modulation":
        return "scale_shift_table"

    block_modulation_match = re.fullmatch(r"blocks\.(\d+)\.modulation", key)
    if block_modulation_match:
        return f"blocks.{block_modulation_match.group(1)}.scale_shift_table"

    replacements = (
        (".self_attn.q.", ".attn1.to_q."),
        (".self_attn.k.", ".attn1.to_k."),
        (".self_attn.v.", ".attn1.to_v."),
        (".self_attn.o.", ".attn1.to_out.0."),
        (".self_attn.norm_q.", ".attn1.norm_q."),
        (".self_attn.norm_k.", ".attn1.norm_k."),
        (".cross_attn.q.", ".attn2.to_q."),
        (".cross_attn.k.", ".attn2.to_k."),
        (".cross_attn.v.", ".attn2.to_v."),
        (".cross_attn.o.", ".attn2.to_out.0."),
        (".cross_attn.norm_q.", ".attn2.norm_q."),
        (".cross_attn.norm_k.", ".attn2.norm_k."),
        (".ffn.0.", ".ffn.net.0.proj."),
        (".ffn.2.", ".ffn.net.2."),
        (".norm3.", ".norm2."),
    )
    mapped_key = key
    for source, target in replacements:
        mapped_key = mapped_key.replace(source, target)
    return mapped_key


def load_vidar_transformer_weights(
    transformer: torch.nn.Module,
    vidar_path: str | None,
) -> None:
    weight_path = Path(str(vidar_path))
    state_dict = torch.load(
        weight_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    converted_state_dict = {}
    for key, value in state_dict.items():
        mapped_key = _map_vidar_transformer_key(key)
        if mapped_key is not None:
            converted_state_dict[mapped_key] = value

    missing_keys, unexpected_keys = transformer.load_state_dict(
        converted_state_dict,
        strict=True,
    )
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Failed to load converted Vidar transformer weights: "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )


__all__ = ["load_vidar_transformer_weights"]
