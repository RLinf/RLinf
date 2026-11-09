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

"""Shared native-to-Diffusers names for export and live weight updates."""

# Same attention names as cosmos-framework's native -> Diffusers converter.
_ATTENTION_NAMES = (
    ("q_proj_moe_gen", "add_q_proj"),
    ("k_proj_moe_gen", "add_k_proj"),
    ("v_proj_moe_gen", "add_v_proj"),
    ("o_proj_moe_gen", "to_add_out"),
    ("q_norm_moe_gen", "norm_added_q"),
    ("k_norm_moe_gen", "norm_added_k"),
    ("q_proj", "to_q"),
    ("k_proj", "to_k"),
    ("v_proj", "to_v"),
    ("o_proj", "to_out"),
    ("q_norm", "norm_q"),
    ("k_norm", "norm_k"),
)
_ROOT_NAMES = (
    ("time_embedder.mlp.0.", "time_embedder.linear_1."),
    ("time_embedder.mlp.2.", "time_embedder.linear_2."),
    ("vae2llm.", "proj_in."),
    ("llm2vae.", "proj_out."),
    ("action2llm.", "action_proj_in."),
    ("llm2action.", "action_proj_out."),
)


def native_to_diffusers_key(name: str) -> str:
    if name.startswith("language_model."):
        name = name.removeprefix("language_model.").removeprefix("model.")
        for native, exported in _ATTENTION_NAMES:
            name = name.replace(f".self_attn.{native}.", f".self_attn.{exported}.")
        return name
    for native, exported in _ROOT_NAMES:
        if name.startswith(native):
            return exported + name.removeprefix(native)
    return name


def actor_to_sglang_key(name: str) -> str | None:
    """Select the online transformer and map its native actor key for serving."""
    if not name.startswith("omni.net."):
        return None
    return native_to_diffusers_key(name.removeprefix("omni.net."))
