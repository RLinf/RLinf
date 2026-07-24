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

"""Convert an OLD-format PyTorch OpenPI 0.5 checkpoint to the NEW self-contained layout.

The old format uses the ``paligemma_with_expert.*`` key layout of the previous
PyTorch model (and the BEHAVIOR eval checkpoint); the new format is the bare
``Pi0`` layout the self-contained package loads. ``old_to_new_state_dict`` owns
the key renaming and weight transforms (SigLIP Q/K/V concat, LLM MLP
transpose+stack, norm-prefix rewrites). When the source dir carries a
``config.json`` it is copied verbatim; the norm-stats file is copied across too.
"""

from __future__ import annotations

import pathlib

import torch

from rlinf.utils.ckpt_convertor.openpi._core import (
    copy_config_json,
    copy_norm_stats,
    load_safetensors,
    resolve_model_safetensors,
    save_safetensors,
)


def old_to_new_state_dict(old_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert an old-format flat state dict to the new-format ``Pi0`` layout.

    Handles key renaming and weight transformations:
      - SigLIP Q/K/V concat -> in_proj_weight/bias
      - LLM MLP gate/up transpose+stack -> w_gating (2, features, hidden_dim)
      - LLM MLP down transpose -> w_linear
    """
    new_sd: dict[str, torch.Tensor] = {}

    _SIGLIP_OLD = "paligemma_with_expert.paligemma.model.vision_tower.vision_model."

    # Stem
    for suf in (".weight", ".bias"):
        ok = _SIGLIP_OLD + "embeddings.patch_embedding" + suf
        if ok in old_sd:
            new_sd["img.stem" + suf] = old_sd[ok]

    # Pos embedding. The old SigLIP stores this as an (num_patches, width)
    # nn.Embedding weight; the new SigLIPViT holds a (1, num_patches, width)
    # parameter, so add the leading broadcast dimension on import.
    ok = _SIGLIP_OLD + "embeddings.position_embedding.weight"
    if ok in old_sd:
        pos = old_sd[ok]
        new_sd["img.pos_embedding"] = pos.unsqueeze(0) if pos.dim() == 2 else pos

    # Encoder layers (0..26)
    for i in range(27):
        op = f"{_SIGLIP_OLD}encoder.layers.{i}."
        np = f"img.encoder.layers.{i}."

        for old_n, new_n in [("layer_norm1", "norm1"), ("layer_norm2", "norm2")]:
            for suf in (".weight", ".bias"):
                ok = f"{op}{old_n}{suf}"
                if ok in old_sd:
                    new_sd[f"{np}{new_n}{suf}"] = old_sd[ok]

        qkv_w = []
        qkv_b = []
        for proj in ("q_proj", "k_proj", "v_proj"):
            wk = f"{op}self_attn.{proj}.weight"
            bk = f"{op}self_attn.{proj}.bias"
            if wk in old_sd:
                qkv_w.append(old_sd[wk])
            if bk in old_sd:
                qkv_b.append(old_sd[bk])
        if qkv_w:
            new_sd[f"{np}attn.in_proj_weight"] = torch.cat(qkv_w, dim=0)
        if qkv_b:
            new_sd[f"{np}attn.in_proj_bias"] = torch.cat(qkv_b, dim=0)

        for suf in (".weight", ".bias"):
            ok = f"{op}self_attn.out_proj{suf}"
            if ok in old_sd:
                new_sd[f"{np}attn.out_proj{suf}"] = old_sd[ok]

        for name in ("fc1", "fc2"):
            for suf in (".weight", ".bias"):
                ok = f"{op}mlp.{name}{suf}"
                if ok in old_sd:
                    new_sd[f"{np}mlp.{name}{suf}"] = old_sd[ok]

    # Post layernorm
    for suf in (".weight", ".bias"):
        ok = _SIGLIP_OLD + "post_layernorm" + suf
        if ok in old_sd:
            new_sd["img.encoder.norm" + suf] = old_sd[ok]

    # Multi-modal projector
    for suf in (".weight", ".bias"):
        ok = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear" + suf
        if ok in old_sd:
            new_sd["img.head" + suf] = old_sd[ok]

    # --- PaliGemma LLM (expert 0) ---
    _PALI_LLM = "paligemma_with_expert.paligemma.model.language_model."
    for i in range(18):
        op = f"{_PALI_LLM}layers.{i}."
        np = f"llm.layers.{i}."

        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            ok = f"{op}self_attn.{proj}.weight"
            if ok in old_sd:
                new_sd[f"{np}attn.{proj}.0.weight"] = old_sd[ok]

        gk = f"{op}mlp.gate_proj.weight"
        uk = f"{op}mlp.up_proj.weight"
        if gk in old_sd and uk in old_sd:
            gate_t = old_sd[gk].T.contiguous()
            up_t = old_sd[uk].T.contiguous()
            new_sd[f"{np}mlps.0.w_gating"] = torch.stack([gate_t, up_t], dim=0)

        dk = f"{op}mlp.down_proj.weight"
        if dk in old_sd:
            new_sd[f"{np}mlps.0.w_linear"] = old_sd[dk].T.contiguous()

        for old_n, new_n in [
            ("input_layernorm", "pre_attention_norms"),
            ("post_attention_layernorm", "pre_ffw_norms"),
        ]:
            ok = f"{op}{old_n}.weight"
            if ok in old_sd:
                new_sd[f"{np}{new_n}.0.scale"] = old_sd[ok]

    ok = _PALI_LLM + "norm.weight"
    if ok in old_sd:
        new_sd["llm.final_norms.0.scale"] = old_sd[ok]

    # --- Gemma Action Expert (expert 1) ---
    _GEMMA_EXP = "paligemma_with_expert.gemma_expert.model."
    for i in range(18):
        op = f"{_GEMMA_EXP}layers.{i}."
        np = f"llm.layers.{i}."

        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            ok = f"{op}self_attn.{proj}.weight"
            if ok in old_sd:
                new_sd[f"{np}attn.{proj}.1.weight"] = old_sd[ok]

        gk = f"{op}mlp.gate_proj.weight"
        uk = f"{op}mlp.up_proj.weight"
        if gk in old_sd and uk in old_sd:
            gate_t = old_sd[gk].T.contiguous()
            up_t = old_sd[uk].T.contiguous()
            new_sd[f"{np}mlps.1.w_gating"] = torch.stack([gate_t, up_t], dim=0)

        dk = f"{op}mlp.down_proj.weight"
        if dk in old_sd:
            new_sd[f"{np}mlps.1.w_linear"] = old_sd[dk].T.contiguous()

        for old_n, new_n in [
            ("input_layernorm", "pre_attention_norms"),
            ("post_attention_layernorm", "pre_ffw_norms"),
        ]:
            for suf in (".weight", ".bias"):
                ok = f"{op}{old_n}.dense{suf}"
                if ok in old_sd:
                    new_sd[f"{np}{new_n}.1.ada_modulation{suf}"] = old_sd[ok]

    for suf in (".weight", ".bias"):
        ok = _GEMMA_EXP + "norm.dense" + suf
        if ok in old_sd:
            new_sd["llm.final_norms.1.ada_modulation" + suf] = old_sd[ok]

    # --- lm_head -> embedder ---
    # The new model's shared token embedder is PaliGemma's embedding (tied with
    # ``paligemma.lm_head``, width = paligemma width, e.g. 2048). The action
    # expert's ``gemma_expert.lm_head`` is a separate, narrower head (e.g. 1024)
    # and must NOT be used for the embedder, so prefer the PaliGemma key.
    lm_head_key = None
    if "paligemma_with_expert.paligemma.lm_head.weight" in old_sd:
        lm_head_key = "paligemma_with_expert.paligemma.lm_head.weight"
    elif "paligemma_with_expert.gemma_expert.lm_head.weight" in old_sd:
        lm_head_key = "paligemma_with_expert.gemma_expert.lm_head.weight"
    if lm_head_key is not None:
        new_sd["llm.embedder.embedding.weight"] = old_sd[lm_head_key]

    # --- Action head (same names in both formats) ---
    for k in old_sd:
        if k.startswith(
            (
                "action_in_proj",
                "action_out_proj",
                "time_mlp_",
                "state_proj",
                "action_time_mlp_",
                "pointnet.",
            )
        ):
            new_sd[k] = old_sd[k]

    return new_sd


def convert(
    input_model: str | pathlib.Path,
    input_norm_stats: str | pathlib.Path,
    output_model: str | pathlib.Path,
    output_norm_stats: str | pathlib.Path,
) -> pathlib.Path:
    """Convert an old-format checkpoint to the new layout.

    Loads the old ``model.safetensors`` from ``input_model`` (a directory or a
    file), converts it via :func:`old_to_new_state_dict`, writes
    ``output_model/model.safetensors`` (copying ``config.json`` if present), and
    copies ``input_norm_stats`` verbatim to ``output_norm_stats``.
    """
    input_model = pathlib.Path(input_model)
    output_model = pathlib.Path(output_model)
    old_path = resolve_model_safetensors(input_model)
    if not old_path.exists():
        raise FileNotFoundError(f"old checkpoint not found: {old_path}")

    old_sd = load_safetensors(old_path)
    new_sd = old_to_new_state_dict(old_sd)

    save_safetensors(new_sd, output_model / "model.safetensors")

    config_src = input_model if input_model.is_dir() else input_model.parent
    copy_config_json(config_src, output_model)

    copy_norm_stats(input_norm_stats, output_norm_stats)
    return output_model


def add_arguments(parser) -> None:
    """Register the ``old2new`` mode arguments on ``parser``."""
    parser.add_argument(
        "--input-model", required=True, help="old checkpoint dir or model.safetensors"
    )
    parser.add_argument(
        "--input-norm-stats", required=True, help="norm_stats.json to copy across"
    )
    parser.add_argument(
        "--output-model", required=True, help="output (new-format) checkpoint dir"
    )
    parser.add_argument(
        "--output-norm-stats", required=True, help="destination norm_stats.json path"
    )


def run(args) -> None:
    """Execute the ``old2new`` mode from parsed ``args``."""
    convert(
        args.input_model,
        args.input_norm_stats,
        args.output_model,
        args.output_norm_stats,
    )
