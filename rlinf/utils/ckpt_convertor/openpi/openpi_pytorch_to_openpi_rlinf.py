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

"""Convert an OpenPI PyTorch checkpoint to the OpenPI_RLinf layout.

The OpenPI PyTorch layout uses ``paligemma_with_expert.*`` keys; the
OpenPI_RLinf layout uses bare ``Pi0`` keys. The
``old_to_new_state_dict`` function owns the key renaming
and weight transforms (SigLIP Q/K/V concat, LLM MLP transpose+stack,
norm-prefix rewrites). When the source directory carries a ``config.json`` it
is copied verbatim; the norm-stats file is copied across too.

Within this module, ``old`` denotes the OpenPI PyTorch layout and ``new``
denotes the OpenPI_RLinf layout. The public CLI mode remains
``openpi_pytorch_to_openpi_rlinf``.
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


def old_to_new_state_dict(
    old_sd: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert an old-format state dict to the new ``Pi0`` layout.

    Handles key renaming and weight transformations:
      - SigLIP Q/K/V concat -> in_proj_weight/bias
      - LLM MLP gate/up transpose+stack -> w_gating (2, features, hidden_dim)
      - LLM MLP down transpose -> w_linear
      - Action-expert RMSNorm: Pi0 ``*.weight`` -> ``*.1.scale``;
        Pi0.5 AdaRMS ``*.dense.*`` -> ``*.1.ada_modulation.*``
    """
    openpi_pytorch_state_dict = old_sd
    openpi_rlinf_state_dict: dict[str, torch.Tensor] = {}

    _OPENPI_PYTORCH_SIGLIP = (
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model."
    )

    # Stem
    for suffix in (".weight", ".bias"):
        source_key = _OPENPI_PYTORCH_SIGLIP + "embeddings.patch_embedding" + suffix
        if source_key in openpi_pytorch_state_dict:
            openpi_rlinf_state_dict["img.stem" + suffix] = openpi_pytorch_state_dict[
                source_key
            ]

    # OpenPI stores this as an (num_patches, width) nn.Embedding weight; RLinf
    # holds a (1, num_patches, width) parameter, so add the broadcast dimension.
    source_key = _OPENPI_PYTORCH_SIGLIP + "embeddings.position_embedding.weight"
    if source_key in openpi_pytorch_state_dict:
        position_embedding = openpi_pytorch_state_dict[source_key]
        openpi_rlinf_state_dict["img.pos_embedding"] = (
            position_embedding.unsqueeze(0)
            if position_embedding.dim() == 2
            else position_embedding
        )

    # Encoder layers (0..26)
    for layer_index in range(27):
        source_prefix = f"{_OPENPI_PYTORCH_SIGLIP}encoder.layers.{layer_index}."
        target_prefix = f"img.encoder.layers.{layer_index}."

        for source_name, target_name in [
            ("layer_norm1", "norm1"),
            ("layer_norm2", "norm2"),
        ]:
            for suffix in (".weight", ".bias"):
                source_key = f"{source_prefix}{source_name}{suffix}"
                if source_key in openpi_pytorch_state_dict:
                    openpi_rlinf_state_dict[f"{target_prefix}{target_name}{suffix}"] = (
                        openpi_pytorch_state_dict[source_key]
                    )

        qkv_weights = []
        qkv_biases = []
        for projection in ("q_proj", "k_proj", "v_proj"):
            weight_key = f"{source_prefix}self_attn.{projection}.weight"
            bias_key = f"{source_prefix}self_attn.{projection}.bias"
            if weight_key in openpi_pytorch_state_dict:
                qkv_weights.append(openpi_pytorch_state_dict[weight_key])
            if bias_key in openpi_pytorch_state_dict:
                qkv_biases.append(openpi_pytorch_state_dict[bias_key])
        if qkv_weights:
            openpi_rlinf_state_dict[f"{target_prefix}attn.in_proj_weight"] = torch.cat(
                qkv_weights, dim=0
            )
        if qkv_biases:
            openpi_rlinf_state_dict[f"{target_prefix}attn.in_proj_bias"] = torch.cat(
                qkv_biases, dim=0
            )

        for suffix in (".weight", ".bias"):
            source_key = f"{source_prefix}self_attn.out_proj{suffix}"
            if source_key in openpi_pytorch_state_dict:
                openpi_rlinf_state_dict[f"{target_prefix}attn.out_proj{suffix}"] = (
                    openpi_pytorch_state_dict[source_key]
                )

        for name in ("fc1", "fc2"):
            for suffix in (".weight", ".bias"):
                source_key = f"{source_prefix}mlp.{name}{suffix}"
                if source_key in openpi_pytorch_state_dict:
                    openpi_rlinf_state_dict[f"{target_prefix}mlp.{name}{suffix}"] = (
                        openpi_pytorch_state_dict[source_key]
                    )

    # Post layernorm
    for suffix in (".weight", ".bias"):
        source_key = _OPENPI_PYTORCH_SIGLIP + "post_layernorm" + suffix
        if source_key in openpi_pytorch_state_dict:
            openpi_rlinf_state_dict["img.encoder.norm" + suffix] = (
                openpi_pytorch_state_dict[source_key]
            )

    # Multi-modal projector
    for suffix in (".weight", ".bias"):
        source_key = (
            "paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
            + suffix
        )
        if source_key in openpi_pytorch_state_dict:
            openpi_rlinf_state_dict["img.head" + suffix] = openpi_pytorch_state_dict[
                source_key
            ]

    # PaliGemma LLM (expert 0)
    _PALI_LLM = "paligemma_with_expert.paligemma.model.language_model."
    for layer_index in range(18):
        source_prefix = f"{_PALI_LLM}layers.{layer_index}."
        target_prefix = f"llm.layers.{layer_index}."

        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            source_key = f"{source_prefix}self_attn.{projection}.weight"
            if source_key in openpi_pytorch_state_dict:
                openpi_rlinf_state_dict[
                    f"{target_prefix}attn.{projection}.0.weight"
                ] = openpi_pytorch_state_dict[source_key]

        gate_key = f"{source_prefix}mlp.gate_proj.weight"
        up_key = f"{source_prefix}mlp.up_proj.weight"
        if (
            gate_key in openpi_pytorch_state_dict
            and up_key in openpi_pytorch_state_dict
        ):
            gate_transposed = openpi_pytorch_state_dict[gate_key].T.contiguous()
            up_transposed = openpi_pytorch_state_dict[up_key].T.contiguous()
            openpi_rlinf_state_dict[f"{target_prefix}mlps.0.w_gating"] = torch.stack(
                [gate_transposed, up_transposed], dim=0
            )

        down_key = f"{source_prefix}mlp.down_proj.weight"
        if down_key in openpi_pytorch_state_dict:
            openpi_rlinf_state_dict[f"{target_prefix}mlps.0.w_linear"] = (
                openpi_pytorch_state_dict[down_key].T.contiguous()
            )

        for source_name, target_name in [
            ("input_layernorm", "pre_attention_norms"),
            ("post_attention_layernorm", "pre_ffw_norms"),
        ]:
            source_key = f"{source_prefix}{source_name}.weight"
            if source_key in openpi_pytorch_state_dict:
                openpi_rlinf_state_dict[f"{target_prefix}{target_name}.0.scale"] = (
                    openpi_pytorch_state_dict[source_key]
                )

    source_key = _PALI_LLM + "norm.weight"
    if source_key in openpi_pytorch_state_dict:
        openpi_rlinf_state_dict["llm.final_norms.0.scale"] = openpi_pytorch_state_dict[
            source_key
        ]

    # Gemma action expert (expert 1)
    _GEMMA_EXPERT = "paligemma_with_expert.gemma_expert.model."
    for layer_index in range(18):
        source_prefix = f"{_GEMMA_EXPERT}layers.{layer_index}."
        target_prefix = f"llm.layers.{layer_index}."

        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            source_key = f"{source_prefix}self_attn.{projection}.weight"
            if source_key in openpi_pytorch_state_dict:
                openpi_rlinf_state_dict[
                    f"{target_prefix}attn.{projection}.1.weight"
                ] = openpi_pytorch_state_dict[source_key]

        gate_key = f"{source_prefix}mlp.gate_proj.weight"
        up_key = f"{source_prefix}mlp.up_proj.weight"
        if (
            gate_key in openpi_pytorch_state_dict
            and up_key in openpi_pytorch_state_dict
        ):
            gate_transposed = openpi_pytorch_state_dict[gate_key].T.contiguous()
            up_transposed = openpi_pytorch_state_dict[up_key].T.contiguous()
            openpi_rlinf_state_dict[f"{target_prefix}mlps.1.w_gating"] = torch.stack(
                [gate_transposed, up_transposed], dim=0
            )

        down_key = f"{source_prefix}mlp.down_proj.weight"
        if down_key in openpi_pytorch_state_dict:
            openpi_rlinf_state_dict[f"{target_prefix}mlps.1.w_linear"] = (
                openpi_pytorch_state_dict[down_key].T.contiguous()
            )

        for source_name, target_name in [
            ("input_layernorm", "pre_attention_norms"),
            ("post_attention_layernorm", "pre_ffw_norms"),
        ]:
            # Pi0.5: AdaRMS stored as ``*.dense.{weight,bias}``.
            for suffix in (".weight", ".bias"):
                source_key = f"{source_prefix}{source_name}.dense{suffix}"
                if source_key in openpi_pytorch_state_dict:
                    openpi_rlinf_state_dict[
                        f"{target_prefix}{target_name}.1.ada_modulation{suffix}"
                    ] = openpi_pytorch_state_dict[source_key]
            # Pi0: regular RMSNorm stored as ``*.weight``.
            source_key = f"{source_prefix}{source_name}.weight"
            if source_key in openpi_pytorch_state_dict:
                openpi_rlinf_state_dict[f"{target_prefix}{target_name}.1.scale"] = (
                    openpi_pytorch_state_dict[source_key]
                )

    for suffix in (".weight", ".bias"):
        source_key = _GEMMA_EXPERT + "norm.dense" + suffix
        if source_key in openpi_pytorch_state_dict:
            openpi_rlinf_state_dict["llm.final_norms.1.ada_modulation" + suffix] = (
                openpi_pytorch_state_dict[source_key]
            )
    source_key = _GEMMA_EXPERT + "norm.weight"
    if source_key in openpi_pytorch_state_dict:
        openpi_rlinf_state_dict["llm.final_norms.1.scale"] = openpi_pytorch_state_dict[
            source_key
        ]

    # The RLinf shared token embedder is PaliGemma's embedding (tied with
    # ``paligemma.lm_head``, width = PaliGemma width, e.g. 2048). The action
    # expert's 1024-wide head must not be used as the embedder.
    lm_head_key = None
    if "paligemma_with_expert.paligemma.lm_head.weight" in openpi_pytorch_state_dict:
        lm_head_key = "paligemma_with_expert.paligemma.lm_head.weight"
    elif (
        "paligemma_with_expert.gemma_expert.lm_head.weight" in openpi_pytorch_state_dict
    ):
        lm_head_key = "paligemma_with_expert.gemma_expert.lm_head.weight"
    if lm_head_key is not None:
        openpi_rlinf_state_dict["llm.embedder.embedding.weight"] = (
            openpi_pytorch_state_dict[lm_head_key]
        )

    # Action head (same names in both layouts)
    for key in openpi_pytorch_state_dict:
        if key.startswith(
            (
                "action_in_proj",
                "action_out_proj",
                "time_mlp_",
                "state_proj",
                "action_time_mlp_",
                "pointnet.",
            )
        ):
            openpi_rlinf_state_dict[key] = openpi_pytorch_state_dict[key]

    return openpi_rlinf_state_dict


def convert(
    input_model: str | pathlib.Path,
    input_norm_stats: str | pathlib.Path,
    output_model: str | pathlib.Path,
    output_norm_stats: str | pathlib.Path,
) -> pathlib.Path:
    """Convert an OpenPI PyTorch checkpoint to the OpenPI_RLinf layout.

    Loads ``model.safetensors`` from ``input_model`` (a directory or file),
    converts it via :func:`old_to_new_state_dict`, writes
    ``output_model/model.safetensors`` (copying ``config.json`` if present), and
    copies ``input_norm_stats`` verbatim to ``output_norm_stats``.
    """
    input_model = pathlib.Path(input_model)
    output_model = pathlib.Path(output_model)
    openpi_pytorch_path = resolve_model_safetensors(input_model)
    if not openpi_pytorch_path.exists():
        raise FileNotFoundError(
            f"OpenPI PyTorch checkpoint not found: {openpi_pytorch_path}"
        )

    openpi_pytorch_state_dict = load_safetensors(openpi_pytorch_path)
    new_sd = old_to_new_state_dict(openpi_pytorch_state_dict)
    save_safetensors(new_sd, output_model / "model.safetensors")

    config_src = input_model if input_model.is_dir() else input_model.parent
    copy_config_json(config_src, output_model)
    copy_norm_stats(input_norm_stats, output_norm_stats)
    return output_model


def add_arguments(parser) -> None:
    """Register the ``openpi_pytorch_to_openpi_rlinf`` mode arguments."""
    parser.add_argument(
        "--input-model",
        required=True,
        help="OpenPI PyTorch checkpoint directory or model.safetensors",
    )
    parser.add_argument(
        "--input-norm-stats", required=True, help="norm_stats.json to copy across"
    )
    parser.add_argument(
        "--output-model", required=True, help="output OpenPI_RLinf checkpoint dir"
    )
    parser.add_argument(
        "--output-norm-stats", required=True, help="destination norm_stats.json path"
    )


def run(args) -> None:
    """Execute the ``openpi_pytorch_to_openpi_rlinf`` mode."""
    convert(
        args.input_model,
        args.input_norm_stats,
        args.output_model,
        args.output_norm_stats,
    )
