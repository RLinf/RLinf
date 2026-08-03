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

"""Convert an RLinf PyTorch checkpoint to the OpenPI PyTorch layout.

This is the inverse of :mod:`openpi_pytorch2rlinf_pytorch`. RLinf PyTorch uses
bare ``Pi0`` keys while OpenPI PyTorch uses ``paligemma_with_expert.*`` keys.
RLinf PyTorch does not retain OpenPI's separate 1024-wide action-expert token
head, so an OpenPI PyTorch reference model is required to produce a complete
checkpoint.
"""

from __future__ import annotations

import os
import pathlib
import shutil

import torch

from rlinf.utils.ckpt_convertor.openpi._core import (
    NORM_STATS_SUBDIR,
    copy_norm_stats,
    load_safetensors,
    resolve_model_safetensors,
    save_safetensors,
)

ACTION_EXPERT_LM_HEAD = "paligemma_with_expert.gemma_expert.lm_head.weight"
_OPENPI_SIGLIP = "paligemma_with_expert.paligemma.model.vision_tower.vision_model."
_OPENPI_PALIGEMMA_LLM = "paligemma_with_expert.paligemma.model.language_model."
_OPENPI_GEMMA_EXPERT = "paligemma_with_expert.gemma_expert.model."


def rlinf_pytorch_to_openpi_pytorch_state_dict(
    rlinf_pytorch_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map every representable RLinf PyTorch tensor to OpenPI PyTorch keys."""
    openpi_pytorch_state_dict: dict[str, torch.Tensor] = {}

    for suffix in (".weight", ".bias"):
        rlinf_key = "img.stem" + suffix
        if rlinf_key in rlinf_pytorch_state_dict:
            openpi_pytorch_state_dict[
                _OPENPI_SIGLIP + "embeddings.patch_embedding" + suffix
            ] = rlinf_pytorch_state_dict[rlinf_key]

    if "img.pos_embedding" in rlinf_pytorch_state_dict:
        position_embedding = rlinf_pytorch_state_dict["img.pos_embedding"]
        if position_embedding.dim() == 3 and position_embedding.shape[0] == 1:
            position_embedding = position_embedding.squeeze(0)
        openpi_pytorch_state_dict[
            _OPENPI_SIGLIP + "embeddings.position_embedding.weight"
        ] = position_embedding

    for layer_index in range(27):
        openpi_prefix = f"{_OPENPI_SIGLIP}encoder.layers.{layer_index}."
        rlinf_prefix = f"img.encoder.layers.{layer_index}."
        for openpi_name, rlinf_name in [
            ("layer_norm1", "norm1"),
            ("layer_norm2", "norm2"),
        ]:
            for suffix in (".weight", ".bias"):
                rlinf_key = f"{rlinf_prefix}{rlinf_name}{suffix}"
                if rlinf_key in rlinf_pytorch_state_dict:
                    openpi_pytorch_state_dict[
                        f"{openpi_prefix}{openpi_name}{suffix}"
                    ] = rlinf_pytorch_state_dict[rlinf_key]

        rlinf_key = f"{rlinf_prefix}attn.in_proj_weight"
        if rlinf_key in rlinf_pytorch_state_dict:
            query, key, value = torch.chunk(
                rlinf_pytorch_state_dict[rlinf_key], 3, dim=0
            )
            openpi_pytorch_state_dict[f"{openpi_prefix}self_attn.q_proj.weight"] = (
                query.contiguous()
            )
            openpi_pytorch_state_dict[f"{openpi_prefix}self_attn.k_proj.weight"] = (
                key.contiguous()
            )
            openpi_pytorch_state_dict[f"{openpi_prefix}self_attn.v_proj.weight"] = (
                value.contiguous()
            )
        rlinf_key = f"{rlinf_prefix}attn.in_proj_bias"
        if rlinf_key in rlinf_pytorch_state_dict:
            query, key, value = torch.chunk(
                rlinf_pytorch_state_dict[rlinf_key], 3, dim=0
            )
            openpi_pytorch_state_dict[f"{openpi_prefix}self_attn.q_proj.bias"] = (
                query.contiguous()
            )
            openpi_pytorch_state_dict[f"{openpi_prefix}self_attn.k_proj.bias"] = (
                key.contiguous()
            )
            openpi_pytorch_state_dict[f"{openpi_prefix}self_attn.v_proj.bias"] = (
                value.contiguous()
            )

        for suffix in (".weight", ".bias"):
            rlinf_key = f"{rlinf_prefix}attn.out_proj{suffix}"
            if rlinf_key in rlinf_pytorch_state_dict:
                openpi_pytorch_state_dict[
                    f"{openpi_prefix}self_attn.out_proj{suffix}"
                ] = rlinf_pytorch_state_dict[rlinf_key]
        for name in ("fc1", "fc2"):
            for suffix in (".weight", ".bias"):
                rlinf_key = f"{rlinf_prefix}mlp.{name}{suffix}"
                if rlinf_key in rlinf_pytorch_state_dict:
                    openpi_pytorch_state_dict[f"{openpi_prefix}mlp.{name}{suffix}"] = (
                        rlinf_pytorch_state_dict[rlinf_key]
                    )

    for suffix in (".weight", ".bias"):
        rlinf_key = "img.encoder.norm" + suffix
        if rlinf_key in rlinf_pytorch_state_dict:
            openpi_pytorch_state_dict[_OPENPI_SIGLIP + "post_layernorm" + suffix] = (
                rlinf_pytorch_state_dict[rlinf_key]
            )
        rlinf_key = "img.head" + suffix
        if rlinf_key in rlinf_pytorch_state_dict:
            openpi_pytorch_state_dict[
                "paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
                + suffix
            ] = rlinf_pytorch_state_dict[rlinf_key]

    _convert_llm_expert(
        rlinf_pytorch_state_dict,
        openpi_pytorch_state_dict,
        _OPENPI_PALIGEMMA_LLM,
        expert_index=0,
        action_expert=False,
    )
    _convert_llm_expert(
        rlinf_pytorch_state_dict,
        openpi_pytorch_state_dict,
        _OPENPI_GEMMA_EXPERT,
        expert_index=1,
        action_expert=True,
    )

    rlinf_key = "llm.final_norms.0.scale"
    if rlinf_key in rlinf_pytorch_state_dict:
        openpi_pytorch_state_dict[_OPENPI_PALIGEMMA_LLM + "norm.weight"] = (
            rlinf_pytorch_state_dict[rlinf_key]
        )
    for suffix in (".weight", ".bias"):
        rlinf_key = f"llm.final_norms.1.ada_modulation{suffix}"
        if rlinf_key in rlinf_pytorch_state_dict:
            openpi_pytorch_state_dict[_OPENPI_GEMMA_EXPERT + "norm.dense" + suffix] = (
                rlinf_pytorch_state_dict[rlinf_key]
            )

    rlinf_key = "llm.embedder.embedding.weight"
    if rlinf_key in rlinf_pytorch_state_dict:
        openpi_pytorch_state_dict["paligemma_with_expert.paligemma.lm_head.weight"] = (
            rlinf_pytorch_state_dict[rlinf_key]
        )

    for key, tensor in rlinf_pytorch_state_dict.items():
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
            openpi_pytorch_state_dict[key] = tensor
    return openpi_pytorch_state_dict


def _convert_llm_expert(
    rlinf_pytorch_state_dict: dict[str, torch.Tensor],
    openpi_pytorch_state_dict: dict[str, torch.Tensor],
    openpi_expert_prefix: str,
    *,
    expert_index: int,
    action_expert: bool,
) -> None:
    """Convert one of the two RLinf Pi0 LLM experts to OpenPI PyTorch keys."""
    for layer_index in range(18):
        rlinf_prefix = f"llm.layers.{layer_index}."
        openpi_prefix = f"{openpi_expert_prefix}layers.{layer_index}."
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            rlinf_key = f"{rlinf_prefix}attn.{projection}.{expert_index}.weight"
            if rlinf_key in rlinf_pytorch_state_dict:
                openpi_pytorch_state_dict[
                    f"{openpi_prefix}self_attn.{projection}.weight"
                ] = rlinf_pytorch_state_dict[rlinf_key]

        rlinf_key = f"{rlinf_prefix}mlps.{expert_index}.w_gating"
        if rlinf_key in rlinf_pytorch_state_dict:
            gating = rlinf_pytorch_state_dict[rlinf_key]
            openpi_pytorch_state_dict[f"{openpi_prefix}mlp.gate_proj.weight"] = gating[
                0
            ].T.contiguous()
            openpi_pytorch_state_dict[f"{openpi_prefix}mlp.up_proj.weight"] = gating[
                1
            ].T.contiguous()
        rlinf_key = f"{rlinf_prefix}mlps.{expert_index}.w_linear"
        if rlinf_key in rlinf_pytorch_state_dict:
            openpi_pytorch_state_dict[f"{openpi_prefix}mlp.down_proj.weight"] = (
                rlinf_pytorch_state_dict[rlinf_key].T.contiguous()
            )

        for openpi_name, rlinf_name in [
            ("input_layernorm", "pre_attention_norms"),
            ("post_attention_layernorm", "pre_ffw_norms"),
        ]:
            if action_expert:
                for suffix in (".weight", ".bias"):
                    rlinf_key = (
                        f"{rlinf_prefix}{rlinf_name}.{expert_index}.ada_modulation"
                        f"{suffix}"
                    )
                    if rlinf_key in rlinf_pytorch_state_dict:
                        openpi_pytorch_state_dict[
                            f"{openpi_prefix}{openpi_name}.dense{suffix}"
                        ] = rlinf_pytorch_state_dict[rlinf_key]
            else:
                rlinf_key = f"{rlinf_prefix}{rlinf_name}.{expert_index}.scale"
                if rlinf_key in rlinf_pytorch_state_dict:
                    openpi_pytorch_state_dict[
                        f"{openpi_prefix}{openpi_name}.weight"
                    ] = rlinf_pytorch_state_dict[rlinf_key]


def convert_trained_ckpt(
    input_ckpt: str,
    output_dir: str,
    reference_model: str,
    norm_stats: str | None = None,
) -> None:
    """Convert RLinf PyTorch trained weights with an OpenPI PyTorch reference."""
    import safetensors.torch

    if str(input_ckpt).endswith(".safetensors"):
        rlinf_pytorch_state_dict = safetensors.torch.load_file(input_ckpt, device="cpu")
    else:
        rlinf_pytorch_state_dict = torch.load(
            input_ckpt, map_location="cpu", weights_only=True
        )
    rlinf_pytorch_state_dict = {
        key.removeprefix("_orig_mod."): tensor
        for key, tensor in rlinf_pytorch_state_dict.items()
    }
    openpi_pytorch_state_dict = rlinf_pytorch_to_openpi_pytorch_state_dict(
        rlinf_pytorch_state_dict
    )

    reference_safetensors = os.path.join(reference_model, "model.safetensors")
    reference_state_dict = safetensors.torch.load_file(reference_safetensors)
    reference_head = reference_state_dict[ACTION_EXPERT_LM_HEAD]
    if (
        ACTION_EXPERT_LM_HEAD not in openpi_pytorch_state_dict
        or openpi_pytorch_state_dict[ACTION_EXPERT_LM_HEAD].shape
        != reference_head.shape
    ):
        openpi_pytorch_state_dict[ACTION_EXPERT_LM_HEAD] = reference_head.clone()
    for key, tensor in openpi_pytorch_state_dict.items():
        if tensor.dtype != torch.bfloat16:
            openpi_pytorch_state_dict[key] = tensor.to(torch.bfloat16)

    reference_keys = set(reference_state_dict)
    converted_keys = set(openpi_pytorch_state_dict)
    missing = reference_keys - converted_keys
    extra = converted_keys - reference_keys
    shape_mismatches = [
        (
            key,
            tuple(reference_state_dict[key].shape),
            tuple(openpi_pytorch_state_dict[key].shape),
        )
        for key in sorted(reference_keys & converted_keys)
        if openpi_pytorch_state_dict[key].shape != reference_state_dict[key].shape
    ]
    if missing or extra or shape_mismatches:
        raise RuntimeError(
            "Validation failed — keys/shapes do not match the OpenPI PyTorch "
            f"reference: missing={sorted(missing)} extra={sorted(extra)} "
            f"shape_mismatches={shape_mismatches}"
        )

    os.makedirs(output_dir, exist_ok=True)
    safetensors.torch.save_file(
        openpi_pytorch_state_dict,
        os.path.join(output_dir, "model.safetensors"),
    )
    reference_config = os.path.join(reference_model, "config.json")
    if os.path.exists(reference_config):
        shutil.copy2(reference_config, os.path.join(output_dir, "config.json"))
    if norm_stats and os.path.exists(norm_stats):
        norm_dst_dir = os.path.join(output_dir, *NORM_STATS_SUBDIR.parts)
        os.makedirs(norm_dst_dir, exist_ok=True)
        shutil.copy2(norm_stats, os.path.join(norm_dst_dir, "norm_stats.json"))


def convert(
    input_model: str | pathlib.Path,
    input_norm_stats: str | pathlib.Path,
    output_model: str | pathlib.Path,
    output_norm_stats: str | pathlib.Path,
) -> pathlib.Path:
    """Convert RLinf PyTorch weights to OpenPI PyTorch without a reference.

    This always raises because an OpenPI PyTorch checkpoint needs the separate
    action-expert head, which RLinf PyTorch cannot reconstruct. Use
    ``--reference-model`` to call :func:`convert_trained_ckpt` instead.
    """
    input_model = pathlib.Path(input_model)
    output_model = pathlib.Path(output_model)
    rlinf_pytorch_path = resolve_model_safetensors(input_model)
    if not rlinf_pytorch_path.exists():
        raise FileNotFoundError(
            f"RLinf PyTorch checkpoint not found: {rlinf_pytorch_path}"
        )
    rlinf_pytorch_state_dict = load_safetensors(rlinf_pytorch_path)
    openpi_pytorch_state_dict = rlinf_pytorch_to_openpi_pytorch_state_dict(
        rlinf_pytorch_state_dict
    )
    if ACTION_EXPERT_LM_HEAD not in openpi_pytorch_state_dict:
        raise RuntimeError(
            "rlinf_pytorch2openpi_pytorch cannot produce a complete OpenPI "
            "PyTorch checkpoint because the action-expert head "
            f"{ACTION_EXPERT_LM_HEAD!r} is not carried by RLinf PyTorch. Pass "
            "--reference-model (an OpenPI PyTorch model dir)."
        )
    save_safetensors(openpi_pytorch_state_dict, output_model / "model.safetensors")
    copy_norm_stats(input_norm_stats, output_norm_stats)
    return output_model


def add_arguments(parser) -> None:
    """Register the ``rlinf_pytorch2openpi_pytorch`` mode arguments."""
    parser.add_argument(
        "--input-model",
        required=True,
        help="RLinf PyTorch checkpoint directory, model.safetensors, or model.pt",
    )
    parser.add_argument(
        "--input-norm-stats", required=True, help="norm_stats.json to copy across"
    )
    parser.add_argument(
        "--output-model", required=True, help="output OpenPI PyTorch checkpoint dir"
    )
    parser.add_argument(
        "--output-norm-stats", required=True, help="destination norm_stats.json path"
    )
    parser.add_argument(
        "--reference-model",
        default=None,
        help=(
            "reference OpenPI PyTorch model directory used to source the "
            "action-expert lm_head and validate keys/shapes"
        ),
    )


def run(args) -> None:
    """Execute ``rlinf_pytorch2openpi_pytorch`` from parsed arguments."""
    if args.reference_model:
        input_path = pathlib.Path(args.input_model)
        if input_path.is_dir():
            input_path = resolve_model_safetensors(input_path)
        convert_trained_ckpt(
            input_ckpt=str(input_path),
            output_dir=args.output_model,
            reference_model=args.reference_model,
        )
        copy_norm_stats(args.input_norm_stats, args.output_norm_stats)
    else:
        convert(
            args.input_model,
            args.input_norm_stats,
            args.output_model,
            args.output_norm_stats,
        )
