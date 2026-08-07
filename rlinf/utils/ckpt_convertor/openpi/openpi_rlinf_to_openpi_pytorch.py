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

"""Convert an OpenPI_RLinf checkpoint to the OpenPI PyTorch layout.

This implements the internal new-to-old conversion. OpenPI_RLinf uses
bare ``Pi0`` keys while OpenPI PyTorch uses ``paligemma_with_expert.*`` keys.
OpenPI_RLinf does not retain OpenPI's separate 1024-wide action-expert token
head, so an OpenPI PyTorch reference model is required to produce a complete
checkpoint.

Within this module, ``new`` denotes the OpenPI_RLinf layout and ``old``
denotes the OpenPI PyTorch layout. The public CLI mode remains
``openpi_rlinf_to_openpi_pytorch``.
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


def new_to_old_state_dict(
    new_sd: dict[str, torch.Tensor],
    *,
    action_expert_uses_adarms: bool = True,
) -> dict[str, torch.Tensor]:
    """Convert a new-format state dict to the OpenPI PyTorch layout.

    Pi0.5 uses adaRMSNorm for the action expert, while Pi0 uses RMSNorm. The
    caller derives the action-expert layout from the reference checkpoint.
    """
    openpi_rlinf_state_dict = new_sd
    openpi_pytorch_state_dict: dict[str, torch.Tensor] = {}

    for suffix in (".weight", ".bias"):
        rlinf_key = "img.stem" + suffix
        if rlinf_key in openpi_rlinf_state_dict:
            openpi_pytorch_state_dict[
                _OPENPI_SIGLIP + "embeddings.patch_embedding" + suffix
            ] = openpi_rlinf_state_dict[rlinf_key]

    if "img.pos_embedding" in openpi_rlinf_state_dict:
        position_embedding = openpi_rlinf_state_dict["img.pos_embedding"]
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
                if rlinf_key in openpi_rlinf_state_dict:
                    openpi_pytorch_state_dict[
                        f"{openpi_prefix}{openpi_name}{suffix}"
                    ] = openpi_rlinf_state_dict[rlinf_key]

        rlinf_key = f"{rlinf_prefix}attn.in_proj_weight"
        if rlinf_key in openpi_rlinf_state_dict:
            query, key, value = torch.chunk(
                openpi_rlinf_state_dict[rlinf_key], 3, dim=0
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
        if rlinf_key in openpi_rlinf_state_dict:
            query, key, value = torch.chunk(
                openpi_rlinf_state_dict[rlinf_key], 3, dim=0
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
            if rlinf_key in openpi_rlinf_state_dict:
                openpi_pytorch_state_dict[
                    f"{openpi_prefix}self_attn.out_proj{suffix}"
                ] = openpi_rlinf_state_dict[rlinf_key]
        for name in ("fc1", "fc2"):
            for suffix in (".weight", ".bias"):
                rlinf_key = f"{rlinf_prefix}mlp.{name}{suffix}"
                if rlinf_key in openpi_rlinf_state_dict:
                    openpi_pytorch_state_dict[f"{openpi_prefix}mlp.{name}{suffix}"] = (
                        openpi_rlinf_state_dict[rlinf_key]
                    )

    for suffix in (".weight", ".bias"):
        rlinf_key = "img.encoder.norm" + suffix
        if rlinf_key in openpi_rlinf_state_dict:
            openpi_pytorch_state_dict[_OPENPI_SIGLIP + "post_layernorm" + suffix] = (
                openpi_rlinf_state_dict[rlinf_key]
            )
        rlinf_key = "img.head" + suffix
        if rlinf_key in openpi_rlinf_state_dict:
            openpi_pytorch_state_dict[
                "paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
                + suffix
            ] = openpi_rlinf_state_dict[rlinf_key]

    _convert_llm_expert(
        openpi_rlinf_state_dict,
        openpi_pytorch_state_dict,
        _OPENPI_PALIGEMMA_LLM,
        expert_index=0,
        action_expert=False,
    )
    _convert_llm_expert(
        openpi_rlinf_state_dict,
        openpi_pytorch_state_dict,
        _OPENPI_GEMMA_EXPERT,
        expert_index=1,
        action_expert=action_expert_uses_adarms,
    )

    rlinf_key = "llm.final_norms.0.scale"
    if rlinf_key in openpi_rlinf_state_dict:
        openpi_pytorch_state_dict[_OPENPI_PALIGEMMA_LLM + "norm.weight"] = (
            openpi_rlinf_state_dict[rlinf_key]
        )
    if action_expert_uses_adarms:
        for suffix in (".weight", ".bias"):
            rlinf_key = f"llm.final_norms.1.ada_modulation{suffix}"
            if rlinf_key in openpi_rlinf_state_dict:
                openpi_pytorch_state_dict[
                    _OPENPI_GEMMA_EXPERT + "norm.dense" + suffix
                ] = openpi_rlinf_state_dict[rlinf_key]
    else:
        rlinf_key = "llm.final_norms.1.scale"
        if rlinf_key in openpi_rlinf_state_dict:
            openpi_pytorch_state_dict[_OPENPI_GEMMA_EXPERT + "norm.weight"] = (
                openpi_rlinf_state_dict[rlinf_key]
            )

    rlinf_key = "llm.embedder.embedding.weight"
    if rlinf_key in openpi_rlinf_state_dict:
        openpi_pytorch_state_dict["paligemma_with_expert.paligemma.lm_head.weight"] = (
            openpi_rlinf_state_dict[rlinf_key]
        )

    for key, tensor in openpi_rlinf_state_dict.items():
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


def _reference_uses_action_expert_adarms(
    reference_state_dict: dict[str, torch.Tensor],
) -> bool:
    """Infer Pi0.5's action-expert norm layout from a reference checkpoint."""
    return any(
        key.startswith(_OPENPI_GEMMA_EXPERT)
        and (
            ".input_layernorm.dense." in key
            or ".post_attention_layernorm.dense." in key
        )
        for key in reference_state_dict
    )


def _convert_llm_expert(
    openpi_rlinf_state_dict: dict[str, torch.Tensor],
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
            if rlinf_key in openpi_rlinf_state_dict:
                openpi_pytorch_state_dict[
                    f"{openpi_prefix}self_attn.{projection}.weight"
                ] = openpi_rlinf_state_dict[rlinf_key]

        rlinf_key = f"{rlinf_prefix}mlps.{expert_index}.w_gating"
        if rlinf_key in openpi_rlinf_state_dict:
            gating = openpi_rlinf_state_dict[rlinf_key]
            openpi_pytorch_state_dict[f"{openpi_prefix}mlp.gate_proj.weight"] = gating[
                0
            ].T.contiguous()
            openpi_pytorch_state_dict[f"{openpi_prefix}mlp.up_proj.weight"] = gating[
                1
            ].T.contiguous()
        rlinf_key = f"{rlinf_prefix}mlps.{expert_index}.w_linear"
        if rlinf_key in openpi_rlinf_state_dict:
            openpi_pytorch_state_dict[f"{openpi_prefix}mlp.down_proj.weight"] = (
                openpi_rlinf_state_dict[rlinf_key].T.contiguous()
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
                    if rlinf_key in openpi_rlinf_state_dict:
                        openpi_pytorch_state_dict[
                            f"{openpi_prefix}{openpi_name}.dense{suffix}"
                        ] = openpi_rlinf_state_dict[rlinf_key]
            else:
                rlinf_key = f"{rlinf_prefix}{rlinf_name}.{expert_index}.scale"
                if rlinf_key in openpi_rlinf_state_dict:
                    openpi_pytorch_state_dict[
                        f"{openpi_prefix}{openpi_name}.weight"
                    ] = openpi_rlinf_state_dict[rlinf_key]


def convert_trained_ckpt(
    input_ckpt: str,
    output_dir: str,
    reference_model: str,
    norm_stats: str | None = None,
) -> None:
    """Convert OpenPI_RLinf trained weights with an OpenPI PyTorch reference."""
    import safetensors.torch

    if str(input_ckpt).endswith(".safetensors"):
        openpi_rlinf_state_dict = safetensors.torch.load_file(input_ckpt, device="cpu")
    else:
        openpi_rlinf_state_dict = torch.load(
            input_ckpt, map_location="cpu", weights_only=True
        )
    openpi_rlinf_state_dict = {
        key.removeprefix("_orig_mod."): tensor
        for key, tensor in openpi_rlinf_state_dict.items()
    }

    reference_safetensors = os.path.join(reference_model, "model.safetensors")
    reference_state_dict = safetensors.torch.load_file(reference_safetensors)
    openpi_pytorch_state_dict = new_to_old_state_dict(
        openpi_rlinf_state_dict,
        action_expert_uses_adarms=_reference_uses_action_expert_adarms(
            reference_state_dict
        ),
    )
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
    """Convert OpenPI_RLinf weights to OpenPI PyTorch without a reference.

    This always raises because an OpenPI PyTorch checkpoint needs the separate
    action-expert head, which OpenPI_RLinf cannot reconstruct. Use
    ``--reference-model`` to call :func:`convert_trained_ckpt` instead.
    """
    input_model = pathlib.Path(input_model)
    output_model = pathlib.Path(output_model)
    openpi_rlinf_path = resolve_model_safetensors(input_model)
    if not openpi_rlinf_path.exists():
        raise FileNotFoundError(
            f"OpenPI_RLinf checkpoint not found: {openpi_rlinf_path}"
        )
    openpi_rlinf_state_dict = load_safetensors(openpi_rlinf_path)
    openpi_pytorch_state_dict = new_to_old_state_dict(openpi_rlinf_state_dict)
    if ACTION_EXPERT_LM_HEAD not in openpi_pytorch_state_dict:
        raise RuntimeError(
            "openpi_rlinf_to_openpi_pytorch cannot produce a complete OpenPI "
            "PyTorch checkpoint because the action-expert head "
            f"{ACTION_EXPERT_LM_HEAD!r} is not carried by OpenPI_RLinf. Pass "
            "--reference-model (an OpenPI PyTorch model dir)."
        )
    save_safetensors(openpi_pytorch_state_dict, output_model / "model.safetensors")
    copy_norm_stats(input_norm_stats, output_norm_stats)
    return output_model


def add_arguments(parser) -> None:
    """Register the ``openpi_rlinf_to_openpi_pytorch`` mode arguments."""
    parser.add_argument(
        "--input-model",
        required=True,
        help="OpenPI_RLinf checkpoint directory, model.safetensors, or model.pt",
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
    """Execute ``openpi_rlinf_to_openpi_pytorch`` from parsed arguments."""
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
