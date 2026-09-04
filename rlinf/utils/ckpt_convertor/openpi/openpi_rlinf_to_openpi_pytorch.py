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

import json
import os
import pathlib
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping
from typing import Any

import torch

from rlinf.utils.ckpt_convertor.openpi._core import (
    NORM_STATS_SUBDIR,
    as_state_dict,
    copy_norm_stats,
    load_safetensors,
    resolve_model_safetensors,
    save_safetensors,
    strip_wrapper_prefix,
)

ACTION_EXPERT_LM_HEAD = "paligemma_with_expert.gemma_expert.lm_head.weight"
EMBED_TOKENS_KEY = (
    "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
)
_OPENPI_SIGLIP = "paligemma_with_expert.paligemma.model.vision_tower.vision_model."
_OPENPI_PALIGEMMA_LLM = "paligemma_with_expert.paligemma.model.language_model."
_OPENPI_GEMMA_EXPERT = "paligemma_with_expert.gemma_expert.model."
_WEIGHTS_CANDIDATES = (
    "actor/model_state_dict/full_weights.pt",
    "model_state_dict/full_weights.pt",
    "full_weights.pt",
)
_DEPLOY_WEIGHTS_REL = pathlib.Path("actor/model_state_dict/full_weights.pt")


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
    reference_state_dict: Mapping[str, Any],
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


def _resolve_full_weights(path: str | pathlib.Path) -> pathlib.Path:
    """Resolve a checkpoint file or directory to ``full_weights.pt``."""
    path = pathlib.Path(path)
    if path.is_file():
        return path
    for relative in _WEIGHTS_CANDIDATES:
        candidate = path / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No full_weights.pt found under {path}")


def _resolve_output_pt(output: str | pathlib.Path) -> pathlib.Path:
    """Resolve an output file or deployment directory to ``full_weights.pt``."""
    output = pathlib.Path(output)
    if output.suffix == ".pt":
        return output
    return output / _DEPLOY_WEIGHTS_REL


def _detect_variant(state_dict: Mapping[str, Any]) -> str:
    """Detect Pi0 or Pi0.5 from the model projection keys."""
    keys = set(state_dict)
    is_pi0 = {
        "state_proj.weight",
        "action_time_mlp_in.weight",
        "action_time_mlp_out.weight",
    }.issubset(keys)
    is_pi05 = {
        "time_mlp_in.weight",
        "time_mlp_out.weight",
    }.issubset(keys) and "state_proj.weight" not in keys
    if is_pi0 == is_pi05:
        raise ValueError(
            "Cannot unambiguously detect Pi0/Pi0.5 from projection keys: "
            f"state_proj={'state_proj.weight' in keys}, "
            f"action_time_mlp={'action_time_mlp_in.weight' in keys}, "
            f"time_mlp={'time_mlp_in.weight' in keys}"
        )
    return "pi0" if is_pi0 else "pi05"


def _load_reference(
    reference: str | pathlib.Path,
) -> tuple[dict[str, tuple[int, ...]], torch.Tensor, str]:
    """Load a reference schema, action-expert head, and model variant."""
    reference = pathlib.Path(reference)
    if reference.is_dir():
        safetensors_path = reference / "model.safetensors"
        reference = (
            safetensors_path
            if safetensors_path.is_file()
            else _resolve_full_weights(reference)
        )

    if reference.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(reference), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            if ACTION_EXPERT_LM_HEAD not in keys:
                raise KeyError(f"Reference {reference} lacks {ACTION_EXPERT_LM_HEAD!r}")
            schema = {key: tuple(handle.get_slice(key).get_shape()) for key in keys}
            head = handle.get_tensor(ACTION_EXPERT_LM_HEAD).float().contiguous()
            return schema, head, _detect_variant(schema)

    loaded = torch.load(
        str(reference), map_location="cpu", weights_only=False, mmap=True
    )
    state_dict = as_state_dict(loaded)
    if ACTION_EXPERT_LM_HEAD not in state_dict:
        raise KeyError(f"Reference {reference} lacks {ACTION_EXPERT_LM_HEAD!r}")
    schema = {key: tuple(tensor.shape) for key, tensor in state_dict.items()}
    head = state_dict[ACTION_EXPERT_LM_HEAD].detach().float().cpu().contiguous()
    return schema, head, _detect_variant(schema)


def _validate_source_fp32(state_dict: Mapping[str, torch.Tensor]) -> None:
    """Validate that every floating-point source tensor uses FP32."""
    bad = {
        key: str(tensor.dtype)
        for key, tensor in state_dict.items()
        if tensor.is_floating_point() and tensor.dtype != torch.float32
    }
    if bad:
        raise TypeError(
            f"Source is not an all-fp32 SFT checkpoint; {len(bad)} floating "
            f"tensors are non-fp32, examples={list(bad.items())[:10]}"
        )


def _validate_fp32_output(
    state_dict: Mapping[str, torch.Tensor],
    shape_schema: Mapping[str, tuple[int, ...]],
) -> dict[str, Any]:
    """Validate FP32 deployment keys and shapes against a reference schema."""
    expected = set(shape_schema)
    actual = set(state_dict)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise RuntimeError(
            f"Deploy key mismatch: missing={missing[:20]}, extra={extra[:20]}"
        )

    shape_bad = [
        (key, tuple(state_dict[key].shape), shape_schema[key])
        for key in sorted(expected)
        if tuple(state_dict[key].shape) != shape_schema[key]
    ]
    if shape_bad:
        raise RuntimeError(f"Deploy shape mismatch: {shape_bad[:20]}")

    dtype_bad = [
        (key, str(tensor.dtype))
        for key, tensor in state_dict.items()
        if tensor.is_floating_point() and tensor.dtype != torch.float32
    ]
    if dtype_bad:
        raise RuntimeError(f"Non-fp32 deploy tensors: {dtype_bad[:20]}")

    dtype_counts = Counter(str(tensor.dtype) for tensor in state_dict.values())
    tensor_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in state_dict.values()
    )
    return {
        "keys": len(state_dict),
        "dtype_counts": dict(dtype_counts),
        "tensor_bytes": tensor_bytes,
        "tensor_gib": tensor_bytes / 1024**3,
    }


def convert_fp32_pt(
    checkpoint: str | pathlib.Path,
    reference_model: str | pathlib.Path,
    output: str | pathlib.Path,
) -> pathlib.Path:
    """Convert an RLinf Pi0/Pi0.5 SFT checkpoint to OpenPI PyTorch FP32 PT."""
    source_path = _resolve_full_weights(checkpoint)
    output_path = _resolve_output_pt(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    loaded = torch.load(
        str(source_path), map_location="cpu", weights_only=False, mmap=True
    )
    bare = strip_wrapper_prefix(as_state_dict(loaded), cast_dtype=None)
    _validate_source_fp32(bare)
    variant = _detect_variant(bare)

    reference_schema, reference_head, reference_variant = _load_reference(
        reference_model
    )
    if reference_variant != variant:
        raise ValueError(
            "Source/reference model variant mismatch: "
            f"source={variant}, reference={reference_variant}"
        )

    action_expert_uses_adarms = _reference_uses_action_expert_adarms(reference_schema)
    if action_expert_uses_adarms != (variant == "pi05"):
        raise ValueError(
            "Reference normalization layout does not match the detected model "
            f"variant: variant={variant}, "
            f"action_expert_uses_adarms={action_expert_uses_adarms}"
        )

    deploy = new_to_old_state_dict(
        bare, action_expert_uses_adarms=action_expert_uses_adarms
    )
    if "llm.embedder.embedding.weight" not in bare:
        raise KeyError("SFT checkpoint lacks llm.embedder.embedding.weight")
    deploy[EMBED_TOKENS_KEY] = bare["llm.embedder.embedding.weight"]
    deploy[ACTION_EXPERT_LM_HEAD] = reference_head
    deploy = {
        key: (
            tensor.detach().float().cpu().contiguous()
            if tensor.is_floating_point()
            else tensor.detach().cpu().contiguous()
        )
        for key, tensor in deploy.items()
    }

    reference_schema[EMBED_TOKENS_KEY] = tuple(deploy[EMBED_TOKENS_KEY].shape)
    report = _validate_fp32_output(deploy, reference_schema)
    report.update(
        {
            "variant": variant,
            "source": str(source_path),
            "reference": str(reference_model),
            "output": str(output_path),
        }
    )

    fd, temporary_name = tempfile.mkstemp(
        prefix=output_path.name + ".tmp.", dir=str(output_path.parent)
    )
    os.close(fd)
    temporary_path = pathlib.Path(temporary_name)
    try:
        torch.save(deploy, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    report_path = output_path.with_suffix(output_path.suffix + ".report.json")
    report["file_bytes"] = output_path.stat().st_size
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return output_path


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
    dtype: torch.dtype = torch.bfloat16,
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
        if tensor.is_floating_point() and tensor.dtype != dtype:
            openpi_pytorch_state_dict[key] = tensor.to(dtype)

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
        "--ckpt",
        dest="input_model",
        required=True,
        help="OpenPI_RLinf checkpoint directory, full_weights.pt, or safetensors",
    )
    parser.add_argument(
        "--input-norm-stats", default=None, help="norm_stats.json to copy across"
    )
    parser.add_argument(
        "--output-model",
        "--output",
        dest="output_model",
        required=True,
        help="output OpenPI PyTorch checkpoint path or directory",
    )
    parser.add_argument(
        "--output-norm-stats",
        default=None,
        help="destination norm_stats.json path",
    )
    parser.add_argument(
        "--reference-model",
        default=None,
        help=(
            "reference OpenPI PyTorch model directory used to source the "
            "action-expert lm_head and validate keys/shapes"
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=("safetensors", "pt"),
        default="safetensors",
        help="output checkpoint format; PT output is written as full_weights.pt",
    )
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp32"),
        default="bf16",
        help="floating-point dtype for the converted checkpoint",
    )


def run(args) -> None:
    """Execute ``openpi_rlinf_to_openpi_pytorch`` from parsed arguments."""
    if args.output_format == "pt":
        if not args.reference_model:
            raise ValueError("--reference-model is required for PT output")
        if args.dtype != "fp32":
            raise ValueError("PT output currently requires --dtype fp32")
        convert_fp32_pt(args.input_model, args.reference_model, args.output_model)
        return

    if not args.input_norm_stats or not args.output_norm_stats:
        raise ValueError(
            "--input-norm-stats and --output-norm-stats are required for "
            "safetensors output"
        )

    if args.reference_model:
        input_path = pathlib.Path(args.input_model)
        if input_path.is_dir():
            input_path = resolve_model_safetensors(input_path)
        convert_trained_ckpt(
            input_ckpt=str(input_path),
            output_dir=args.output_model,
            reference_model=args.reference_model,
            dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float32,
        )
        copy_norm_stats(args.input_norm_stats, args.output_norm_stats)
    else:
        convert(
            args.input_model,
            args.input_norm_stats,
            args.output_model,
            args.output_norm_stats,
        )
