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

"""Convert an RLinf SFT checkpoint to the OpenPI_RLinf layout.

The ``--config-name`` argument is the single source of truth for the target
Pi0/Pi0.5 architecture. It is resolved through
``rlinf.models.embodiment.openpi.dataconfig.get_openpi_config`` so the
converter uses the same action horizon, action dimension, token length, and
state-input semantics as SFT and eval.
"""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Mapping
from typing import Any

import torch

from rlinf.utils.ckpt_convertor.openpi._core import (
    as_state_dict,
    copy_norm_stats,
    load_safetensors,
    resolve_model_safetensors,
    save_safetensors,
    strip_wrapper_prefix,
    write_config_json,
)

_REQUIRED_PI0_KEYS = (
    "img.stem.weight",
    "llm.embedder.embedding.weight",
    "action_in_proj.weight",
    "action_out_proj.weight",
    "state_proj.weight",
    "action_time_mlp_in.weight",
    "action_time_mlp_out.weight",
)

_REQUIRED_PI05_KEYS = (
    "img.stem.weight",
    "llm.embedder.embedding.weight",
    "action_in_proj.weight",
    "action_out_proj.weight",
    "time_mlp_in.weight",
    "time_mlp_out.weight",
)

_DTYPES = {
    "fp32": (torch.float32, "float32"),
    "bf16": (torch.bfloat16, "bfloat16"),
}

_WEIGHTS_CANDIDATES = (
    "actor/model_state_dict/full_weights.pt",
    "model_state_dict/full_weights.pt",
    "full_weights.pt",
)


@dataclasses.dataclass(frozen=True)
class SftToOpenPIRLinfModelSpec:
    """Architecture contract derived from one OpenPI TrainConfig."""

    config_name: str
    pi05: bool
    config: Mapping[str, Any]
    required_keys: tuple[str, ...]
    forbidden_keys: tuple[str, ...]


def _get_openpi_train_config(config_name: str) -> Any:
    """Lazily import the shared OpenPI config registry for a conversion."""
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    return get_openpi_config(config_name)


def _model_attr(model_config: Any, name: str) -> Any:
    try:
        return getattr(model_config, name)
    except AttributeError as exc:
        raise ValueError(
            f"OpenPI config {type(model_config).__name__} does not define "
            f"the required model field {name!r}."
        ) from exc


def resolve_model_spec(config_name: str) -> SftToOpenPIRLinfModelSpec:
    """Resolve model shape and Pi0/Pi0.5 semantics from ``config_name``."""
    train_config = _get_openpi_train_config(config_name)
    model_config = train_config.model
    pi05 = bool(getattr(model_config, "pi05", False))
    discrete_state_input = getattr(model_config, "discrete_state_input", pi05)
    if discrete_state_input is None:
        discrete_state_input = pi05

    config = {
        "action_dim": int(_model_attr(model_config, "action_dim")),
        "action_horizon": int(_model_attr(model_config, "action_horizon")),
        "max_token_len": int(_model_attr(model_config, "max_token_len")),
        "paligemma_variant": str(_model_attr(model_config, "paligemma_variant")),
        "action_expert_variant": str(
            _model_attr(model_config, "action_expert_variant")
        ),
        "pi05": pi05,
        "discrete_state_input": bool(discrete_state_input),
        "pcd": bool(getattr(model_config, "pcd", False)),
    }
    if pi05:
        required_keys = _REQUIRED_PI05_KEYS
        forbidden_keys = (
            "state_proj.weight",
            "action_time_mlp_in.weight",
            "action_time_mlp_out.weight",
        )
    else:
        required_keys = _REQUIRED_PI0_KEYS
        forbidden_keys = ("time_mlp_in.weight", "time_mlp_out.weight")

    return SftToOpenPIRLinfModelSpec(
        config_name=config_name,
        pi05=pi05,
        config=config,
        required_keys=required_keys,
        forbidden_keys=forbidden_keys,
    )


def _resolve_full_weights(ckpt: str | pathlib.Path) -> pathlib.Path:
    """Find the consolidated ``full_weights.pt`` checkpoint file."""
    ckpt = pathlib.Path(ckpt)
    if ckpt.is_file():
        return ckpt
    for relative_path in _WEIGHTS_CANDIDATES:
        candidate = ckpt / relative_path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No full_weights.pt found under {ckpt}; looked at "
        f"{[str(ckpt / path) for path in _WEIGHTS_CANDIDATES]}."
    )


def _validate_state_dict(
    state_dict: Mapping[str, torch.Tensor], spec: SftToOpenPIRLinfModelSpec
) -> None:
    """Reject checkpoints whose architecture differs from the selected config."""
    missing = [key for key in spec.required_keys if key not in state_dict]
    if missing:
        raise ValueError(
            "The checkpoint does not look like the "
            f"{'Pi0.5' if spec.pi05 else 'Pi0'} "
            f"architecture selected by --config-name {spec.config_name!r}. "
            f"Missing required bare keys: {missing}"
        )
    present_forbidden = [key for key in spec.forbidden_keys if key in state_dict]
    if present_forbidden:
        raise ValueError(
            "The checkpoint contains architecture-incompatible keys "
            f"{present_forbidden}; check --config-name {spec.config_name!r}."
        )


def _validate_against_reference(
    state_dict: Mapping[str, torch.Tensor],
    reference_model: str | pathlib.Path,
    spec: SftToOpenPIRLinfModelSpec,
) -> None:
    """Validate keys and tensor shapes against a matching OpenPI_RLinf model."""
    reference_path = resolve_model_safetensors(reference_model)
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"reference model must contain model.safetensors: {reference_path}"
        )
    reference = load_safetensors(reference_path)
    actual_keys = set(state_dict)
    reference_keys = set(reference)
    missing = sorted(reference_keys - actual_keys)
    unexpected = sorted(actual_keys - reference_keys)
    if missing or unexpected:
        raise ValueError(
            "SFT checkpoint keys do not match the OpenPI_RLinf reference model for "
            f"--config-name {spec.config_name!r}: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    shape_mismatches = [
        f"{key}: got {tuple(state_dict[key].shape)}, "
        f"expected {tuple(reference[key].shape)}"
        for key in sorted(reference_keys)
        if tuple(state_dict[key].shape) != tuple(reference[key].shape)
    ]
    if shape_mismatches:
        raise ValueError(
            "SFT checkpoint tensor shapes do not match the reference model for "
            f"--config-name {spec.config_name!r}: {shape_mismatches[:8]}"
        )


def convert(
    ckpt: str | pathlib.Path,
    input_norm_stats: str | pathlib.Path,
    output_model: str | pathlib.Path,
    output_norm_stats: str | pathlib.Path,
    *,
    config_name: str,
    dtype: str,
    reference_model: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Convert an SFT checkpoint using the specified OpenPI TrainConfig."""
    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {dtype!r}")
    spec = resolve_model_spec(config_name)

    weights_path = _resolve_full_weights(ckpt)
    loaded = torch.load(
        str(weights_path), map_location="cpu", weights_only=False, mmap=True
    )
    state_dict = as_state_dict(loaded)
    bare_state = strip_wrapper_prefix(
        state_dict,
        cast_dtype=_DTYPES[dtype][0],
    )
    _validate_state_dict(bare_state, spec)
    if reference_model is not None:
        _validate_against_reference(bare_state, reference_model, spec)

    output_model = pathlib.Path(output_model)
    save_safetensors(bare_state, output_model / "model.safetensors")
    config = dict(spec.config)
    config["dtype"] = _DTYPES[dtype][1]
    write_config_json(config, output_model)
    copy_norm_stats(input_norm_stats, output_norm_stats)
    print(
        f"Converted {weights_path} -> {output_model} "
        f"(config_name={spec.config_name}, pi05={spec.pi05}, "
        f"{len(bare_state)} {dtype} tensors); "
        f"norm stats -> {output_norm_stats}"
    )
    return output_model


def add_arguments(parser) -> None:
    """Register SFT conversion arguments on the shared CLI parser."""
    parser.add_argument(
        "--ckpt",
        required=True,
        help="SFT checkpoint dir, actor/model_state_dict dir, or full_weights.pt",
    )
    parser.add_argument(
        "--input-norm-stats", required=True, help="norm_stats.json to copy across"
    )
    parser.add_argument(
        "--output-model",
        required=True,
        help="output OpenPI_RLinf checkpoint dir with config.json + model.safetensors",
    )
    parser.add_argument("--output-norm-stats", required=True)
    parser.add_argument(
        "--config-name",
        required=True,
        help=(
            "OpenPI TrainConfig name used by SFT/eval, e.g. "
            "pi05_behavior or pi0_aloha_robotwin"
        ),
    )
    parser.add_argument(
        "--dtype",
        required=True,
        choices=sorted(_DTYPES),
        help="storage dtype for output weights; fp32 preserves SFT master weights",
    )
    parser.add_argument(
        "--reference-model",
        default=None,
        help="optional matching OpenPI_RLinf model used to validate keys and shapes",
    )


def run(args) -> None:
    """Execute the unified SFT conversion from parsed arguments."""
    convert(
        args.ckpt,
        args.input_norm_stats,
        args.output_model,
        args.output_norm_stats,
        config_name=args.config_name,
        dtype=args.dtype,
        reference_model=args.reference_model,
    )
