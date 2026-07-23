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

"""Convert an RLinf RoboTwin Pi0 SFT checkpoint to the new HF-style layout.

The output is the self-contained layout consumed by ``openpi_pytorch``::

    model.safetensors
    config.json
    physical - intelligence / robotwin / norm_stats.json

Unlike the legacy ``sft2new`` mode, this converter targets the non-Pi05
RoboTwin Pi0 architecture (action horizon 50, model action dimension 32, and
maximum prompt length 48). Floating-point weights are kept in fp32 by default
to avoid introducing an additional cast after SFT; ``--dtype bf16`` is
available when a smaller deployment artifact is preferred.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping

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

_ROBOTWIN_PI0_CONFIG = {
    "action_dim": 32,
    "action_horizon": 50,
    "max_token_len": 48,
    "paligemma_variant": "gemma_2b",
    "action_expert_variant": "gemma_300m",
    "pi05": False,
    "discrete_state_input": False,
    "pcd": False,
}

_WEIGHTS_CANDIDATES = (
    "actor/model_state_dict/full_weights.pt",
    "model_state_dict/full_weights.pt",
    "full_weights.pt",
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

_DTYPES = {
    "fp32": (torch.float32, "float32"),
    "bf16": (torch.bfloat16, "bfloat16"),
}


def _resolve_full_weights(ckpt: str | pathlib.Path) -> pathlib.Path:
    """Find consolidated FSDP weights under an SFT checkpoint path."""
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


def _validate_pi0_state_dict(state_dict: Mapping[str, torch.Tensor]) -> None:
    """Reject checkpoints that are not the non-Pi05 RoboTwin Pi0 layout."""
    missing = [key for key in _REQUIRED_PI0_KEYS if key not in state_dict]
    if missing:
        raise ValueError(
            "The SFT checkpoint does not look like a RoboTwin Pi0 checkpoint. "
            f"Missing required bare keys: {missing}"
        )

    pi05_only = ("time_mlp_in.weight", "time_mlp_out.weight")
    present_pi05_keys = [key for key in pi05_only if key in state_dict]
    if present_pi05_keys:
        raise ValueError(
            "The checkpoint contains Pi0.5-only keys "
            f"{present_pi05_keys}; use the legacy sft2new mode for Pi0.5."
        )


def _validate_against_reference(
    state_dict: Mapping[str, torch.Tensor], reference_model: str | pathlib.Path
) -> None:
    """Validate keys and shapes against a new-format Pi0 base model."""
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
            "SFT checkpoint keys do not match the reference Pi0 model: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    shape_mismatches = [
        f"{key}: got {tuple(state_dict[key].shape)}, expected {tuple(reference[key].shape)}"
        for key in sorted(reference_keys)
        if tuple(state_dict[key].shape) != tuple(reference[key].shape)
    ]
    if shape_mismatches:
        raise ValueError(
            "SFT checkpoint tensor shapes do not match the reference Pi0 model: "
            f"{shape_mismatches[:8]}"
        )


def convert(
    ckpt: str | pathlib.Path,
    input_norm_stats: str | pathlib.Path,
    output_model: str | pathlib.Path,
    output_norm_stats: str | pathlib.Path,
    *,
    dtype: str = "fp32",
    reference_model: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Convert RoboTwin Pi0 SFT weights to a new-format model directory."""
    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {dtype!r}")

    weights_path = _resolve_full_weights(ckpt)
    loaded = torch.load(
        str(weights_path), map_location="cpu", weights_only=False, mmap=True
    )
    state_dict = as_state_dict(loaded)
    bare_state = strip_wrapper_prefix(
        state_dict,
        cast_dtype=_DTYPES[dtype][0],
    )
    _validate_pi0_state_dict(bare_state)
    if reference_model is not None:
        _validate_against_reference(bare_state, reference_model)

    output_model = pathlib.Path(output_model)
    save_safetensors(bare_state, output_model / "model.safetensors")
    config = dict(_ROBOTWIN_PI0_CONFIG)
    config["dtype"] = _DTYPES[dtype][1]
    write_config_json(config, output_model)
    copy_norm_stats(input_norm_stats, output_norm_stats)

    print(
        f"Converted {weights_path} -> {output_model} "
        f"({len(bare_state)} {dtype} tensors); norm stats -> {output_norm_stats}"
    )
    return output_model


def add_arguments(parser) -> None:
    """Register RoboTwin Pi0 SFT conversion arguments."""
    parser.add_argument(
        "--ckpt",
        required=True,
        help="SFT checkpoint dir, actor/ dir, model_state_dict/ dir, or full_weights.pt",
    )
    parser.add_argument(
        "--input-norm-stats", required=True, help="norm_stats.json to copy across"
    )
    parser.add_argument(
        "--output-model",
        required=True,
        help="output HF-style model dir with model.safetensors + config.json",
    )
    parser.add_argument(
        "--output-norm-stats",
        required=True,
        help="destination, normally OUTPUT_MODEL/physical-intelligence/robotwin/norm_stats.json",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(_DTYPES),
        default="fp32",
        help="output floating-point dtype; fp32 preserves the SFT master weights",
    )
    parser.add_argument(
        "--reference-model",
        default=None,
        help="optional new-format Pi0 base model used to validate keys and shapes",
    )


def run(args) -> None:
    """Execute the RoboTwin Pi0 SFT conversion."""
    convert(
        args.ckpt,
        args.input_norm_stats,
        args.output_model,
        args.output_norm_stats,
        dtype=args.dtype,
        reference_model=args.reference_model,
    )
