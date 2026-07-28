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

"""Convert an RLinf SFT-trained checkpoint to the NEW-format PyTorch layout.

An SFT run (``bash examples/sft/run_vla_sft.sh behavior_pi05_vla``) saves the
``OpenPiPytorchActionModel`` wrapper state dict, consolidated by the FSDP worker
into::

    checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt

Every tensor key is prefixed ``model.`` because the vendored ``Pi0`` lives at
``wrapper.model`` (FSDP/compile may add ``_fsdp_wrapped_module.`` / ``_orig_mod.``
/ ``module.`` in front of that). This mode strips those wrapper prefixes, casts
float weights to bf16, and writes a directory identical in structure and key
namespace to a new-format ``pi05_base_pytorch_new`` checkpoint — i.e.
``model.safetensors`` (bf16, bare ``Pi0`` keys) plus ``config.json`` — so the
trained model can be evaluated through the unchanged new-format eval path. The
norm-stats file is copied across verbatim.
"""

from __future__ import annotations

import pathlib

import torch

from rlinf.utils.ckpt_convertor.openpi._core import (
    as_state_dict,
    copy_norm_stats,
    save_safetensors,
    strip_wrapper_prefix,
    write_config_json,
)

# The BEHAVIOR pi0.5 architecture config — identical to a new-format
# ``pi05_base_pytorch_new/config.json``; the SFT fine-tunes the same
# architecture, so the converted checkpoint shares this exact config (and bf16
# weights).
_PI05_BEHAVIOR_CONFIG = {
    "action_dim": 32,
    "action_horizon": 32,
    "max_token_len": 200,
    "paligemma_variant": "gemma_2b",
    "action_expert_variant": "gemma_300m",
    "pi05": True,
    "pcd": False,
    "dtype": "bfloat16",
}

# Relative paths to the consolidated full-weights file, tried in order, so
# ``--ckpt`` may point at the ``global_step_<N>`` dir, its ``actor`` subdir, the
# ``model_state_dict`` dir, or the ``full_weights.pt`` file itself.
_WEIGHTS_CANDIDATES = (
    "actor/model_state_dict/full_weights.pt",
    "model_state_dict/full_weights.pt",
    "full_weights.pt",
)


def _resolve_full_weights(ckpt: pathlib.Path) -> pathlib.Path:
    """Find the consolidated ``full_weights.pt`` for a saved SFT checkpoint path."""
    if ckpt.is_file():
        return ckpt
    candidates = [ckpt / rel for rel in _WEIGHTS_CANDIDATES]
    weights = next((c for c in candidates if c.is_file()), None)
    if weights is None:
        raise FileNotFoundError(
            f"No full_weights.pt found under {ckpt} "
            f"(looked at {[str(c) for c in candidates]})."
        )
    return weights


def convert(
    ckpt: str | pathlib.Path,
    input_norm_stats: str | pathlib.Path,
    output_model: str | pathlib.Path,
    output_norm_stats: str | pathlib.Path,
) -> pathlib.Path:
    """Convert an SFT checkpoint to the new-format layout.

    Loads the consolidated ``full_weights.pt`` under ``ckpt``, strips the wrapper /
    FSDP key prefixes and casts float weights to bf16 to recover the bare ``Pi0``
    state dict, writes ``output_model/model.safetensors`` + ``output_model/config.json``
    (the new-format structure), and copies ``input_norm_stats`` verbatim to
    ``output_norm_stats``.
    """
    weights_path = _resolve_full_weights(pathlib.Path(ckpt))
    loaded = torch.load(
        str(weights_path), map_location="cpu", weights_only=False, mmap=True
    )
    state_dict = as_state_dict(loaded)
    bare_state = strip_wrapper_prefix(state_dict, cast_dtype=torch.bfloat16)

    output_model = pathlib.Path(output_model)
    save_safetensors(bare_state, output_model / "model.safetensors")
    write_config_json(_PI05_BEHAVIOR_CONFIG, output_model)

    copy_norm_stats(input_norm_stats, output_norm_stats)
    print(
        f"Converted {weights_path} -> {output_model} "
        f"({len(bare_state)} bf16 tensors); norm stats -> {output_norm_stats}"
    )
    return output_model


def add_arguments(parser) -> None:
    """Register the ``sft2new`` mode arguments on ``parser``."""
    parser.add_argument(
        "--ckpt",
        required=True,
        help="saved SFT checkpoint (global_step_<N> dir, its actor/ subdir, the "
        "model_state_dict/ dir, or the full_weights.pt file)",
    )
    parser.add_argument(
        "--input-norm-stats", required=True, help="norm_stats.json to copy across"
    )
    parser.add_argument(
        "--output-model",
        required=True,
        help="output new-format checkpoint dir (config.json + model.safetensors)",
    )
    parser.add_argument(
        "--output-norm-stats", required=True, help="destination norm_stats.json path"
    )


def run(args) -> None:
    """Execute the ``sft2new`` mode from parsed ``args``."""
    convert(
        args.ckpt,
        args.input_norm_stats,
        args.output_model,
        args.output_norm_stats,
    )
