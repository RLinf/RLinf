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

"""Shared helpers for the LingBot-VA eval backend and the SFT training path."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from rlinf.utils.logging import get_logger

logger = get_logger()

# State-dict key prefixes used by upstream LingBot-VA checkpoints that we may
# need to strip before loading the weights into a bare transformer module.
TRANSFORMER_STATE_PREFIXES: tuple[str, ...] = (
    "_sft_core.transformer.",
    "transformer.",
)


def extend_import_path(repo_path: Path | str) -> None:
    """Make `wan_va.*` importable by prepending the LingBot-VA repo to sys.path."""
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def extract_transformer_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip optional `transformer.` / `_sft_core.transformer.` key prefixes."""
    for prefix in TRANSFORMER_STATE_PREFIXES:
        extracted = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if extracted:
            return extracted
    return state_dict


def load_transformer_state_dict(
    transformer: torch.nn.Module,
    checkpoint_path: Path | str,
) -> None:
    """Load a LingBot-VA transformer checkpoint into `transformer`.

    Accepts either a directory containing a HuggingFace-style
    `transformer/diffusion_pytorch_model.safetensors` (the LingBot-VA SFT
    checkpoint layout) or a single `.pt`/`.pth` file with optional
    `transformer.`-prefixed keys.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"LingBot-VA transformer state dict path does not exist: {checkpoint_path}"
        )

    if checkpoint_path.is_dir():
        transformer_dir = (
            checkpoint_path / "transformer"
            if (checkpoint_path / "transformer" / "config.json").exists()
            else checkpoint_path
        )
        state_path = transformer_dir / "diffusion_pytorch_model.safetensors"
        if not state_path.exists():
            raise FileNotFoundError(
                "LingBot-VA transformer checkpoint directory must contain "
                f"diffusion_pytorch_model.safetensors: {transformer_dir}"
            )
        transformer_state = load_file(str(state_path), device="cpu")
    else:
        raw_state = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if not isinstance(raw_state, dict):
            raise TypeError(
                "LingBot-VA transformer state dict must deserialize to a dict, got "
                f"{type(raw_state)!r}."
            )
        transformer_state = extract_transformer_state_dict(raw_state)

    missing_keys, unexpected_keys = transformer.load_state_dict(
        transformer_state, strict=False
    )
    if missing_keys or unexpected_keys:
        preview_missing = ", ".join(missing_keys[:5])
        preview_unexpected = ", ".join(unexpected_keys[:5])
        logger.warning(
            "LingBot-VA transformer checkpoint partial match "
            "(missing=%d, unexpected=%d). Missing preview: [%s]. "
            "Unexpected preview: [%s].",
            len(missing_keys),
            len(unexpected_keys),
            preview_missing,
            preview_unexpected,
        )
    logger.info(
        "Loaded LingBot-VA Libero transformer checkpoint from %s "
        "(missing=%d, unexpected=%d).",
        checkpoint_path,
        len(missing_keys),
        len(unexpected_keys),
    )
