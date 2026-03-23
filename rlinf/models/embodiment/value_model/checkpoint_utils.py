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

"""Checkpoint loading and input transform utilities for value models."""

import glob
import json
import logging
import pathlib
from typing import Optional, Sequence

import numpy as np
import safetensors.torch
import torch

logger = logging.getLogger(__name__)


def load_state_dict_from_checkpoint(checkpoint_path: pathlib.Path) -> dict:
    """Load state dict from checkpoint directory or file.

    Supports:
    - Directory with .safetensors files
    - Directory with .pt/.pth files
    - Single .safetensors file
    - Single .pt/.pth file

    Args:
        checkpoint_path: Path to checkpoint directory or file

    Returns:
        Combined state dict from all files
    """
    if checkpoint_path.is_file():
        if str(checkpoint_path).endswith(".safetensors"):
            return safetensors.torch.load_file(str(checkpoint_path), device="cpu")
        else:
            return torch.load(
                str(checkpoint_path), map_location="cpu", weights_only=False
            )

    # Directory
    safetensor_files = sorted(glob.glob(str(checkpoint_path / "*.safetensors")))
    if safetensor_files:
        state_dict = {}
        for f in safetensor_files:
            state_dict.update(safetensors.torch.load_file(f, device="cpu"))
        return state_dict

    pt_files = sorted(glob.glob(str(checkpoint_path / "*.pt"))) + sorted(
        glob.glob(str(checkpoint_path / "*.pth"))
    )
    if pt_files:
        state_dict = {}
        for f in pt_files:
            state_dict.update(torch.load(f, map_location="cpu", weights_only=False))
        return state_dict

    raise FileNotFoundError(f"No checkpoint files found in {checkpoint_path}")


def has_tokenizer_files(checkpoint_dir: pathlib.Path) -> bool:
    """Check if checkpoint directory has tokenizer files."""
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    return any((checkpoint_dir / f).exists() for f in tokenizer_files)


def load_norm_stats(
    checkpoint_dir: pathlib.Path, asset_id: str = "libero"
) -> dict:
    """Load normalization statistics from checkpoint assets.

    Args:
        checkpoint_dir: Checkpoint directory containing assets
        asset_id: Asset identifier (e.g., "libero", "droid")

    Returns:
        Dictionary mapping stat names to NormStats objects
    """
    from rlinf.datasets.lerobot.normalize import NormStats

    possible_paths = [
        checkpoint_dir / "norm_stats" / asset_id / "norm_stats.json",
        checkpoint_dir / "stats" / asset_id / "norm_stats.json",
        checkpoint_dir / "norm_stats.json",
    ]

    # Also search for any subdirectory in norm_stats/
    norm_stats_dir = checkpoint_dir / "norm_stats"
    if norm_stats_dir.exists():
        for subdir in norm_stats_dir.iterdir():
            if subdir.is_dir():
                candidate = subdir / "norm_stats.json"
                if candidate.exists() and candidate not in possible_paths:
                    possible_paths.append(candidate)

    for path in possible_paths:
        if path.exists():
            logger.info(f"Loading norm stats from {path}")
            with open(path) as f:
                data = json.load(f)

            if "norm_stats" in data:
                data = data["norm_stats"]

            result = {}
            for k, v in data.items():
                result[k] = NormStats(
                    mean=np.asarray(v["mean"], dtype=np.float32),
                    std=np.asarray(v["std"], dtype=np.float32),
                    q01=(
                        np.asarray(v["q01"], dtype=np.float32)
                        if v.get("q01") is not None
                        else None
                    ),
                    q99=(
                        np.asarray(v["q99"], dtype=np.float32)
                        if v.get("q99") is not None
                        else None
                    ),
                    min=(
                        np.asarray(v["min"], dtype=np.float32)
                        if v.get("min") is not None
                        else None
                    ),
                    max=(
                        np.asarray(v["max"], dtype=np.float32)
                        if v.get("max") is not None
                        else None
                    ),
                )
            return result

    raise FileNotFoundError(f"Could not find norm_stats.json in {checkpoint_dir}")


def build_input_transforms(
    env_type: str,
    model_type: str,
    action_dim: int,
    default_prompt: Optional[str],
    norm_stats: Optional[dict],
    use_quantile_norm: bool,
    action_norm_skip_dims: Optional[dict[str, list[int]]] = None,
) -> Sequence:
    """Build input transforms for value inference.

    Similar to action policy but without output transforms since we only need to
    preprocess observations, not postprocess actions.
    """
    from rlinf.datasets.lerobot.transforms import (
        InjectDefaultPrompt,
        Normalize,
        PadStatesAndActions,
        ResizeImages,
    )

    input_transforms = []

    if env_type == "libero":
        from rlinf.datasets.lerobot.libero import LiberoInputs

        input_transforms.append(InjectDefaultPrompt(default_prompt))
        input_transforms.append(LiberoInputs(mask_padding=True, model_type=model_type))

    elif env_type == "franka":
        from rlinf.datasets.lerobot.franka import FrankaEEInputs

        input_transforms.append(InjectDefaultPrompt(default_prompt))
        input_transforms.append(FrankaEEInputs(mask_padding=True, model_type=model_type))

    else:
        raise ValueError(f"Unknown environment type: {env_type}")

    if norm_stats is not None:
        input_transforms.append(
            Normalize(
                norm_stats,
                use_quantiles=use_quantile_norm,
                skip_dims=action_norm_skip_dims,
            )
        )

    input_transforms.extend(
        [
            ResizeImages(224, 224),
            PadStatesAndActions(model_action_dim=action_dim),
        ]
    )

    return input_transforms
