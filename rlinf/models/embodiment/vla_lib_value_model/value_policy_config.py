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

"""
Value policy configuration and creation utilities.

This module provides functions to create value policies from trained value checkpoints.
Unlike action policies, value policies predict return values instead of actions.

Uses expert-based categorical value prediction via PI05ValueCritic.
"""

import glob
import json
import logging
import pathlib
from typing import Any, Optional, Sequence

import numpy as np
import safetensors.torch
import torch
from transformers import AutoTokenizer
from rlinf.datasets.vla_lib.lerobot_datasets.normalize import NormStats
from rlinf.datasets.vla_lib.lerobot_datasets.transforms import (
    DataTransformFn,
    InjectDefaultPrompt,
    Normalize,
    PadStatesAndActions,
    ResizeImages,
)
from rlinf.models.embodiment.vla_lib_value_model.configuration import PI05CriticConfig
from rlinf.models.embodiment.vla_lib_value_model.modeling_pi05_critic import (
    PI05ValueCritic,
)
from rlinf.models.embodiment.vla_lib_value_model.processing import PI05Processor

from .value_policy import ValuePolicy

logger = logging.getLogger(__name__)


def _load_state_dict_from_checkpoint(checkpoint_path: pathlib.Path) -> dict:
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
            return torch.load(str(checkpoint_path), map_location="cpu")

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
            state_dict.update(torch.load(f, map_location="cpu"))
        return state_dict

    raise FileNotFoundError(f"No checkpoint files found in {checkpoint_path}")


def _has_tokenizer_files(checkpoint_dir: pathlib.Path) -> bool:
    """Check if checkpoint directory has tokenizer files."""
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    return any((checkpoint_dir / f).exists() for f in tokenizer_files)


def load_norm_stats(
    checkpoint_dir: pathlib.Path, asset_id: str = "libero"
) -> dict[str, NormStats]:
    """Load normalization statistics from checkpoint assets.

    Args:
        checkpoint_dir: Checkpoint directory containing assets
        asset_id: Asset identifier (e.g., "libero", "droid")

    Returns:
        Dictionary mapping stat names to NormStats objects
    """
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


def create_trained_value_policy(
    checkpoint_dir: pathlib.Path | str,
    *,
    env_type: str = "libero",
    model_type: str = "pi05",
    default_prompt: Optional[str] = None,
    norm_stats: Optional[dict[str, NormStats]] = None,
    device: str = "cuda",
    metadata: Optional[dict[str, Any]] = None,
    num_return_bins: int = 201,
    return_min: float = -1.0,
    return_max: float = 0.0,
    action_norm_skip_dims: Optional[dict[str, list[int]]] = None,
    critic_expert_variant: str = "gemma_100m",
    **kwargs,
) -> ValuePolicy:
    """Create a value policy from a trained checkpoint.

    Args:
        checkpoint_dir: Path to checkpoint directory
        env_type: Environment type (e.g., "libero")
        model_type: Model type ("pi0", "pi05")
        default_prompt: Default prompt to inject if none provided
        norm_stats: Normalization stats (loaded from checkpoint if not provided)
        device: Device to run inference on
        metadata: Policy metadata
        num_return_bins: Number of return bins
        return_min: Minimum return value
        return_max: Maximum return value
        action_norm_skip_dims: Dims to skip in normalization (e.g., gripper)
        critic_expert_variant: Gemma variant (e.g., "gemma_100m")

    Returns:
        Configured ValuePolicy instance with batch inference support
    """
    checkpoint_dir = pathlib.Path(checkpoint_dir)

    logger.info(f"Loading value model from {checkpoint_dir}")

    # Load processor: try to load tokenizer from checkpoint, fallback to default
    logger.info("Loading processor...")
    if _has_tokenizer_files(checkpoint_dir):
        logger.info("  Found tokenizer files in checkpoint, loading from checkpoint")
        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint_dir), add_bos_token=True
        )
        processor = PI05Processor(
            tokenizer=tokenizer,
            max_token_len=200,
            discrete_state_input=False,
        )
    else:
        logger.info("  No tokenizer files in checkpoint, using default processor")
        processor = PI05Processor(
            max_token_len=200,
            discrete_state_input=False,
        )

    vocab_size = len(processor.tokenizer)
    logger.info(f"Processor vocab_size: {vocab_size}")

    # Load model
    critic_config = PI05CriticConfig(
        critic_expert_variant=critic_expert_variant,
        num_bins=num_return_bins,
        v_min=return_min,
        v_max=return_max,
    )
    logger.info("Loading PI05ValueCritic (expert categorical)")

    # Try from_pretrained first (for HuggingFace-style checkpoints)
    # If it fails, fallback to creating model from config and loading state dict
    try:
        model = PI05ValueCritic.from_pretrained(
            str(checkpoint_dir),
            config=critic_config,
            torch_dtype=torch.bfloat16,
            device_map=None,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )
        logger.info("  Loaded model using from_pretrained")
    except (OSError, ValueError, AttributeError) as e:
        # Fallback: create model from config and load state dict
        logger.info(
            f"  from_pretrained failed ({type(e).__name__}: {e}), loading state dict directly"
        )
        model = PI05ValueCritic(critic_config)
        state_dict = _load_state_dict_from_checkpoint(checkpoint_dir)

        # Handle key prefix mismatch between rlinf's ValueCriticModel and vla_lib's PI05ValueCritic
        # rlinf saves: "paligemma_with_expert.xxx"
        # vla_lib expects: "model.paligemma_with_expert.xxx"
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())

        # Check if we need to add "model." prefix
        if len(model_keys & ckpt_keys) == 0:
            # No direct matches, try adding "model." prefix to checkpoint keys
            remapped_state_dict = {}
            for k, v in state_dict.items():
                new_key = f"model.{k}"
                if new_key in model_keys:
                    remapped_state_dict[new_key] = v
                else:
                    remapped_state_dict[k] = v  # Keep original if no match

            if len(set(remapped_state_dict.keys()) & model_keys) > len(
                ckpt_keys & model_keys
            ):
                logger.info("  Remapped checkpoint keys: added 'model.' prefix")
                state_dict = remapped_state_dict

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(
            f"  Loaded state dict (missing={len(missing)}, unexpected={len(unexpected)})"
        )
        if missing:
            logger.debug(f"  Missing keys: {missing[:10]}...")
        if unexpected:
            logger.debug(f"  Unexpected keys: {unexpected[:10]}...")

    logger.info("Model ready (expert categorical)")

    # Attach processor to model for easy access
    object.__setattr__(model, "processor", processor)

    model = model.to(device)  # type: ignore[assignment]
    model.eval()

    if norm_stats is None:
        try:
            asset_id = env_type
            norm_stats = load_norm_stats(checkpoint_dir, asset_id)
            logger.info(
                f"Loaded norm stats from {checkpoint_dir} with asset_id={asset_id}"
            )
        except FileNotFoundError:
            logger.warning(
                f"Could not find norm stats in {checkpoint_dir}, proceeding without normalization"
            )
            norm_stats = None

    # Exclude 'return' from normalization (value model handles return discretization separately)
    if norm_stats and "return" in norm_stats:
        norm_stats = {k: v for k, v in norm_stats.items() if k != "return"}

    use_quantile_norm = model_type.lower() != "pi0"

    input_transforms = _build_input_transforms(
        env_type=env_type,
        model_type=model_type,
        action_dim=getattr(model.config, "action_dim", 32),
        default_prompt=default_prompt,
        norm_stats=norm_stats,
        use_quantile_norm=use_quantile_norm,
        action_norm_skip_dims=action_norm_skip_dims,
    )

    policy_metadata = {
        "model_type": model_type,
        "env_type": env_type,
        "checkpoint_dir": str(checkpoint_dir),
        "num_return_bins": num_return_bins,
        "return_range": [return_min, return_max],
        **(metadata or {}),
    }

    return ValuePolicy(
        model=model,
        transforms=input_transforms,
        metadata=policy_metadata,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        num_return_bins=num_return_bins,
        return_min=return_min,
        return_max=return_max,
    )


def _build_input_transforms(
    env_type: str,
    model_type: str,
    action_dim: int,
    default_prompt: Optional[str],
    norm_stats: Optional[dict[str, NormStats]],
    use_quantile_norm: bool,
    action_norm_skip_dims: Optional[dict[str, list[int]]] = None,
) -> Sequence[DataTransformFn]:
    """Build input transforms for value policy.

    Similar to action policy but without output transforms since we only need to
    preprocess observations, not postprocess actions.
    """
    input_transforms = []

    if env_type == "libero":
        from rlinf.datasets.vla_lib.lerobot_datasets.io_processing.libero import (
            LiberoInputs,
        )

        input_transforms.append(InjectDefaultPrompt(default_prompt))
        input_transforms.append(LiberoInputs(mask_padding=True, model_type=model_type))

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

    else:
        raise ValueError(f"Unknown environment type: {env_type}")

    return input_transforms
