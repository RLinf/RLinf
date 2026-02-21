# Copyright 2025 The RLinf Authors.
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

Copied from vla_lib and modified to use local ValuePolicy with batch inference support.

Supports two model types:
1. PI05ForConditionalGeneration: VLM-based token prediction (continuous/discrete modes)
2. PI05ValueCritic: Expert-based direct value prediction (expert_* modes)
"""

import glob
import json
import logging
import pathlib
from typing import Any, Optional, Sequence

import numpy as np
import safetensors.torch
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from vla_lib.datasets.vla_datasets.lerobot_datasets.normalize import NormStats
from vla_lib.datasets.vla_datasets.lerobot_datasets.transforms import (
    DataTransformFn,
    InjectDefaultPrompt,
    Normalize,
    PadStatesAndActions,
    ResizeImages,
)
from vla_lib.models.vlas.openpi05.modeling_critic import (
    PI05CriticConfig,
    PI05ValueCritic,
)
from vla_lib.models.vlas.openpi05.modeling_pi05 import PI05ForConditionalGeneration
from vla_lib.models.vlas.openpi05.processing_pi05 import PI05Processor

from .value_policy import ValuePolicy

logger = logging.getLogger(__name__)

# Expert modes that require PI05ValueCritic
EXPERT_VALUE_MODES = ("expert_mse", "expert_categorical", "expert_distributional")


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
    value_mode: str = "continuous",
    top_p: float = 0.9,
    num_return_bins: int = 201,
    return_min: float = -1.0,
    return_max: float = 0.0,
    use_ar_generation: bool = False,
    action_norm_skip_dims: Optional[dict[str, list[int]]] = None,
    # Expert mode settings
    critic_expert_variant: str = "gemma_100m",
) -> ValuePolicy:
    """Create a value policy from a trained checkpoint.

    Args:
        checkpoint_dir: Path to checkpoint directory
        env_type: Environment type ("libero", "droid", "aloha", "franka", "franka_3cam")
        model_type: Model type ("pi0", "pi05")
        default_prompt: Default prompt to inject if none provided
        norm_stats: Normalization stats (loaded from checkpoint if not provided)
        device: Device to run inference on
        metadata: Policy metadata
        value_mode: Inference mode:
            - "continuous": VLM token logits, expectation over bins
            - "discrete": VLM token logits, argmax bin
            - "expert_mse": Expert model, continuous scalar output
            - "expert_categorical": Expert model, categorical output
            - "expert_distributional": Expert model, distributional output
        top_p: Deprecated parameter (kept for backward compatibility)
        num_return_bins: Number of return bins
        return_min: Minimum return value
        return_max: Maximum return value
        use_ar_generation: If True and mode is "discrete", use AR generation
        action_norm_skip_dims: Dims to skip in normalization (e.g., gripper)
        critic_expert_variant: Gemma variant for expert modes (e.g., "gemma_100m")

    Returns:
        Configured ValuePolicy instance with batch inference support
    """
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    is_expert_mode = value_mode in EXPERT_VALUE_MODES

    logger.info(f"Loading value model from {checkpoint_dir}")
    logger.info(f"  value_mode: {value_mode} ({'expert' if is_expert_mode else 'VLM'})")

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

    # Add value tokens for VLM modes (needed for token ID mapping)
    if not is_expert_mode:
        processor.add_value_tokens(num_bins=num_return_bins)
    vocab_size = len(processor.tokenizer)
    logger.info(f"Processor vocab_size: {vocab_size}")

    # Load model based on value_mode
    if is_expert_mode:
        # Expert mode: use PI05ValueCritic
        expert_loss_type = value_mode.replace(
            "expert_", ""
        )  # "mse", "categorical", "distributional"
        critic_config = PI05CriticConfig(
            critic_expert_variant=critic_expert_variant,
            critic_forward_mode="expert",
            expert_loss_type=expert_loss_type,
            num_bins=num_return_bins,
            v_min=return_min,
            v_max=return_max,
        )
        logger.info(f"Loading PI05ValueCritic with expert_loss_type={expert_loss_type}")

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
        except (OSError, ValueError, AttributeError, RuntimeError) as e:
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
    else:
        # VLM mode: use PI05ForConditionalGeneration
        model = PI05ForConditionalGeneration.from_pretrained(
            str(checkpoint_dir),
            torch_dtype=torch.bfloat16,
            device_map=None,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )

        # Handle extended embeddings for VLM mode
        safetensors_files = sorted(
            glob.glob(str(checkpoint_dir / "model-*.safetensors"))
        )
        if safetensors_files:
            logger.info(
                f"Loading extended embeddings from {len(safetensors_files)} shard(s)..."
            )

            embed_tokens_weight = None
            lm_head_weight = None

            for shard_file in safetensors_files:
                with safe_open(shard_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if "language_model.embed_tokens.weight" in key:
                            embed_tokens_weight = f.get_tensor(key)
                        elif "paligemma.lm_head.weight" in key:
                            lm_head_weight = f.get_tensor(key)

            if embed_tokens_weight is not None:
                new_vocab_size = embed_tokens_weight.shape[0]
                logger.info(f"Resizing model embeddings to {new_vocab_size}...")
                model.resize_token_embeddings(new_vocab_size)

                model.get_input_embeddings().weight.data.copy_(
                    embed_tokens_weight.to(model.get_input_embeddings().weight.dtype)
                )
                logger.info(
                    f"  Copied embed_tokens weights: {embed_tokens_weight.shape}"
                )

                if lm_head_weight is not None:
                    output_emb = model.get_output_embeddings()
                    if output_emb.weight.shape == lm_head_weight.shape:
                        output_emb.weight.data.copy_(
                            lm_head_weight.to(output_emb.weight.dtype)
                        )
                        logger.info(f"  Copied lm_head weights: {lm_head_weight.shape}")
                    else:
                        logger.warning(
                            f"  lm_head shape mismatch: model={output_emb.weight.shape}, ckpt={lm_head_weight.shape}"
                        )
                        if output_emb.weight.shape[0] == embed_tokens_weight.shape[0]:
                            output_emb.weight.data.copy_(
                                embed_tokens_weight.to(output_emb.weight.dtype)
                            )
                            logger.info(
                                "  Used embed_tokens weights for lm_head (tied)"
                            )

    # Expert modes predict values directly via ValueHead (no token vocab needed)
    # VLM modes predict value tokens and need extended vocab_size
    if is_expert_mode:
        logger.info(f"Model ready (expert mode: {value_mode}, outputs values directly)")
    else:
        vocab_size = getattr(model.config, "vocab_size", len(processor.tokenizer))
        logger.info(f"Model ready (VLM mode: {value_mode}, vocab_size={vocab_size})")

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
        "value_mode": value_mode,
        "top_p": top_p,
        "num_return_bins": num_return_bins,
        "return_range": [return_min, return_max],
        "use_ar_generation": use_ar_generation,
        **(metadata or {}),
    }

    return ValuePolicy(
        model=model,
        transforms=input_transforms,
        metadata=policy_metadata,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        value_mode=value_mode,
        num_return_bins=num_return_bins,
        return_min=return_min,
        return_max=return_max,
        use_ar_generation=use_ar_generation,
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
        from vla_lib.datasets.vla_datasets.lerobot_datasets.io_processing.libero import (
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

    elif env_type == "droid":
        from vla_lib.datasets.vla_datasets.lerobot_datasets.io_processing.droid import (
            DroidInputs,
        )

        input_transforms.append(InjectDefaultPrompt(default_prompt))
        input_transforms.append(DroidInputs())

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

    elif env_type == "aloha":
        from vla_lib.datasets.vla_datasets.lerobot_datasets.io_processing.aloha import (
            AlohaInputs,
        )

        input_transforms.append(InjectDefaultPrompt(default_prompt))
        input_transforms.append(AlohaInputs(adapt_to_pi=True))

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

    elif env_type in ["franka", "franka_demo"]:
        from vla_lib.datasets.vla_datasets.lerobot_datasets.io_processing.franka import (
            FrankaInputs,
        )

        input_transforms.append(InjectDefaultPrompt(default_prompt))
        input_transforms.append(FrankaInputs(model_type=model_type))

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

    elif env_type == "franka_3cam":
        from vla_lib.datasets.vla_datasets.lerobot_datasets.io_processing.franka_3cam import (
            FrankaInputs as Franka3CamInputs,
        )

        input_transforms.append(InjectDefaultPrompt(default_prompt))
        input_transforms.append(Franka3CamInputs())

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
