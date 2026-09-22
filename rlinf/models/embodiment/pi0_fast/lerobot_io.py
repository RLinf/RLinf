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

"""RLinf ↔ LeRobot I/O for PI0-Fast.

Converts environment observations into LeRobot batches, and loads the
checkpoint config plus optional preprocessor / postprocessor pipelines.
"""

from __future__ import annotations

import importlib
from typing import Any

import torch
from omegaconf import DictConfig


def _image_to_chw_float(
    image: torch.Tensor, image_size: int | None = None
) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(
            f"Expected image [B,H,W,C] or [B,C,H,W], got {tuple(image.shape)}"
        )
    needs_rescale = not image.is_floating_point()
    if not needs_rescale and image.numel() > 0:
        valid_unit_range = (
            torch.isfinite(image).all() & (image >= 0).all() & (image <= 1).all()
        )
        if not bool(valid_unit_range):
            raise ValueError(
                "Floating-point images must contain finite values in [0, 1]"
            )
    if image.shape[-1] in (1, 3):
        image = image.permute(0, 3, 1, 2).contiguous()
    elif image.shape[1] not in (1, 3):
        raise ValueError(
            f"Expected image channel dimension to be 1 or 3, got {tuple(image.shape)}"
        )
    if image_size is not None and tuple(image.shape[-2:]) != (image_size, image_size):
        image = torch.nn.functional.interpolate(
            image.to(dtype=torch.float32),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
    else:
        image = image.to(dtype=torch.float32)
    if needs_rescale:
        image = image / 255.0
    return image


def build_lerobot_batch_from_env_obs(
    env_obs: dict[str, Any],
    *,
    image_size: int | None = None,
) -> dict[str, Any]:
    """Convert RLinf environment observations to LeRobot policy inputs.

    Args:
        env_obs: Batched images, robot states, and task descriptions.
        image_size: Optional square image size expected by the policy.

    Returns:
        A LeRobot-compatible policy input batch.
    """
    batch: dict[str, Any] = {}
    batch["observation.images.image"] = _image_to_chw_float(
        env_obs["main_images"], image_size
    )
    if env_obs.get("wrist_images") is not None:
        batch["observation.images.image2"] = _image_to_chw_float(
            env_obs["wrist_images"], image_size
        )
    batch["observation.state"] = env_obs["states"].to(dtype=torch.float32)
    batch["task"] = list(env_obs["task_descriptions"])
    return batch


def _checkpoint_action_dim(policy_config) -> int | None:
    from lerobot.utils.constants import ACTION

    features = getattr(policy_config, "output_features", None) or {}
    feature = features.get(ACTION)
    shape = getattr(feature, "shape", None) if feature is not None else None
    if not shape:
        return None
    return int(shape[0])


def _align_pi0_fast_policy_config(policy_config, cfg: DictConfig):
    """Match LeRobot decode knobs to RLinf yaml, and reject silent shape drift."""
    model_cfg = cfg.get("pi0_fast", {})
    num_action_chunks = int(cfg.num_action_chunks)
    action_dim = int(cfg.action_dim)
    n_action_steps = int(getattr(policy_config, "n_action_steps", 0) or 0)
    if n_action_steps != num_action_chunks:
        raise ValueError(
            "pi0_fast model.num_action_chunks must match checkpoint "
            f"n_action_steps ({n_action_steps}), got {num_action_chunks}."
        )

    checkpoint_action_dim = _checkpoint_action_dim(policy_config)
    if checkpoint_action_dim is None:
        raise ValueError(
            "pi0_fast checkpoint is missing output_features['action'].shape."
        )
    if checkpoint_action_dim != action_dim:
        raise ValueError(
            "pi0_fast model.action_dim must match checkpoint action feature "
            f"shape[0] ({checkpoint_action_dim}), got {action_dim}."
        )

    max_action_tokens = model_cfg.get("max_action_tokens")
    if max_action_tokens is None:
        max_action_tokens = int(getattr(policy_config, "max_action_tokens", 256))
    else:
        max_action_tokens = int(max_action_tokens)
    if max_action_tokens <= 0:
        raise ValueError(
            f"pi0_fast max_action_tokens must be > 0, got {max_action_tokens}."
        )

    yaml_max_decoding_steps = model_cfg.get("max_decoding_steps")
    if (
        yaml_max_decoding_steps is not None
        and int(yaml_max_decoding_steps) != max_action_tokens
    ):
        raise ValueError(
            "pi0_fast max_decoding_steps must equal max_action_tokens "
            f"({max_action_tokens}), got {yaml_max_decoding_steps}."
        )

    policy_config.n_action_steps = num_action_chunks
    policy_config.max_action_tokens = max_action_tokens
    policy_config.max_decoding_steps = max_action_tokens
    # Train temperature comes from rollout.sampling_params. Keep the LeRobot
    # config greedy so leftover native inference cannot silently sample.
    policy_config.temperature = 0.0
    return policy_config


def load_policy_config(model_path: str, cfg: DictConfig):
    try:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pi0_fast import PI0FastConfig
    except ImportError as exc:
        raise ImportError(
            "pi0_fast requires LeRobot with pi0_fast support. "
            "Install with: bash requirements/install.sh embodied "
            "--model pi0_fast --env libero"
        ) from exc

    model_cfg = cfg.get("pi0_fast", {})
    policy_config = PreTrainedConfig.from_pretrained(model_path)
    if not isinstance(policy_config, PI0FastConfig):
        raise TypeError(
            "Expected a LeRobot PI0FastConfig for pi0_fast model_path, got "
            f"{type(policy_config).__name__}."
        )

    for name in ("text_tokenizer_name", "action_tokenizer_name"):
        value = model_cfg.get(name)
        if value is None:
            raise ValueError(f"pi0_fast requires a local {name} path.")
        setattr(policy_config, name, str(value))

    override_names = (
        "fast_skip_tokens",
        "use_kv_cache",
        "gradient_checkpointing",
    )
    for name in override_names:
        if model_cfg.get(name) is not None:
            setattr(policy_config, name, model_cfg.get(name))
    if model_cfg.get("require_action_token_prefix") is not None:
        policy_config.validate_action_token_prefix = bool(
            model_cfg.require_action_token_prefix
        )

    if model_cfg.get("device") is not None:
        policy_config.device = str(model_cfg.device)
    elif not cfg.get("load_to_device", True):
        policy_config.device = "cpu"

    return _align_pi0_fast_policy_config(policy_config, cfg)


def load_optional_processor(
    model_path: str, processor_type: str, model_cfg: DictConfig, policy_config=None
):
    try:
        from lerobot.processor import PolicyProcessorPipeline
    except ImportError:
        return None

    loader_kwargs = {}
    overrides = {}
    if processor_type == "pre":
        importlib.import_module("lerobot.policies.pi0_fast.processor_pi0_fast")

        tokenizer_override = {}
        text_tokenizer_name = getattr(policy_config, "text_tokenizer_name", None)
        if text_tokenizer_name is not None:
            tokenizer_override["tokenizer_name"] = str(text_tokenizer_name)
        if tokenizer_override:
            overrides["tokenizer_processor"] = tokenizer_override

        action_tokenizer_override = {}
        action_tokenizer_name = getattr(policy_config, "action_tokenizer_name", None)
        if action_tokenizer_name is not None:
            action_tokenizer_override["action_tokenizer_name"] = str(
                action_tokenizer_name
            )
        if text_tokenizer_name is not None:
            action_tokenizer_override["paligemma_tokenizer_name"] = str(
                text_tokenizer_name
            )
        if model_cfg.get("max_action_tokens") is not None:
            action_tokenizer_override["max_action_tokens"] = int(
                model_cfg.max_action_tokens
            )
        if model_cfg.get("fast_skip_tokens") is not None:
            action_tokenizer_override["fast_skip_tokens"] = int(
                model_cfg.fast_skip_tokens
            )
        if action_tokenizer_override:
            overrides["action_tokenizer_processor"] = action_tokenizer_override

        if model_cfg.get("device") is not None:
            overrides["device_processor"] = {"device": str(model_cfg.device)}

    if processor_type == "post":
        from lerobot.processor.converters import (
            policy_action_to_transition,
            transition_to_policy_action,
        )

        loader_kwargs.update(
            {
                "to_transition": policy_action_to_transition,
                "to_output": transition_to_policy_action,
            }
        )

    config_filenames = {
        "pre": ("policy_preprocessor.json", "preprocessor_config.json"),
        "post": ("policy_postprocessor.json", "postprocessor_config.json"),
    }[processor_type]
    for config_filename in config_filenames:
        try:
            processor = PolicyProcessorPipeline.from_pretrained(
                model_path,
                config_filename=config_filename,
                overrides=overrides,
                **loader_kwargs,
            )
            if processor_type == "pre":
                processor.steps = [
                    step
                    for step in processor.steps
                    if step.__class__.__name__ != "AddBatchDimensionProcessorStep"
                ]
            return processor
        except FileNotFoundError:
            continue
        except OSError as exc:
            message = str(exc).lower()
            if any(
                marker in message
                for marker in (
                    "not found",
                    "does not exist",
                    "404",
                    "entry not found",
                )
            ):
                continue
            raise
    return None
