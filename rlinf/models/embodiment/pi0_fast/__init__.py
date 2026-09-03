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

from __future__ import annotations

import importlib

import torch
import torch.nn as nn
from omegaconf import DictConfig

from rlinf.models.embodiment.pi0_fast.pi0_fast_action_model import (
    PI0FastForRLActionPrediction,
)


def _load_lerobot_pi0_fast():
    try:
        module = importlib.import_module("lerobot.policies.pi0_fast")
    except ImportError as exc:
        raise ImportError(
            "pi0_fast requires LeRobot with pi0_fast support. "
            "Install with: bash requirements/install.sh embodied "
            "--model pi0_fast --env libero"
        ) from exc
    return module


def _load_policy_config(pi0_fast_module, model_path: str, cfg: DictConfig):
    try:
        from lerobot.configs.policies import PreTrainedConfig
    except ImportError as exc:
        raise ImportError(
            "pi0_fast requires LeRobot policy config support. "
            "Install with: bash requirements/install.sh embodied "
            "--model pi0_fast --env libero"
        ) from exc

    model_cfg = cfg.get("pi0_fast", {})
    policy_config = PreTrainedConfig.from_pretrained(model_path)
    if not isinstance(policy_config, pi0_fast_module.PI0FastConfig):
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
        "max_action_tokens",
        "max_decoding_steps",
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

    return policy_config


def _processor_config_is_missing(exc: OSError) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in ("not found", "does not exist", "404", "entry not found")
    )


def _load_optional_processor(
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
            if _processor_config_is_missing(exc):
                continue
            raise
    return None


def _cast_model_to_dtype(
    model: nn.Module, torch_dtype: torch.dtype | None
) -> nn.Module:
    # LeRobot PI0-Fast checkpoints use their own mixed dtype layout; recasting the
    # policy changes greedy FAST tokens. Keep the checkpoint's original layout.
    del torch_dtype
    return model


def _validate_action_shape(cfg: DictConfig) -> None:
    for field in ("num_action_chunks", "action_dim"):
        value = cfg.get(field, 0)
        if not value or int(value) <= 0:
            raise ValueError(f"pi0_fast requires model.{field} > 0, got {value!r}.")


def get_model(
    cfg: DictConfig,
    torch_dtype: torch.dtype | None = None,
) -> PI0FastForRLActionPrediction:
    """Build an RLinf policy wrapper around LeRobot's PI0FastPolicy.

    Args:
        cfg: PI0-Fast model configuration.
        torch_dtype: Requested model dtype. The checkpoint's native mixed-dtype
            layout is preserved regardless of this value.

    Returns:
        The PI0-Fast policy adapted to RLinf's embodied policy interface.
    """
    pi0_fast_module = _load_lerobot_pi0_fast()
    model_cfg = cfg.get("pi0_fast", {})
    configured_model_path = str(cfg.model_path)
    _validate_action_shape(cfg)
    model_path = configured_model_path
    policy_config = _load_policy_config(pi0_fast_module, model_path, cfg)
    policy = pi0_fast_module.PI0FastPolicy.from_pretrained(
        model_path,
        config=policy_config,
    )
    model = PI0FastForRLActionPrediction(
        policy,
        action_dim=int(cfg.action_dim),
        num_action_chunks=int(cfg.num_action_chunks),
        max_action_tokens=int(model_cfg.get("max_action_tokens", 256)),
        image_size=model_cfg.get("image_size", None),
        temperature_train=float(model_cfg.get("temperature_train", 0.3)),
        temperature_eval=float(model_cfg.get("temperature_eval", 0.0)),
        preprocessor=_load_optional_processor(
            model_path, "pre", model_cfg, policy_config
        ),
        postprocessor=_load_optional_processor(model_path, "post", model_cfg),
    )
    return _cast_model_to_dtype(model, torch_dtype)
