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
import logging
import os

import torch
import torch.nn as nn
from omegaconf import DictConfig

from rlinf.models.embodiment.pi0_fast.pi0_fast_action_model import (
    PI0FastForRLActionPrediction,
)

_TEXT_TOKENIZER_ALLOW_PATTERNS = (
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "added_tokens.json",
    "special_tokens_map.json",
)

PI0_FAST_MODEL_ID = "lerobot/pi0fast-libero"
PI0_FAST_TEXT_TOKENIZER_ID = "google/paligemma-3b-pt-224"


def _resolve_pi0_fast_lora_target_modules(target_scope: str) -> tuple[str, str]:
    normalized_scope = str(target_scope).lower().replace("-", "_")
    if normalized_scope == "all_linear":
        return normalized_scope, "all-linear"
    raise ValueError(
        "pi0_fast currently supports only lora_target_scope='all_linear'; "
        f"got {target_scope!r}."
    )


def _apply_pi0_fast_lora(
    policy: nn.Module,
    rank: int,
    target_scope: str = "all_linear",
) -> nn.Module:
    if rank <= 0:
        raise ValueError(f"pi0_fast lora_rank must be positive, got {rank}.")

    target_scope, target_modules = _resolve_pi0_fast_lora_target_modules(target_scope)

    try:
        from peft import LoraConfig, inject_adapter_in_model
    except ImportError as exc:
        raise ImportError(
            "PI0-Fast LoRA requires peft. Reinstall the pi0_fast model environment."
        ) from exc

    for param in policy.parameters():
        param.requires_grad_(False)

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        lora_dropout=0.0,
        target_modules=target_modules,
        init_lora_weights="gaussian",
        bias="none",
    )
    policy = inject_adapter_in_model(lora_config, policy)

    trainable_params = sum(
        param.numel() for param in policy.parameters() if param.requires_grad
    )
    if trainable_params == 0:
        raise RuntimeError("PI0-Fast LoRA did not match any trainable modules.")
    total_params = sum(param.numel() for param in policy.parameters())
    logging.info(
        "PI0-Fast LoRA enabled: rank=%d target_scope=%s "
        "trainable_params=%d total_params=%d",
        rank,
        target_scope,
        trainable_params,
        total_params,
    )
    return policy


def _looks_like_hf_repo_id(name_or_path: str) -> bool:
    expanded = os.path.expanduser(str(name_or_path))
    if os.path.exists(expanded):
        return False
    if expanded.startswith(("/", "./", "../", "~")):
        return False
    if "://" in expanded:
        return False
    return "/" in expanded


def _hf_load_kwargs(model_cfg: DictConfig) -> dict:
    load_kwargs = {}
    if model_cfg.get("cache_dir") is not None:
        load_kwargs["cache_dir"] = str(model_cfg.cache_dir)
    if model_cfg.get("revision") is not None:
        load_kwargs["revision"] = str(model_cfg.revision)
    if model_cfg.get("local_files_only") is not None:
        load_kwargs["local_files_only"] = bool(model_cfg.local_files_only)
    return load_kwargs


def _resolve_hf_snapshot(
    name_or_path: str,
    *,
    cache_dir: str | None = None,
    revision: str | None = None,
    local_files_only: bool | None = None,
    allow_patterns: tuple[str, ...] | None = None,
) -> str:
    if revision is None and not local_files_only:
        return str(name_or_path)
    if not _looks_like_hf_repo_id(str(name_or_path)):
        return str(name_or_path)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "pi0_fast requires huggingface_hub to resolve pinned HF artifacts."
        ) from exc

    kwargs = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    if revision is not None:
        kwargs["revision"] = revision
    if local_files_only is not None:
        kwargs["local_files_only"] = local_files_only
    if allow_patterns is not None:
        kwargs["allow_patterns"] = list(allow_patterns)
    return snapshot_download(repo_id=str(name_or_path), **kwargs)


def _resolve_model_path(model_path: str, model_cfg: DictConfig) -> str:
    load_kwargs = _hf_load_kwargs(model_cfg)
    return _resolve_hf_snapshot(
        model_path,
        cache_dir=load_kwargs.get("cache_dir"),
        revision=load_kwargs.get("revision"),
        local_files_only=load_kwargs.get("local_files_only"),
    )


def _resolve_policy_tokenizers(policy_config, model_cfg: DictConfig) -> None:
    cache_dir = None
    if model_cfg.get("cache_dir") is not None:
        cache_dir = str(model_cfg.cache_dir)
    local_files_only = None
    if model_cfg.get("local_files_only") is not None:
        local_files_only = bool(model_cfg.local_files_only)
    text_revision = model_cfg.get("text_tokenizer_revision")
    if model_cfg.get("text_tokenizer_name") is not None:
        policy_config.text_tokenizer_name = str(model_cfg.text_tokenizer_name)
    policy_config.text_tokenizer_name = _resolve_hf_snapshot(
        policy_config.text_tokenizer_name,
        cache_dir=cache_dir,
        revision=str(text_revision) if text_revision is not None else None,
        local_files_only=local_files_only,
        allow_patterns=_TEXT_TOKENIZER_ALLOW_PATTERNS,
    )

    action_revision = model_cfg.get("action_tokenizer_revision")
    if model_cfg.get("action_tokenizer_name") is not None:
        policy_config.action_tokenizer_name = str(model_cfg.action_tokenizer_name)
    policy_config.action_tokenizer_name = _resolve_hf_snapshot(
        policy_config.action_tokenizer_name,
        cache_dir=cache_dir,
        revision=str(action_revision) if action_revision is not None else None,
        local_files_only=local_files_only,
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
    load_kwargs = _hf_load_kwargs(model_cfg)
    if not _looks_like_hf_repo_id(model_path):
        load_kwargs = {}

    policy_config = PreTrainedConfig.from_pretrained(model_path, **load_kwargs)
    if not isinstance(policy_config, pi0_fast_module.PI0FastConfig):
        raise TypeError(
            "Expected a LeRobot PI0FastConfig for pi0_fast model_path, got "
            f"{type(policy_config).__name__}."
        )

    if not policy_config.text_tokenizer_name:
        policy_config.text_tokenizer_name = PI0_FAST_TEXT_TOKENIZER_ID

    _resolve_policy_tokenizers(policy_config, model_cfg)

    override_names = (
        "max_action_tokens",
        "temperature",
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

    loader_kwargs = _hf_load_kwargs(model_cfg)
    if not _looks_like_hf_repo_id(model_path):
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


def _validate_artifact_pins(model_path: str, model_cfg: DictConfig) -> None:
    required = {
        "revision": model_cfg.get("revision"),
        "text_tokenizer_name": model_cfg.get("text_tokenizer_name"),
        "text_tokenizer_revision": model_cfg.get("text_tokenizer_revision"),
        "action_tokenizer_name": model_cfg.get("action_tokenizer_name"),
        "action_tokenizer_revision": model_cfg.get("action_tokenizer_revision"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(
            "pi0_fast requires pinned public artifacts; missing: " + ", ".join(missing)
        )
    if _looks_like_hf_repo_id(model_path) and model_path != PI0_FAST_MODEL_ID:
        logging.warning(
            "Using non-default PI0-Fast model repository %s with explicit revisions.",
            model_path,
        )


def get_model(cfg: DictConfig, torch_dtype=None):
    pi0_fast_module = _load_lerobot_pi0_fast()
    model_cfg = cfg.get("pi0_fast", {})
    configured_model_path = str(cfg.model_path)
    _validate_artifact_pins(configured_model_path, model_cfg)
    model_path = _resolve_model_path(configured_model_path, model_cfg)
    policy_config = _load_policy_config(pi0_fast_module, model_path, cfg)
    policy = pi0_fast_module.PI0FastPolicy.from_pretrained(
        model_path,
        config=policy_config,
    )
    if cfg.get("is_lora", False):
        policy = _apply_pi0_fast_lora(
            policy,
            rank=int(cfg.get("lora_rank", 32)),
            target_scope=str(cfg.get("lora_target_scope", "all_linear")),
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
