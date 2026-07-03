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

"""STEAM value model factory.

Provides ``get_model(cfg)`` with the same signature as
:func:`rlinf.models.embodiment.value_model.recap.get_model` so the router in
``rlinf/models/__init__.py`` can dispatch between value-model flavours
uniformly.
"""

import glob
import logging
import os
from typing import Any, Union

import safetensors.torch
import torch
from omegaconf import DictConfig

from .configuration import SteamConfig
from .ensemble_modeling_critic import (
    EnsembleCriticOutput,
    EnsembleSteamCriticModel,
    clone_ensemble_members,
    reinitialize_member_value_heads,
)
from .modeling_critic import CriticOutput, SteamCriticModel

logger = logging.getLogger(__name__)


_STEAM_CONFIG_DEFAULTS: dict[str, Any] = {
    "vision_repo_id": "",
    "language_repo_id": "",
    "vision_revision": None,
    "language_revision": None,
    "fusion_hidden_dim": 512,
    "dropout": 0.1,
    "label_smoothing": 0.05,
    "num_frames_per_pair": 2,
    "num_bins": 2,
    "stride_k": None,
    "ensemble_size": 1,
    "ensemble_head_seed_base": None,
    "freeze_vision_encoder": False,
    "freeze_language_model": True,
    "use_gradient_checkpointing": False,
    "max_token_len": 200,
}

assert all(
    not isinstance(v, (dict, DictConfig)) for v in _STEAM_CONFIG_DEFAULTS.values()
), (
    "_STEAM_CONFIG_DEFAULTS must stay flat: build_steam_config "
    "merges only top-level fields and does not recurse."
)


def _is_override(value: Any) -> bool:
    """True if ``value`` is a real, non-empty override (skip ``None`` / ``""``)."""
    if value is None:
        return False
    if isinstance(value, str) and value == "":
        return False
    return True


def _candidate_checkpoint_dirs(model_path: str | os.PathLike[str] | None) -> list[str]:
    """Return candidate directories that may hold a ``config.json``.

    Accepts either a directory (typical: ``<step>`` or ``<step>/actor``) or a
    file path (e.g. ``<step>/actor/model_state_dict/full_weights.pt``), since
    RLinf's FSDP checkpoint layout nests the model artefacts under
    ``<step>/actor/``. Walks the input and up to two parents, adding each
    dir's ``actor/`` subdir when present.
    """
    if model_path is None:
        return []

    resolved = os.path.abspath(os.fspath(model_path))
    start = resolved if os.path.isdir(resolved) else os.path.dirname(resolved)

    candidates: list[str] = []
    current = start
    for _ in range(3):  # start, parent, grandparent
        if not current or not os.path.isdir(current):
            break
        if current not in candidates:
            candidates.append(current)
        actor_subdir = os.path.join(current, "actor")
        if os.path.isdir(actor_subdir) and actor_subdir not in candidates:
            candidates.append(actor_subdir)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return candidates


def load_steam_checkpoint_config(
    model_path: str | os.PathLike[str] | None,
) -> SteamConfig | None:
    """Load ``SteamConfig`` saved alongside a checkpoint if available."""
    for checkpoint_dir in _candidate_checkpoint_dirs(model_path):
        config_path = os.path.join(checkpoint_dir, "config.json")
        if not os.path.exists(config_path):
            continue
        try:
            config = SteamConfig.from_pretrained(
                checkpoint_dir,
                local_files_only=True,
            )
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.warning(
                "Failed to load SteamConfig from %s: %s",
                checkpoint_dir,
                exc,
            )
            continue
        logger.info(
            "Loaded SteamConfig metadata from checkpoint directory %s",
            checkpoint_dir,
        )
        return config
    return None


def _strip_model_prefix(state_dict: dict, model) -> dict:
    """Strip a ``model.`` prefix from checkpoint keys if required."""
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    if len(model_keys & ckpt_keys) == 0:
        stripped = {
            k.removeprefix("model."): v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }
        if len(set(stripped.keys()) & model_keys) > 0:
            logger.info("Stripped 'model.' prefix from checkpoint keys")
            return stripped
    return state_dict


def _load_state_dict(path: str) -> dict:
    """Load a state dict from .safetensors / .pt / .pth file or directory."""
    if path.endswith(".safetensors"):
        return safetensors.torch.load_file(path, device="cpu")
    elif path.endswith((".pt", ".pth")):
        return torch.load(path, map_location="cpu", weights_only=False)
    elif os.path.isdir(path):
        weight_paths = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if not weight_paths:
            weight_paths = sorted(glob.glob(os.path.join(path, "*.pt")))
        sd = {}
        for wp in weight_paths:
            if wp.endswith(".safetensors"):
                sd.update(safetensors.torch.load_file(wp, device="cpu"))
            else:
                sd.update(torch.load(wp, map_location="cpu", weights_only=False))
        return sd
    return {}


def _resolve_precision_dtype(precision) -> str:
    """Normalise a precision / dtype string to one of bfloat16/float32/float16.

    Accepts both Hydra-style short forms (``bf16``/``fp32``/``fp16`` and their
    Lightning-mixed variants) and already-normalised torch-style names
    (``bfloat16``/``float32``/``float16``). The latter matters when reading
    the value back from a saved ``SteamConfig.dtype`` in
    ``config.json`` — without that branch the rehydration path silently
    downgrades a float16 / float32 checkpoint to bfloat16.
    """
    if precision in ("bf16", "bf16-mixed", "bfloat16"):
        return "bfloat16"
    if precision in ("fp32", "32", "32-true", "float32"):
        return "float32"
    if precision in ("fp16", "16", "16-mixed", "float16"):
        return "float16"
    return "bfloat16"


def build_steam_config(
    cfg: DictConfig,
    checkpoint_config: SteamConfig | None = None,
) -> SteamConfig:
    """Build config from live Hydra values plus checkpoint metadata fallback.

    Precedence: ``_STEAM_CONFIG_DEFAULTS`` ← ``checkpoint_config`` ← ``cfg``.
    ``None`` and empty-string values in either source are skipped so that
    partially-specified Hydra configs do not erase real checkpoint metadata.
    """
    if checkpoint_config is None:
        checkpoint_config = load_steam_checkpoint_config(
            getattr(cfg, "model_path", None)
        )

    merged = dict[str, Any](_STEAM_CONFIG_DEFAULTS)
    for source_label, source in (
        ("checkpoint_config", checkpoint_config),
        ("cfg", cfg),
    ):  # cfg wins over checkpoint
        if source is None:
            continue
        for field in _STEAM_CONFIG_DEFAULTS:
            value = getattr(source, field, None)
            if not _is_override(value):
                continue
            if isinstance(value, (dict, DictConfig)):
                raise TypeError(
                    f"{source_label}.{field} is {type(value).__name__}; "
                    f"build_steam_config merges non-recursively and "
                    f"requires scalar fields. Pass the flat model sub-config "
                    f"(e.g. cfg.actor.model), not a nested parent."
                )
            merged[field] = value

    # ``dtype`` is derived from ``precision`` on either source, not a direct field.
    precision = getattr(cfg, "precision", None)
    if precision is None and checkpoint_config is not None:
        precision = getattr(checkpoint_config, "precision", None) or getattr(
            checkpoint_config, "dtype", None
        )
    merged["dtype"] = _resolve_precision_dtype(precision or "bf16")

    return SteamConfig(**merged)


def save_steam_checkpoint_assets(
    save_path: str,
    cfg: DictConfig | SteamConfig,
    processor: Any | None = None,
) -> None:
    """Persist config/tokenizer/image-processor assets next to checkpoint weights."""
    os.makedirs(save_path, exist_ok=True)
    model_config = cfg if isinstance(cfg, SteamConfig) else build_steam_config(cfg)
    model_config.to_json_file(os.path.join(save_path, "config.json"), use_diff=False)

    if processor is not None and hasattr(processor, "save_pretrained"):
        processor.save_pretrained(save_path)
        logger.info("Saved binary value processor assets to %s", save_path)

    logger.info("Saved binary value checkpoint metadata to %s", save_path)


def _resolve_model_path(model_path: str | None) -> str | None:
    if model_path is None:
        return None

    full_weights_path = os.path.join(model_path, "model_state_dict", "full_weights.pt")
    actor_full_weights_path = os.path.join(
        model_path,
        "actor",
        "model_state_dict",
        "full_weights.pt",
    )
    if os.path.exists(full_weights_path):
        return full_weights_path
    if os.path.exists(actor_full_weights_path):
        return actor_full_weights_path
    return model_path


def _num_matching_keys(state_dict: dict, model: torch.nn.Module) -> int:
    return len(set(state_dict.keys()) & set(model.state_dict().keys()))


def _load_checkpoint_into_model(
    model: torch.nn.Module, state_dict: dict, model_path: str
):
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(
        "Loaded fine-tuned checkpoint from %s (missing=%d, unexpected=%d)",
        model_path,
        len(missing),
        len(unexpected),
    )


def get_model(
    cfg: DictConfig,
    torch_dtype=None,
) -> Union[SteamCriticModel, EnsembleSteamCriticModel]:
    """Build a binary value critic or its ensemble wrapper.

    Signature matches :func:`rlinf.models.embodiment.value_model.recap.get_model`
    so the router in ``rlinf/models/__init__.py`` can dispatch via
    ``from rlinf.models.embodiment.value_model.steam import get_model``.

    Args:
        cfg: Hydra model config. Expected keys:
            - vision_repo_id, language_repo_id (local paths or HF repo ids)
            - label_smoothing, num_frames_per_pair, fusion_hidden_dim, dropout
            - precision (one of bf16/fp32/fp16)
            - freeze_vision_encoder, freeze_language_model
            - use_gradient_checkpointing
            - max_token_len
            - model_path: optional fine-tuned checkpoint path
        torch_dtype: unused, kept for interface parity with
            ``recap.get_value_model``.

    Returns:
        A :class:`SteamCriticModel` or
        :class:`EnsembleSteamCriticModel` instance.
    """
    del torch_dtype

    checkpoint_config = load_steam_checkpoint_config(getattr(cfg, "model_path", None))
    config = build_steam_config(cfg, checkpoint_config=checkpoint_config)
    model_path = _resolve_model_path(getattr(cfg, "model_path", None))
    state_dict = {}
    if model_path and os.path.exists(model_path):
        state_dict = _load_state_dict(model_path)

    if config.ensemble_size == 1:
        model = SteamCriticModel(config)
        logger.info("Created SteamCriticModel")

        if state_dict:
            model_state_dict = _strip_model_prefix(state_dict, model)
            _load_checkpoint_into_model(model, model_state_dict, model_path)
        else:
            logger.info(
                "No model_path provided; using from_pretrained() backbone weights."
            )
        return model

    base_member = SteamCriticModel(config)
    members = clone_ensemble_members(base_member, config.ensemble_size)
    model = EnsembleSteamCriticModel(config, members)
    logger.info(
        "Created EnsembleSteamCriticModel (ensemble_size=%d)",
        config.ensemble_size,
    )

    direct_load_applied = False
    if state_dict:
        ensemble_state_dict = _strip_model_prefix(state_dict, model)
        if _num_matching_keys(ensemble_state_dict, model) > 0:
            _load_checkpoint_into_model(model, ensemble_state_dict, model_path)
            direct_load_applied = True
            logger.info("Loaded ensemble checkpoint without reinitializing heads")
        else:
            member_state_dict = _strip_model_prefix(state_dict, base_member)
            if _num_matching_keys(member_state_dict, base_member) > 0:
                for member in members:
                    _load_checkpoint_into_model(member, member_state_dict, model_path)
                logger.info(
                    "Loaded legacy single-model checkpoint into all ensemble members"
                )
            else:
                logger.warning(
                    "Checkpoint at %s did not match ensemble or single-member "
                    "state dicts; falling back to pretrained backbone weights",
                    model_path,
                )
    else:
        logger.info("No model_path provided; using from_pretrained() backbone weights.")

    if not direct_load_applied:
        head_seed_base = (
            0
            if config.ensemble_head_seed_base is None
            else int(config.ensemble_head_seed_base)
        )
        reinitialize_member_value_heads(list(model.members), head_seed_base)
        logger.info(
            "Reinitialized %d ensemble value heads using seeds [%d, %d]",
            len(model.members),
            head_seed_base,
            head_seed_base + len(model.members) - 1,
        )

    return model


__all__ = [
    "SteamConfig",
    "SteamCriticModel",
    "CriticOutput",
    "EnsembleSteamCriticModel",
    "EnsembleCriticOutput",
    "build_steam_config",
    "get_model",
    "load_steam_checkpoint_config",
    "save_steam_checkpoint_assets",
]
