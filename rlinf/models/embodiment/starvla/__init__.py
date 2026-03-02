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

"""starVLA embodied policy wrapper for RLinf.

This module exposes 'get_model', which loads a starVLA checkpoint and returns a
'StarVLAForRLActionPrediction' instance compatible with RLinf.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping

import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.utils.logging import get_logger

from .starvla_action_model import StarVLAForRLActionPrediction
from .utils.profile import resolve_vlm_interface


def get_model(
    cfg: DictConfig,
    torch_dtype: torch.dtype | None = None,
) -> StarVLAForRLActionPrediction:
    """Load a starVLA checkpoint and wrap it into RLinf's embodied policy interface.

    Args:
        cfg: Model config. Must specify a starVLA checkpoint path via
            'actor.model.model_path'.
        torch_dtype: Optional torch dtype to cast the loaded model to.

    Returns:
        A 'StarVLAForRLActionPrediction' instance.

    Raises:
        ValueError: If no checkpoint path is provided in 'cfg'.
    """
    logger = get_logger()
    model_path = getattr(cfg, "model_path", None)
    if model_path is None:
        raise ValueError(
            "starVLA requires 'actor.model.model_path'. Set it to a .pt checkpoint inside "
            "a starVLA run directory."
        )
    if model_path.endswith(".pt"):
        assert os.path.exists(model_path), (
            f"Checkpoint path {model_path} does not exist"
        )
        ckpt_path = model_path
    else:
        # Try to find the latest checkpoint in the checkpoints directory
        model_path = os.path.join(os.fspath(model_path), "checkpoints")
        assert os.path.exists(model_path), (
            f"Checkpoint path {model_path} does not exist"
        )
        ckpt_files = os.listdir(model_path)
        ckpt_files = sorted([f for f in ckpt_files if f.endswith(".pt")])
        assert len(ckpt_files) > 0, f"No checkpoint files found in {model_path}"
        ckpt_path = os.path.join(model_path, ckpt_files[-1])
    logger.info(f"Loading checkpoint file: {ckpt_path}")

    try:
        from starVLA.model.framework.base_framework import baseframework
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "starVLA is required to load starVLA checkpoints. Please install starVLA and "
            "ensure it is importable as the Python module 'starVLA'."
        ) from e

    starvla_model = baseframework.from_pretrained(ckpt_path)

    # Check early whether the loaded model provides a compatible interface.
    resolve_vlm_interface(starvla_model)

    # 'framework_name' is optional but helps infer the expected wiring for some checkpoints.
    starvla_cfg = getattr(cfg, "starvla", None)
    framework_name = getattr(starvla_cfg, "framework_name", None)
    if framework_name is not None:
        framework_name = str(framework_name).strip()
    if framework_name:
        starvla_model.framework_name = framework_name

    # Collect normalization-stat overrides from 'cfg' and merge them into
    # 'starvla_model.norm_stats'. This helps when checkpoints do not persist norm
    # stats (or persist incorrect values).
    extra_stats: dict[str, object] = {}

    inline_norm_stats = getattr(cfg, "norm_stats", None)
    if inline_norm_stats is not None and OmegaConf.is_config(inline_norm_stats):
        inline_norm_stats = OmegaConf.to_container(inline_norm_stats, resolve=True)
    if isinstance(inline_norm_stats, Mapping):
        extra_stats.update({str(k): v for k, v in inline_norm_stats.items()})

    norm_stats_path = getattr(cfg, "norm_stats_path", None)
    if isinstance(norm_stats_path, (str, os.PathLike)):
        norm_stats_path = os.fspath(norm_stats_path)
        if os.path.isfile(norm_stats_path):
            with open(norm_stats_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, Mapping):
                extra_stats.update({str(k): v for k, v in loaded.items()})

    if extra_stats:
        replace_norm_stats = bool(getattr(cfg, "replace_norm_stats", False))
        if replace_norm_stats:
            starvla_model.norm_stats = dict(extra_stats)
        else:
            norm_stats = getattr(starvla_model, "norm_stats", {})
            if norm_stats is not None and OmegaConf.is_config(norm_stats):
                norm_stats = OmegaConf.to_container(norm_stats, resolve=True)
            if not isinstance(norm_stats, Mapping):
                norm_stats = {}
            norm_stats = {str(k): v for k, v in norm_stats.items()}
            norm_stats.update(extra_stats)
            starvla_model.norm_stats = norm_stats

    # Cast to requested dtype.
    if torch_dtype is not None:
        starvla_model = starvla_model.to(dtype=torch_dtype)

    disable_action_unnorm = bool(getattr(cfg, "disable_action_unnormalization", False))

    return StarVLAForRLActionPrediction(
        starvla_model=starvla_model,
        action_dim=cfg.action_dim,
        num_action_chunks=cfg.num_action_chunks,
        add_value_head=getattr(cfg, "add_value_head", True),
        unnorm_key=getattr(cfg, "unnorm_key", None),
        disable_action_unnormalization=disable_action_unnorm,
    )


__all__ = ["StarVLAForRLActionPrediction", "get_model"]
