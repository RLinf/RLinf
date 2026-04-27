# Copyright 2026 The RLinf Authors.
# Signed-off-by: The RLinf Authors.
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

import os

import torch
from omegaconf import DictConfig

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

    enable_state_input = getattr(starvla_cfg, "enable_state_input", None)
    if enable_state_input is None:
        enable_state_input = getattr(cfg, "enable_state_input", True)

    # Cast to requested dtype.
    if torch_dtype is not None:
        starvla_model = starvla_model.to(dtype=torch_dtype)

    return StarVLAForRLActionPrediction(
        starvla_model=starvla_model,
        action_dim=cfg.action_dim,
        num_action_chunks=cfg.num_action_chunks,
        add_value_head=getattr(cfg, "add_value_head", True),
        unnorm_key=getattr(cfg, "unnorm_key", None),
        action_stats_source=getattr(cfg, "action_stats_source", "minmax"),
        enable_state_input=enable_state_input,
        policy_setup=getattr(cfg, "policy_setup", None),
    )


__all__ = ["StarVLAForRLActionPrediction", "get_model"]
