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

"""MolmoAct2 embodied policy wrapper for RLinf (evaluation only).

Exposes ``get_model``, which builds the official LeRobot ``MolmoAct2Policy`` from
a checkpoint directory (local path or HuggingFace repo id) and wraps it in
``MolmoAct2ForRLActionPrediction``.

MolmoAct2 is installed externally (see ``requirements/install.sh --model
molmoact2``), which installs RLinf's LeRobot fork (branch
``RLinf/molmoact2-hf-inference``) providing ``lerobot.policies.molmoact2``.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from rlinf.models.embodiment.molmoact2.molmoact2_policy import (
    MolmoAct2ForRLActionPrediction,
)
from rlinf.utils.logging import get_logger

logger = get_logger()


def get_model(cfg: DictConfig, torch_dtype: torch.dtype | None = None):
    """Load a MolmoAct2 checkpoint and wrap it for RLinf.

    Args:
        cfg: Model config. Requires ``model_path`` (or ``checkpoint_path``);
            MolmoAct2-specific options live under the ``molmoact2`` block.
        torch_dtype: Ignored. MolmoAct2 loads its weights in fp32 upstream; a
            warning is logged when another precision is requested.

    Returns:
        A ``MolmoAct2ForRLActionPrediction`` instance.
    """
    try:
        from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
        from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "MolmoAct2 requires the pinned LeRobot checkout. Install it with "
            "'bash requirements/install.sh embodied --model molmoact2 --env libero'."
        ) from e

    checkpoint_path = cfg.get("checkpoint_path", None) or cfg.get("model_path", None)
    if not checkpoint_path:
        raise ValueError(
            "MolmoAct2 requires 'checkpoint_path' or 'model_path' in the model config."
        )

    if torch_dtype is not None and torch_dtype != torch.float32:
        logger.warning(
            f"MolmoAct2 loads its weights in fp32 upstream; "
            f"model.precision={cfg.get('precision', None)} has no effect."
        )

    molmoact2_cfg = cfg.get("molmoact2", None) or {}
    molmo_cfg = MolmoAct2Config(
        checkpoint_path=checkpoint_path,
        num_steps=molmoact2_cfg.get("num_steps", None),
        inference_action_mode=molmoact2_cfg.get("inference_action_mode", "continuous"),
        discrete_action_tokenizer=molmoact2_cfg.get(
            "discrete_action_tokenizer",
            "allenai/MolmoAct2-FAST-Tokenizer",
        ),
        enable_depth_reasoning=molmoact2_cfg.get("enable_depth_reasoning", False),
        # Empty by default: upstream raises rather than silently skipping action
        # un-normalization when the checkpoint's norm_tag is not configured.
        norm_tag=molmoact2_cfg.get("norm_tag", ""),
    )

    return MolmoAct2ForRLActionPrediction(
        MolmoAct2Policy(molmo_cfg),
        action_dim=cfg.get("action_dim", 7),
        num_action_chunks=cfg.get("num_action_chunks", 1),
    )


__all__ = ["MolmoAct2ForRLActionPrediction", "get_model"]
