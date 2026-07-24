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

"""Evo-1 (evo1-flash) embodied policy wrapper for RLinf.

Exposes ``get_model``, which loads an Evo-1 checkpoint (DeepSpeed-format
directory containing ``config.json``, ``norm_stats.json`` and
``mp_rank_00_model_states.pt``) and returns an ``Evo1ForRLActionPrediction``.

Evo-1 is installed externally (see ``requirements/install.sh``); its repo root
must be importable so that ``import scripts.Evo1`` / ``import config`` resolve.
Configure the repo location via ``cfg.evo1.repo_path`` or the ``EVO1_REPO_PATH``
environment variable.
"""

from __future__ import annotations

import json
import os
import sys

import torch
from omegaconf import DictConfig

from rlinf.models.embodiment.evo1.evo1_action_model import Evo1ForRLActionPrediction
from rlinf.models.embodiment.evo1.utils.normalizer import (
    NormalizationType,
    Normalizer,
)
from rlinf.utils.logging import get_logger

logger = get_logger()


def _ensure_evo1_importable(cfg: DictConfig) -> None:
    """Put the Evo-1 repo root on sys.path so ``scripts``/``config`` resolve."""
    evo1_cfg = getattr(cfg, "evo1", None)
    # Consider BOTH the config's repo_path and the EVO1_REPO_PATH env var, and
    # only add directories that actually exist. This way a stale placeholder
    # repo_path (e.g. the template default "/path/to/Evo-1") does not shadow a
    # valid env var, and vice versa.
    roots = []
    if evo1_cfg is not None:
        rp = getattr(evo1_cfg, "repo_path", None)
        if rp:
            roots.append(rp)
    env_rp = os.environ.get("EVO1_REPO_PATH")
    if env_rp:
        roots.append(env_rp)
    # Evo-1's modules (scripts/, config.py, model/, dataset/) live directly under
    # Evo_1/, so that directory is the import root.
    for root in roots:
        for cand in (root, os.path.join(root, "Evo_1")):
            if os.path.isdir(cand) and cand not in sys.path:
                sys.path.insert(0, cand)


def get_model(cfg: DictConfig, torch_dtype: torch.dtype | None = None):
    """Load an Evo-1 checkpoint and wrap it for RLinf.

    Args:
        cfg: Model config. Must set ``model_path`` to an Evo-1 checkpoint dir.
        torch_dtype: Optional dtype to cast the loaded model to.

    Returns:
        An ``Evo1ForRLActionPrediction`` instance.
    """
    _ensure_evo1_importable(cfg)

    try:
        from config import EvoConfig  # type: ignore
        from scripts.Evo1 import EVO1  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Evo-1 is required. Install it (requirements/install.sh --model evo1) "
            "and set cfg.evo1.repo_path or EVO1_REPO_PATH to the Evo-1 repo root "
            "so that 'import scripts.Evo1' and 'import config' resolve."
        ) from e

    ckpt_dir = getattr(cfg, "model_path", None)
    if ckpt_dir is None or not os.path.isdir(ckpt_dir):
        raise ValueError(
            f"Evo-1 requires 'model_path' to be a checkpoint directory. Got: {ckpt_dir}"
        )

    config_path = os.path.join(ckpt_dir, "config.json")
    stats_path = os.path.join(ckpt_dir, "norm_stats.json")
    weights_path = os.path.join(ckpt_dir, "mp_rank_00_model_states.pt")
    for p in (config_path, stats_path, weights_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing Evo-1 checkpoint file: {p}")

    with open(config_path) as f:
        config_dict = json.load(f)

    config_dict.setdefault("num_inference_timesteps", 32)

    # Optional overrides from the RLinf config's nested evo1 block.
    evo1_cfg = getattr(cfg, "evo1", None)
    if evo1_cfg is not None:
        for key in (
            "horizon",
            "action_horizon",
            "per_action_dim",
            "state_dim",
            "num_inference_timesteps",
            "num_layers",
            "num_categories",
            "vlm_name",
            "image_size",
        ):
            val = getattr(evo1_cfg, key, None)
            if val is not None:
                config_dict[key] = val

    # RL fine-tuning scope (mirrors Evo-1 SFT set_finetune_flags). Default freezes
    # the InternVL3 VLM (embedder) and trains only the flow-matching DiT action head
    # -- the SFT "stage1" recipe. Set evo1.rl_trainable_scope="all" for a full
    # fine-tune. Full-finetuning a 1B VLM under sparse RL reward is unstable.
    rl_scope = "action_head"
    if evo1_cfg is not None:
        rl_scope = getattr(evo1_cfg, "rl_trainable_scope", None) or "action_head"
    rl_scope = str(rl_scope).strip().lower()
    config_dict["finetune_language_model"] = False
    config_dict["finetune_vision_model"] = False
    config_dict["finetune_action_head"] = True
    if rl_scope == "all":
        config_dict["finetune_vlm"] = True
    else:
        config_dict["finetune_vlm"] = False
        # A frozen VLM needs no activation checkpointing; disabling it also avoids
        # reentrant-checkpoint "no input requires grad" errors during training.
        config_dict["enable_gradient_checkpointing"] = False

    evo_config = EvoConfig.from_dict(config_dict)
    logger.info(
        f"Loading Evo-1 from {ckpt_dir} "
        f"(horizon={evo_config.horizon}, per_action_dim={evo_config.per_action_dim}, "
        f"state_dim={evo_config.state_dim}, steps={evo_config.num_inference_timesteps})"
    )

    model = EVO1(evo_config).eval()
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("module", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Evo-1 load: {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        logger.warning(
            f"Evo-1 load: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})"
        )

    # Apply the RL fine-tuning scope by reusing Evo-1's own set_finetune_flags()
    # (reads the finetune_* flags set on evo_config above). build_optimizer filters
    # by requires_grad, so freezing here is what makes RL train the DiT head only.
    if hasattr(model, "set_finetune_flags"):
        model.set_finetune_flags()
    _n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Evo-1 RL trainable scope='{rl_scope}': {_n_train / 1e6:.1f}M / "
        f"{_n_total / 1e6:.1f}M params trainable "
        f"({100.0 * _n_train / max(1, _n_total):.1f}%)"
    )

    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)

    with open(stats_path) as f:
        stats = json.load(f)
    normalization_type = config_dict.get(
        "normalization_type", NormalizationType.BOUNDS.value
    )
    normalizer = Normalizer(stats, normalization_type=normalization_type)

    def _get(key, default):
        if evo1_cfg is not None:
            v = getattr(evo1_cfg, key, None)
            if v is not None:
                return v
        return default

    # RL (SDE-replay) config: an optional ``rl_head_config`` block on the model
    # config enables/parameterizes the flow-matching RL path (GRPO-first).
    rl_head_config = getattr(cfg, "rl_head_config", None)
    rl_denoising_steps = getattr(cfg, "denoising_steps", None)
    if rl_denoising_steps is None and rl_head_config is not None:
        rl_denoising_steps = getattr(rl_head_config, "denoising_steps", None)

    return Evo1ForRLActionPrediction(
        evo1_model=model,
        normalizer=normalizer,
        action_dim=cfg.action_dim,
        num_action_chunks=cfg.num_action_chunks,
        arm_key=_get("arm_key", "franka_ee_pose_delta"),
        dataset_key=_get("dataset_key", "libero_10_no_noops_lerobot"),
        policy_setup=getattr(cfg, "policy_setup", "libero"),
        image_size=int(_get("image_size", evo_config.image_size)),
        num_view_slots=int(_get("num_view_slots", 3)),
        binarize_gripper=bool(_get("binarize_gripper", True)),
        rl_head_config=rl_head_config,
        rl_denoising_steps=rl_denoising_steps,
        valid_action_dim=getattr(cfg, "action_dim", None),
    )


__all__ = ["Evo1ForRLActionPrediction", "get_model"]
