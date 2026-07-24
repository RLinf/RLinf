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

"""FastWAM model factory for RLinf.

``get_model`` composes the upstream FastWAM Hydra configs (model + data/processor),
instantiates the model and processor exactly as the official LIBERO eval does, loads
the fine-tuned checkpoint, and wraps everything in :class:`FastWAMPolicy`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from rlinf.models.embodiment.fastwam.fastwam_policy import FastWAMPolicy
from rlinf.utils.logging import get_logger

logger = get_logger()


@dataclass
class FastWAMInferConfig:
    """Resolved rollout-time hyper-parameters (defaults mirror ``configs/sim_libero.yaml``)."""

    action_horizon: int = 32
    num_action_chunks: int = 8
    num_inference_steps: int = 10
    binarize_gripper: bool = True
    text_cfg_scale: float = 1.0
    negative_prompt: str = ""
    sigma_shift: Optional[float] = None
    seed: Optional[int] = None
    rand_device: str = "cpu"
    tiled: bool = False
    concat_multi_camera: str = "horizontal"
    # Optional world-model future-video generation during eval (uses infer_joint,
    # which also decodes pixels -> slower; applied to env 0 only, capped count).
    visualize_future_video: bool = False
    future_video_dir: Optional[str] = None
    num_video_frames: int = 9
    max_video_saves: int = 12


def _default_fastwam_config_dir() -> str:
    """Locate FastWAM's ``configs`` dir (editable install keeps the repo intact)."""
    env = os.environ.get("FASTWAM_CONFIG_DIR")
    if env:
        return env
    import fastwam

    repo_root = Path(fastwam.__file__).resolve().parents[2]
    cfg_dir = repo_root / "configs"
    if not cfg_dir.is_dir():
        raise FileNotFoundError(
            f"FastWAM configs dir not found at {cfg_dir}. Set FASTWAM_CONFIG_DIR or "
            "model.fastwam.config_dir."
        )
    return str(cfg_dir)


def _register_fastwam_resolvers() -> None:
    """Register the OmegaConf resolvers FastWAM configs rely on (idempotent)."""
    for name, fn in (
        ("eval", eval),
        ("max", lambda x: max(x)),
        ("split", lambda s, idx: s.split("/")[int(idx)]),
    ):
        try:
            OmegaConf.register_new_resolver(name, fn)
        except ValueError:
            pass  # already registered


def _compose_fastwam_cfg(config_dir: str, config_name: str, overrides) -> DictConfig:
    """Compose a FastWAM Hydra config without disturbing RLinf's own Hydra state."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    _register_fastwam_resolvers()
    already = GlobalHydra.instance().is_initialized()
    if already:
        GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(config_name=config_name, overrides=list(overrides or []))
    finally:
        GlobalHydra.instance().clear()
    return cfg


def _resolve_dataset_stats_path(cfg: DictConfig, ckpt_path: Optional[str]) -> str:
    explicit = cfg.get("dataset_stats_path", None)
    candidates = []
    if explicit:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))
    if ckpt_path:
        ckpt = Path(os.path.expanduser(os.path.expandvars(str(ckpt_path))))
        # released checkpoint ships ``<name>_dataset_stats.json`` next to ``<name>.pt``
        candidates.append(ckpt.with_name(ckpt.stem + "_dataset_stats.json"))
        for parent in list(ckpt.parents)[:4]:
            candidates.append(parent / "dataset_stats.json")
    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        "Could not locate FastWAM dataset_stats.json. Set "
        "model.dataset_stats_path to the *_dataset_stats.json shipped with the checkpoint. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def get_model(cfg: DictConfig, torch_dtype=None) -> nn.Module:
    """Build a :class:`FastWAMPolicy` from an RLinf model config.

    Expected ``cfg`` keys (under ``rollout.model`` / ``actor.model``):
        model_type: "fastwam"
        precision: "bf16"
        checkpoint_path: path to ``libero_uncond_2cam224.pt`` (None for SFT-from-base)
        dataset_stats_path: optional explicit ``*_dataset_stats.json``
        num_action_chunks / action_horizon / num_inference_steps / binarize_gripper / ...
        fastwam:
            config_dir: optional FastWAM configs dir (auto-detected otherwise)
            config_name: FastWAM Hydra config (default "sim_libero")
            overrides: optional list[str] of extra Hydra overrides
    """
    # Point DiffSynth at HuggingFace + the shared checkpoint dir (idempotent).
    os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")
    os.environ.setdefault(
        "DIFFSYNTH_MODEL_BASE_PATH",
        os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "/workspace/checkpoints"),
    )

    fw = cfg.get("fastwam", {}) or {}
    config_dir = fw.get("config_dir", None) or _default_fastwam_config_dir()
    config_name = fw.get("config_name", "sim_libero")
    overrides = fw.get("overrides", None)

    fcfg = _compose_fastwam_cfg(config_dir, config_name, overrides)

    if torch_dtype is None:
        torch_dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from hydra.utils import instantiate

    # Instantiate the FastWAM model (resolves ${data...}/${model...} against fcfg root).
    model = instantiate(fcfg.model, model_dtype=torch_dtype, device=device)

    ckpt_path = cfg.get("checkpoint_path", None) or cfg.get("model_path", None)
    if ckpt_path:
        ckpt_path = os.path.expanduser(os.path.expandvars(str(ckpt_path)))
        if Path(ckpt_path).exists():
            logger.info("Loading FastWAM checkpoint: %s", ckpt_path)
            model.load_checkpoint(ckpt_path)
        else:
            raise FileNotFoundError(f"FastWAM checkpoint_path not found: {ckpt_path}")
    else:
        logger.warning(
            "FastWAM get_model called without checkpoint_path; using base/random "
            "weights (only valid for SFT-from-base)."
        )

    # Train only the MoT experts (+ proprio encoder), matching FastWAM's
    # ``_apply_dit_only_train_mode``. Frozen modules (VAE / text encoder) are not
    # updated during SFT; for eval this is a harmless no-op under torch.no_grad.
    if bool(cfg.get("freeze_non_dit", True)):
        model.requires_grad_(False)
        if hasattr(model, "dit"):
            model.dit.requires_grad_(True)
        if getattr(model, "proprio_encoder", None) is not None:
            model.proprio_encoder.requires_grad_(True)

    # Build the processor + normalizer from dataset stats (action/state min-max).
    processor = instantiate(fcfg.data.train.processor).eval()
    stats_path = _resolve_dataset_stats_path(cfg, ckpt_path)
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

    processor.set_normalizer_from_stats(load_dataset_stats_from_json(stats_path))
    logger.info("FastWAM processor using dataset stats: %s", stats_path)

    # Resolve rollout hyper-parameters (RLinf cfg wins over sim_libero defaults).
    eval_cfg = fcfg.get("EVALUATION", {})
    num_frames = int(fcfg.data.train.num_frames)
    infer_cfg = FastWAMInferConfig(
        action_horizon=int(cfg.get("action_horizon", None) or (num_frames - 1)),
        num_action_chunks=int(cfg.get("num_action_chunks", eval_cfg.get("replan_steps", 8))),
        num_inference_steps=int(
            cfg.get("num_inference_steps", eval_cfg.get("num_inference_steps", 10))
        ),
        binarize_gripper=bool(cfg.get("binarize_gripper", eval_cfg.get("binarize_gripper", True))),
        text_cfg_scale=float(cfg.get("text_cfg_scale", eval_cfg.get("text_cfg_scale", 1.0))),
        negative_prompt=str(cfg.get("negative_prompt", eval_cfg.get("negative_prompt", ""))),
        sigma_shift=cfg.get("sigma_shift", None),
        seed=cfg.get("seed", None),
        rand_device=str(cfg.get("rand_device", "cpu")),
        tiled=bool(cfg.get("tiled", False)),
        concat_multi_camera=str(fcfg.data.train.get("concat_multi_camera", "horizontal")),
        visualize_future_video=bool(cfg.get("visualize_future_video", False)),
        future_video_dir=cfg.get("future_video_dir", None),
        num_video_frames=(num_frames - 1)
        // int(fcfg.data.train.get("action_video_freq_ratio", 1))
        + 1,
        max_video_saves=int(cfg.get("max_video_saves", 12)),
    )

    policy = FastWAMPolicy(model=model, processor=processor, infer_cfg=infer_cfg)
    policy = policy.to(dtype=torch_dtype)
    return policy
