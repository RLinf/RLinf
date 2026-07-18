# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging

from rlinf.config import torch_dtype_from_precision

logger = logging.getLogger(__name__)


def get_model(cfg, torch_dtype=None):
    """Build the BEHAVIOR pi05 model from a model config (factory entry).

    ``cfg`` is ``actor.model``; ``cfg.model_path`` points at a *new-format*
    checkpoint directory containing ``model.safetensors``. The Pi0 model shape is
    built entirely from YAML fields (``num_action_chunks`` plus
    ``openpi.model_action_dim`` / ``openpi.paligemma_variant`` /
    ``openpi.action_expert_variant``); a checkpoint ``config.json`` is never read.

    The concrete wrapper class is selected by ``cfg.openpi.task``:

    * ``task: sft`` → :class:`OpenPiPytorchSFTActionModel` (built by
      :func:`_build_sft_model`). Flow-matching loss over the ``(Observation,
      actions)`` the BEHAVIOR SFT data loader already builds through the
      openpi transform pipeline — no processor lives on the model.
    * ``task: rl``  → :class:`OpenPiPytorchRLActionModel` (built by
      :func:`_build_rl_model`). Uses the openpi.transforms pipeline, adds the
      PPO chain-collecting SDE sampler + VLM value head.
    * ``task: eval`` → :class:`OpenPiPytorchEvalActionModel` (built by
      :func:`_build_eval_model`). Same openpi.transforms pipeline as ``rl``,
      but with no value head, no chain collection, no training-mode forward.
    """
    import pathlib

    import safetensors.torch
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_pytorch.pi0_model import gemma as pi0_gemma
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config
    from rlinf.models.embodiment.openpi_pytorch.utils.model_builders import (
        _build_eval_model,
        _build_rl_model,
        _build_sft_model,
    )

    model_cfg = cfg.openpi

    target_dtype = (
        torch_dtype
        if torch_dtype is not None
        else torch_dtype_from_precision(cfg.precision)
    )

    model_path = pathlib.Path(cfg.model_path)
    weights_path = model_path / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"openpi_pytorch checkpoint not found: {weights_path}")

    pi0_config = Pi0Config(
        pi05=True,
        action_horizon=int(cfg.num_action_chunks),
        action_dim=int(model_cfg.model_action_dim),
        paligemma_variant=str(model_cfg.paligemma_variant),
        action_expert_variant=str(model_cfg.action_expert_variant),
        dtype="bfloat16",
        pcd=False,
    )
    model = pi0_config.create()
    state_dict = safetensors.torch.load_file(str(weights_path), device="cpu")
    model.load_state_dict(state_dict, strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    if target_dtype is not None:
        model = model.to(target_dtype)

    num_steps = int(cfg.num_steps)
    action_chunk = int(cfg.num_action_chunks)
    action_env_dim = int(cfg.action_dim)

    task = OmegaConf.select(model_cfg, "task", default=None)
    if task is None:
        raise ValueError(
            "actor.model.openpi.task is required: set it to 'sft', 'rl', or 'eval' "
            "to pick the concrete OpenPI PyTorch model variant."
        )
    task = str(task).lower()

    logger.info(
        "openpi_pytorch[%s]: loaded %s (%.2fB params) strict from %s precision=%s "
        "num_steps=%s",
        task,
        pi0_config,
        n_params / 1e9,
        weights_path,
        cfg.precision,
        num_steps,
    )

    if task == "eval":
        return _build_eval_model(
            cfg,
            model_cfg,
            model,
            num_steps=num_steps,
            action_chunk=action_chunk,
            action_env_dim=action_env_dim,
        )

    if task == "sft":
        return _build_sft_model(
            model,
            num_steps=num_steps,
            action_env_dim=action_env_dim,
        )

    if task == "rl":
        paligemma_width = pi0_gemma.get_config(pi0_config.paligemma_variant).width
        return _build_rl_model(
            cfg,
            model_cfg,
            model,
            num_steps=num_steps,
            action_chunk=action_chunk,
            action_env_dim=action_env_dim,
            paligemma_width=paligemma_width,
        )

    raise ValueError(
        f"actor.model.openpi.task={task!r} is not supported; "
        "use 'eval', 'sft', or 'rl'."
    )
