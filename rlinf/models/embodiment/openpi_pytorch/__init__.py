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

"""Self-contained PyTorch OpenPI 0.5 model package for embodied BEHAVIOR.

This package vendors the optimized PyTorch OpenPI 0.5 implementation so that the
eval / action-generation path is fully self-contained: it does not import the
externally installed ``openpi`` package and does not patch ``transformers``.

Layout:
  openpi_action_model.py  eval action sampling and SFT-loss entry point.
  pi0_model/              vendored model core (pi0, gemma, siglip, ...).
  utils/                  normalization, tokenizer, and image tooling.
  policies/               BEHAVIOR input/output transforms.

The BEHAVIOR streaming SFT dataset / data loader lives under
``rlinf.data.datasets.openpi_pytorch.behavior``; checkpoint conversion lives
under ``rlinf.utils.ckpt_convertor.openpi``.
"""

from __future__ import annotations

import logging
import pathlib

from rlinf.config import torch_dtype_from_precision

logger = logging.getLogger(__name__)


def get_model(cfg, torch_dtype=None):
    """Build the BEHAVIOR pi05 model from a model config (factory entry).

    ``cfg`` is ``actor.model``; ``cfg.model_path`` points at a *new-format*
    checkpoint directory containing ``model.safetensors``. The Pi0 model shape is
    built entirely from YAML fields (``num_action_chunks`` plus
    ``openpi.model_action_dim`` / ``openpi.paligemma_variant`` /
    ``openpi.action_expert_variant``); a checkpoint ``config.json`` is never read.

    The model dtype is precision-driven: ``precision: fp32`` keeps fp32 weights as
    the FSDP master (FSDP MixedPrecision casts to bf16 for compute and the
    optimizer updates the fp32 master, so warmup-LR updates are not lost to bf16
    rounding), while ``precision: bf16`` casts the weights to bf16 for eval. Norm
    stats and the PaliGemma tokenizer are resolved from YAML (``openpi.assets_dir``
    + ``openpi.asset_id`` and ``openpi.paligemma_tokenizer``); gradient
    checkpointing is governed by the FSDP manager
    (``fsdp_config.gradient_checkpointing``), not here.
    """
    import safetensors.torch

    from rlinf.data.datasets.openpi_pytorch import get_eval_processor
    from rlinf.models.embodiment.openpi_pytorch.openpi_action_model import (
        OpenPiPytorchActionModel,
    )
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config
    from rlinf.models.embodiment.openpi_pytorch.utils.normalize import load_norm_stats
    from rlinf.models.embodiment.openpi_pytorch.utils.tokenizer import (
        PaligemmaTokenizer,
    )

    model_cfg = cfg.openpi

    # Precision drives the weight dtype; the compute dtype (FSDP MixedPrecision
    # param_dtype) is a separate knob configured in the experiment YAML.
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
    # Strict load enforces key/shape parity. Weights are materialized in fp32, so a
    # bf16 base checkpoint widens losslessly into the fp32 master (the intended SFT
    # init); the dtype cast below then sets the requested weight precision.
    state_dict = safetensors.torch.load_file(str(weights_path), device="cpu")
    model.load_state_dict(state_dict, strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    if target_dtype is not None:
        model = model.to(target_dtype)

    num_steps = int(cfg.num_steps)
    action_chunk = int(cfg.num_action_chunks)
    action_env_dim = int(cfg.action_dim)

    # Norm stats + tokenizer resolve strictly from YAML (the SAME canonical
    # task-0000 stats the SFT data loader resolves), so eval and SFT share one
    # norm-stats distribution and there is no hard-coded asset/tokenizer path.
    norm_stats = load_norm_stats(model_cfg.assets_dir, model_cfg.asset_id)
    tokenizer = PaligemmaTokenizer(
        model_cfg.paligemma_tokenizer, max_len=pi0_config.max_token_len
    )
    # The eval processor is selected by env so the factory is not coupled to a
    # single environment; ``openpi.env`` defaults to "behavior" (the only env
    # registered today) when absent.
    env_type = model_cfg.get("env", "behavior")
    processor = get_eval_processor(
        env_type,
        norm_stats,
        tokenizer,
        action_chunk=action_chunk,
        action_env_dim=action_env_dim,
        model_action_dim=pi0_config.action_dim,
    )

    logger.info(
        "openpi_pytorch: loaded %s (%.2fB params) strict from %s precision=%s "
        "num_steps=%s",
        pi0_config,
        n_params / 1e9,
        weights_path,
        cfg.precision,
        num_steps,
    )
    return OpenPiPytorchActionModel(
        model,
        processor,
        num_steps=num_steps,
        action_chunk=action_chunk,
        action_env_dim=action_env_dim,
    )
