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

"""Build a Cosmos3 action-policy for RLinf SFT.

Builds cosmos's *training* config directly, then disables only its internal
FSDP/CP/CFGP (via ``enable_inference_mode`` so AC + EMA stay on) -- RLinf's
FSDPModelManager owns the single ``fully_shard``. The model is materialized on
CPU (net bf16 ~27GB + net_ema fp32 ~54GB ~= 81GB > a single 80GB GPU) and
sharded to GPU later by FSDP2. Model defaults live in the Hydra config group
``examples/sft/config/model/cosmos3.yaml``.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import re
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig

from rlinf.models.embodiment.cosmos3.cosmos3_policy import Cosmos3Policy
from rlinf.utils.logging import get_logger

logger = get_logger()


def _cosmos_framework_root() -> Path:
    """cosmos-framework repo root (its configs reference files relative to it)."""
    import cosmos_framework

    return Path(cosmos_framework.__file__).resolve().parent.parent


def _apply_checkpoint_model_config(cosmos_cfg, model_cfg: DictConfig) -> None:
    """Use checkpoint architecture and processor with the SFT recipe settings."""
    from cosmos_framework.inference.common.config import structure_config
    from cosmos_framework.inference.common.public_model_config import (
        load_model_config_from_hf_config,
    )

    model_path = Path(str(model_cfg.get("model_path", ""))).expanduser()
    for config_path in (
        model_path / "model" / "config.json",
        model_path / "config.json",
    ):
        if not config_path.is_file():
            continue
        checkpoint_config = json.loads(config_path.read_text())
        if "model" not in checkpoint_config:
            continue
        checkpoint_model = structure_config(
            load_model_config_from_hf_config(checkpoint_config), DictConfig
        )
        recipe_config = cosmos_cfg.model.config
        # Architecture, action width and tokenizer come from the same checkpoint.
        # Training and memory settings remain owned by the selected SFT recipe.
        for key in (
            "action_gen",
            "activation_checkpointing",
            "compile",
            "ema",
            "max_num_tokens_after_packing",
            "parallelism",
            "rectified_flow_training_config",
        ):
            checkpoint_model.config[key] = copy.deepcopy(getattr(recipe_config, key))
        checkpoint_model.config.diffusion_expert_config.load_weights_from_pretrained = (
            recipe_config.diffusion_expert_config.load_weights_from_pretrained
        )
        num_domains = model_cfg.get("num_embodiment_domains")
        if num_domains is not None:
            checkpoint_model.config.num_embodiment_domains = int(num_domains)
        cosmos_cfg.model = checkpoint_model
        logger.info(
            "Cosmos3: using architecture from %s (backbone=%s, max_action_dim=%s)",
            config_path,
            checkpoint_model.config.vlm_config.model_name,
            checkpoint_model.config.max_action_dim,
        )
        break

    tokenizer_path = model_cfg.get("tokenizer_path") or os.environ.get(
        "COSMOS_TOKENIZER_PATH"
    )
    if tokenizer_path:
        from cosmos_framework.data.generator.processors import build_processor
        from cosmos_framework.utils.lazy_config import LazyCall

        tokenizer_path = Path(str(tokenizer_path)).expanduser().resolve()
        cosmos_cfg.model.config.vlm_config.tokenizer = LazyCall(build_processor)(
            tokenizer_type=str(tokenizer_path)
        )
        logger.info("Cosmos3: using local VLM processor from %s", tokenizer_path)

    wan_vae_path = model_cfg.get("wan_vae_path") or os.environ.get("WAN_VAE_PATH")
    if wan_vae_path:
        tokenizer = cosmos_cfg.model.config.tokenizer
        tokenizer.vae_path = str(wan_vae_path)
        # A bucket would turn even an absolute local path into an S3 URI.
        tokenizer.bucket_name = ""
        tokenizer.object_store_credential_path_pretrained = ""


def _promote_scalar_params_to_1d(model: nn.Module) -> None:
    """FSDP2 cannot shard 0-d Parameters; promote scalars to shape [1].

    Preserves dtype (so fp32 ``net_ema`` stays fp32) and ``requires_grad``.
    """
    # Collect first: mutating params during named_parameters() iteration is unsafe.
    scalar_names = [name for name, p in model.named_parameters() if p.ndim == 0]
    for full_name in scalar_names:
        module, param_name = model, full_name
        if "." in full_name:
            module_name, param_name = full_name.rsplit(".", 1)
            module = model.get_submodule(module_name)
        old_p = getattr(module, param_name)
        setattr(
            module,
            param_name,
            nn.Parameter(old_p.detach().reshape(1), requires_grad=old_p.requires_grad),
        )


def _apply_freeze(cosmos3_model: nn.Module, keys_to_select: list[str]) -> None:
    """Freeze the understanding tower: any ``net.*`` param whose name contains
    none of ``keys_to_select`` is frozen (mirrors cosmos's optimizer rule).
    ``net_ema`` is a separate frozen copy, left untouched."""
    if not keys_to_select:
        return
    net = getattr(cosmos3_model, "net", None)
    if net is None:
        logger.warning("Cosmos3: cosmos3_model has no `net`; skipping freeze.")
        return
    frozen = trainable = 0
    for name, p in net.named_parameters():
        if any(k in name for k in keys_to_select):
            trainable += p.numel()
        else:
            p.requires_grad = False
            frozen += p.numel()
    logger.info(
        "Cosmos3: froze understanding tower -- trainable=%.2fM frozen=%.2fM (keys_to_select=%s)",
        trainable / 1e6,
        frozen / 1e6,
        keys_to_select,
    )


def _load_base_weights(
    cosmos3_model: nn.Module, model_path: str, keys_to_skip: list[str]
) -> None:
    """Warm-start ``cosmos3_model`` from a base checkpoint, skipping action heads + net_ema.

    Reuses cosmos's loaders so the AC prefix and DTensor resharding are handled:
    ``load_vfm_model`` (HF safetensors dir, regex skip on ``cosmos3_model.net``) or
    ``_load_model`` (DCP dir, substring skip via ``CustomLoadPlanner``).
    """
    from cosmos_framework.utils.generator.model_loader import (
        _is_safetensors_checkpoint,
        _load_model,
    )

    if _is_safetensors_checkpoint(model_path, None):
        from cosmos_framework.model.generator.utils.safetensors_loader import (
            load_vfm_model,
        )

        skip_patterns = [f".*{re.escape(s)}.*" for s in keys_to_skip]
        vfm = getattr(cosmos3_model, "net", cosmos3_model)
        logger.info(
            "Cosmos3: loading safetensors base into cosmos3_model.net from %s (skip=%s)",
            model_path,
            skip_patterns,
        )
        load_vfm_model(
            vfm,
            model_path,
            credential_path=None,
            parallel_dims=getattr(cosmos3_model, "parallel_dims", None),
            skip_patterns=skip_patterns or None,
        )
        return

    # DCP dir; accept either model_path or model_path/model layouts.
    dcp_dir = (
        os.path.join(model_path, "model")
        if os.path.isdir(os.path.join(model_path, "model"))
        else model_path
    )
    logger.info(
        "Cosmos3: loading DCP base into cosmos3_model from %s (skip=%s)",
        dcp_dir,
        keys_to_skip,
    )
    _load_model(
        cosmos3_model,
        dcp_dir,
        credential_path=None,
        keys_to_skip_loading=keys_to_skip or None,
    )


def load_inference_checkpoint(model: Cosmos3Policy, checkpoint_path: str) -> None:
    """Load RLinf weights, excluding EMA state when the policy disables EMA."""
    state_dict = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True, mmap=True
    )
    ema_keys = []
    if not model._ema_enabled:
        ema_keys = [key for key in state_dict if key.startswith("omni.net_ema.")]
        for key in ema_keys:
            del state_dict[key]
    model.load_state_dict(state_dict, strict=True)
    logger.info(
        "Cosmos3: loaded inference checkpoint %s (excluded EMA keys=%d)",
        checkpoint_path,
        len(ema_keys),
    )


def _configure_sft_objective(model_config, cfg: DictConfig) -> None:
    """Select native loss weights without changing the Cosmos loss implementation."""
    objective = cfg.get("sft_objective", "joint")
    rf_config = model_config.rectified_flow_training_config
    if objective == "joint":
        rf_config.normalize_loss_by_active = False
        if hasattr(rf_config, "sample_level_loss_averaging"):
            rf_config.sample_level_loss_averaging = False
        if hasattr(rf_config, "loss_reduction_mode"):
            rf_config.loss_reduction_mode = "local_item_mean"
        return
    rf_config.normalize_loss_by_active = True
    # The current image remains a clean condition. Preserve native dummy graph
    # connections and MoE regularization, while disabling both vision FM weights.
    rf_config.loss_scale = 0.0
    rf_config.image_loss_scale = 0.0


def get_model(cfg: DictConfig, torch_dtype=None):
    """Build and warm-start a ``Cosmos3Policy`` for RLinf SFT.

    ``torch_dtype`` is accepted only for model-builder registry signature parity;
    cosmos sets precision from its own config, so it is unused here.

    Returns a ``Cosmos3Policy`` wrapping a cosmos-FSDP-disabled ``OmniMoTModel``
    with AC + EMA on, ready for RLinf ``FSDPModelManager`` to ``fully_shard``.
    """
    import cosmos_framework.model.generator.omni_mot_model as omni_mot_model
    from cosmos_framework.configs.base.config import make_config
    from cosmos_framework.utils import distributed as cosmos_distributed
    from cosmos_framework.utils.config_helper import override
    from cosmos_framework.utils.flags import Device as _CosmosDevice
    from cosmos_framework.utils.lazy_config import instantiate

    model_path = cfg.get("model_path")

    # Build + instantiate under cosmos-framework root (its configs use paths
    # relative to that root).
    with contextlib.chdir(_cosmos_framework_root()):
        cosmos3_cfg = make_config()
        experiment_name = "action_policy_libero_nano"
        cosmos3_cfg = override(cosmos3_cfg, ["--", f"experiment={experiment_name}"])
        _apply_checkpoint_model_config(cosmos3_cfg, cfg)

        # Disable cosmos internal FSDP/CP/CFGP at any world size; RLinf FSDP owns
        # fully_shard. enable_inference_mode avoids degrees=1 re-enabling cosmos
        # FSDP under multi-GPU, without touching AC/EMA.
        if bool(cfg.get("disable_cosmos_parallelism", True)):
            p = cosmos3_cfg.model.config.parallelism
            p.enable_inference_mode = True
            p.data_parallel_shard_degree = 1
            p.context_parallel_shard_degree = 1
            p.cfg_parallel_shard_degree = 1
        if not bool(cfg.get("ema_enabled", True)):
            cosmos3_cfg.model.config.ema.enabled = False
        cosmos3_cfg.model.config.compile.enabled = bool(
            cfg.get("compile_enabled", False)
        )
        _configure_sft_objective(cosmos3_cfg.model.config, cfg)

        # cosmos probes on CPU unless get_backend() == "nccl", which RLinf's lazily
        # bound group never reports; the probe is diagnostic only, so skip it.
        _ensure_world_communicator = cosmos_distributed.ensure_world_communicator
        cosmos_distributed.ensure_world_communicator = lambda *args, **kwargs: None
        try:
            cosmos3_cfg.validate()
        finally:
            cosmos_distributed.ensure_world_communicator = _ensure_world_communicator
        cosmos3_cfg.freeze()

        # Materialize on CPU: net(bf16 ~27GB) + net_ema(fp32 ~54GB) ~= 81GB would
        # OOM a single 80GB GPU before FSDP2 can shard. FSDP2 shards it to GPU.
        _orig_device = omni_mot_model.DEVICE
        omni_mot_model.DEVICE = _CosmosDevice.CPU
        try:
            cosmos3_model = instantiate(cosmos3_cfg.model)
        finally:
            omni_mot_model.DEVICE = _orig_device

        # tensor_kwargs were captured as CPU during the CPU build; restore to CUDA
        # (cosmos builds condition_mask/noise from them in training_step).
        cosmos3_model.tensor_kwargs = {
            "device": _orig_device,
            "dtype": cosmos3_model.precision,
        }
        cosmos3_model.tensor_kwargs_fp32 = {
            "device": _orig_device,
            "dtype": torch.float32,
        }
        # Qwen3-VL asserts rotary inv_freq is fp32; bf16 build cast it to bf16.
        for _mod in cosmos3_model.modules():
            if getattr(_mod, "inv_freq", None) is not None:
                _mod.inv_freq = _mod.inv_freq.float()
        cosmos3_model.net.init_weights(buffer_device=_CosmosDevice.CPU)

    # Warm-start from the non-action base, skipping the fresh action heads + EMA.
    _load_base_weights(
        cosmos3_model, model_path, list(cfg.get("keys_to_skip_loading", []))
    )
    # Freeze the understanding tower (only keys_to_select train).
    _apply_freeze(cosmos3_model, list(cfg.get("keys_to_select", [])))
    # FSDP2 cannot shard 0-d params.
    _promote_scalar_params_to_1d(cosmos3_model)

    return Cosmos3Policy(
        cosmos3_model,
        lr_multipliers=dict(cfg.get("lr_multipliers", {})),
        ema_enabled=bool(cfg.get("ema_enabled", True)),
        rollout_config=cfg,
    )


__all__ = ["Cosmos3Policy", "get_model", "load_inference_checkpoint"]
