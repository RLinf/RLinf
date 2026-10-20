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

"""Build the native Cosmos3 action model for RLinf training and HF evaluation.

Builds cosmos's *training* config directly, then disables only its internal
FSDP/CP/CFGP (via ``enable_inference_mode`` so AC + EMA stay on) -- RLinf's
FSDPModelManager owns the single ``fully_shard``. The network and FP32 EMA are
materialized on CPU and sharded to GPU by FSDP2. Shared model defaults live in
``examples/embodiment/config/model/cosmos3.yaml``.
"""

from __future__ import annotations

import contextlib
import os

import torch
import torch.nn as nn
from omegaconf import DictConfig

from rlinf.models.embodiment.cosmos3.checkpoint_loader import (
    load_pretrained_checkpoint,
    native_checkpoint_covers_language_model,
)
from rlinf.models.embodiment.cosmos3.cosmos3_action_model import Cosmos3ActionModel
from rlinf.models.embodiment.cosmos3.cosmos3_config import (
    build_cosmos3_config,
)
from rlinf.models.embodiment.cosmos3.cosmos3_config import (
    cosmos_framework_root,
)
from rlinf.models.embodiment.cosmos3.cosmos3_policy import Cosmos3Policy
from rlinf.utils.logging import get_logger


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
    logger = get_logger()
    net = getattr(cosmos3_model, "net", None)
    if net is None:
        logger.warning("Cosmos3: cosmos3_model has no `net`; skipping freeze.")
        return
    for name, p in net.named_parameters():
        if not any(k in name for k in keys_to_select):
            p.requires_grad = False


def _initialize_rotary_buffers(net: nn.Module) -> None:
    """Restore nonpersistent RoPE buffers after meta materialization."""
    from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.nemotron_3_dense_vl import (
        MultiModalRotaryEmbedding,
    )
    from cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl import (
        Qwen3VLTextRotaryEmbedding,
        Qwen3VLVisionRotaryEmbedding,
    )

    for module in net.modules():
        if isinstance(
            module,
            (
                Qwen3VLTextRotaryEmbedding,
                Qwen3VLVisionRotaryEmbedding,
                MultiModalRotaryEmbedding,
            ),
        ):
            module.init_weights(buffer_device=torch.device("cpu"))
            if hasattr(module, "original_inv_freq"):
                module.original_inv_freq = module.inv_freq.clone()


def _initialize_native_weights(
    model: nn.Module, model_path: str, keys_to_skip_loading: list[str]
) -> None:
    """Initialize fresh heads and buffers before loading a native checkpoint."""
    if not native_checkpoint_covers_language_model(
        model, model_path, keys_to_skip_loading
    ):
        model.net.init_weights(buffer_device=torch.device("cpu"))
        return

    language_model = model.net.language_model
    original = language_model.init_weights
    had_instance_initializer = "init_weights" in language_model.__dict__
    # The native initializer still owns fresh action heads; DCP owns the backbone.
    language_model.init_weights = lambda buffer_device=None: _initialize_rotary_buffers(
        language_model
    )
    try:
        model.net.init_weights(buffer_device=torch.device("cpu"))
    finally:
        if had_instance_initializer:
            language_model.init_weights = original
        else:
            del language_model.init_weights


def get_model(cfg: DictConfig, torch_dtype=None):
    """Build a trainable Cosmos3 model for SFT, actor and HF rollout.

    ``torch_dtype`` is accepted only for model-builder registry signature parity;
    cosmos sets precision from its own config, so it is unused here.

    ``checkpoint_format`` only selects how weights are read into the native
    implementation; it never changes the inference or training code.
    """
    logger = get_logger()
    if str(cfg.get("model_backend", "native")).lower() != "native":
        raise ValueError(
            "Cosmos3 uses one native model for actor, HF rollout and eval; "
            "remove model_backend=diffusers and keep checkpoint_format=diffusers "
            "when loading a Diffusers checkpoint."
        )
    checkpoint_format = str(cfg.get("checkpoint_format", "native")).lower()

    import cosmos_framework.model.generator.omni_mot_model as omni_mot_model
    from cosmos_framework.utils import distributed as cosmos_distributed
    from cosmos_framework.utils.flags import Device as _CosmosDevice
    from cosmos_framework.utils.lazy_config import instantiate

    model_path = cfg.get("model_path")

    # Native architecture configs contain paths relative to cosmos-framework.
    cosmos_root = cosmos_framework_root()
    with contextlib.chdir(cosmos_root):
        cosmos3_cfg = build_cosmos3_config(cfg)

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
        if cfg.get("activation_checkpointing_mode") is not None:
            cosmos3_cfg.model.config.activation_checkpointing.mode = str(
                cfg.activation_checkpointing_mode
            )
        torch.backends.cudnn.benchmark = True

        # Wan2.2 VAE: the cookbook injects via ${oc.env:WAN_VAE_PATH}; we build the
        # config directly, so set the local file path here instead.
        wan_vae_path = cfg.get("wan_vae_path") or os.environ.get("WAN_VAE_PATH")
        if wan_vae_path:
            try:
                cosmos3_cfg.model.config.tokenizer.vae_path = str(wan_vae_path)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Cosmos3: could not set tokenizer.vae_path=%s", wan_vae_path
                )

        # cosmos probes on CPU unless get_backend() == "nccl", which RLinf's lazily
        # bound group never reports; the probe is diagnostic only, so skip it.
        _ensure_world_communicator = getattr(
            cosmos_distributed, "ensure_world_communicator", None
        )
        if _ensure_world_communicator is not None:
            cosmos_distributed.ensure_world_communicator = lambda *args, **kwargs: None
        try:
            cosmos3_cfg.validate()
        finally:
            if _ensure_world_communicator is not None:
                cosmos_distributed.ensure_world_communicator = (
                    _ensure_world_communicator
                )
        cosmos3_cfg.freeze()

        # Build on CPU to avoid allocating the full network and EMA on one GPU.
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
        if checkpoint_format != "diffusers":
            _initialize_native_weights(
                cosmos3_model,
                str(model_path),
                list(cfg.get("keys_to_skip_loading", [])),
            )
        # Nonpersistent buffers are absent from both DCP and Diffusers weights.
        _initialize_rotary_buffers(cosmos3_model.net)
        net_ema = getattr(cosmos3_model, "net_ema", None)
        if net_ema is not None:
            _initialize_rotary_buffers(net_ema)

    # Native checkpoints warm-start from a base model; Diffusers packages load
    # the complete trained policy, including action heads, into the same native
    # model used by the actor and HF rollout worker.
    load_pretrained_checkpoint(
        cosmos3_model,
        model_path,
        checkpoint_format,
        list(cfg.get("keys_to_skip_loading", [])),
    )
    # Freeze the understanding tower (only keys_to_select train).
    _apply_freeze(cosmos3_model, list(cfg.get("keys_to_select", [])))
    # FSDP2 cannot shard 0-d params.
    _promote_scalar_params_to_1d(cosmos3_model)

    return Cosmos3ActionModel(
        cosmos3_model,
        model_cfg=cfg,
        lr_multipliers=dict(cfg.get("lr_multipliers", {})),
        ema_enabled=bool(cfg.get("ema_enabled", True)),
    )


__all__ = ["Cosmos3ActionModel", "Cosmos3Policy", "get_model"]
