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

import glob
import os

from omegaconf import DictConfig


def get_model(cfg: DictConfig, torch_dtype=None):
    import safetensors.torch
    import torch
    from dexbotic.data.dataset.transform.action import ActionNorm, PadState
    from dexbotic.data.dataset.transform.common import ToNumpy, ToTensor
    from dexbotic.data.dataset.transform.output import AbsoluteAction, ActionDenorm
    from dexbotic.model.dm0.dm0_arch import DM0Config
    from dexbotic.tokenization.process import DM0Tokenization
    from transformers import AutoTokenizer

    from rlinf.models.embodiment.dexbotic_dm0.dm0_policy import (
        DexboticDM0ForRLActionPrediction,
    )
    from rlinf.utils.logging import get_logger

    logger = get_logger()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    if not cfg.model_path or not os.path.exists(cfg.model_path):
        raise ValueError(f"Model path does not exist: {cfg.model_path}")

    try:
        config = DM0Config.from_pretrained(cfg.model_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_path, use_fast=False, local_files_only=True
        )
        config.num_steps = cfg.get("num_steps", 10)
        config.action_env_dim = cfg.action_dim
        config.add_value_head = cfg.get("add_value_head", True)
        config.noise_level = cfg.get("dexbotic", {}).get("noise_level", 0.5)
        config.noise_method = cfg.get("dexbotic", {}).get("noise_method", "flow_sde")
        config.detach_critic_input = cfg.get("dexbotic", {}).get(
            "detach_critic_input", True
        )
        config.train_expert_only = cfg.get("dexbotic", {}).get(
            "train_expert_only", False
        )
        config.action_horizon = config.chunk_size
        config.output_action_chunks = cfg.num_action_chunks
        config.safe_get_logprob = cfg.get("safe_get_logprob", False)
        config.chunk_critic_input = cfg.get("chunk_critic_input", True)
        config.noise_anneal = cfg.get("noise_anneal", False)
        config.joint_logprob = cfg.get("joint_logprob", False)
        config.value_after_vlm = cfg.get("value_after_vlm", False)
        config.processor_config = cfg.model_path

        # Force SDPA attention to avoid eager attention's O(S²) memory usage
        if hasattr(config, "llm_config") and config.llm_config is not None:
            config.llm_config._attn_implementation = "sdpa"
        if hasattr(config, "action_config") and config.action_config is not None:
            config.action_config._attn_implementation = "sdpa"

        original_offline = os.environ.get("HF_HUB_OFFLINE", None)
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            model = DexboticDM0ForRLActionPrediction(config)
        finally:
            if original_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = original_offline

        model.tokenizer = tokenizer
        model.dm0_tokenization = DM0Tokenization(tokenizer)

        weight_paths = sorted(glob.glob(os.path.join(cfg.model_path, "*.safetensors")))
        weight_paths = [p for p in weight_paths if not p.endswith(".index.json")]
        if not weight_paths:
            weight_path = os.path.join(cfg.model_path, "model.safetensors")
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"No weights found in {cfg.model_path}")
            weight_paths = [weight_path]
        for weight_path in weight_paths:
            safetensors.torch.load_model(model, weight_path, strict=False)

        # Weights loaded from checkpoint may restore float32 params that
        # DM0's to_bfloat16_for_selected_params() intentionally kept in fp32
        # (layernorms, etc.).  FSDP requires all params within a wrapped unit
        # to share the same dtype, so we enforce uniform dtype across the
        # entire model (including lm_head, value_head, etc.) after load.
        target_dtype = (
            torch.bfloat16 if cfg.get("precision", "bf16") == "bf16" else torch.float32
        )
        model = model.to(dtype=target_dtype)

        # PEVisionTower.device and .dtype use list(self.vision_tower.parameters())[-1]
        # which returns empty when FSDP (use_orig_params=False) has consumed the
        # parameters into a flat tensor.  Patch them with a dynamic fallback:
        # - Try the original parameters()-based lookup first (works for plain models
        #   and FSDP with use_orig_params=True).
        # - Fall back to a cached value only when the parameter list is empty (FSDP
        #   with use_orig_params=False).  The cached dtype is fixed at load time;
        #   the cached device is read from the model's mm_projector (which is always
        #   accessible, even under FSDP) so it follows model.to(device) calls.
        _pe_dtype_fallback = target_dtype
        _pe_model_ref = model  # weak-ish ref; the closure keeps the model alive anyway

        from dexbotic.model.modules.mm_vision.pe.pe_encoder import PEVisionTower

        def _pe_device_property(self):
            params = list(self.vision_tower.parameters())
            if params:
                return params[-1].device
            # FSDP flattened the params; ask the projector instead.
            try:
                return next(_pe_model_ref.model.mm_projector.parameters()).device
            except StopIteration:
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def _pe_dtype_property(self):
            params = list(self.vision_tower.parameters())
            if params:
                return params[-1].dtype
            return _pe_dtype_fallback

        PEVisionTower.device = property(_pe_device_property)
        PEVisionTower.dtype = property(_pe_dtype_property)

        norm_stats_file = os.path.join(cfg.model_path, "norm_stats.json")
        if os.path.exists(norm_stats_file):
            model.norm_stats = model._read_normalization_stats(norm_stats_file)
        else:
            model.norm_stats = None

        model._train_expert_only = getattr(config, "train_expert_only", False)

    except Exception as e:
        logger.error(f"Failed to load pretrained DM0 model: {e}")
        raise

    input_transforms_list = []
    if model.norm_stats is not None:
        input_transforms_list = [
            PadState(ndim=config.action_dim, axis=-1),
            ActionNorm(statistic_mapping=model.norm_stats, strict=False),
            ToTensor(),
        ]
    output_transforms_list = []
    if model.norm_stats is not None:
        output_transforms_list = [
            ToNumpy(),
            ActionDenorm(statistic_mapping=model.norm_stats, strict=False),
            AbsoluteAction(),
        ]
    model.setup_wrappers(
        transforms=input_transforms_list, output_transforms=output_transforms_list
    )
    return model
