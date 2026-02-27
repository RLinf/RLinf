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
# dexbotic model configs

import glob
import os

from omegaconf import DictConfig


def get_model(cfg: DictConfig, torch_dtype=None):
    import safetensors.torch
    import torch
    from dexbotic.data.dataset.transform.action import ActionNorm, PadState
    from dexbotic.data.dataset.transform.common import ToNumpy, ToTensor
    from dexbotic.data.dataset.transform.output import AbsoluteAction, ActionDenorm
    from dexbotic.model.pi0.pi0_arch import Pi0Config
    from dexbotic.tokenization.process import Pi0Tokenization
    from transformers import AutoTokenizer

    from rlinf.models.embodiment.dexbotic_pi.dexbotic_pi_policy import (
        DexboticPi0ForRLActionPrediction,
    )
    from rlinf.utils.logging import get_logger

    logger = get_logger()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    if not cfg.model_path or not os.path.exists(cfg.model_path):
        raise ValueError(f"Model path does not exist: {cfg.model_path}")

    try:
        config = Pi0Config.from_pretrained(cfg.model_path, local_files_only=True)
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
        original_offline = os.environ.get("HF_HUB_OFFLINE", None)
        os.environ["HF_HUB_OFFLINE"] = "1"

        try:
            model = DexboticPi0ForRLActionPrediction(config)
        finally:
            if original_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = original_offline
        model.tokenizer = tokenizer

        model.pi0_tokenization = Pi0Tokenization(tokenizer)
        weight_paths = sorted(glob.glob(os.path.join(cfg.model_path, "*.safetensors")))
        weight_paths = [p for p in weight_paths if not p.endswith(".index.json")]
        if not weight_paths:
            weight_path = os.path.join(cfg.model_path, "model.safetensors")
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"No weights found in {cfg.model_path}")
            weight_paths = [weight_path]
        for weight_path in weight_paths:
            safetensors.torch.load_model(model, weight_path, strict=False)
        norm_stats_file = os.path.join(cfg.model_path, "norm_stats.json")
        if os.path.exists(norm_stats_file):
            model.norm_stats = model._read_normalization_stats(norm_stats_file)
        else:
            model.norm_stats = None

        model._train_expert_only = getattr(config, "train_expert_only", False)

    except Exception as e:
        logger.error(f"Failed to load pretrained model: {e}")
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
