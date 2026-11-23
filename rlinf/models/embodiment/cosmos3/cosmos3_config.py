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

"""Configure Cosmos3 models for RLinf SFT and DAgger."""

import json
import os
from pathlib import Path

from omegaconf import DictConfig

from rlinf.utils.logging import get_logger

logger = get_logger()


def validate_cosmos3_sft_model_cfg(model_cfg: DictConfig) -> DictConfig:
    """Validate a Cosmos3 SFT cfg."""

    assert model_cfg.get("model_path"), "Cosmos3 SFT requires actor.model.model_path"
    return model_cfg


def _apply_checkpoint_model_config(cosmos_cfg, model_cfg: DictConfig) -> None:
    """Use checkpoint architecture and processor with the SFT recipe settings."""
    from cosmos_framework.inference.common.config import structure_config
    from cosmos_framework.inference.common.public_model_config import (
        load_model_config_from_hf_config,
    )

    model_path = Path(str(model_cfg.model_path)).expanduser()
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
            checkpoint_model.config[key] = getattr(recipe_config, key)
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


def _configure_sft_objective(model_config, cfg: DictConfig) -> None:
    """Select native loss weights without changing the Cosmos loss implementation."""
    objective = cfg.get("sft_objective", "joint")
    rf_config = model_config.rectified_flow_training_config
    if objective == "joint":
        rf_config.normalize_loss_by_active = False
        # Cosmos releases use different fields for the local loss reduction.
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
