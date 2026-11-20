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

"""Configure and validate the Cosmos3 SFT model.

The Cosmos3 model is built from cosmos-framework's own experiment recipe
(``make_config`` + ``override(experiment=<SKU>)`` + ``instantiate``).

Model defaults (model_type, precision, load_to_device, ema_enabled,
disable_cosmos_parallelism, keys_to_skip_loading, keys_to_select,
lr_multipliers, action_normalization, ...) live in the Hydra config group
``examples/sft/config/model/cosmos3.yaml`` -- pull them in via
``defaults: - model/cosmos3@actor.model``. Recipe yamls override only
deployment-specific fields (``model_path``, ``wan_vae_path``,
``max_samples_per_batch``).

Dataset-recipe knobs (``image_size``, ``chunk_length``, ``action_space``,
``fps``, ...) are owned by the cosmos experiment recipe (resolved by cosmos);
rlinf does not read them from ``actor.model``, so they are deliberately NOT
duplicated here or in ``model/cosmos3.yaml``.
"""

import copy
import json
import os
from pathlib import Path

from omegaconf import DictConfig, open_dict

from rlinf.utils.logging import get_logger

logger = get_logger()


def validate_cosmos3_sft_model_cfg(model_cfg: DictConfig) -> DictConfig:
    """Validate a Cosmos3 SFT cfg."""

    with open_dict(model_cfg):
        model_path = model_cfg.get("model_path", None)
        assert model_path, "Cosmos3 SFT requires actor.model.model_path"

    logger.info(
        "Cosmos3 SFT model cfg validated: model_path=%s "
        "ema_enabled=%s disable_cosmos_parallelism=%s (keys_to_select=%d, "
        "lr_multipliers=%d, skip=%d)",
        model_path,
        model_cfg.get("ema_enabled"),
        model_cfg.get("disable_cosmos_parallelism"),
        len(model_cfg.get("keys_to_select", [])),
        len(model_cfg.get("lr_multipliers", {})),
        len(model_cfg.get("keys_to_skip_loading", [])),
    )
    return model_cfg


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
