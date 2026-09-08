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

"""Cosmos3 SFT model-config validation.

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
