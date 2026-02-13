# Copyright 2025 The RLinf Authors.
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

import torch
from omegaconf import DictConfig


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    """
    Load and initialize the VLA model from checkpoint.

    Args:
        cfg: DictConfig configuration object

    Returns:
        torch.nn.Module: The initialized OpenVLAOFT model
    """
    print("Instantiating pretrained OpenVLAOFT policy...")

    from prismatic.extern.hf.processing_prismatic import (
        PrismaticImageProcessor,
        PrismaticProcessor,
    )
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForVision2Seq,
        AutoProcessor,
    )

    from rlinf.models.embodiment.openvla_oft.official.openvla_oft_action_model import (
        OpenVLAOFTForRLActionPrediction,
        OpenVLAOFTRLConfig,
    )
    from rlinf.models.embodiment.openvla_oft.openvla_utils import (
        apply_film_to_vla,
        load_dataset_stats,
    )

    AutoConfig.register("openvla", OpenVLAOFTRLConfig)
    AutoImageProcessor.register(OpenVLAOFTRLConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAOFTRLConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAOFTRLConfig, OpenVLAOFTForRLActionPrediction)

    # Load the config first
    model_config = AutoConfig.from_pretrained(
        cfg.model_path,
        trust_remote_code=True,
    )

    if not isinstance(model_config, OpenVLAOFTRLConfig):
        # Create a new OpenVLAOFTRLConfig from the loaded config
        config_dict = model_config.to_dict()
        model_config = OpenVLAOFTRLConfig(**config_dict)

    # Set custom parameters from cfg to config
    # These parameters are required for OpenVLAOFTRLConfig
    custom_params = {
        "action_dim": cfg.action_dim,
        "num_action_chunks": cfg.num_action_chunks,
        "add_value_head": cfg.add_value_head,
        "value_type": cfg.value_type,
        "proprio_dim": cfg.proprio_dim,
        "use_proprio": cfg.use_proprio,
        "use_film": cfg.use_film,
        "max_prompt_length": cfg.max_prompt_length,
        "unnorm_key": cfg.unnorm_key,
    }
    for key, val in custom_params.items():
        setattr(model_config, key, val)

    # Load the model with the updated config
    vla = OpenVLAOFTForRLActionPrediction.from_pretrained(
        cfg.model_path,
        config=model_config,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    if cfg.use_proprio:
        # Load proprio projector weights if available
        vla.load_proprio_projector_weights(cfg.model_path)

    # If using FiLM, wrap the vision backbone to allow for infusion of language inputs
    if cfg.use_film:
        vla = apply_film_to_vla(vla, cfg)

    # Set number of images in model input
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # Load dataset stats for action normalization
    norm_stats = load_dataset_stats(cfg)
    vla.norm_stats = norm_stats

    if vla.unnorm_key not in norm_stats and f"{vla.unnorm_key}_no_noops" in norm_stats:
        vla.unnorm_key = f"{vla.unnorm_key}_no_noops"
    assert vla.unnorm_key in norm_stats, (
        f"Action un-norm key {vla.unnorm_key} not found in VLA `norm_stats`!"
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(cfg.model_path)
    vla.set_processor(processor)

    vla.to(torch_dtype)

    return vla
