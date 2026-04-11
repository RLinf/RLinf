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

import json
import os

import torch
from omegaconf import DictConfig
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoProcessor,
    AutoTokenizer,
)


def get_model_config_and_input_processor(cfg: DictConfig):
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig

    from rlinf.models.embodiment.prismatic.processing_prismatic import (
        PrismaticImageProcessor,
        PrismaticProcessor,
    )

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)

    model_config = AutoConfig.from_pretrained(cfg.model_path)

    dataset_statistics_path = os.path.join(cfg.model_path, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            new_norm_stats = json.load(f)
            norm_stats = getattr(model_config, "norm_stats", {})
            norm_stats.update(new_norm_stats)
            setattr(model_config, "norm_stats", norm_stats)
    image_processor = PrismaticImageProcessor.from_pretrained(
        cfg.model_path, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path, trust_remote_code=True, padding_side="left"
    )
    input_processor = PrismaticProcessor.from_pretrained(
        cfg.model_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        trust_remote_code=True,
    )

    return model_config, input_processor


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig

    actor_model_config = OpenVLAConfig.from_pretrained(
        cfg.model_path, trust_remote_code=cfg.trust_remote_code
    )

    dataset_statistics_path = os.path.join(cfg.model_path, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            new_norm_stats = json.load(f)
            norm_stats = getattr(actor_model_config, "norm_stats", {})
            norm_stats.update(new_norm_stats)
            setattr(actor_model_config, "norm_stats", norm_stats)

    from rlinf.models.embodiment.openvla.openvla_action_model import (
        OpenVLAForRLActionPrediction,
    )

    model = OpenVLAForRLActionPrediction.from_pretrained(
        cfg.model_path,
        torch_dtype=torch_dtype,
        hidden_size=cfg.hidden_size,
        unnorm_key=cfg.unnorm_key,
        config=actor_model_config,
        add_value_head=cfg.add_value_head,
        max_prompt_length=cfg.max_prompt_length,
        action_dim=cfg.action_dim,
        num_action_chunks=cfg.num_action_chunks,
        attn_implementation=cfg.attn_implementation,
        low_cpu_mem_usage=cfg.low_cpu_mem_usage,
        trust_remote_code=cfg.trust_remote_code,
    )

    model.to(torch_dtype)

    model_config, input_processor = get_model_config_and_input_processor(cfg)
    model.setup_config_and_processor(model_config, input_processor)

    return model
