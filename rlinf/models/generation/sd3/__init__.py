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

from diffusers import StableDiffusion3Pipeline
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model

from rlinf.config import torch_dtype_from_precision
from rlinf.models.generation.sd3.stable_diffusion3 import (
    StableDiffusion3,
    StableDiffusion3Config,
)
from rlinf.models.generation.sd3.utils import (
    configure_pipeline_trainability,
    move_pipeline_modules,
)


def get_model(cfg: DictConfig, torch_dtype=None):
    model_config = StableDiffusion3Config(model_path=str(cfg.model_path))
    model_config.update_from_dict(
        OmegaConf.to_container(cfg.get("sd3", {}), resolve=True) or {}
    )
    inference_dtype = torch_dtype or torch_dtype_from_precision(cfg.precision)
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_config.model_path,
        torch_dtype=inference_dtype,
    )
    pipeline.safety_checker = None
    configure_pipeline_trainability(
        pipeline,
        train_transformer=not model_config.use_lora,
    )

    if model_config.use_lora:
        if model_config.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(
                pipeline.transformer,
                str(model_config.lora_path),
                is_trainable=True,
            )
            pipeline.transformer.set_adapter("default")
        else:
            lora_config = LoraConfig(
                r=model_config.lora_rank,
                lora_alpha=model_config.lora_alpha,
                init_lora_weights=model_config.init_lora_weights,
                target_modules=model_config.target_modules,
            )
            pipeline.transformer = get_peft_model(pipeline.transformer, lora_config)

    move_pipeline_modules(pipeline, inference_dtype=inference_dtype)

    return StableDiffusion3(model_config, pipeline=pipeline)


__all__ = ["StableDiffusion3", "StableDiffusion3Config", "get_model"]
