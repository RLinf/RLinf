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

from __future__ import annotations

import torch
from diffusers import WanImageToVideoPipeline, WanPipeline
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model

from rlinf.config import torch_dtype_from_precision
from rlinf.models.generation.wan.base import Wan22Config, Wan22Model
from rlinf.models.generation.wan.t2v import Wan22T2VModel
from rlinf.models.generation.wan.ti2v import Wan22TI2VModel
from rlinf.models.generation.wan.vidar_weights import load_vidar_transformer_weights


def _build_pipeline(model_config: Wan22Config, inference_dtype: torch.dtype):
    if model_config.condition_mode == "ti2v":
        pipeline_cls = WanImageToVideoPipeline
    elif model_config.condition_mode == "t2v":
        pipeline_cls = WanPipeline
    else:
        raise ValueError(f"Unknown Wan22 condition_mode: {model_config.condition_mode}")
    pipeline = pipeline_cls.from_pretrained(
        model_config.model_path,
        torch_dtype=inference_dtype,
    )
    if model_config.weight_format == "vidar":
        load_vidar_transformer_weights(pipeline.transformer, model_config.vidar_path)
    return pipeline


def _prepare_pipeline(
    pipeline, model_config: Wan22Config, inference_dtype: torch.dtype
):
    pipeline.set_progress_bar_config(disable=True)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    if getattr(pipeline, "image_encoder", None) is not None:
        pipeline.image_encoder.requires_grad_(False)
    pipeline.transformer.requires_grad_(not model_config.use_lora)

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

    pipeline.vae.to(dtype=torch.float32)
    pipeline.text_encoder.to(dtype=inference_dtype)
    if getattr(pipeline, "image_encoder", None) is not None:
        pipeline.image_encoder.to(dtype=inference_dtype)
    pipeline.transformer.to(dtype=inference_dtype)
    return pipeline


def get_model(cfg: DictConfig, torch_dtype=None) -> Wan22Model:
    model_options = OmegaConf.to_container(cfg.wan22_ti2v_5b, resolve=True)
    model_config = Wan22Config(model_path=str(cfg.model_path), **model_options)
    inference_dtype = torch_dtype or torch_dtype_from_precision(cfg.precision)
    pipeline = _build_pipeline(model_config, inference_dtype)
    pipeline = _prepare_pipeline(pipeline, model_config, inference_dtype)
    if model_config.condition_mode == "ti2v":
        model_cls = Wan22TI2VModel
    else:
        model_cls = Wan22T2VModel
    return model_cls(model_config, pipeline=pipeline)


__all__ = [
    "Wan22Config",
    "Wan22Model",
    "Wan22T2VModel",
    "Wan22TI2VModel",
    "get_model",
]
