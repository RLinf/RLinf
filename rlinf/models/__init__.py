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

from rlinf.config import SupportedModel, get_supported_model, torch_dtype_from_precision


def get_model(cfg: DictConfig):
    model_type = get_supported_model(cfg.model_type)
    if model_type == SupportedModel.OPENVLA:
        from rlinf.models.embodiment.openvla import get_model
    elif model_type == SupportedModel.OPENVLA_OFT:
        from rlinf.models.embodiment.openvla_oft import get_model
    elif model_type == SupportedModel.OPENPI:
        from rlinf.models.embodiment.openpi import get_model
    elif model_type == SupportedModel.DEXBOTIC_PI:
        from rlinf.models.embodiment.dexbotic_pi import get_model
    elif model_type == SupportedModel.MLP_POLICY:
        from rlinf.models.embodiment.mlp_policy import get_model
    elif model_type == SupportedModel.GR00T:
        from rlinf.models.embodiment.gr00t import get_model
    elif model_type == SupportedModel.CNN_POLICY:
        from rlinf.models.embodiment.cnn_policy import get_model
    elif model_type == SupportedModel.FLOW_POLICY:
        from rlinf.models.embodiment.flow_policy import get_model
    else:
        return None

    torch_dtype = torch_dtype_from_precision(cfg.precision)
    model = get_model(cfg, torch_dtype)

    if torch.cuda.is_available():
        model = model.cuda()

    if cfg.is_lora:
        from peft import LoraConfig, PeftModel, get_peft_model

        if not hasattr(cfg, "lora_path") or cfg.lora_path is None:
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_rank,
                lora_dropout=0.0,
                target_modules=[
                    "proj",
                    "qkv",
                    "fc1",
                    "fc2",  # vision
                    "q",
                    "kv",
                    "fc3",
                    "out_proj",  # project
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",  # llm
                ],
                init_lora_weights="gaussian",
            )
            if model_type == SupportedModel.OPENPI:
                module_to_lora = model.paligemma_with_expert.paligemma
                module_to_lora = get_peft_model(module_to_lora, lora_config)
                tag_vlm_subtree(model, False)
                tag_vlm_subtree(module_to_lora, True)
                model.paligemma_with_expert.paligemma = module_to_lora
            else:
                model = get_peft_model(model, lora_config)
        else:
            model = PeftModel.from_pretrained(model, cfg.lora_path, is_trainable=True)

        if hasattr(model, "value_head"):
            for param in model.value_head.parameters():
                param.requires_grad = True

    return model


def tag_vlm_subtree(model, is_vlm: bool):
    for n, m in model.named_modules():
        setattr(m, "_to_lora", is_vlm)
