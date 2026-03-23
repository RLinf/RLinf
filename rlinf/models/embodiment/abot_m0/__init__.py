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
    from pathlib import Path

    from rlinf.models.embodiment.abot_m0.abot_m0_action_model import (
        ABotM0ForRLActionPrediction,
    )

    model_path = Path(cfg.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    model = ABotM0ForRLActionPrediction.from_pretrained(
        pretrained_checkpoint=str(model_path),
        rl_head_config=cfg.rl_head_config,
        image_size=cfg.image_size,
        num_action_chunks=cfg.num_action_chunks,
        denoising_steps=cfg.denoising_steps,
        qwen_max_length=cfg.get("qwen_max_length", 256),
    )
    model.to(torch_dtype)

    if cfg.rl_head_config.add_value_head:
        model.action_head_rl.value_head._init_weights()

    return model
