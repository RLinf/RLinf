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

import os

import torch
from omegaconf import DictConfig


def get_model(cfg: DictConfig, torch_dtype=None):
    """Factory function to instantiate the LingbotVLA Action Model for RLinf."""

    from rlinf.models.embodiment.lingbotvla.lingbotvla_action_model import (
        LingbotvlaActionModel,
    )

    if torch_dtype is None:
        torch_dtype = torch.bfloat16

    model = LingbotvlaActionModel(cfg, torch_dtype=torch_dtype)

    checkpoint_dir = str(cfg.model_path)

    full_weights_path = os.path.join(
        checkpoint_dir, "model_state_dict", "full_weights.pt"
    )
    actor_full_weights_path = os.path.join(
        checkpoint_dir, "actor", "model_state_dict", "full_weights.pt"
    )

    if os.path.exists(full_weights_path):
        print(
            f"[LingbotVLA] Loading RLinf FSDP specific weights from {full_weights_path}"
        )
        model_state_dict = torch.load(full_weights_path, map_location="cpu")
        model.load_state_dict(model_state_dict, strict=False)
    elif os.path.exists(actor_full_weights_path):
        print(
            f"[LingbotVLA] Loading RLinf FSDP actor weights from {actor_full_weights_path}"
        )
        model_state_dict = torch.load(actor_full_weights_path, map_location="cpu")
        model.load_state_dict(model_state_dict, strict=False)

    lingbotvla_cfg = getattr(cfg, "lingbotvla", getattr(cfg, "lingbot", cfg))
    train_expert_only = getattr(lingbotvla_cfg, "train_expert_only", False)
    if train_expert_only:
        print(
            "[LingbotVLA] train_expert_only is True, freezing VLM (PaliGemma/Qwen) backbone..."
        )
        model.vla_model.model.qwenvl_with_expert.qwenvl.eval()
        for param in model.vla_model.model.qwenvl_with_expert.qwenvl.parameters():
            param.requires_grad = False

    return model
