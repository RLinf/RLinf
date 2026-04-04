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
# WITHOUT WARRANTIES OR CONDITIONS FOR ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DreamZero model entry point for RLinf.

Called by rlinf/models/__init__.py when model_type == "dreamzero".
Constructs a DreamZeroPolicy wrapping the groot VLA model.
"""

import os

from omegaconf import DictConfig

from rlinf.models.embodiment.dreamzero.dreamzero_policy import DreamZeroPolicy


def get_model(cfg: DictConfig, torch_dtype=None):
    """Build DreamZeroPolicy from Hydra config.

    Args:
        cfg: actor.model config block with fields:
            model_path      - checkpoint dir containing config.json and optional safetensors
            tokenizer_path  - local path or HF hub ID for umt5-xxl tokenizer
            max_seq_len     - T5 token sequence length (default 512)
            is_lora         - if False, train_architecture="full" (all params trainable)
            cpu_init        - if True, build on CPU; FSDP shards to GPUs later (avoids OOM)
        torch_dtype: unused, kept for interface compatibility.
    """
    model_path = cfg.get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"DreamZero model_path does not exist: {model_path}. "
            "Please provide a valid checkpoint directory."
        )

    tokenizer_path = cfg.get("tokenizer_path", "google/umt5-xxl")
    max_seq_len = cfg.get("max_seq_len", 512)
    cpu_init = cfg.get("cpu_init", True)
    device = "cpu" if cpu_init else (
        "cuda" if __import__("torch").cuda.is_available() else "cpu"
    )

    train_architecture = None
    if not cfg.get("is_lora", False):
        train_architecture = "full"

    action_loss_scale = float(cfg.get("action_loss_scale", 1.0))

    pretrained_model_path = cfg.get("pretrained_model_path", None)
    if pretrained_model_path in ("null", "None", ""):
        pretrained_model_path = None

    model = DreamZeroPolicy(
        model_path=model_path,
        device=device,
        eval_bf16=False,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        cpu_init=cpu_init,
        train_architecture=train_architecture,
        action_loss_scale=action_loss_scale,
        pretrained_model_path=pretrained_model_path,
    )

    return model
