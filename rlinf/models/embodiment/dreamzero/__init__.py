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

"""DreamZero policy for RLinf embodied evaluation."""

import os

from omegaconf import DictConfig

from rlinf.models.embodiment.dreamzero.dreamzero_policy import (
    DreamZeroForRLActionPrediction,
)
from rlinf.utils.logging import get_logger

logger = get_logger()


def get_model(cfg: DictConfig, torch_dtype=None):
    """Load DreamZero policy from checkpoint.

    Expected cfg fields:
        - model_path: path to DreamZero checkpoint (with experiment_cfg/, config, etc.)
        - embodiment_tag: e.g. "oxe_droid" (default)
        - num_action_chunks: action chunk size (default 24)
        - action_dim: 8 for DROID (7 joints + gripper)
    """
    model_path = cfg.get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"DreamZero model_path does not exist: {model_path}. "
            "Please provide a valid checkpoint directory."
        )

    embodiment_tag = cfg.get("embodiment_tag", "oxe_droid")
    num_action_chunks = cfg.get("num_action_chunks", 24)
    action_dim = cfg.get("action_dim", 8)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    model = DreamZeroForRLActionPrediction(
        model_path=model_path,
        embodiment_tag=embodiment_tag,
        device=device,
        num_action_chunks=num_action_chunks,
        action_dim=action_dim,
    )

    return model
