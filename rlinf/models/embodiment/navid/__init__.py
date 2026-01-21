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

"""
NaVid embodied model integration.

This module provides `get_model(cfg, torch_dtype)` so that NaVid can be constructed via
`rlinf.models.get_model(...)` and used by `rlinf/workers/rollout/hf/huggingface_worker.py`.
"""

from __future__ import annotations

from omegaconf import DictConfig


def get_model(cfg: DictConfig, torch_dtype=None):
    """
    Build a NaVid model wrapper that exposes RLinf embodied policy interfaces:
      - preprocess_env_obs(env_obs)
      - predict_action_batch(env_obs=..., **kwargs) -> (chunk_actions, result_dict)
    """

    from rlinf.models.embodiment.navid.navid_action_model import (
        NaVidForRLActionPrediction,
    )

    model = NaVidForRLActionPrediction.from_pretrained(
        model_path=str(cfg.model_path),
        model_base=getattr(cfg, "model_base", None),
        torch_dtype=torch_dtype,
        action_dim=int(cfg.action_dim),
        num_action_chunks=int(cfg.num_action_chunks),
        max_new_tokens=int(getattr(cfg, "max_new_tokens", 32)),
        conversation_template=getattr(cfg, "conversation_template", "imgsp_v1"),
    )
    return model
