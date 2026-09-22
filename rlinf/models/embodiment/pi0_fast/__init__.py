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

"""PI0-Fast embodied policy wrapper for RLinf.

Exposes ``get_model``, which loads LeRobot's ``PI0FastPolicy`` from a checkpoint
directory and wraps it in ``PI0FastForRLActionPrediction``.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from rlinf.models.embodiment.pi0_fast.lerobot_io import (
    load_optional_processor,
    load_policy_config,
)
from rlinf.models.embodiment.pi0_fast.pi0_fast_action_model import (
    PI0FastForRLActionPrediction,
)


def get_model(
    cfg: DictConfig,
    torch_dtype: torch.dtype | None = None,
) -> PI0FastForRLActionPrediction:
    """Build an RLinf policy wrapper around LeRobot's PI0FastPolicy.

    Args:
        cfg: PI0-Fast model configuration.
        torch_dtype: Requested model dtype. The checkpoint's native mixed-dtype
            layout is preserved regardless of this value.

    Returns:
        The PI0-Fast policy adapted to RLinf's embodied policy interface.
    """
    # Mixed-dtype checkpoint; recasting changes greedy FAST tokens.
    del torch_dtype
    try:
        from lerobot.policies.pi0_fast import PI0FastPolicy
    except ImportError as exc:
        raise ImportError(
            "pi0_fast requires LeRobot with pi0_fast support. "
            "Install with: bash requirements/install.sh embodied "
            "--model pi0_fast --env libero"
        ) from exc
    model_cfg = cfg.get("pi0_fast", {})
    model_path = str(cfg.model_path)
    for field in ("num_action_chunks", "action_dim"):
        value = cfg.get(field, 0)
        if not value or int(value) <= 0:
            raise ValueError(f"pi0_fast requires model.{field} > 0, got {value!r}.")
    policy_config = load_policy_config(model_path, cfg)
    policy = PI0FastPolicy.from_pretrained(
        model_path,
        config=policy_config,
    )
    model = PI0FastForRLActionPrediction(
        policy,
        action_dim=int(cfg.action_dim),
        num_action_chunks=int(cfg.num_action_chunks),
        max_action_tokens=int(policy_config.max_action_tokens),
        image_size=model_cfg.get("image_size", None),
        temperature_train=float(model_cfg.get("temperature_train", 0.3)),
        temperature_eval=float(model_cfg.get("temperature_eval", 0.0)),
        preprocessor=load_optional_processor(
            model_path, "pre", model_cfg, policy_config
        ),
        postprocessor=load_optional_processor(model_path, "post", model_cfg),
    )
    return model


__all__ = ["PI0FastForRLActionPrediction", "get_model"]
