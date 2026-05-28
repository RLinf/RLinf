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

"""XR0 VLA embodied policy wrapper for RLinf.

This module exposes ``get_model``, which instantiates the XR0 model and wraps
it into an ``XR0ForRLActionPrediction`` instance compatible with RLinf.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from rlinf.utils.logging import get_logger

from .utils import ACTION_DIM
from .xr0_action_model import XR0ForRLActionPrediction


class _StubXR0(nn.Module):
    """Lightweight stub that mimics the XR0 interface without downloading weights.

    Used when ``model_path`` is ``"dummy"`` so tests and CI can run without
    network access or large model downloads.
    """

    def __init__(self, action_shape=(30, 32), num_steps=5):
        super().__init__()
        self.action_shape = action_shape
        self.num_steps = num_steps
        # Minimal parameter so the module is not empty
        self._dummy = nn.Linear(1, 1)

    @torch.no_grad()
    def generate(self, batch: dict) -> torch.Tensor:
        """Return random action predictions matching the expected shape."""
        # Infer batch size from any tensor in the batch
        batch_size = 1
        for v in batch.values():
            if isinstance(v, torch.Tensor) and v.ndim > 0:
                batch_size = v.shape[0]
                break
        device = self._dummy.weight.device
        return torch.randn(
            (batch_size, *self.action_shape),
            device=device,
            dtype=torch.bfloat16,
        )


def get_model(
    cfg: DictConfig,
    torch_dtype: torch.dtype | None = None,
) -> XR0ForRLActionPrediction:
    """Instantiate the XR0 model and wrap it for RLinf.

    When ``cfg.model_path`` is ``"dummy"``, a lightweight stub is used instead
    of the full XR0 model (no HuggingFace download required).

    Args:
        cfg: Model config.  Expected keys include ``action_dim``,
            ``num_action_chunks``, ``num_steps``, and XR0-specific params
            under ``cfg.xr0``.
        torch_dtype: Optional torch dtype for the model.

    Returns:
        An ``XR0ForRLActionPrediction`` instance.
    """
    logger = get_logger()

    action_dim = getattr(cfg, "action_dim", ACTION_DIM)
    num_action_chunks = getattr(cfg, "num_action_chunks", 30)
    num_steps = getattr(cfg, "num_steps", 5)
    noise_level = getattr(cfg, "noise_level", 0.5)

    # XR0-specific config (with defaults matching the original XR0 config)
    xr0_cfg = getattr(cfg, "xr0", cfg)
    action_shape = tuple(
        getattr(xr0_cfg, "action_shape", [num_action_chunks, action_dim])
    )

    model_path = getattr(cfg, "model_path", None)

    if model_path == "dummy":
        logger.info("Using stub XR0 model (model_path=dummy)")
        xr0_model = _StubXR0(action_shape=action_shape, num_steps=num_steps)
    else:
        from transformers import AutoModel

        logger.info("Loading XR0 model from %s", model_path)
        xr0_model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch_dtype or torch.bfloat16,
        )

    # Load action normalization stats (optional)
    action_mean, action_std = None, None
    stats_path = getattr(cfg, "stats_path", None) or getattr(
        xr0_cfg, "stats_path", None
    )
    if stats_path:
        import yaml

        logger.info("Loading action normalization stats from %s", stats_path)
        with open(stats_path) as f:
            stats = yaml.safe_load(f)
        action_mean = np.array(stats["mean"], dtype=np.float32)
        action_std = np.array(stats["std"], dtype=np.float32)

    # TODO: When add_value_head=True, initialize ValueHead and attach
    # to policy. See lingbotvla __init__.py for reference pattern.
    # from rlinf.models.embodiment.modules.value_head import ValueHead
    # if getattr(cfg, "add_value_head", False):
    #     policy.value_head = ValueHead(input_dim=dit_hidden_size, ...)

    policy = XR0ForRLActionPrediction(
        xr0_model=xr0_model,
        action_dim=action_dim,
        num_action_chunks=num_action_chunks,
        num_steps=num_steps,
        action_mean=action_mean,
        action_std=action_std,
        noise_level=noise_level,
    )

    return policy


__all__ = ["XR0ForRLActionPrediction", "get_model"]
