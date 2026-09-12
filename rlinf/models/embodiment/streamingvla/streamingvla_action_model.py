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

"""RLinf SFT wrapper for the training-only StreamingVLA model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType

from .streamingvla_pytorch import StreamingVLAPytorch


class StreamingVLAForSFTActionPrediction(nn.Module, BasePolicy):
    """Expose StreamingVLA through RLinf's existing VLA SFT contract."""

    @property
    def _no_split_modules(self) -> list[str]:
        """Return transformer block classes used by FSDP auto-wrapping."""
        return ["Block", "Encoder1DBlock"]

    @property
    def _no_split_names(self) -> list[str]:
        """Return leaf projections that need their own FSDP hooks."""
        return [
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
        ]

    def __init__(
        self,
        config: Any,
        *,
        sigma: float = 0.16,
        noise_decay: float = 4.0,
        require_action_states: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.svla_model = StreamingVLAPytorch(
            config,
            sigma=sigma,
            noise_decay=noise_decay,
            require_action_states=require_action_states,
        )
        self.global_step = 0
        for name, module in self.named_modules():
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

    def set_global_step(self, global_step: int) -> None:
        """Record the optimizer step for the standard RLinf worker interface."""
        self.global_step = int(global_step)

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict[str, Any] | None = None, **_: Any
    ) -> None:
        """Enable activation checkpointing in the StreamingVLA core."""
        self.svla_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self, **_: Any) -> None:
        """Disable activation checkpointing in the StreamingVLA core."""
        self.svla_model.gradient_checkpointing_disable()

    def is_gradient_checkpointing_enabled(self) -> bool:
        """Return whether activation checkpointing is enabled."""
        return self.svla_model.is_gradient_checkpointing_enabled()

    def forward(
        self, forward_type: ForwardType = ForwardType.SFT, **kwargs: Any
    ) -> torch.Tensor:
        """Dispatch only the SFT training forward."""
        if forward_type != ForwardType.SFT:
            raise NotImplementedError(
                "StreamingVLA currently supports SFT training only; "
                f"got forward_type={forward_type!r}."
            )
        return self.sft_forward(**kwargs)

    def sft_forward(
        self,
        data: Any,
        *,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        """Compute the scalar SFP loss for one transformed training batch."""
        if isinstance(data, dict):
            observation = data["observation"]
            actions = data["actions"]
        elif isinstance(data, (tuple, list)) and len(data) == 2:
            observation, actions = data
        else:
            raise TypeError(
                "StreamingVLA SFT data must be an (observation, actions) pair "
                "or a dict with those keys."
            )

        device = next(self.parameters()).device
        observation = observation.to(device)
        actions = torch.as_tensor(actions, device=device, dtype=torch.float32)
        losses = self.svla_model(
            observation,
            actions,
            noise=noise,
            time=time,
        )
        return losses.mean()

    def default_forward(self, **_: Any) -> Any:
        """Reject rollout/RL calls because this integration is training-only."""
        raise NotImplementedError("StreamingVLA rollout and RL are not implemented.")

    def predict_action_batch(self, **_: Any) -> Any:
        """Reject inference calls because this integration is training-only."""
        raise NotImplementedError("StreamingVLA inference is not implemented.")
