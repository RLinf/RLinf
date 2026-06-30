# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entry point for the self-contained PyTorch OpenPI 0.5 BEHAVIOR model.

This model focuses on exactly two responsibilities: eval action sampling and
SFT flow-matching loss.
Its high-level interface mirrors the old ``OpenPi0ForRLActionPrediction`` so the eval
rollout worker can call it unchanged via the OpenPI dispatch path:

    actions, result = model.predict_action_batch(env_obs=env_obs, mode="eval")

``actions`` has shape ``[B, action_chunk, action_env_dim]`` (e.g. ``[B, 32, 23]``).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn

from rlinf.data.datasets.openpi_pytorch.eval_processor import EvalProcessor
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0 import Pi0
from rlinf.models.embodiment.openpi_pytorch.utils.normalize import (
    normalize_quantile,
)


class OpenPiPytorchActionModel(nn.Module):
    """Wrap the vendored ``Pi0`` model with BEHAVIOR eval pre/post-processing."""

    def __init__(
        self,
        pi0_model: Pi0,
        processor: EvalProcessor | None,
        *,
        num_steps: int,
        action_chunk: int,
        action_env_dim: int,
    ):
        super().__init__()
        self.model = pi0_model
        self.processor = processor
        self.num_steps = num_steps
        self.action_chunk = action_chunk
        self.action_env_dim = action_env_dim

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        compute_values: bool = False,
        *,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample env actions for a batch of observations (eval / action generation).

        When ``noise`` (shape ``[B, action_horizon, model_action_dim]``) and/or a
        ``rng`` generator are provided, they are passed verbatim to the
        flow-matching sampler so the same observation yields the same actions —
        used for deterministic / paired evaluation. The default (both ``None``)
        preserves the stochastic production sampling unchanged.
        """
        if self.processor is None:
            raise RuntimeError(
                "predict_action_batch requires an eval processor; "
                "the current model was built for SFT training only."
            )
        observation = self.processor.build_observation(env_obs, self.device)
        model_actions = self.model.sample_actions(
            observation, num_steps=self.num_steps, noise=noise, rng=rng
        )
        actions = self.processor.postprocess_actions(model_actions).to(self.device)

        batch = actions.shape[0]
        result = {
            "prev_logprobs": None,
            "prev_values": None,
            "forward_inputs": {
                "action": actions.reshape(batch, -1).contiguous(),
                "model_action": model_actions.reshape(batch, -1).contiguous(),
            },
        }
        return actions, result

    # --- SFT training (BEHAVIOR supervised fine-tuning) ---
    def forward(self, forward_type: ForwardType = ForwardType.SFT, **kwargs):
        """Dispatch a training forward pass.

        Eval/action-generation goes through :meth:`predict_action_batch`; the
        SFT runner calls this with ``forward_type=ForwardType.SFT, data=batch``.
        """
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        raise NotImplementedError(
            "OpenPiPytorchActionModel supports eval (predict_action_batch) and "
            f"SFT (ForwardType.SFT); got forward_type={forward_type!r}."
        )

    def sft_forward(self, data: Any) -> torch.Tensor:
        """Compute the flow-matching SFT loss for one batch.

        ``data`` is either a ``(observation, actions)`` tuple or a dict with
        ``observation`` and ``actions`` (the dataloader already normalizes and
        pads actions to the model action dim). Returns the scalar mean of the
        ``(B, action_horizon)`` per-timestep loss from :meth:`Pi0.compute_loss`
        (which samples the flow-matching noise/time internally).
        """
        observation, actions = self._unpack_sft_batch(data)
        observation = self._observation_to_device(observation)
        actions = self._actions_to_device(actions)
        per_timestep_loss = self.model.compute_loss(observation, actions, train=True)
        return per_timestep_loss.mean()

    def compute_loss(self, data: Any) -> torch.Tensor:
        """Alias kept for interface parity with the old action model."""
        return self.sft_forward(data)

    @staticmethod
    def _unpack_sft_batch(data: Any) -> tuple[Any, Any]:
        if isinstance(data, (tuple, list)):
            if len(data) != 2:
                raise ValueError(
                    "SFT batch tuple must be (observation, actions); "
                    f"got length {len(data)}."
                )
            observation, actions = data
        elif isinstance(data, dict):
            if "observation" not in data or "actions" not in data:
                raise ValueError(
                    "SFT batch dict must contain 'observation' and 'actions'; "
                    f"got keys {sorted(data)}."
                )
            observation, actions = data["observation"], data["actions"]
        else:
            raise TypeError(f"Unsupported SFT batch type: {type(data)!r}.")
        if observation is None or actions is None:
            raise ValueError("SFT batch is missing observation or actions.")
        return observation, actions

    def _observation_to_device(self, observation: Any) -> Observation:
        if isinstance(observation, dict):
            observation = Observation.from_dict(observation)
        if not isinstance(observation, Observation):
            raise TypeError(
                f"SFT observation must be an Observation or dict; "
                f"got {type(observation)!r}."
            )
        device = self.device

        def _move(x):
            return x.to(device) if isinstance(x, torch.Tensor) else x

        return Observation(
            images={k: _move(v) for k, v in observation.images.items()},
            image_masks={k: _move(v) for k, v in observation.image_masks.items()},
            state=_move(observation.state),
            tokenized_prompt=_move(observation.tokenized_prompt),
            tokenized_prompt_mask=_move(observation.tokenized_prompt_mask),
            token_ar_mask=_move(observation.token_ar_mask),
            token_loss_mask=_move(observation.token_loss_mask),
            pcd_xyz=_move(observation.pcd_xyz),
        )

    def _actions_to_device(self, actions: Any) -> torch.Tensor:
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        model_action_dim = self.model.action_dim
        if actions.dim() != 3:
            raise ValueError(
                "SFT actions must have shape [B, action_horizon, D]; "
                f"got {tuple(actions.shape)}."
            )
        if actions.shape[-1] == model_action_dim:
            return actions.to(device=self.device, dtype=torch.float32)
        if actions.shape[-1] == self.action_env_dim:
            return self._normalize_and_pad_env_actions(actions)
        raise ValueError(
            "SFT actions must have shape [B, action_horizon, "
            f"{model_action_dim}] (normalized + padded) or [B, action_horizon, "
            f"{self.action_env_dim}] (raw env actions); got {tuple(actions.shape)}."
        )

    def _normalize_and_pad_env_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.processor is None:
            raise ValueError(
                "SFT actions were provided at env action dim, but this model "
                "has no action normalization stats. Supply normalized/padded "
                "actions from the dataloader or build the model with a processor."
            )
        actions_np = actions.detach().cpu().numpy()
        normalized = normalize_quantile(
            actions_np.astype(np.float32), self.processor.action_stats
        )
        pad_width = [(0, 0)] * normalized.ndim
        pad_width[-1] = (0, self.model.action_dim - normalized.shape[-1])
        padded = np.pad(normalized, pad_width, constant_values=0.0)
        return torch.as_tensor(padded, device=self.device, dtype=torch.float32)

    # --- Gradient checkpointing pass-through (used by the FSDP training path) ---
    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict | None = None, **kwargs
    ) -> None:
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self, **kwargs) -> None:
        self.model.gradient_checkpointing_disable()
