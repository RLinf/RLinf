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

from __future__ import annotations

from typing import Any

import torch

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.openpi_pytorch.openpi_action_model import (
    OpenPiPytorchActionModel,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0 import Pi0


class OpenPiPytorchSFTActionModel(OpenPiPytorchActionModel):
    """SFT variant of :class:`OpenPiPytorchActionModel` (flow-matching loss)."""

    def __init__(
        self,
        pi0_model: Pi0,
        *,
        num_steps: int,
        action_env_dim: int,
    ):
        super().__init__(
            pi0_model,
            num_steps=num_steps,
            action_env_dim=action_env_dim,
        )

    def forward(self, forward_type: ForwardType = ForwardType.SFT, **kwargs):
        """Dispatch — SFT variant only supports :attr:`ForwardType.SFT`."""
        if forward_type != ForwardType.SFT:
            raise NotImplementedError(
                f"{type(self).__name__} only supports ForwardType.SFT; "
                f"got forward_type={forward_type!r}. "
                "Use the RL subclass (actor.model.openpi.task='rl') for PPO."
            )
        return self.sft_forward(**kwargs)

    def sft_forward(self, data: Any) -> torch.Tensor:
        """Compute the flow-matching SFT loss for one batch.

        ``data`` is either a ``(observation, actions)`` tuple or a dict with
        ``observation`` and ``actions`` keys. The data loader has already run
        the openpi transform pipeline, so ``actions`` arrive normalised and
        padded to the model action dim. Returns the scalar mean of the
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
        raise ValueError(
            "SFT actions must arrive normalized + padded to the model action "
            f"dim {model_action_dim} (the BEHAVIOR SFT data loader applies the "
            f"openpi transform pipeline before collation); got last dim "
            f"{actions.shape[-1]}."
        )
