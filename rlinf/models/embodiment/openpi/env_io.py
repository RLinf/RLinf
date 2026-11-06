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

"""Non-Module mixin: env worker dict ↔ Observation ↔ decoded env actions."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from rlinf.models.embodiment.openpi.modules.model import Observation
from rlinf.models.embodiment.openpi.transforms.apply import (
    apply_input_transform,
    apply_output_transform,
)
from rlinf.models.embodiment.openpi.transforms.env import repack_env_obs
from rlinf.models.embodiment.openpi.transforms.pipeline import (
    split_short_memory_transforms,
)


class EnvIO:
    """Shared env I/O for task subclasses that inherit :class:`Pi0`.

    Not an ``nn.Module`` — mix in *before* ``Pi0`` so ``super().__init__`` still
    reaches the network constructor.
    """

    _input_transform_fn = None
    _output_transform_fn = None

    def setup_transforms(
        self,
        transforms: Sequence,
        output_transforms: Sequence,
    ) -> None:
        """Install the openpi.transforms input/output pipelines."""
        from openpi.transforms import compose

        self._input_transform_fn = compose(transforms)
        self._output_transform_fn = compose(output_transforms)
        if self._config.short_memory:
            self._history_frame_transform, self._history_current_transform = (
                split_short_memory_transforms(transforms)
            )

    def _ensure_transforms(self) -> None:
        if self._input_transform_fn is None or self._output_transform_fn is None:
            raise RuntimeError(
                f"{type(self).__name__}.setup_transforms(...) must be called "
                "after construction (the factory does this); the openpi "
                "transforms pipeline is not yet installed."
            )

    def _select_configured_state(self, states):
        """Select a configured subset of the raw env state (openpi parity)."""
        indices = self.state_indices
        if not indices:
            return states

        if hasattr(states, "shape"):
            state_dim = states.shape[-1]
        else:
            state_dim = np.asarray(states).shape[-1]
        if state_dim == len(indices):
            return states
        if state_dim <= max(indices):
            raise ValueError(
                f"Cannot select state_indices={indices} from state dim {state_dim}."
            )

        if torch.is_tensor(states):
            index_tensor = torch.as_tensor(indices, device=states.device)
            return states.index_select(-1, index_tensor)
        return np.asarray(states)[..., indices]

    def input_transform(self, obs: dict, transpose: bool = False) -> dict:
        self._ensure_transforms()
        return apply_input_transform(self._input_transform_fn, obs, transpose=transpose)

    def output_transform(self, outputs: dict) -> dict:
        self._ensure_transforms()
        return apply_output_transform(
            self._output_transform_fn,
            outputs,
            action_chunk=self.action_chunk,
        )

    def decode_actions(
        self, model_actions: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """Run the output transform and return env-space actions on this device."""
        env_outputs = self.output_transform({"actions": model_actions, "state": state})
        return env_outputs["actions"].to(device=self.device, dtype=torch.float32)

    def _observation_dict_to_device(self, processed: dict) -> Observation:
        """Convert a per-key dict into a device-resident Observation."""
        device = self.device
        obs = Observation.from_dict(processed)

        def _move(x):
            return x.to(device) if isinstance(x, torch.Tensor) else x

        def _move_state(x):
            return (
                x.to(device=device, dtype=torch.float32)
                if isinstance(x, torch.Tensor)
                else x
            )

        return Observation(
            images={k: _move(v) for k, v in obs.images.items()},
            image_masks={k: _move(v) for k, v in obs.image_masks.items()},
            state=_move_state(obs.state),
            tokenized_prompt=_move(obs.tokenized_prompt),
            tokenized_prompt_mask=_move(obs.tokenized_prompt_mask),
            token_ar_mask=_move(obs.token_ar_mask),
            token_loss_mask=_move(obs.token_loss_mask),
            pcd_xyz=_move(obs.pcd_xyz),
            history_time_offsets=_move(obs.history_time_offsets),
        )

    def env_obs_to_observation(self, env_obs: dict[str, Any]) -> Observation:
        """Repack env obs through the transform pipeline onto this device."""
        self._ensure_transforms()
        if self._config.short_memory:
            return self._history_env_obs_to_observation(env_obs)
        repacked = repack_env_obs(
            self.config_name,
            env_obs,
            select_state=self._select_configured_state,
        )
        processed = self.input_transform(repacked, transpose=False)
        return self._observation_dict_to_device(processed)

    def _history_env_obs_to_observation(self, env_obs: dict[str, Any]) -> Observation:
        """Transform camera history framewise, keeping the current state and prompt."""
        main_images = env_obs["history_main_images"]
        batch_size, num_frames = main_images.shape[:2]
        history_obs = {
            "main_images": main_images.flatten(0, 1),
            "wrist_images": env_obs["history_wrist_images"].flatten(0, 1),
            "states": env_obs["states"].repeat_interleave(num_frames, dim=0),
            "task_descriptions": [
                prompt
                for prompt in env_obs["task_descriptions"]
                for _ in range(num_frames)
            ],
        }
        repacked = repack_env_obs(
            self.config_name,
            history_obs,
            select_state=self._select_configured_state,
        )
        processed = apply_input_transform(self._history_frame_transform, repacked)
        prompts = processed.pop("prompt")
        from torch.utils._pytree import tree_map

        current = tree_map(lambda value: value[num_frames - 1 :: num_frames], processed)
        current["prompt"] = prompts[num_frames - 1 :: num_frames]
        current = apply_input_transform(self._history_current_transform, current)
        frame_mask = env_obs["history_frame_mask"].bool()
        current["image"] = {
            key: value.reshape(batch_size, num_frames, *value.shape[1:])
            for key, value in processed["image"].items()
        }
        current["image_mask"] = {
            key: value.reshape(batch_size, num_frames).bool() & frame_mask
            for key, value in processed["image_mask"].items()
        }
        current["history_time_offsets"] = env_obs["history_time_offsets"]
        return self._observation_dict_to_device(current)

    def _predict_eval(
        self,
        observation: Observation,
        *,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Deterministic Euler ODE sampler shared by eval / RL eval / DAgger."""
        model_actions = self.sample_actions(
            observation, num_steps=self.num_steps, noise=noise, rng=rng
        )
        actions = self.decode_actions(model_actions, observation.state)
        B = actions.shape[0]
        result = {
            "prev_logprobs": None,
            "prev_values": None,
            "forward_inputs": {
                "action": actions.reshape(B, -1).contiguous(),
                "model_action": model_actions.reshape(B, -1).contiguous(),
            },
            "model_actions": model_actions,
        }
        return actions, result
