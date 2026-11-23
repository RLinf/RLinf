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

from dataclasses import dataclass
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


@dataclass(frozen=True)
class EnvIOSpec:
    """Per-env OpenPI transform profile registered under ``model.envs``."""

    config_name: str
    action_env_dim: int


class EnvIO:
    """Shared env I/O for task subclasses that inherit :class:`Pi0`.

    Not an ``nn.Module`` — mix in *before* ``Pi0`` so ``super().__init__`` still
    reaches the network constructor.
    """

    _input_transform_fn = None
    _output_transform_fn = None
    _io_pipelines: dict[str, tuple[Any, Any]] | None = None
    _env_io_specs: dict[str, EnvIOSpec] | None = None

    def setup_transforms(
        self,
        transforms: Sequence,
        output_transforms: Sequence,
        *,
        env_name: str | None = None,
    ) -> None:
        """Install one openpi.transforms input/output pipeline.

        ``env_name`` selects a per-env profile from ``model.envs``. The unnamed
        call installs the default pipeline of ``config_name`` used when a batch
        does not name a per-env profile.
        """
        from openpi.transforms import compose

        pipeline = (compose(transforms), compose(output_transforms))
        if self._io_pipelines is None:
            self._io_pipelines = {}
        key = self.config_name if env_name is None else env_name
        self._io_pipelines[key] = pipeline
        if self._input_transform_fn is None or env_name is None:
            self._input_transform_fn, self._output_transform_fn = pipeline
        if env_name is None and self._config.short_memory:
            self._frame_transform, self._sample_transform = (
                split_short_memory_transforms(transforms)
            )

    def register_env_io(
        self,
        env_name: str,
        *,
        config_name: str,
        action_env_dim: int,
        transforms: Sequence,
        output_transforms: Sequence,
    ) -> None:
        """Install the transforms one env uses on this shared network."""
        self.setup_transforms(transforms, output_transforms, env_name=env_name)
        if self._env_io_specs is None:
            self._env_io_specs = {}
        self._env_io_specs[env_name] = EnvIOSpec(
            config_name=config_name,
            action_env_dim=int(action_env_dim),
        )

    def _io_pipeline(self, env_name: str | None):
        """Return the input and output transforms for ``env_name`` or the default."""
        self._ensure_transforms()
        if not env_name:
            return self._input_transform_fn, self._output_transform_fn
        pipelines = self._io_pipelines or {}
        if env_name not in pipelines:
            raise KeyError(
                f"OpenPI env I/O {env_name!r} is not installed. Known pipelines: "
                f"{sorted(pipelines)}."
            )
        return pipelines[env_name]

    def _env_io(self, env_name: str | None) -> EnvIOSpec | None:
        """Return the registered spec of ``env_name``, or None for the default I/O."""
        if not env_name:
            return None
        specs = self._env_io_specs or {}
        if env_name not in specs:
            raise KeyError(
                f"OpenPI env I/O {env_name!r} is not registered. Known envs: "
                f"{sorted(specs)}."
            )
        return specs[env_name]

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

    def input_transform(
        self, obs: dict, transpose: bool = False, *, env_name: str | None = None
    ) -> dict:
        self._ensure_transforms()
        transform, _ = self._io_pipeline(env_name)
        return apply_input_transform(transform, obs, transpose=transpose)

    def output_transform(self, outputs: dict, *, env_name: str | None = None) -> dict:
        self._ensure_transforms()
        _, transform = self._io_pipeline(env_name)
        return apply_output_transform(
            transform,
            outputs,
            action_chunk=self.action_chunk,
        )

    def decode_actions(
        self,
        model_actions: torch.Tensor,
        state: torch.Tensor,
        *,
        env_name: str | None = None,
    ) -> torch.Tensor:
        """Run the output transform and return env-space actions on this device."""
        env_outputs = self.output_transform(
            {"actions": model_actions, "state": state}, env_name=env_name
        )
        return env_outputs["actions"].to(device=self.device, dtype=torch.float32)

    def _observation_dict_to_device(self, processed: dict) -> Observation:
        """Convert a per-key dict into a device-resident Observation."""
        device = self.device
        if self._config.pytorch_numerics:
            from torch.utils._pytree import tree_map

            processed = tree_map(
                lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, processed
            )
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

    def env_obs_to_observation(
        self, env_obs: dict[str, Any], *, env_name: str | None = None
    ) -> Observation:
        """Repack env obs through the transform pipeline onto this device."""
        self._ensure_transforms()
        if self._config.short_memory:
            return self._history_to_observation(env_obs)
        io_spec = self._env_io(env_name)
        repack_name = self.config_name if io_spec is None else io_spec.config_name
        repacked = repack_env_obs(
            repack_name,
            env_obs,
            select_state=self._select_configured_state,
        )
        processed = self.input_transform(repacked, transpose=False, env_name=env_name)
        return self._observation_dict_to_device(processed)

    def _history_to_observation(self, env_obs: dict[str, Any]) -> Observation:
        """Transform camera history framewise, keeping the current state and prompt."""
        history = env_obs["history"]
        main_images = history["main_images"]
        batch_size, num_frames = main_images.shape[:2]
        history_obs = {
            "main_images": main_images.flatten(0, 1),
            "wrist_images": history["wrist_images"].flatten(0, 1),
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
        processed = apply_input_transform(self._frame_transform, repacked)
        prompts = processed.pop("prompt")
        from torch.utils._pytree import tree_map

        current = tree_map(lambda value: value[num_frames - 1 :: num_frames], processed)
        current["prompt"] = prompts[num_frames - 1 :: num_frames]
        current = apply_input_transform(self._sample_transform, current)
        frame_mask = history["frame_mask"].bool()
        current["image"] = {
            key: value.reshape(batch_size, num_frames, *value.shape[1:])
            for key, value in processed["image"].items()
        }
        current["image_mask"] = {
            key: value.reshape(batch_size, num_frames).bool() & frame_mask
            for key, value in processed["image_mask"].items()
        }
        current["history_time_offsets"] = history["time_offsets"]
        return self._observation_dict_to_device(current)

    def _predict_eval(
        self,
        observation: Observation,
        *,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
        env_name: str | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Deterministic Euler ODE sampler shared by eval / RL eval / DAgger."""
        model_actions = self.sample_actions(
            observation, num_steps=self.num_steps, noise=noise, rng=rng
        )
        actions = self.decode_actions(
            model_actions, observation.state, env_name=env_name
        )
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
