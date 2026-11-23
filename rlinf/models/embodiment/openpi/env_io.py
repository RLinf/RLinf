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
        *,
        name: str | None = None,
    ) -> None:
        """Install one openpi.transforms input/output pipeline.

        ``name`` is an env I/O key (``actor.model.openpi.envs``). The unnamed
        call installs the pipeline of ``config_name`` and is the one used when
        a batch does not name an env.
        """
        from openpi.transforms import compose

        pipeline = (compose(transforms), compose(output_transforms))
        pipelines = getattr(self, "_io_pipelines", None)
        if pipelines is None:
            pipelines = {}
            self._io_pipelines = pipelines
        key = self.config_name if name is None else name
        pipelines[key] = pipeline
        if self._input_transform_fn is None or name is None:
            self._input_transform_fn, self._output_transform_fn = pipeline
        if name is None and self._config.short_memory:
            self._frame_transform, self._sample_transform = (
                split_short_memory_transforms(transforms)
            )

    def register_env_io(
        self,
        name: str,
        *,
        config_name: str,
        action_env_dim: int,
        transforms: Sequence,
        output_transforms: Sequence,
    ) -> None:
        """Install the transforms one env uses on this shared network."""
        self.setup_transforms(transforms, output_transforms, name=name)
        specs = getattr(self, "_env_io_specs", None)
        if specs is None:
            specs = {}
            self._env_io_specs = specs
        specs[name] = {
            "config_name": config_name,
            "action_env_dim": int(action_env_dim),
        }
        self._env_io_order = list(specs)

    def _io_pipeline(self, name: str | None):
        """Return the input and output transforms for ``name`` or the default."""
        self._ensure_transforms()
        if not name:
            return self._input_transform_fn, self._output_transform_fn
        pipelines = getattr(self, "_io_pipelines", {})
        try:
            return pipelines[name]
        except KeyError as exc:
            raise KeyError(
                f"OpenPI I/O {name!r} is not installed. Known pipelines: "
                f"{sorted(pipelines)}."
            ) from exc

    def _env_io(self, name: str | None) -> dict[str, Any] | None:
        """Return the registered spec of ``name``, or None for the default I/O."""
        if not name:
            return None
        specs = getattr(self, "_env_io_specs", {})
        try:
            return specs[name]
        except KeyError as exc:
            raise KeyError(
                f"OpenPI I/O {name!r} is not registered. Known envs: "
                f"{list(getattr(self, '_env_io_order', []))}."
            ) from exc

    def _resolve_stored_io(
        self, forward_inputs: dict[str, Any]
    ) -> tuple[str | None, int | None]:
        """Read the env I/O stamped on a training batch.

        Returns the env name and its action dim, or ``(None, None)`` when the
        batch used the default pipeline.
        """
        io_ids = forward_inputs.get("openpi_io_id")
        if io_ids is None:
            return None, None
        if not torch.is_tensor(io_ids):
            raise TypeError(
                f"openpi_io_id must be a tensor, got {type(io_ids).__name__}."
            )
        unique = torch.unique(io_ids.detach().reshape(-1))
        if unique.numel() != 1:
            raise ValueError(
                f"One OpenPI forward must use one env I/O, got ids {unique.tolist()}."
            )
        order = getattr(self, "_env_io_order", [])
        index = int(unique.item())
        if index < 0 or index >= len(order):
            raise ValueError(
                f"openpi_io_id {index} is outside the registered envs {order}."
            )
        name = order[index]
        return name, int(self._env_io(name)["action_env_dim"])

    def _stamp_openpi_io(
        self, forward_inputs: dict[str, Any], io_name: str | None, batch_size: int
    ) -> None:
        """Record which env I/O produced ``forward_inputs`` so training matches."""
        if not io_name:
            return
        order = getattr(self, "_env_io_order", [])
        forward_inputs["openpi_io_id"] = torch.full(
            (batch_size,), order.index(io_name), dtype=torch.long
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

    def input_transform(
        self, obs: dict, transpose: bool = False, *, io_name: str | None = None
    ) -> dict:
        self._ensure_transforms()
        transform, _ = self._io_pipeline(io_name)
        return apply_input_transform(transform, obs, transpose=transpose)

    def output_transform(self, outputs: dict, *, io_name: str | None = None) -> dict:
        self._ensure_transforms()
        _, transform = self._io_pipeline(io_name)
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
        io_name: str | None = None,
    ) -> torch.Tensor:
        """Run the output transform and return env-space actions on this device."""
        env_outputs = self.output_transform(
            {"actions": model_actions, "state": state}, io_name=io_name
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

    def env_obs_to_observation(self, env_obs: dict[str, Any]) -> Observation:
        """Repack env obs through the transform pipeline onto this device."""
        self._ensure_transforms()
        if self._config.short_memory:
            return self._history_to_observation(env_obs)
        repacked = repack_env_obs(
            self.config_name,
            env_obs,
            select_state=self._select_configured_state,
        )
        processed = self.input_transform(repacked, transpose=False)
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
        io_name: str | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Deterministic Euler ODE sampler shared by eval / RL eval / DAgger."""
        model_actions = self.sample_actions(
            observation, num_steps=self.num_steps, noise=noise, rng=rng
        )
        actions = self.decode_actions(model_actions, observation.state, io_name=io_name)
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
