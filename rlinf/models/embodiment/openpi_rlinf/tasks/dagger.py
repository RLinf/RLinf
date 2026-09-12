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

from typing import Any, Literal, Sequence

import numpy as np
import torch

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.openpi_rlinf.env_io import EnvIO
from rlinf.models.embodiment.openpi_rlinf.modules.model import Observation
from rlinf.models.embodiment.openpi_rlinf.pi0 import Pi0
from rlinf.models.embodiment.openpi_rlinf.pi0_config import Pi0Config
from rlinf.models.embodiment.openpi_rlinf.transforms.env import repack_env_obs


class Pi0DAgger(EnvIO, Pi0):
    """Imitation: Euler rollout + SFT loss on expert-relabeled chunks."""

    def __init__(
        self,
        config: Pi0Config,
        *,
        num_steps: int = 10,
        action_env_dim: int | None = None,
        action_chunk: int | None = None,
        config_name: str = "",
        state_indices: Sequence[int] | None = None,
    ):
        super().__init__(
            config,
            num_steps=num_steps,
            action_env_dim=action_env_dim,
            action_chunk=action_chunk,
            config_name=config_name,
            state_indices=state_indices,
        )

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del mode, kwargs
        repacked = repack_env_obs(
            self.config_name,
            env_obs,
            select_state=self._select_configured_state,
        )
        processed = self.input_transform(repacked, transpose=False)
        observation = self._observation_dict_to_device(processed)
        actions, result = self._predict_eval(observation)
        # Replay DAgger reads observation/* + tokenized_prompt from
        # forward_inputs. The HF worker always calls DAgger with mode="eval".
        result["forward_inputs"].update(_clone_replay_obs(repacked))
        if observation.tokenized_prompt is not None:
            result["forward_inputs"]["tokenized_prompt"] = (
                observation.tokenized_prompt.contiguous()
            )
            result["forward_inputs"]["tokenized_prompt_mask"] = (
                observation.tokenized_prompt_mask.contiguous()
            )
        return actions, result

    def forward(self, forward_type: ForwardType = ForwardType.SFT, **kwargs):
        if forward_type != ForwardType.SFT:
            raise NotImplementedError(
                f"{type(self).__name__} only supports ForwardType.SFT; "
                f"got forward_type={forward_type!r}."
            )
        return super().sft_forward(**kwargs)

    def prepare_dagger_sft_batch(self, batch):
        """Prepare replay-buffer samples for DAgger SFT updates."""
        device = self.device
        obs_dict = {k: batch[k] for k in batch.keys() if k.startswith("observation/")}
        if not obs_dict:
            raise ValueError(
                "Replay DAgger batch has no observation/* keys. "
                "Pi0DAgger.predict_action_batch(mode='train') must stash the "
                "env observation into forward_inputs."
            )
        if "tokenized_prompt" in batch:
            obs_dict["tokenized_prompt"] = batch["tokenized_prompt"]
        if "tokenized_prompt_mask" in batch:
            obs_dict["tokenized_prompt_mask"] = batch["tokenized_prompt_mask"]

        bsz = batch["action"].shape[0]
        horizon = self.action_horizon
        action_dim = self.action_dim
        if "model_action" in batch:
            actions = batch["model_action"].reshape(bsz, horizon, action_dim).clone()
            processed_obs = self.input_transform(obs_dict, transpose=False)
            observation = Observation.from_dict(processed_obs)
        else:
            chunk = self.action_chunk or horizon
            obs_dict["actions"] = batch["action"].reshape(bsz, chunk, -1)
            obs_dict["prompt"] = ["empty" for _ in range(bsz)]
            processed_obs = self.input_transform(obs_dict, transpose=False)
            if "tokenized_prompt" in batch:
                processed_obs["tokenized_prompt"] = batch["tokenized_prompt"]
            if "tokenized_prompt_mask" in batch:
                processed_obs["tokenized_prompt_mask"] = batch["tokenized_prompt_mask"]
            observation = Observation.from_dict(processed_obs)
            actions = processed_obs["actions"].clone()
            processed_obs.pop("actions", None)

        return {
            "observation": self._observation_to_device(observation),
            "actions": actions.to(torch.float32).to(device),
        }

    def prepare_lerobot_sft_batch(self, batch):
        """Prepare LeRobot-style samples for DAgger SFT updates."""
        device = self.device
        raw_obs_keys = [
            k
            for k in batch.keys()
            if k
            in [
                "image",
                "wrist_image",
                "extra_view_image",
                "extra_view_image-0",
                "extra_view_image-1",
                "state",
            ]
        ]
        merge_keys = ["extra_view_image-0", "extra_view_image-1"]
        merge_extra = "extra_view_image" not in raw_obs_keys and all(
            k in raw_obs_keys for k in merge_keys
        )
        obs_dict = {}
        for key in raw_obs_keys:
            if merge_extra and key in merge_keys:
                continue
            obs_dict[f"observation/{key}"] = batch[key]
        if merge_extra:
            obs_dict["observation/extra_view_image"] = torch.stack(
                [batch[k] for k in merge_keys], dim=1
            )

        bsz = batch["actions"].shape[0]
        chunk = self.action_chunk or self.action_horizon
        obs_dict["actions"] = batch["actions"].reshape(bsz, chunk, -1)
        obs_dict["prompt"] = batch["task"]
        processed_obs = self.input_transform(obs_dict, transpose=False)
        actions = processed_obs["actions"].clone()
        processed_obs.pop("actions")
        observation = Observation.from_dict(processed_obs)
        return {
            "observation": self._observation_to_device(observation),
            "actions": actions.to(torch.float32).to(device),
        }


def _clone_replay_obs(repacked: dict[str, Any]) -> dict[str, Any]:
    """Clone env-repacked tensors for ``forward_inputs`` (skip the prompt)."""
    cloned: dict[str, Any] = {}
    for key, value in repacked.items():
        if key == "prompt":
            continue
        if torch.is_tensor(value):
            cloned[key] = value.detach().contiguous()
        elif isinstance(value, np.ndarray):
            cloned[key] = torch.from_numpy(np.ascontiguousarray(value))
        else:
            cloned[key] = value
    return cloned
