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

"""OpenPI wrapper reused by RLT Stage 2 for ManiSkill simulation."""

from __future__ import annotations

import pathlib
from typing import Any

import torch
from openpi.models import model as _model
from omegaconf import OmegaConf, open_dict

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.openpi import get_model as get_openpi_model

from .proprio import select_proprio


class Stage2VLAWrapper:
    """Frozen OpenPI VLA wrapper for embedding extraction and reference actions."""

    def __init__(
        self,
        *,
        model_path: str,
        config_name: str,
        norm_stats_path: str | None = None,
        num_images_in_input: int = 2,
        num_action_chunks: int = 10,
        action_dim: int = 8,
        num_steps: int = 5,
        device: torch.device | str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        cfg_dict = {
            "model_type": "openpi",
            "model_path": model_path,
            "precision": None,
            "is_lora": False,
            "load_to_device": True,
            "openpi": {
                "config_name": config_name,
                "num_images_in_input": num_images_in_input,
                "action_chunk": int(num_action_chunks),
                "num_steps": int(num_steps),
                "action_env_dim": int(action_dim),
                "train_expert_only": False,
                "add_value_head": False,
            },
        }
        if norm_stats_path is not None:
            cfg_dict["openpi_data"] = {"norm_stats_path": norm_stats_path}
        cfg = OmegaConf.create(cfg_dict)
        with open_dict(cfg):
            cfg.num_action_chunks = int(num_action_chunks)
            cfg.action_dim = int(action_dim)
        self.model = get_openpi_model(cfg, torch_dtype=None)

        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        checkpoint_dir = pathlib.Path(model_path)
        self.embedding_dim = 2048
        self.action_horizon = self.model.config.action_horizon
        self.action_dim = self.model.config.action_dim

    def prepare_obs(
        self,
        obs: dict[str, Any],
    ) -> tuple[_model.Observation, dict[str, Any]]:
        if self._has_pre_tokenized_env_obs(obs):
            to_process_obs = self._openpi_obs_from_pre_tokenized_env_obs(obs)
        elif "task_descriptions" in obs and obs["task_descriptions"] is not None:
            to_process_obs = self.model.obs_processor(obs)
        else:
            raise KeyError(
                "RLT Stage 2 observation must contain either task_descriptions or pre-tokenized prompt tensors."
            )
        processed = self.model.input_transform(to_process_obs, transpose=False)
        processed = self.model.precision_processor(processed)
        return _model.Observation.from_dict(processed), processed

    @staticmethod
    def _has_pre_tokenized_env_obs(obs: dict[str, Any]) -> bool:
        return (
            "tokenized_prompt" in obs
            and "tokenized_prompt_mask" in obs
            and "main_images" in obs
            and "states" in obs
        )

    @staticmethod
    def _openpi_obs_from_pre_tokenized_env_obs(obs: dict[str, Any]) -> dict[str, Any]:
        to_process_obs = {
            "observation/image": obs["main_images"],
            "observation/state": obs["states"],
            "tokenized_prompt": obs["tokenized_prompt"],
            "tokenized_prompt_mask": obs["tokenized_prompt_mask"],
        }
        if obs.get("wrist_images") is not None:
            to_process_obs["observation/wrist_image"] = obs["wrist_images"]
        if obs.get("extra_view_images") is not None:
            to_process_obs["observation/extra_view_image"] = obs["extra_view_images"]
        return to_process_obs

    def extract_proprio(
        self,
        observation: _model.Observation,
        proprio_dim: int | None = None,
        proprio_mode: str | None = None,
    ) -> torch.Tensor:
        return select_proprio(
            observation.state,
            proprio_dim=proprio_dim,
            proprio_mode=proprio_mode,
        ).to(
            device=self.device,
            dtype=torch.float32,
        )

    def preprocess_obs(self, obs: dict[str, Any]) -> _model.Observation:
        observation, _ = self.prepare_obs(obs)
        return observation

    @torch.no_grad()
    def extract_embeddings(
        self,
        observation: _model.Observation,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images, img_masks, lang_tokens, lang_masks, _ = (
            self.model._preprocess_observation(observation, train=False)
        )
        images = [img.to(self.device) for img in images]
        img_masks = [mask.to(self.device) for mask in img_masks]
        prefix_output, prefix_pad_masks, _ = self.model._build_prefix_cache(
            images, img_masks, lang_tokens, lang_masks
        )
        return prefix_output.detach(), prefix_pad_masks.detach()

    @torch.no_grad()
    def get_rl_chunk_reference(
        self,
        observation: _model.Observation,
        chunk_length: int,
    ) -> torch.Tensor:
        outputs = self.model.sample_actions(
            observation,
            mode="eval",
            compute_values=False,
        )
        action_dict = self.model.output_transform(
            {"actions": outputs["actions"], "state": observation.state}
        )
        actions = action_dict["actions"].to(self.device, dtype=torch.float32)
        return actions[:, :chunk_length]
