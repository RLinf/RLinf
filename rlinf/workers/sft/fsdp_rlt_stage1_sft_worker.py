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
import os

import torch
from omegaconf import DictConfig
from torch.utils._pytree import tree_map

from rlinf.config import SupportedModel
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.pytree import register_pytree_dataclasses
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker


class FSDPRLTStage1SftWorker(FSDPVlaSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def get_train_model_output(self, batch):
        if SupportedModel(self.cfg.actor.model.model_type) != SupportedModel.RLT_STAGE1:
            return super().get_train_model_output(batch)

        observation, actions = batch
        register_pytree_dataclasses(observation)
        observation = tree_map(
            lambda x: (
                torch.as_tensor(x, device=self.device).contiguous().clone()
                if x is not None
                else x
            ),
            observation,
        )
        actions = actions.to(torch.float32).to(self.device)

        with self.amp_context:
            losses = self.model(
                forward_type=ForwardType.SFT,
                data={"observation": observation, "actions": actions},
            )

        step_metrics = {"loss": losses["loss"].detach().item()}
        for key in ("l_ro", "l_vla"):
            if key in losses:
                step_metrics[key] = losses[key].detach().item()
        return losses["loss"], step_metrics

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        super().save_checkpoint(save_path, step)
        if self._rank != 0:
            return
        rl_token_dir = os.path.join(save_path, "rl_token")
        os.makedirs(rl_token_dir, exist_ok=True)
        model_state = self._strategy.get_model_state_dict(
            self.model, cpu_offload=True, full_state_dict=True
        )
        rl_token_state = {
            key.replace("rl_token_model.", "", 1): value
            for key, value in model_state.items()
            if key.startswith("rl_token_model.")
        }
        torch.save(
            {"model_state_dict": rl_token_state, "step": step},
            os.path.join(rl_token_dir, "rl_token_model.pt"),
        )
