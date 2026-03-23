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
import json
import os
from typing import Any

import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.utils import _pytree

from rlinf.config import SupportedModel
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.dreamzero.sft_builder import (
    build_dreamzero_sft_dataloader,
    build_dreamzero_sft_model,
)
from rlinf.utils.pytree import register_pytree_dataclasses
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def model_provider_func(self):
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.DREAMZERO_SFT
        ]:
            return build_dreamzero_sft_model(self.cfg.actor.model)
        return super().model_provider_func()

    def _save_data_state(self, save_path: str):
        state = {
            "data_epoch": self._data_epoch,
            "data_iter_offset": self._data_iter_offset,
        }
        with open(os.path.join(save_path, "data_state.json"), "w") as f:
            json.dump(state, f)

    def _load_data_state(self, load_path: str):
        if self.data_loader is None:
            return

        path = os.path.join(load_path, "data_state.json")
        if not os.path.exists(path):
            return

        with open(path, "r") as f:
            state = json.load(f)

        self._data_epoch = int(state.get("data_epoch", 0))
        self._data_iter_offset = int(state.get("data_iter_offset", 0))

        if hasattr(self.data_loader, "sampler") and hasattr(
            self.data_loader.sampler, "set_epoch"
        ):
            self.data_loader.sampler.set_epoch(self._data_epoch)

        self.data_iter = iter(self.data_loader)
        for _ in range(self._data_iter_offset):
            try:
                next(self.data_iter)
            except StopIteration:
                self._data_epoch += 1
                if hasattr(self.data_loader, "sampler") and hasattr(
                    self.data_loader.sampler, "set_epoch"
                ):
                    self.data_loader.sampler.set_epoch(self._data_epoch)
                self.data_iter = iter(self.data_loader)

    def _save_dreamzero_artifacts(self, save_path: str):
        if SupportedModel(self.cfg.actor.model.model_type) not in [
            SupportedModel.DREAMZERO_SFT
        ]:
            return
        if not isinstance(getattr(self, "data_config", None), dict):
            return

        train_cfg = self.data_config.get("dreamzero_train_cfg")
        metadata = self.data_config.get("dreamzero_metadata")
        model_config = self.data_config.get("dreamzero_model_config")

        if train_cfg is None and metadata is None and model_config is None:
            return

        exp_cfg_dir = os.path.join(save_path, "experiment_cfg")
        os.makedirs(exp_cfg_dir, exist_ok=True)

        if train_cfg is not None:
            OmegaConf.save(train_cfg, os.path.join(exp_cfg_dir, "conf.yaml"), resolve=True)

        if metadata is not None:
            with open(os.path.join(exp_cfg_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

        if model_config is not None:
            with open(os.path.join(save_path, "config.json"), "w") as f:
                json.dump(model_config, f, indent=2)

    def save_checkpoint(self, save_path: str, step: int = 0):
        super().save_checkpoint(save_path, step)
        if self._rank == 0:
            self._save_data_state(save_path)
            self._save_dreamzero_artifacts(save_path)

    def load_checkpoint(self, load_path: str):
        super().load_checkpoint(load_path)
        self._load_data_state(load_path)

    def build_dataloader(self, data_paths: list[str], eval_dataset: bool = False):
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            import openpi.training.data_loader as openpi_data_loader

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            config = get_openpi_config(
                self.cfg.actor.model.openpi.config_name,
                model_path=self.cfg.actor.model.model_path,
                batch_size=self.cfg.actor.micro_batch_size * self._world_size,
                data_kwargs=getattr(self.cfg.actor, "openpi_data", None),
            )
            data_loader = openpi_data_loader.create_data_loader(
                config, framework="pytorch", shuffle=True
            )
            return data_loader, data_loader.data_config()
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.LINGBOTVLA
        ]:
            from rlinf.models.embodiment.lingbotvla.sft_builder import (
                build_lingbot_sft_dataloader,
            )

            return build_lingbot_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.DREAMZERO_SFT
        ]:
            return build_dreamzero_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def get_eval_model_output(self, batch: dict[str, Any]):
        # now the eval is not supported for embodied sft
        raise NotImplementedError("eval is not supported for embodied sft right now.")

    def get_train_model_output(self, batch: dict[str, Any]):
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.LINGBOTVLA
        ]:
            batch_data = batch
            batch_data = _pytree.tree_map(
                lambda x: torch.as_tensor(x, device=self.device).contiguous().clone()
                if isinstance(x, torch.Tensor)
                else x,
                batch_data,
            )
            with self.amp_context:
                losses_dict = self.model(forward_type=ForwardType.SFT, data=batch_data)
            return losses_dict["loss"]
            
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.DREAMZERO_SFT
        ]:
            batch_data = batch
            batch_data = _pytree.tree_map(
                lambda x: torch.as_tensor(x, device=self.device).contiguous().clone()
                if isinstance(x, torch.Tensor)
                else x,
                batch_data,
            )
            with self.amp_context:
                losses_dict = self.model(batch_data)
            return losses_dict["loss"]

        observation, actions = next(self.data_iter)
        register_pytree_dataclasses(observation)
        observation = _pytree.tree_map(
            lambda x: torch.as_tensor(x, device=self.device).contiguous().clone()
            if x is not None
            else x,
            observation,
        )
        actions = actions.to(torch.float32)
        actions = actions.to(self.device)

        with self.amp_context:
            losses = self.model(
                forward_type=ForwardType.SFT,
                data={"observation": observation, "actions": actions},
            )

        # train model return the loss
        return losses
