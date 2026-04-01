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
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils import _pytree

from rlinf.config import SupportedModel
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.pytree import register_pytree_dataclasses
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker

import rlinf.models.embodiment.gr00t_1_6.gr00t_16_sft_model


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def build_dataloader(self, data_paths: list[str], eval_dataset: bool = False):
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            import openpi.training.data_loader as openpi_data_loader

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            config = get_openpi_config(
                self.cfg.actor.model.openpi.config_name,
                model_path=self.cfg.actor.model.model_path,
                batch_size=self.cfg.actor.micro_batch_size * self._world_size,
            )
            data_loader = openpi_data_loader.create_data_loader(
                config, framework="pytorch", shuffle=True
            )
            return data_loader, data_loader.data_config()
        elif SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.GR00T_1_6_SFT]:
            import torch
            from torch.utils.data import DataLoader
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            print(f" [Custom DataLoader] Loading LeRobot dataset from: {self.cfg.data.train_data_paths}")

            dataset = LeRobotDataset(
                repo_id=self.cfg.data.train_data_paths,
                # local_files_only=True
                video_backend="pyav"
            )

            def gr00t_collate_fn(batch):
                action_key = next((k for k in batch[0].keys() if "action" in k.lower()), None)
                if action_key is None:
                    raise KeyError("Could not find an action key!")

                actions = torch.stack([item[action_key] for item in batch])
                if actions.dim() == 2:
                    actions = actions.unsqueeze(1) 

                obs = {}
                for key in batch[0].keys():
                    if key != action_key:
                        item_val = batch[0][key]
                        
                        if isinstance(item_val, str) or (isinstance(item_val, (list, tuple)) and len(item_val)>0 and isinstance(item_val[0], str)):
                            byte_tensors = []
                            for item in batch:
                                text = item[key]
                                if isinstance(text, (list, tuple)): text = text[0]
                                byte_tensors.append(torch.tensor(list(text.encode('utf-8')), dtype=torch.uint8))
                            
                            obs[key] = torch.nn.utils.rnn.pad_sequence(byte_tensors, batch_first=True, padding_value=0)
                            
                        elif isinstance(item_val, torch.Tensor):
                            obs[key] = torch.stack([item[key] for item in batch])
                        else:
                            try:
                                obs[key] = torch.tensor([item[key] for item in batch])
                            except:
                                obs[key] = [item[key] for item in batch]

                return obs, actions

            dataloader = DataLoader(
                dataset,
                batch_size=self.cfg.actor.micro_batch_size * self._world_size,
                shuffle=not eval_dataset,
                num_workers=4,
                collate_fn=gr00t_collate_fn,
                pin_memory=True,
                drop_last=True
            )
            
            print(" [Custom DataLoader] Successfully built for GR00T-1.6!")
            return dataloader, None
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def get_eval_model_output(self, batch: dict[str, Any]):
        # now the eval is not supported for embodied sft
        raise NotImplementedError("eval is not supported for embodied sft right now.")

    def get_train_model_output(self, batch: dict[str, Any]):
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
