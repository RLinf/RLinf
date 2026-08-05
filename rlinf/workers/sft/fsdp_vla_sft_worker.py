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
import dataclasses
import os
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.config import SupportedModel
from rlinf.data.lerobot_paths import resolve_lerobot_repo_id
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.utils import get_rng_state, set_rng_state
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def build_dataloader(self, data_paths: Any, eval_dataset: bool = False):
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        if model_type == SupportedModel.OPENPI_PYTORCH:
            if "robotwin" in str(self.cfg.actor.model.openpi.config_name).lower():
                return self._build_official_openpi_dataloader(
                    data_paths, eval_dataset=eval_dataset
                )
            from rlinf.data.datasets.openpi_pytorch import (
                build_openpi_pytorch_sft_dataloader,
            )

            return build_openpi_pytorch_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        elif model_type == SupportedModel.OPENPI:
            return self._build_official_openpi_dataloader(
                data_paths, eval_dataset=eval_dataset
            )
        elif model_type == SupportedModel.LINGBOTVLA:
            from rlinf.models.embodiment.lingbotvla.sft_builder import (
                build_lingbot_sft_dataloader,
            )

            return build_lingbot_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        elif model_type == SupportedModel.DREAMZERO:
            from rlinf.data.datasets.dreamzero import (
                build_dreamzero_sft_dataloader,
            )

            return build_dreamzero_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        elif model_type == SupportedModel.EVO1:
            from rlinf.models.embodiment.evo1.sft_builder import (
                build_evo1_sft_dataloader,
            )

            return build_evo1_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def _build_official_openpi_dataloader(
        self, data_paths: Any, *, eval_dataset: bool = False
    ):
        """Build the original RLinf SFT path backed by OpenPI's loader."""
        repo_id = resolve_lerobot_repo_id(data_paths)
        if repo_id is None:
            raise ValueError(
                "OpenPI SFT requires data.train_data_paths to be set to a local "
                "dataset path or LeRobot repo id."
            )

        import openpi.training.data_loader as openpi_data_loader

        from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

        model_type = SupportedModel(self.cfg.actor.model.model_type)
        batch_size = self.cfg.actor.micro_batch_size
        if eval_dataset:
            batch_size = self.cfg.actor.get("eval_batch_size", batch_size)

        config = get_openpi_config(
            self.cfg.actor.model.openpi.config_name,
            model_path=self.cfg.actor.model.model_path,
            batch_size=batch_size * self._world_size,
            repo_id=repo_id,
            data_kwargs=getattr(self.cfg.actor.model, "openpi_data", None),
        )
        if model_type == SupportedModel.OPENPI_PYTORCH:
            config = dataclasses.replace(
                config,
                num_workers=int(
                    OmegaConf.select(
                        self.cfg, "data.num_workers", default=config.num_workers
                    )
                ),
                seed=int(OmegaConf.select(self.cfg, "actor.seed", default=config.seed)),
            )
            self._validate_openpi_pytorch_model_shape(config)

        data_loader = openpi_data_loader.create_data_loader(
            config, framework="pytorch", shuffle=not eval_dataset
        )
        return data_loader, data_loader.data_config()

    def _validate_openpi_pytorch_model_shape(self, openpi_config: Any) -> None:
        """Keep the local Pi0 architecture consistent with the OpenPI config."""
        model_cfg = self.cfg.actor.model
        local_horizon = int(model_cfg.num_action_chunks)
        official_horizon = int(openpi_config.model.action_horizon)
        if local_horizon != official_horizon:
            raise ValueError(
                "openpi_pytorch SFT action horizon must match the official OpenPI "
                f"config: actor.model.num_action_chunks={local_horizon}, "
                f"{model_cfg.openpi.config_name}.model.action_horizon="
                f"{official_horizon}."
            )

        local_action_dim = int(model_cfg.openpi.model_action_dim)
        official_action_dim = int(openpi_config.model.action_dim)
        if local_action_dim != official_action_dim:
            raise ValueError(
                "openpi_pytorch SFT model action dim must match the official OpenPI "
                f"config: actor.model.openpi.model_action_dim={local_action_dim}, "
                f"{model_cfg.openpi.config_name}.model.action_dim="
                f"{official_action_dim}."
            )

    def get_eval_model_output(self, batch: dict[str, Any]):
        # now the eval is not supported for embodied sft
        raise NotImplementedError("eval is not supported for embodied sft right now.")

    def get_train_model_output(self, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        with self.amp_context:
            output = self.model(forward_type=ForwardType.SFT, data=batch)

        if isinstance(output, torch.Tensor):
            loss = output
        else:
            loss = output["loss"]

        step_metrics = {"loss": loss.detach().item()}
        if isinstance(output, dict):
            for key, value in output.items():
                if key == "loss":
                    continue
                if torch.is_tensor(value):
                    if value.numel() == 1:
                        step_metrics[key] = value.detach().item()
                elif isinstance(value, (float, int)):
                    step_metrics[key] = value
        return loss, step_metrics

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        super().save_checkpoint(save_path, step)

        if isinstance(self.data_loader, StatefulDataLoader):
            state = self.data_loader.state_dict()

            all_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_states, state)

            if self._rank == 0:
                torch.save(all_states, os.path.join(save_path, "data.pt"))

            torch.distributed.barrier()

            rng_state = get_rng_state()
            all_rng_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_rng_states, rng_state)
            if self._rank == 0:
                torch.save(all_rng_states, os.path.join(save_path, "rng.pt"))

            torch.distributed.barrier()

    def load_checkpoint(self, load_path: str) -> None:
        super().load_checkpoint(load_path)

        if isinstance(self.data_loader, StatefulDataLoader):
            all_states = torch.load(
                os.path.join(load_path, "data.pt"), weights_only=False
            )
            state = all_states[self._rank]
            self.data_loader.load_state_dict(state)
            self.data_iter = iter(self.data_loader)

            rng_path = os.path.join(load_path, "rng.pt")
            if os.path.exists(rng_path):
                all_rng_states = torch.load(rng_path, weights_only=False)
                set_rng_state(all_rng_states[self._rank])

            torch.distributed.barrier()

    def get_max_steps_per_epoch(self):
        if self.data_loader is None:
            return 0
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        if model_type == SupportedModel.OPENPI_PYTORCH:
            if "robotwin" in str(self.cfg.actor.model.openpi.config_name).lower():
                num_batches = len(self._openpi_pytorch_dataloader(self.data_loader))
            else:
                num_batches = len(self.data_loader)
        elif model_type == SupportedModel.OPENPI:
            num_batches = len(self._openpi_pytorch_dataloader(self.data_loader))
        else:
            return super().get_max_steps_per_epoch()
        return max(1, num_batches // self.gradient_accumulation)

    @staticmethod
    def _openpi_pytorch_dataloader(openpi_dataloader: Any):
        """Unwrap OpenPI `DataLoaderImpl` to the inner PyTorch DataLoader.

        OpenPI torch path:
          DataLoaderImpl._data_loader -> TorchDataLoader
          TorchDataLoader._data_loader / .torch_loader -> torch.utils.data.DataLoader

        """
        torch_data_loader = getattr(openpi_dataloader, "_data_loader", None)
        pytorch_dl = getattr(torch_data_loader, "_data_loader", None) or getattr(
            torch_data_loader, "torch_loader", None
        )
        if pytorch_dl is None:
            raise TypeError(
                "OpenPI dataloader does not expose an inner torch DataLoader; cannot infer steps per epoch from len()."
            )
        return pytorch_dl
