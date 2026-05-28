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
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils._pytree import tree_map
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.config import SupportedModel
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.pytree import register_pytree_dataclasses
from rlinf.utils.utils import get_rng_state, set_rng_state
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


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
            SupportedModel.DREAMZERO
        ]:
            self._dreamzero_loss = None
            from rlinf.data.datasets.dreamzero import (
                build_dreamzero_sft_dataloader,
            )

            return build_dreamzero_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.LINGBOTVA
        ]:
            self._lingbotva_loss = None
            self._lingbotva_loss_accum: dict[str, list[torch.Tensor]] = {
                "latent_loss": [],
                "action_loss": [],
            }
            from rlinf.data.datasets.lingbotva import (
                build_lingbotva_sft_dataloader,
            )

            return build_lingbotva_sft_dataloader(
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
            SupportedModel.LINGBOTVLA,
            SupportedModel.DREAMZERO,
            SupportedModel.LINGBOTVA,
        ]:
            with self.amp_context:
                losses_dict = self.model(forward_type=ForwardType.SFT, data=batch)
            if losses_dict.get("dynamics_loss", None) is not None:
                self._dreamzero_loss = {
                    "dynamics_loss": losses_dict["dynamics_loss"],
                    "action_loss": losses_dict["action_loss"],
                }
            if losses_dict.get("latent_loss", None) is not None:
                # Accumulate per-micro-batch latent/action loss so the metric we
                # report per optimizer step is the mean across grad-accum
                # micro-batches, matching upstream lingbot-va's reporting (which
                # logs `sum(loss_i / grad_accum)` over grad_accum micro-batches).
                self._lingbotva_loss_accum["latent_loss"].append(
                    losses_dict["latent_loss"]
                )
                self._lingbotva_loss_accum["action_loss"].append(
                    losses_dict["action_loss"]
                )
            return losses_dict["loss"]
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
        actions = actions.to(torch.float32)
        actions = actions.to(self.device)

        with self.amp_context:
            losses = self.model(
                forward_type=ForwardType.SFT,
                data={"observation": observation, "actions": actions},
            )

        # train model return the loss
        return losses

    def run_training(self):
        train_metrics = super().run_training()
        if (
            SupportedModel(self.cfg.actor.model.model_type)
            in [SupportedModel.DREAMZERO]
            and self._dreamzero_loss is not None
        ):
            train_metrics.update(
                {
                    "dynamics_loss": self._dreamzero_loss["dynamics_loss"],
                    "action_loss": self._dreamzero_loss["action_loss"],
                }
            )
            self._dreamzero_loss = None
        if (
            SupportedModel(self.cfg.actor.model.model_type)
            in [SupportedModel.LINGBOTVA]
            and getattr(self, "_lingbotva_loss_accum", None) is not None
            and self._lingbotva_loss_accum["latent_loss"]
        ):
            # Report the per-component mean across grad-accum micro-batches,
            # matching upstream lingbot-va's reporting. Without this we'd log
            # only the last micro-batch's noisy single-sample value, which is
            # far below the mean for the heavy-tailed action-timestep weights.
            latent_mean = torch.stack(
                self._lingbotva_loss_accum["latent_loss"]
            ).mean()
            action_mean = torch.stack(
                self._lingbotva_loss_accum["action_loss"]
            ).mean()
            train_metrics.update(
                {"latent_loss": latent_mean, "action_loss": action_mean}
            )
            self._lingbotva_loss_accum["latent_loss"].clear()
            self._lingbotva_loss_accum["action_loss"].clear()
        return train_metrics

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
        if SupportedModel(self.cfg.actor.model.model_type) == SupportedModel.OPENPI:
            num_batches = len(self._openpi_pytorch_dataloader(self.data_loader))
            return max(1, num_batches // self.gradient_accumulation)
        return super().get_max_steps_per_epoch()

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
