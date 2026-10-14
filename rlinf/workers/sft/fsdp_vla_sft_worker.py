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
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.config import SupportedModel
from rlinf.data.datasets.openpi.official_sft_data_loader import StatefulOpenPIDataLoader
from rlinf.hybrid_engines.fsdp.strategy.checkpoint import checkpoint_communication_group
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.utils import get_rng_state, seed_everything, set_rng_state
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def init_worker(self) -> None:
        """Seed OpenPI model initialization, then each rank's training stream."""
        if SupportedModel(self.cfg.actor.model.model_type) != SupportedModel.OPENPI:
            super().init_worker()
            return
        seed = int(self.cfg.actor.seed)
        seed_everything(seed)
        super().init_worker()
        seed_everything(seed + self._rank)

    def build_dataloader(self, data_paths: Any, eval_dataset: bool = False):
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        if model_type == SupportedModel.OPENPI:
            from rlinf.data.datasets.openpi import (
                build_openpi_sft_dataloader,
            )

            return build_openpi_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
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
        elif model_type == SupportedModel.COSMOS3:
            from rlinf.data.datasets.cosmos3 import (
                build_cosmos3_sft_dataloader,
            )

            return build_cosmos3_sft_dataloader(self.cfg, data_paths, eval_dataset)
        elif model_type == SupportedModel.EVO1:
            from rlinf.models.embodiment.evo1.sft_builder import (
                build_evo1_sft_dataloader,
            )

            return build_evo1_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        elif model_type == SupportedModel.FASTWAM:
            from rlinf.data.datasets.fastwam import build_fastwam_sft_dataloader

            return build_fastwam_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
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

    def save_checkpoint(
        self, save_path: str, step: int = 0, *, force_training_state: bool = False
    ) -> None:
        if isinstance(self.data_loader, StatefulOpenPIDataLoader) and self._rank == 0:
            (Path(save_path) / "openpi_loader_state.json").unlink(missing_ok=True)
        super().save_checkpoint(
            save_path, step, force_training_state=force_training_state
        )
        if not self._should_save_training_state(step, force_training_state):
            return

        if isinstance(self.data_loader, (StatefulDataLoader, StatefulOpenPIDataLoader)):
            with checkpoint_communication_group(
                self._cfg.fsdp_config.get("checkpoint_communication_backend"),
                self._cfg.fsdp_config.get("checkpoint_format", "dcp"),
            ) as checkpoint_group:
                snapshots = [None] * self._world_size
                torch.distributed.all_gather_object(
                    snapshots,
                    (self.data_loader.state_dict(), get_rng_state()),
                    group=checkpoint_group,
                )
                if self._rank == 0:
                    all_states, all_rng_states = map(list, zip(*snapshots))
                    torch.save(all_states, os.path.join(save_path, "data.pt"))
                    torch.save(all_rng_states, os.path.join(save_path, "rng.pt"))
                    if isinstance(self.data_loader, StatefulOpenPIDataLoader):
                        self.data_loader.mark_checkpoint_complete(save_path, all_states)
                torch.distributed.barrier()

    def load_checkpoint(self, load_path: str) -> None:
        if isinstance(self.data_loader, StatefulOpenPIDataLoader):
            self.data_loader.validate_checkpoint(load_path)
        super().load_checkpoint(load_path)

        if isinstance(self.data_loader, (StatefulDataLoader, StatefulOpenPIDataLoader)):
            all_states = torch.load(
                os.path.join(load_path, "data.pt"), weights_only=False
            )
            if (
                isinstance(self.data_loader, StatefulOpenPIDataLoader)
                and len(all_states) != self._world_size
            ):
                raise ValueError("OpenPI loader restore requires unchanged world size.")
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
        if model_type == SupportedModel.OPENPI:
            from rlinf.data.datasets.openpi import (
                get_official_openpi_sft_num_batches,
                is_official_openpi_sft_dataloader,
            )

            num_batches = (
                get_official_openpi_sft_num_batches(self.data_loader)
                if is_official_openpi_sft_dataloader(self.data_loader)
                else len(self.data_loader)
            )
        else:
            return super().get_max_steps_per_epoch()
        return max(1, num_batches // self.gradient_accumulation)
