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
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.config import SupportedModel
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.utils import get_rng_state, set_rng_state
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        self._is_streamingvla = (
            SupportedModel(cfg.actor.model.model_type) == SupportedModel.STREAMINGVLA
        )
        self._streamingvla_step_inputs: Any | None = None
        if self._is_streamingvla:
            from rlinf.models.embodiment.streamingvla.training import (
                seed_streamingvla_training,
            )

            rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
            seed_streamingvla_training(cfg.actor.seed, rank)
        super().__init__(cfg)
        if self._is_streamingvla:
            from rlinf.models.embodiment.streamingvla.training import (
                StreamingVLAStepInputBuffer,
            )

            self._configure_streamingvla_grad_norm_group()
            self._streamingvla_step_inputs = StreamingVLAStepInputBuffer(
                seed=cfg.actor.seed,
                rank=self._rank,
                local_batch_size=self.global_batch_size // self._world_size,
                action_dim=int(cfg.actor.model.streamingvla.model_action_dim),
                device=self.device,
            )

    def _configure_streamingvla_grad_norm_group(
        self, process_group: Any | None = None
    ) -> None:
        """Use the full FSDP world group for StreamingVLA gradient clipping."""
        if not self._is_streamingvla:
            return
        if self._strategy._dp_group is not None:
            return
        if process_group is None:
            if not torch.distributed.is_initialized():
                raise RuntimeError(
                    "StreamingVLA FSDP gradient clipping requires an initialized "
                    "distributed process group."
                )
            process_group = torch.distributed.group.WORLD
        self._strategy._dp_group = process_group

    def set_global_step(self, global_step: int) -> None:
        """Set the step and prepare partition-invariant StreamingVLA RNG inputs."""
        super().set_global_step(global_step)
        if not self._is_streamingvla:
            return

        if self._streamingvla_step_inputs is None:
            raise RuntimeError("StreamingVLA step-input buffer was not initialized.")
        self._streamingvla_step_inputs.set_step(global_step)

    def build_dataloader(self, data_paths: Any, eval_dataset: bool = False):
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        if model_type == SupportedModel.STREAMINGVLA:
            from rlinf.models.embodiment.streamingvla.data import (
                build_streamingvla_dataloader,
            )

            if eval_dataset:
                raise NotImplementedError(
                    "StreamingVLA validation is not implemented in the SFT-only integration."
                )
            return build_streamingvla_dataloader(self.cfg, self._world_size)
        elif model_type == SupportedModel.OPENPI_RLINF:
            from rlinf.data.datasets.openpi_rlinf import (
                build_openpi_rlinf_sft_dataloader,
            )

            return build_openpi_rlinf_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        elif model_type == SupportedModel.OPENPI:
            from rlinf.data.datasets.openpi_rlinf import (
                build_official_openpi_sft_dataloader,
            )

            return build_official_openpi_sft_dataloader(
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
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def get_eval_model_output(self, batch: dict[str, Any]):
        # now the eval is not supported for embodied sft
        raise NotImplementedError("eval is not supported for embodied sft right now.")

    def get_train_model_output(self, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        model_kwargs: dict[str, Any] = {}
        if self._is_streamingvla:
            if self._streamingvla_step_inputs is None:
                raise RuntimeError(
                    "StreamingVLA step-input buffer was not initialized."
                )
            actions = batch[1] if isinstance(batch, (tuple, list)) else batch["actions"]
            time, noise = self._streamingvla_step_inputs.next_micro_batch(
                int(actions.shape[0])
            )
            model_kwargs = {
                "time": time,
                "noise": noise,
            }

        with self.amp_context:
            output = self.model(
                forward_type=ForwardType.SFT, data=batch, **model_kwargs
            )

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

        if self._is_streamingvla:
            checkpoint_dir = os.path.basename(os.path.dirname(load_path))
            prefix = "global_step_"
            if checkpoint_dir.startswith(prefix):
                if self._streamingvla_step_inputs is None:
                    raise RuntimeError(
                        "StreamingVLA step-input buffer was not initialized."
                    )
                self._streamingvla_step_inputs.set_step(
                    int(checkpoint_dir.removeprefix(prefix))
                )

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
        if model_type in (SupportedModel.OPENPI_RLINF, SupportedModel.OPENPI):
            if model_type == SupportedModel.OPENPI_RLINF:
                from rlinf.data.datasets.openpi_rlinf import (
                    get_official_openpi_sft_num_batches,
                    is_official_openpi_sft_dataloader,
                )

                num_batches = (
                    get_official_openpi_sft_num_batches(self.data_loader)
                    if is_official_openpi_sft_dataloader(self.data_loader)
                    else len(self.data_loader)
                )
            else:
                from rlinf.data.datasets.openpi_rlinf import (
                    get_official_openpi_sft_num_batches,
                )

                num_batches = get_official_openpi_sft_num_batches(self.data_loader)
        else:
            return super().get_max_steps_per_epoch()
        return max(1, num_batches // self.gradient_accumulation)
