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
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.config import SupportedModel, torch_dtype_from_precision
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.utils import get_rng_state, set_rng_state
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._openwam_normalization_stats_path: str | None = None
        if self._is_openwam():
            train_info = getattr(self, "data_config", None) or {}
            eval_info = getattr(self, "eval_data_config", None) or {}
            train_digest = train_info.get("normalization_stats_digest")
            eval_digest = eval_info.get("normalization_stats_digest")
            if train_digest and eval_digest and train_digest != eval_digest:
                raise ValueError(
                    "OpenWAM train and validation data use different normalization "
                    "stats; configure one shared artifact for both splits."
                )
            self._openwam_normalization_stats_path = train_info.get(
                "normalization_stats_path"
            ) or eval_info.get("normalization_stats_path")

    def model_provider_func(self):
        """Build VLA models with OpenWAM's FSDP compute dtype."""
        if self._is_openwam():
            model_cfg = OmegaConf.create(
                OmegaConf.to_container(self.cfg.actor.model, resolve=False)
            )
            model_cfg.runtime_dtype = self.cfg.actor.fsdp_config.mixed_precision.get(
                "param_dtype", None
            )
            # Keep fp32 (or the configured master dtype) in parameter storage;
            # OpenWAM's cached input/backbone runtime dtype is passed separately
            # above so FSDP mixed precision can control compute without changing
            # optimizer/master weights.
            return get_model(
                model_cfg,
                torch_dtype=torch_dtype_from_precision(model_cfg.get("precision")),
            )
        return super().model_provider_func()

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
        elif model_type == SupportedModel.OPENWAM:
            from rlinf.data.datasets.openwam import build_openwam_sft_dataloader

            return build_openwam_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
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

    def _is_openwam(self) -> bool:
        return SupportedModel(self.cfg.actor.model.model_type) == SupportedModel.OPENWAM

    def get_eval_model_output(self, batch: Any) -> dict[str, float]:
        """Return the per-batch validation losses of an OpenWAM SFT model."""
        if not self._is_openwam():
            # now the eval is not supported for the other embodied sft models
            raise NotImplementedError(
                "eval is not supported for embodied sft right now."
            )
        with torch.no_grad(), self.amp_context:
            output = self.model(forward_type=ForwardType.SFT, data=batch)
        if isinstance(output, torch.Tensor):
            return {"loss": float(output.detach().item())}
        return {
            key: float(value.detach().item())
            for key, value in output.items()
            if torch.is_tensor(value) and value.numel() == 1
        }

    def run_eval(self):
        """Average OpenWAM's native SFT losses over the validation loader.

        The base implementation counts token-level hits, which has no meaning
        for a video/action diffusion loss, so OpenWAM reports ``loss``,
        ``loss_video`` and ``loss_action`` (logged under ``eval/``) instead.
        ``actor.eval_max_batches`` caps the number of validation batches per
        rank for large datasets.
        """
        if not self._is_openwam():
            return super().run_eval()
        assert self.eval_data_loader is not None, "eval_data_loader is not set"
        max_batches = self.cfg.actor.get("eval_max_batches", None)
        from rlinf.data.datasets.openwam.dataloader import ValidationSample

        with self.worker_timer():
            self.model.eval()
            # Native losses are already means over each local batch. Accumulate
            # weighted sums so uneven validation shards and a final short batch
            # contribute according to their actual number of samples.
            loss_keys = ("loss", "loss_video", "loss_action")
            sums = dict.fromkeys(loss_keys, 0.0)
            num_samples = 0
            num_batches = 0
            for index, batch in enumerate(self.eval_data_loader):
                if max_batches is not None and index >= int(max_batches):
                    break
                # The validation sampler pads short shards so every rank executes
                # the same number of FSDP forwards. Run one sample per forward: a
                # padded sample still participates in FSDP collectives, while
                # its loss is excluded from the global metric.
                for item in batch:
                    if isinstance(item, ValidationSample):
                        sample, is_padding = item.sample, item.is_padding
                    else:
                        sample, is_padding = item, False
                    outputs = self.get_eval_model_output([sample])
                    if is_padding:
                        continue
                    for key in loss_keys:
                        sums[key] += float(outputs.get(key, 0.0))
                    num_samples += 1
                num_batches += 1
            self.model.train()
            reduced = all_reduce_dict(
                {
                    **sums,
                    "num_samples": float(num_samples),
                    "num_batches": float(num_batches),
                },
                op=torch.distributed.ReduceOp.SUM,
            )
            total_samples = max(1.0, reduced.pop("num_samples"))
            for key in loss_keys:
                reduced[key] /= total_samples
            return reduced

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

        if self._is_openwam() and self._openwam_normalization_stats_path:
            stats_path = Path(self._openwam_normalization_stats_path)
            if not stats_path.is_file():
                raise FileNotFoundError(
                    f"OpenWAM normalization stats artifact not found at checkpoint time: {stats_path}"
                )
            if self._rank == 0:
                shutil.copy2(stats_path, Path(save_path) / "normalization_stats.npy")
            torch.distributed.barrier()

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
            runner_cfg = getattr(getattr(self, "cfg", None), "runner", None)
            strict_resume = bool(
                runner_cfg.get("strict_resume", False)
                if runner_cfg is not None
                else False
            )
            data_path = os.path.join(load_path, "data.pt")
            if os.path.exists(data_path):
                all_states = torch.load(data_path, weights_only=False)
                state = all_states[self._rank]
                self.data_loader.load_state_dict(state)
                self.data_iter = iter(self.data_loader)
                # Creating the iterator applies the sampler state. Continue the
                # shuffle epoch it recorded instead of replaying epoch 0 after
                # the first exhaustion (samplers without an epoch keep the
                # default).
                epoch = getattr(
                    getattr(self.data_loader, "sampler", None), "epoch", None
                )
                if epoch is not None:
                    self._data_epoch = int(epoch)
            else:
                # Checkpoints written before the data loader became stateful
                # (or by a plain DataLoader) carry no data position.
                message = (
                    f"{load_path} has no data.pt; the data loader restarts from "
                    "the beginning of the epoch, so samples seen before the "
                    "checkpoint may repeat."
                )
                if strict_resume:
                    raise FileNotFoundError(message)
                logging.warning(message)

            rng_path = os.path.join(load_path, "rng.pt")
            if os.path.exists(rng_path):
                all_rng_states = torch.load(rng_path, weights_only=False)
                set_rng_state(all_rng_states[self._rank])
            elif strict_resume:
                raise FileNotFoundError(
                    f"{load_path} has no rng.pt; strict resume requires the "
                    "checkpoint RNG state."
                )

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
