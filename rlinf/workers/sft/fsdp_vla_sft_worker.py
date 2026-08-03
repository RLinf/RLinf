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
import itertools
import os
from collections.abc import Iterable
from typing import Any

import torch
from omegaconf import DictConfig
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.config import SupportedModel
from rlinf.data.lerobot_paths import resolve_lerobot_repo_id
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.utils import get_rng_state, set_rng_state
from rlinf.workers.sft._index_recording_dataset import _IndexRecordingDataset
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        self._robotwin_ddp_sharding_audit_enabled = False
        self._robotwin_ddp_sharding_audit_sample_reads = 0
        super().__init__(cfg)

    def init_worker(self):
        super().init_worker()
        if self._robotwin_ddp_sharding_audit_enabled:
            # ``init_worker`` is invoked as a synchronized ActorGroup call. Do
            # the cross-rank collective here rather than in the constructor,
            # where individual Ray actors can be created at different times.
            self._audit_robotwin_ddp_sharding(
                self.data_loader,
                num_sample_reads=self._robotwin_ddp_sharding_audit_sample_reads,
            )

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
        if model_type == SupportedModel.OPENPI:
            return self._build_official_openpi_dataloader(data_paths)
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
            from rlinf.data.datasets.dreamzero import (
                build_dreamzero_sft_dataloader,
            )

            return build_dreamzero_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
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
            from omegaconf import OmegaConf

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
        if (
            model_type == SupportedModel.OPENPI_PYTORCH
            and "robotwin" in str(self.cfg.actor.model.openpi.config_name).lower()
            and not eval_dataset
            and bool(self.cfg.data.get("audit_ddp_sharding", False))
        ):
            self._robotwin_ddp_sharding_audit_enabled = True
            self._robotwin_ddp_sharding_audit_sample_reads = int(
                self.cfg.data.get("audit_ddp_sharding_sample_reads", 8)
            )
        return data_loader, data_loader.data_config()

    def _audit_robotwin_ddp_sharding(
        self, openpi_loader: Any, *, num_sample_reads: int
    ) -> None:
        """Verify that the actual RoboTwin SFT loader is DDP-sharded.

        This deliberately runs after OpenPI builds its real PyTorch loader and
        before the first training batch. It checks both links in the data path:

        1. ``DistributedSampler`` assigns disjoint indices to every DDP rank.
        2. The transformed RoboTwin dataset passes a sampler index through to
           the underlying LeRobot dataset, rather than ignoring it as a
           streaming dataset would.

        The audit never changes the loader or the batches consumed by SFT.
        """
        if num_sample_reads < 1:
            raise ValueError("data.audit_ddp_sharding_sample_reads must be positive.")

        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            self._logger.warning(
                "[RoboTwin DDP data audit] torch.distributed is not initialized; "
                "skipping sharding audit."
            )
            return

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        if world_size == 1:
            self._logger.info(
                "[RoboTwin DDP data audit] world_size=1; no cross-rank overlap to check."
            )
            return

        torch_loader = self._get_openpi_torch_loader(openpi_loader)
        sampler = getattr(torch_loader, "sampler", None)
        if not isinstance(sampler, torch.utils.data.DistributedSampler):
            raise RuntimeError(
                "[RoboTwin DDP data audit] expected OpenPI's PyTorch loader to use "
                "DistributedSampler, but got "
                f"{type(sampler)!r}."
            )

        # Materializing the sampler is metadata-only: no video or image sample
        # is loaded. It gives an exact, whole-epoch cross-rank overlap check.
        sampler_indices = [int(index) for index in sampler]
        local_report: dict[str, Any]
        try:
            probe_indices = sampler_indices[:num_sample_reads]
            index_mapping = self._probe_openpi_dataset_indices(
                torch_loader.dataset, probe_indices
            )
            local_report = {
                "rank": rank,
                "sampler_count": len(sampler_indices),
                "sampler_indices": sampler_indices,
                "index_mapping": index_mapping,
            }
        except Exception as exc:
            # Every rank still participates in the collective below, preventing
            # a peer from hanging if only one rank cannot read a probe sample.
            local_report = {
                "rank": rank,
                "sampler_count": len(sampler_indices),
                "sampler_indices": sampler_indices,
                "error": f"{type(exc).__name__}: {exc}",
            }

        reports: list[dict[str, Any] | None] = [None] * world_size
        torch.distributed.all_gather_object(reports, local_report)
        complete_reports = [report for report in reports if report is not None]

        errors = [
            f"rank {report['rank']}: {report['error']}"
            for report in complete_reports
            if "error" in report
        ]
        sampler_overlaps = self._find_pairwise_overlaps(
            [report["sampler_indices"] for report in complete_reports]
        )
        mapping_mismatches = [
            (report["rank"], expected, observed_indices)
            for report in complete_reports
            if "index_mapping" in report
            for expected, observed_indices in report["index_mapping"]
            if expected not in observed_indices
        ]

        if rank == 0:
            counts = [report["sampler_count"] for report in complete_reports]
            previews = [
                (report["rank"], report.get("index_mapping", [])[:4])
                for report in complete_reports
            ]
            self._logger.info(
                "[RoboTwin DDP data audit] dataset_size=%s, world_size=%s, "
                "sampler=%s, drop_last=%s, per-rank sampler counts=%s",
                len(torch_loader.dataset),
                world_size,
                type(sampler).__name__,
                sampler.drop_last,
                counts,
            )
            self._logger.info(
                "[RoboTwin DDP data audit] actual __getitem__ index probes "
                "(rank, sampler_index->LeRobot_indices): %s",
                previews,
            )

        if errors or sampler_overlaps or mapping_mismatches:
            details = []
            if errors:
                details.append("probe read errors: " + "; ".join(errors))
            if sampler_overlaps:
                details.append(
                    "sampler index overlap: "
                    + self._format_overlaps(sampler_overlaps)
                )
            if mapping_mismatches:
                details.append(
                    "dataset ignored/remapped sampler index: "
                    + repr(mapping_mismatches[:8])
                )
            raise RuntimeError("[RoboTwin DDP data audit] FAILED: " + " | ".join(details))

        if rank == 0:
            self._logger.info(
                "[RoboTwin DDP data audit] PASS: all sampler indices are disjoint "
                "and every checked dataset read used its sampler index."
            )

    @staticmethod
    def _get_openpi_torch_loader(openpi_loader: Any) -> torch.utils.data.DataLoader:
        """Unwrap OpenPI's public DataLoaderImpl into the backing PyTorch loader."""
        candidates = (
            openpi_loader,
            getattr(openpi_loader, "_data_loader", None),
            getattr(getattr(openpi_loader, "_data_loader", None), "torch_loader", None),
        )
        for candidate in candidates:
            if isinstance(candidate, torch.utils.data.DataLoader):
                return candidate
        raise RuntimeError(
            "[RoboTwin DDP data audit] could not find the backing "
            "torch.utils.data.DataLoader in OpenPI's loader wrapper."
        )

    @staticmethod
    def _probe_openpi_dataset_indices(
        dataset: Any, sampler_indices: Iterable[int]
    ) -> list[tuple[int, list[int]]]:
        """Record raw LeRobot indices used by real transformed-dataset reads."""
        owner = dataset
        seen: set[int] = set()
        while not hasattr(owner, "hf_dataset"):
            if id(owner) in seen:
                raise RuntimeError("cyclic OpenPI dataset wrapper chain")
            seen.add(id(owner))
            owner = getattr(owner, "_dataset", None)
            if owner is None:
                raise RuntimeError(
                    "could not find a LeRobot dataset under OpenPI's transform wrappers"
                )

        original_hf_dataset = owner.hf_dataset
        recorder = _IndexRecordingDataset(original_hf_dataset)
        owner.hf_dataset = recorder
        mappings: list[tuple[int, list[int]]] = []
        try:
            for sampler_index in sampler_indices:
                before = len(recorder.requested_indices)
                dataset[sampler_index]
                read_indices = sorted(
                    {
                        index
                        for request in recorder.requested_indices[before:]
                        for index in request
                    }
                )
                if sampler_index not in read_indices:
                    raise RuntimeError(
                        "raw LeRobot reads did not include the sampler index "
                        f"{sampler_index}; got {read_indices}"
                    )
                mappings.append((sampler_index, read_indices))
        finally:
            owner.hf_dataset = original_hf_dataset
        return mappings

    @staticmethod
    def _find_pairwise_overlaps(
        index_lists: list[list[int]],
    ) -> list[tuple[int, int, list[int]]]:
        overlaps = []
        for left_rank, right_rank in itertools.combinations(range(len(index_lists)), 2):
            intersection = set(index_lists[left_rank]) & set(index_lists[right_rank])
            if intersection:
                overlaps.append((left_rank, right_rank, sorted(intersection)[:8]))
        return overlaps

    @staticmethod
    def _format_overlaps(overlaps: list[tuple[int, int, list[int]]]) -> str:
        return "; ".join(
            f"ranks {left_rank}/{right_rank}: {indices}"
            for left_rank, right_rank, indices in overlaps
        )

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
            if hasattr(self.data_loader, "_data_loader"):
                num_batches = len(self._openpi_pytorch_dataloader(self.data_loader))
                return max(1, num_batches // self.gradient_accumulation)
            return max(1, len(self.data_loader) // self.gradient_accumulation)
        if model_type == SupportedModel.OPENPI:
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
