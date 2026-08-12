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

"""STEAM SFT worker backed by frozen RLT Stage 1 features."""

import copy
import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import open_dict

from rlinf.data.datasets.steam import (
    PairMixtureDataset,
    RLTChunkPairCollator,
    RLTChunkPairDataset,
)
from rlinf.models import get_model
from rlinf.workers.sft.fsdp_steam_sft_worker import (
    FSDPSteamSftWorker,
    _PairDataLoaderImpl,
)


class FSDPRLTSteamSftWorker(FSDPSteamSftWorker):
    """Reuse STEAM's FSDP/ensemble loop with RLT feature-pair inputs."""

    def __init__(self, cfg):
        self.rlt_feature_model = None
        super().__init__(cfg)

    def init_worker(self):
        super().init_worker()
        feature_cfg = self.cfg.data.get("rlt_feature_model", None)
        if feature_cfg is None:
            raise ValueError(
                "RLT STEAM training requires data.rlt_feature_model to point to "
                "the frozen RLT Stage 1 checkpoint."
            )
        self.rlt_feature_model = get_model(copy.deepcopy(feature_cfg)).to(self.device)
        self.rlt_feature_model.eval()
        self.rlt_feature_model.requires_grad_(False)

    def build_dataloader(self):
        data_cfg = self.cfg.data
        model_cfg = self.cfg.actor.model
        chunk_size = int(data_cfg.get("chunk_size", 10))
        k = int(data_cfg.get("k", 4))
        num_bins = int(model_cfg.get("num_bins", 8))
        with open_dict(model_cfg):
            model_cfg.stride_k = k
        collator = RLTChunkPairCollator()

        data_root = data_cfg.get("data_root", None)

        def _resolve(path: str) -> str:
            if data_root and not os.path.isabs(path):
                return os.path.join(data_root, path)
            return path

        def _build(entry: dict[str, Any]) -> RLTChunkPairDataset:
            dataset_type = str(
                entry.get("type", data_cfg.get("dataset_type", "sft"))
            ).lower()
            only_success = bool(
                entry.get("only_success", data_cfg.get("only_success", True))
            )
            return RLTChunkPairDataset(
                _resolve(entry["dataset_path"]),
                chunk_size=chunk_size,
                k=k,
                num_bins=num_bins,
                dataset_type=dataset_type,
                only_success=only_success,
                main_image_key=str(data_cfg.get("main_image_key", "image")),
                wrist_image_key=str(data_cfg.get("wrist_image_key", "wrist_image")),
                state_key=str(data_cfg.get("state_key", "state")),
                default_prompt=str(
                    data_cfg.get("default_prompt", "insert the peg in the hole")
                ),
            )

        train_entries = [
            dict(entry)
            for entry in data_cfg.get("train_data_paths", [])
            if entry.get("dataset_path")
        ]
        if not train_entries:
            raise ValueError("data.train_data_paths must contain a dataset_path")
        datasets_with_weights = [
            (_build(entry), float(entry.get("weight", 1.0))) for entry in train_entries
        ]
        if len(datasets_with_weights) == 1:
            train_dataset: torch.utils.data.Dataset = datasets_with_weights[0][0]
        else:
            train_dataset = PairMixtureDataset(
                datasets=datasets_with_weights,
                mode="train",
                balance_dataset_weights=bool(data_cfg.get("balance_weights", True)),
                seed=int(data_cfg.get("seed", 42)),
            )

        def _worker_kwargs(num_workers: int) -> dict[str, Any]:
            kwargs: dict[str, Any] = {
                "num_workers": num_workers,
                "pin_memory": bool(data_cfg.get("pin_memory", False)),
            }
            if num_workers > 0:
                kwargs["persistent_workers"] = bool(
                    data_cfg.get("persistent_workers", False)
                )
                prefetch_factor = data_cfg.get("prefetch_factor", 2)
                if prefetch_factor is not None:
                    kwargs["prefetch_factor"] = int(prefetch_factor)
            return kwargs

        train_sampler = None
        if torch.distributed.is_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=self._world_size,
                rank=self._rank,
                shuffle=True,
                drop_last=True,
            )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=int(self.cfg.actor.micro_batch_size),
            shuffle=train_sampler is None,
            sampler=train_sampler,
            drop_last=True,
            collate_fn=collator,
            **_worker_kwargs(int(data_cfg.get("train_num_workers", 0))),
        )

        eval_loaders: list[tuple[str, _PairDataLoaderImpl]] = []
        for raw_entry in data_cfg.get("eval_data_paths", []) or []:
            entry = dict(raw_entry)
            if not entry.get("dataset_path"):
                continue
            dataset = _build(entry)
            eval_sampler = None
            if torch.distributed.is_initialized():
                eval_sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset,
                    num_replicas=self._world_size,
                    rank=self._rank,
                    shuffle=False,
                    drop_last=False,
                )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=int(self.cfg.actor.micro_batch_size),
                shuffle=False,
                sampler=eval_sampler,
                drop_last=False,
                collate_fn=collator,
                **_worker_kwargs(int(data_cfg.get("eval_num_workers", 0))),
            )
            name = str(entry.get("name", Path(entry["dataset_path"]).stem))
            eval_loaders.append((name, _PairDataLoaderImpl(loader)))
        return _PairDataLoaderImpl(train_loader), eval_loaders

    def _prepare_input(self, batch: dict[str, Any]):
        if self.rlt_feature_model is None:
            raise RuntimeError("RLT Stage 1 feature model has not been initialized")

        def _to_device(value):
            if isinstance(value, torch.Tensor):
                return value.to(self.device, non_blocking=True)
            if isinstance(value, dict):
                return {key: _to_device(child) for key, child in value.items()}
            return value

        obs_t = _to_device(batch["obs_t"])
        obs_tk = _to_device(batch["obs_tk"])
        with torch.no_grad():
            features_t = self.rlt_feature_model.extract_rlt_obs(obs_t)
            features_tk = self.rlt_feature_model.extract_rlt_obs(obs_tk)
        observation = {
            "z_rl_t": features_t["z_rl"].detach(),
            "proprio_t": features_t["proprio"].detach(),
            "z_rl_tk": features_tk["z_rl"].detach(),
            "proprio_tk": features_tk["proprio"].detach(),
        }
        labels = batch["labels"].to(self.device, non_blocking=True)
        return observation, labels

    def _log_training_batch_diagnostics_once(
        self,
        observation: dict[str, Any],
        labels: torch.Tensor,
        result,
    ) -> None:
        del observation, labels, result


__all__ = ["FSDPRLTSteamSftWorker"]
