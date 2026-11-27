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

"""Mixed-simulator task pools and per-simulator quotas for RPent GRPO."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset

_SIMULATORS = frozenset({"libero", "robotwin", "robocasa", "robodojo", "metaworld"})


@dataclass(frozen=True)
class SimulatorSpec:
    """Endpoint and per-step group quotas for one RPent simulator pool."""

    name: str
    endpoint: str
    train_data_paths: str | list[str]
    rollout_groups: int
    sampling_groups: int


def episode_task(answer: str | list[str]) -> dict:
    """Decode the task manifest carried by one reasoning dataset row."""
    if isinstance(answer, str):
        return json.loads(answer)
    if len(answer) != 1:
        raise ValueError(f"Expected 1 task manifest in answer, got {len(answer)}")
    return json.loads(answer[0])


def simulator_for_answer(answer: str | list[str]) -> str:
    return str(episode_task(answer)["environment"])


def mixed_simulators(cfg: DictConfig) -> dict[str, SimulatorSpec]:
    """Parse ``env.rpent`` simulator pools and validate rollout/sampling group quotas."""
    settings = (cfg.get("env") or {}).get("rpent")
    if not settings:
        return {}
    group_size = int(cfg.algorithm.group_size)
    rollout_total = int(cfg.data.rollout_batch_size)
    sampling_total = int(cfg.data.get("sampling_batch_size", rollout_total))

    quotas: dict[str, list[int]] = {}
    for field_name, total_groups in (
        ("rollout_episodes", rollout_total),
        ("sampling_episodes", sampling_total),
    ):
        values = [entry.get(field_name) for entry in settings.values()]
        if all(v is None for v in values):
            groups, rem = divmod(total_groups, len(settings))
            if rem or groups < 1:
                raise ValueError(
                    f"Specify env.rpent.{field_name} when {total_groups} groups do not split evenly"
                )
            values = [groups * group_size] * len(settings)
        if any(type(v) is not int or v <= 0 or v % group_size for v in values):
            raise ValueError(
                f"Each env.rpent.{field_name} must be a positive multiple of group_size={group_size}"
            )
        if sum(values) != total_groups * group_size:
            raise ValueError(
                f"env.rpent.{field_name} must total {total_groups * group_size} episodes"
            )
        quotas[field_name] = [v // group_size for v in values]

    result: dict[str, SimulatorSpec] = {}
    for idx, (name, entry) in enumerate(settings.items()):
        if name not in _SIMULATORS:
            raise ValueError(f"Unknown RPent simulator {name!r}")
        endpoint = str(entry["endpoint"]).rstrip("/")
        paths = entry["train_data_paths"]
        eff = quotas["rollout_episodes"][idx]
        cand = quotas["sampling_episodes"][idx]
        if cand < eff:
            raise ValueError(f"env.rpent.{name} needs sampling_groups >= {eff}")
        result[name] = SimulatorSpec(
            name=name,
            endpoint=endpoint,
            train_data_paths=paths if isinstance(paths, str) else list(paths),
            rollout_groups=eff,
            sampling_groups=cand,
        )
    return result


def mixed_training_dataset(
    cfg: DictConfig, tokenizer, simulators: dict[str, SimulatorSpec]
):
    """Build a ConcatDataset of per-simulator task pools."""
    from rlinf.data.datasets.reasoning.dataset import ReasoningDataset

    pools = []
    for name, spec in simulators.items():
        pool = ReasoningDataset(spec.train_data_paths, cfg, tokenizer)
        if len(pool) < spec.sampling_groups:
            raise ValueError(
                f"{name} pool has {len(pool)} rows, needs at least {spec.sampling_groups}"
            )
        pools.append(pool)
    val_paths = cfg.data.get("val_data_paths")
    val_ds = ReasoningDataset(val_paths, cfg, tokenizer) if val_paths else None
    return ConcatDataset(pools), val_ds


class SimulatorBatchSampler:
    """Interleave exact per-simulator group counts in every batch."""

    def __init__(
        self, pool_sizes: list[int], quotas: list[int], *, shuffle: bool, seed: int
    ) -> None:
        self.pool_sizes = tuple(pool_sizes)
        self.quotas = tuple(quotas)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.batch = 0

    def __len__(self) -> int:
        return min(
            size // quota
            for size, quota in zip(self.pool_sizes, self.quotas, strict=True)
        )

    def __iter__(self) -> Iterator[list[int]]:
        gen = torch.Generator().manual_seed(self.seed + self.epoch)
        pools, offset = [], 0
        for size in self.pool_sizes:
            order = (
                torch.randperm(size, generator=gen).tolist()
                if self.shuffle
                else range(size)
            )
            pools.append([offset + i for i in order])
            offset += size
        for batch in range(self.batch, len(self)):
            draws = [
                pool[batch * q : (batch + 1) * q]
                for pool, q in zip(pools, self.quotas, strict=True)
            ]
            if batch + 1 == len(self):
                self.epoch, self.batch = self.epoch + 1, 0
            else:
                self.batch = batch + 1
            yield [
                draw[i]
                for i in range(max(self.quotas))
                for draw in draws
                if i < len(draw)
            ]

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "batch": self.batch}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.epoch = state["epoch"]
        self.batch = state["batch"]
        if self.batch == len(self):
            self.epoch, self.batch = self.epoch + 1, 0
