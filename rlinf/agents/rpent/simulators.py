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


"""Task pools and complete-group quotas for mixed RPent training."""

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset

_SIMULATORS = {"libero", "robotwin", "robocasa", "robodojo"}


@dataclass(frozen=True)
class Simulator:
    endpoint: str
    train_data_paths: str | list[str]
    rollout_groups: int
    sampling_groups: int
    group_router: bool = False


def mixed_simulators(cfg: DictConfig) -> dict[str, Simulator]:
    """Read the optional mixed-training config and express quotas in groups."""
    settings = (cfg.get("env") or {}).get("rpent")
    if settings is None:
        return {}
    if not isinstance(settings, DictConfig) or not settings:
        raise ValueError(
            "env.rpent must map simulator names to task pools and endpoints"
        )
    if (
        cfg.runner.get("task_type") != "reasoning"
        or not cfg.runner.get("synchronous")
        or cfg.algorithm.get("adv_type") != "grpo_dynamic"
        or cfg.rollout.get("rollout_backend") != "sglang"
        or cfg.algorithm.get("max_num_gen_batches", 1) != 1
    ):
        raise ValueError("env.rpent requires online synchronous RPent GRPO with SGLang")

    group_size = cfg.algorithm.group_size
    timeout = (
        cfg.tools.rpent.get("full_cohort_ready_timeout_s") if cfg.get("tools") else None
    )
    if timeout is not None:
        if not isinstance(timeout, (int, float)) or not 0 < timeout < float("inf"):
            raise ValueError("full_cohort_ready_timeout_s must be finite and positive")
        if group_size < 2 or not all(
            entry.get("group_router") for entry in settings.values()
        ):
            raise ValueError(
                "Full cohort readiness requires grouped sampling through toolkit routers"
            )
    if group_size < 2 and any(entry.get("group_router") for entry in settings.values()):
        raise ValueError("Toolkit routers require group_size >= 2")
    quotas = {}
    for field, total_groups in (
        ("rollout_episodes", cfg.data.rollout_batch_size),
        (
            "sampling_episodes",
            cfg.data.get("sampling_batch_size", cfg.data.rollout_batch_size),
        ),
    ):
        values = [entry.get(field) for entry in settings.values()]
        if all(value is None for value in values):
            groups, remainder = divmod(total_groups, len(settings))
            if remainder or groups < 1:
                raise ValueError(
                    f"Specify env.rpent {field} when {total_groups} groups cannot be split equally"
                )
            values = [groups * group_size] * len(settings)
        if any(
            type(value) is not int or value <= 0 or value % group_size
            for value in values
        ):
            raise ValueError(
                f"Each env.rpent {field} must be a positive multiple of group_size={group_size}"
            )
        if sum(values) != total_groups * group_size:
            raise ValueError(
                f"env.rpent {field} must total {total_groups * group_size} episodes"
            )
        quotas[field] = [value // group_size for value in values]

    result = {}
    for index, (name, entry) in enumerate(settings.items()):
        if name not in _SIMULATORS:
            raise ValueError(f"Unknown RPent simulator {name!r}")
        endpoint = entry.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint.startswith(
            ("http://", "https://")
        ):
            raise ValueError(
                f"env.rpent.{name}.endpoint must be an HTTP toolkit endpoint"
            )
        paths = entry.get("train_data_paths")
        if not paths:
            raise ValueError(f"env.rpent.{name}.train_data_paths must name a task pool")
        effective = quotas["rollout_episodes"][index]
        candidates = quotas["sampling_episodes"][index]
        if candidates < effective:
            raise ValueError(
                f"env.rpent.{name} needs at least {effective} candidate groups"
            )
        result[name] = Simulator(
            endpoint,
            paths if isinstance(paths, str) else list(paths),
            effective,
            candidates,
            entry.get("group_router", False),
        )
    return result


def simulator_for_answer(answer: str | list[str]) -> str:
    """Identify the simulator in a task manifest or a collated answer."""
    from rlinf.agents.rpent.episode import episode_task

    return episode_task([answer] if isinstance(answer, str) else answer).get(
        "environment"
    )


def mixed_training_dataset(
    cfg: DictConfig, tokenizer, simulators: dict[str, Simulator]
):
    """Load task pools separately, using the existing reasoning item format."""
    from rlinf.data.datasets.reasoning.dataset import ReasoningDataset

    pools = []
    for name, simulator in simulators.items():
        pool = ReasoningDataset(simulator.train_data_paths, cfg, tokenizer)
        for row in pool.data:
            if simulator_for_answer(row[pool.answer_key]) != name:
                raise ValueError(
                    f"Task row in {simulator.train_data_paths} must declare environment={name}"
                )
        if len(pool) < simulator.sampling_groups:
            raise ValueError(
                f"{name} needs {simulator.sampling_groups} task rows per batch, found {len(pool)}"
            )
        pools.append(pool)
    validation_paths = cfg.data.get("val_data_paths")
    validation = (
        ReasoningDataset(validation_paths, cfg, tokenizer) if validation_paths else None
    )
    return ConcatDataset(pools), validation


class SimulatorBatchSampler:
    """Draw a fixed count per simulator, stopping when any pool runs out."""

    def __init__(
        self, pool_sizes: list[int], quotas: list[int], *, shuffle: bool, seed: int
    ):
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
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        pools = []
        offset = 0
        for size in self.pool_sizes:
            indices = (
                torch.randperm(size, generator=generator).tolist()
                if self.shuffle
                else range(size)
            )
            pools.append([offset + index for index in indices])
            offset += size
        for batch in range(self.batch, len(self)):
            draws = [
                pool[batch * quota : (batch + 1) * quota]
                for pool, quota in zip(pools, self.quotas, strict=True)
            ]
            if batch + 1 == len(self):
                self.epoch, self.batch = self.epoch + 1, 0
            else:
                self.batch = batch + 1
            yield [
                draw[index]
                for index in range(max(self.quotas))
                for draw in draws
                if index < len(draw)
            ]

    def state_dict(self) -> dict[str, int]:
        """Store the cursor; the shuffle order is derived from seed and epoch."""
        return {"epoch": self.epoch, "batch": self.batch}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.epoch = state["epoch"]
        self.batch = state["batch"]
        if self.batch == len(self):
            self.epoch, self.batch = self.epoch + 1, 0
