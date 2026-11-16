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
"""Use GR00T's official sharded LeRobot input pipeline for SFT."""

import json
from pathlib import Path
from typing import Any

import torch
from torchdata.stateful_dataloader import StatefulDataLoader


def build_gr00t_sft_dataloader(
    cfg, world_size: int, rank: int, data_paths: Any, eval_dataset: bool = False
):
    """Build the upstream streaming loader and retain its trained processor.

    The upstream mixture owns rank/worker sharding and runs indefinitely. Use an
    explicit update budget; StatefulDataLoader restores this stream by replay,
    rather than promising bitwise restoration of random image augmentation.
    """
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.stats import generate_rel_stats, generate_stats
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    if eval_dataset:
        raise NotImplementedError(
            "GR00T's streaming SFT dataset does not provide validation splits"
        )
    if cfg.runner.max_steps <= 0 or cfg.runner.max_epochs != -1:
        raise ValueError(
            "GR00T streaming SFT requires runner.max_steps > 0 and max_epochs=-1"
        )
    if world_size > 1 and not torch.distributed.is_initialized():
        raise RuntimeError(
            "GR00T distributed input sharding requires an initialized process group"
        )
    model_cfg, data_cfg = cfg.actor.model, cfg.data
    tag = EmbodimentTag(model_cfg.embodiment_tag)
    modality_configs = MODALITY_CONFIGS[tag.value]
    resume_dir = cfg.runner.get("resume_dir")
    processor_path = (
        str(Path(resume_dir) / "actor")
        if resume_dir
        else model_cfg.get("processor_path", model_cfg.model_path)
    )
    processor = Gr00tN1d7Processor.from_pretrained(
        processor_path,
        modality_configs={tag.value: modality_configs},
        model_name=model_cfg.get("backbone_model_path", "nvidia/Cosmos-Reason2-2B"),
    )
    paths = [data_paths] if isinstance(data_paths, str) else list(data_paths)
    datasets = []
    for path in paths:
        if data_cfg.get("format") == "physical_libero":
            from .libero import LiberoEpisodeLoader, LiberoParquetDataset

            task_names = data_cfg.get("task_names")
            reader = LiberoEpisodeLoader(Path(path), task_names)
            statistics_path = Path(data_cfg.statistics_path)
            sources = [
                (
                    episode["episode_index"],
                    reader.episode_path(index).stat().st_size,
                    reader.episode_path(index).stat().st_mtime_ns,
                )
                for index, episode in enumerate(reader.episodes)
            ]
            sources = [list(source) for source in sources]
            if rank == 0:
                cached = (
                    json.loads(statistics_path.read_text())
                    if statistics_path.exists()
                    else None
                )
                if cached is None or cached["sources"] != sources:
                    statistics_path.parent.mkdir(parents=True, exist_ok=True)
                    statistics_path.write_text(
                        json.dumps(
                            {
                                "sources": sources,
                                "statistics": reader.compute_statistics(),
                            }
                        )
                    )
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            statistics = json.loads(statistics_path.read_text())["statistics"]
            datasets.append(
                LiberoParquetDataset(
                    path=path,
                    embodiment_tag=tag,
                    modality_configs=modality_configs,
                    statistics=statistics,
                    task_names=task_names,
                    shard_size=data_cfg.get("shard_size", 1024),
                    episode_sampling_rate=data_cfg.get("episode_sampling_rate", 1.0),
                    seed=cfg.actor.seed,
                    allow_padding=data_cfg.get("allow_padding", False),
                )
            )
            continue
        if rank == 0:
            generate_stats(path)
            generate_rel_stats(path, tag)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        datasets.append(
            ShardedSingleStepDataset(
                dataset_path=Path(path),
                embodiment_tag=tag,
                modality_configs=modality_configs,
                video_backend=data_cfg.get("video_backend", "torchcodec"),
                shard_size=data_cfg.get("shard_size", 1024),
                episode_sampling_rate=data_cfg.get("episode_sampling_rate", 1.0),
                seed=cfg.actor.seed,
                allow_padding=data_cfg.get("allow_padding", False),
            )
        )
    # Match DatasetFactory: paths within one embodiment are weighted by size.
    sizes = [sum(dataset.shard_lengths) for dataset in datasets]
    total = sum(sizes)
    if total == 0:
        raise ValueError("GR00T SFT requires at least one valid action chunk")
    dataset = ShardedMixtureDataset(
        datasets=datasets,
        weights=[size / total for size in sizes],
        processor=processor,
        seed=cfg.actor.seed,
        training=True,
        num_shards_per_epoch=data_cfg.get("num_shards_per_epoch", 10000),
        override_pretraining_statistics=(
            False
            if resume_dir
            else data_cfg.get("override_pretraining_statistics", True)
        ),
    )
    processor.train()
    workers = data_cfg.get("num_workers", 4)
    if len(dataset.shard_sampling_schedule) < world_size * max(1, workers):
        raise ValueError(
            "num_shards_per_epoch must cover every distributed dataloader worker"
        )
    loader = StatefulDataLoader(
        dataset,
        batch_size=cfg.actor.micro_batch_size,
        num_workers=workers,
        collate_fn=processor.collator,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 0,
        **({"prefetch_factor": data_cfg.get("prefetch_factor", 2)} if workers else {}),
    )
    return loader, {"processor": processor, "num_frames": total}
