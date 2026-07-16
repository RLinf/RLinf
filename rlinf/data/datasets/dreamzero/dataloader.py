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

"""DreamZero SFT dataloader builder and batch collator."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.data.datasets.dreamzero.data_transforms import (
    format_training_prompt,
    normalize_instruction_text,
)
from rlinf.utils.logging import get_logger

logger = get_logger()


class DreamZeroCollator:
    """Stack transformed samples and tokenize text (Groot ``DefaultDataCollator``-style)."""

    def __init__(
        self,
        tokenizer_path: str,
        max_seq_len: int,
        embodiment_tag_mapping: dict[str, int],
    ):
        from groot.vla.model.dreamzero.transform.dreamzero_cotrain import (
            HuggingfaceTokenizer,
        )

        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path,
            seq_len=max_seq_len,
            clean="whitespace",
        )
        self.embodiment_tag_mapping = embodiment_tag_mapping

    @staticmethod
    def collate_batch(
        features: list[dict[str, Any]],
        tokenizer: Any,
        embodiment_tag_mapping: dict[str, int],
    ) -> dict[str, Any]:
        batch: dict[str, Any] = {}
        for key in features[0]:
            if key == "text":
                texts = [
                    format_training_prompt(
                        normalize_instruction_text(elem[key]),
                        int(elem["embodiment_id"]),
                        embodiment_tag_mapping,
                    )
                    for elem in features
                ]
                ids, mask = tokenizer(texts, return_mask=True, add_special_tokens=True)
                batch[key] = ids
                batch["text_attention_mask"] = mask
            elif key == "text_negative":
                values = [elem[key] for elem in features]
                ids, mask = tokenizer(values, return_mask=True, add_special_tokens=True)
                batch[key] = ids
                batch["text_attention_mask_negative"] = mask
            else:
                values = [elem[key] for elem in features]
                try:
                    batch[key] = torch.from_numpy(np.stack(values))
                except ValueError as e:
                    shapes = [np.asarray(v).shape for v in values]
                    raise ValueError(
                        f"Shape mismatch in collate for key='{key}': shapes={shapes}"
                    ) from e
        return batch

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        return self.collate_batch(features, self.tokenizer, self.embodiment_tag_mapping)


def build_dreamzero_sft_dataloader(
    cfg,
    world_size: int,
    rank: int,
    data_paths,
    eval_dataset: bool = False,
):
    """Build DreamZero SFT dataloader -- callable from FSDPVlaSftWorker.

    Uses DistributedSampler to shard data across GPUs:
      - Each of the 8 GPUs sees 1/8 of the dataset per epoch
      - micro_batch_size samples are returned per iteration per GPU
      - Global effective batch size = micro_batch_size * world_size * grad_accum_steps
    """
    from rlinf.data.datasets.dreamzero.data_transforms import (
        embodiment_tag_mapping_for_embodiment,
    )
    from rlinf.data.datasets.dreamzero.lerobot_dataset import (
        build_dreamzero_mixture_dataset_from_spec,
        build_single_dreamzero_lerobot_dataset,
        is_dreamzero_mixture_spec,
    )

    model_cfg = cfg.actor.model
    tokenizer_path = model_cfg.get("tokenizer_path", "google/umt5-xxl")
    max_seq_len = int(model_cfg.get("max_seq_len", 512))

    if is_dreamzero_mixture_spec(data_paths):
        dataset, embodiment_tag_mapping = build_dreamzero_mixture_dataset_from_spec(
            cfg,
            data_paths,
            eval_dataset=eval_dataset,
        )
        logger.info("DreamZero mixture dataset:\n%s", dataset)
    else:
        embodiment_tag = model_cfg.embodiment_tag
        dataset = build_single_dreamzero_lerobot_dataset(
            data_path=str(data_paths),
            cfg=cfg,
            eval_dataset=eval_dataset,
        )
        embodiment_tag_mapping = embodiment_tag_mapping_for_embodiment(
            embodiment_tag, model_cfg.get("embodiment_tag_mapping")
        )

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=not eval_dataset,
        drop_last=not eval_dataset,
    )
    num_workers = int(cfg.data.get("num_workers", 4))
    prefetch_factor = int(cfg.data.get("prefetch_factor", 4))
    transform_on_gpu = bool(cfg.data.get("transform_on_gpu", False))
    data_loader = StatefulDataLoader(
        dataset,
        batch_size=cfg.actor.micro_batch_size,  # samples per GPU per step
        sampler=sampler,
        drop_last=not eval_dataset,
        num_workers=num_workers,
        pin_memory=True,  # faster CPU->GPU transfer
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        multiprocessing_context="spawn"
        if transform_on_gpu and num_workers > 0
        else None,
        collate_fn=DreamZeroCollator(
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            embodiment_tag_mapping=dict(embodiment_tag_mapping),
        ),
    )
    return data_loader, {"num_samples": len(dataset)}
