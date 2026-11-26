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

"""Materialize captured SGLang image inputs for Megatron Qwen-VL microbatches."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from typing import ContextManager

import torch

from rlinf.utils.image_replay import REPLAY_FORMAT, replay_pixels, replay_positions


def materialize_multimodal_inputs(
    payloads: list[dict | None],
    device: str | torch.device,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build one microbatch's pixels, grids, image mask, and M-RoPE positions."""
    if not payloads:
        payloads = [None] * input_ids.shape[0]
    if len(payloads) != input_ids.shape[0]:
        raise ValueError("Multimodal media rows must align with the training batch")

    pixels: list[torch.Tensor] = []
    grids: list[torch.Tensor] = []
    positions = torch.zeros((3, *input_ids.shape), dtype=torch.long)
    image_mask: torch.Tensor | None = None

    for row, entry in enumerate(payloads):
        row_mask = attention_mask[row].bool().cpu()
        row_ids = input_ids[row].detach().cpu()[row_mask].tolist()
        if not entry:
            positions[:, row, row_mask] = torch.arange(len(row_ids))
            continue
        if entry.get("format") != REPLAY_FORMAT:
            raise ValueError(
                f"Unsupported multimodal replay format: {entry.get('format')!r}"
            )
        positions[:, row, row_mask] = replay_positions(entry, row_ids)
        if image_mask is None:
            image_mask = torch.zeros_like(
                attention_mask, device="cpu", dtype=torch.bool
            )
        mask = torch.zeros(len(row_ids), dtype=torch.bool)
        for start, end in entry["image_spans"]:
            mask[start:end] = True
        image_mask[row, row_mask] = mask
        pixels.append(replay_pixels(entry))
        grids.append(entry["image_grid_thw"])

    result: dict[str, torch.Tensor] = {"multimodal_position_ids": positions.to(device)}
    if image_mask is not None:
        result["image_input_mask"] = image_mask.to(device)
    if pixels:
        result["pixel_values"] = torch.cat(pixels).to(device)
        result["image_grid_thw"] = torch.cat(grids).to(device)
    return result


class MultimodalBatchPreparer:
    """Prepare learner image tensors on CPU, optionally prefetched on one thread."""

    def __init__(
        self,
        model_path: str,
        *,
        prefetch_batches: int = 0,
        prefetch_bytes: int = 0,
    ) -> None:
        self.model_path = model_path
        self.prefetch_batches = prefetch_batches
        self.prefetch_bytes = prefetch_bytes

    @staticmethod
    def dense_token_inputs(batch: dict) -> dict:
        if torch.is_tensor(batch["input_ids"]):
            return batch
        from torch.nn.utils.rnn import pad_sequence

        return {
            **batch,
            "input_ids": pad_sequence(batch["input_ids"], batch_first=True),
            "attention_mask": pad_sequence(batch["attention_mask"], batch_first=True),
        }

    def for_forward(
        self, batch: dict, device: str | torch.device
    ) -> dict[str, torch.Tensor]:
        if self.prefetch_batches:
            prepared = batch.pop("_prepared_multimodal_inputs")
            return {key: value.to(device) for key, value in prepared.items()}
        return materialize_multimodal_inputs(
            batch.get("multi_modal_inputs") or [],
            device,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

    def iterator_context(
        self, iterator: Iterator[dict]
    ) -> ContextManager[Iterator[dict]]:
        if not self.prefetch_batches:
            return nullcontext(iterator)
        return self._prefetch_context(iterator)

    def _result_bytes(self, batch: dict) -> int:
        ids = batch["input_ids"]
        token_count = (
            ids.numel() if torch.is_tensor(ids) else len(ids) * max(map(len, ids))
        )
        size = 3 * token_count * 8
        entries = batch.get("multi_modal_inputs") or []
        if any(entries):
            size += token_count
        for entry in entries:
            if not entry:
                continue
            size += (
                entry["pixel_values_bytes"]
                if "pixel_refs" in entry
                else entry["pixel_values"].numel() * 2
            )
            size += entry["image_grid_thw"].nbytes
        if size > self.prefetch_bytes:
            raise ValueError(
                f"Image microbatch needs {size} bytes, exceeding prefetch_bytes={self.prefetch_bytes}"
            )
        return size

    def _iterate(self, iterator, executor):
        pending, reserved = deque(), 0
        batch = next(iterator, None)
        while batch is not None or pending:
            while batch is not None and len(pending) < self.prefetch_batches:
                dense = self.dense_token_inputs(batch)
                size = self._result_bytes(dense)
                if reserved + size > self.prefetch_bytes:
                    break
                cpu_batch = {
                    "input_ids": dense["input_ids"].cpu(),
                    "attention_mask": dense["attention_mask"].cpu(),
                    "multi_modal_inputs": [
                        {
                            k: v.cpu() if torch.is_tensor(v) else v
                            for k, v in entry.items()
                        }
                        if entry
                        else entry
                        for entry in batch.get("multi_modal_inputs") or []
                    ],
                }
                future = executor.submit(
                    materialize_multimodal_inputs,
                    cpu_batch["multi_modal_inputs"],
                    "cpu",
                    input_ids=cpu_batch["input_ids"],
                    attention_mask=cpu_batch["attention_mask"],
                )
                pending.append((batch, future, size))
                reserved += size
                batch = next(iterator, None)
            source, future, size = pending.popleft()
            yield {**source, "_prepared_multimodal_inputs": future.result()}
            reserved -= size

    @contextmanager
    def _prefetch_context(self, iterator):
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rlinf-images")
        batches = self._iterate(iter(iterator), executor)
        try:
            yield batches
        finally:
            batches.close()
            executor.shutdown(wait=True, cancel_futures=True)
