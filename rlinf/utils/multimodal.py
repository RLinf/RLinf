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

"""Materialize captured image replay inputs and optional bounded prefetch."""

from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from typing import ContextManager

import torch

from rlinf.utils.image_replay import REPLAY_FORMAT, replay_pixels, replay_positions


def collate_fsdp_multimodal_inputs(
    payloads: list[dict | None],
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    """Collate non-empty per-row multimodal tensors for FSDP forward passes."""
    non_empty = [inputs for inputs in payloads if inputs]
    if not non_empty:
        return {}
    if any(inputs.get("format") == REPLAY_FORMAT for inputs in non_empty):
        raise ValueError(
            "Captured multimodal replay inputs require Megatron-Bridge, not FSDP."
        )
    return {
        key: torch.cat([inputs[key] for inputs in non_empty], dim=0).to(device)
        for key in non_empty[0]
    }


def materialize_multimodal_inputs(
    payloads: list[dict | None],
    device: str | torch.device,
    *,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    capture_replay: bool = False,
) -> dict[str, torch.Tensor]:
    """Build one microbatch's pixels in row order from captured or tensor payloads.

    File-backed replay rows carry SHA-256 checksums and the rollout token grid.
    A missing, changed, or mismatched replay payload is rejected before forward.
    """
    if capture_replay and (input_ids is None or attention_mask is None):
        raise ValueError(
            "capture_replay=True requires both input_ids and attention_mask."
        )
    pixels, grids = [], []
    positions = (
        torch.zeros((3, *input_ids.shape), dtype=torch.long) if capture_replay else None
    )
    if capture_replay and not payloads:
        payloads = [None] * input_ids.shape[0]
    if capture_replay and len(payloads) != input_ids.shape[0]:
        raise ValueError(
            "Captured multi_modal_inputs must align with the training batch: "
            f"expected {input_ids.shape[0]} rows, got {len(payloads)}."
        )
    image_mask = None
    for row, entry in enumerate(payloads):
        if capture_replay:
            row_mask = attention_mask[row].bool().cpu()
            row_ids = input_ids[row].detach().cpu()[row_mask].tolist()
            if not entry:
                positions[:, row, row_mask] = torch.arange(len(row_ids))
            elif entry.get("format") == REPLAY_FORMAT:
                positions[:, row, row_mask] = replay_positions(entry, row_ids)
                if image_mask is None:
                    image_mask = torch.zeros_like(
                        attention_mask, device="cpu", dtype=torch.bool
                    )
                prompt_len = len(entry["prompt_token_ids"])
                mask = torch.zeros(len(row_ids), dtype=torch.bool)
                for start, end in entry["image_spans"]:
                    if not (0 < start < end < prompt_len):
                        raise ValueError(
                            f"Image span [{start}, {end}) must lie strictly inside "
                            f"captured prompt length {prompt_len}."
                        )
                    mask[start:end] = True
                image_mask[row, row_mask] = mask
                pixels.append(replay_pixels(entry))
                grids.append(entry["image_grid_thw"])
                continue
            else:
                raise ValueError(
                    "Captured image alignment requires a captured image row; generate fresh rollouts."
                )
        elif entry and entry.get("format") == REPLAY_FORMAT:
            raise ValueError(
                "Captured image rows require rollout.sglang.capture_multi_modal_inputs=True."
            )
        if not entry:
            continue
        pixels.append(entry["pixel_values"])
        grids.append(entry["image_grid_thw"])
    result = {"multimodal_position_ids": positions.to(device)} if capture_replay else {}
    if image_mask is not None:
        result["image_input_mask"] = image_mask.to(device)
    if not pixels:
        return result
    return {
        **result,
        "pixel_values": torch.cat(pixels).to(device),
        "image_grid_thw": torch.cat(grids).to(device),
    }


class MultimodalBatchPreparer:
    """Prepare learner pixels on CPU, optionally ahead of model forward.

    ``prefetch_batches`` caps how many upcoming microbatches may queue ahead of
    the active microbatch, while ``prefetch_bytes`` bounds the combined prepared
    bytes of the active and queued microbatches. Futures live only inside a
    forward/backward schedule, after distributed exchange.
    """

    def __init__(
        self,
        *,
        capture_replay: bool = False,
        prefetch_batches: int = 0,
        prefetch_bytes: int = 0,
    ):
        self.capture_replay = capture_replay
        self.prefetch_batches = prefetch_batches
        self.prefetch_bytes = prefetch_bytes

    @staticmethod
    def dense_token_inputs(batch: dict) -> dict:
        """Pad only the current image microbatch before Bridge embeds its inputs."""
        if torch.is_tensor(batch["input_ids"]):
            return batch
        from torch.nn.utils.rnn import pad_sequence

        tokens = pad_sequence(batch["input_ids"], batch_first=True)
        if "attention_mask" in batch:
            mask = pad_sequence(batch["attention_mask"], batch_first=True)
        else:
            lengths = torch.tensor(
                [len(row) for row in batch["input_ids"]], device=tokens.device
            )
            mask = (
                torch.arange(tokens.shape[1], device=tokens.device)[None, :]
                < lengths[:, None]
            )
        return {
            **batch,
            "input_ids": tokens,
            "attention_mask": mask,
        }

    def for_forward(
        self, batch: dict, device: str | torch.device
    ) -> dict[str, torch.Tensor]:
        """Move only this microbatch's prepared inputs onto the training device."""
        if self.prefetch_batches:
            prepared = batch.pop("_prepared_multimodal_inputs")
            return {key: value.to(device) for key, value in prepared.items()}
        return self._materialize(self.dense_token_inputs(batch), device)

    def _materialize(self, batch: dict, device) -> dict[str, torch.Tensor]:
        return materialize_multimodal_inputs(
            batch.get("multi_modal_inputs") or [],
            device,
            input_ids=batch.get("input_ids"),
            attention_mask=batch.get("attention_mask"),
            capture_replay=self.capture_replay,
        )

    def iterator_context(
        self, iterator: Iterator[dict]
    ) -> ContextManager[Iterator[dict]]:
        """Own and drain the prefetch thread even when model forward fails."""
        if not self.prefetch_batches:
            return nullcontext(iterator)
        return self._prefetch_context(iterator)

    def _result_bytes(self, batch: dict) -> int:
        size = 3 * batch["input_ids"].numel() * 8 if self.capture_replay else 0
        entries = batch.get("multi_modal_inputs") or []
        if any(entry and entry.get("format") == REPLAY_FORMAT for entry in entries):
            size += batch["input_ids"].numel()
        for entry in entries:
            if not entry:
                continue
            if "pixel_refs" in entry:
                if "pixel_values_bytes" not in entry:
                    raise ValueError(
                        "Image prefetch needs pixel_values_bytes; generate fresh "
                        "captured rollouts or disable multi_modal.prefetch_batches."
                    )
                size += entry["pixel_values_bytes"]
            elif entry.get("format") == REPLAY_FORMAT:
                size += entry["pixel_values"].numel() * torch.bfloat16.itemsize
            else:
                size += entry["pixel_values"].nbytes
            size += entry["image_grid_thw"].nbytes
        if size > self.prefetch_bytes:
            raise ValueError(
                f"Image microbatch needs {size} prepared bytes, exceeding "
                f"multi_modal.prefetch_bytes={self.prefetch_bytes}; reduce the "
                "microbatch or increase this per-rank budget."
            )
        return size

    def _prepare(self, batch: dict, size: int) -> dict[str, torch.Tensor]:
        prepared = self._materialize(batch, "cpu")
        actual = sum(tensor.nbytes for tensor in prepared.values())
        if actual != size:
            raise ValueError(
                f"Prepared image size {actual} differs from recorded size {size}."
            )
        return prepared

    def _iterate(self, iterator, executor):
        pending, reserved = deque(), 0
        batch = next(iterator, None)
        while batch is not None or pending:
            while batch is not None and len(pending) <= self.prefetch_batches:
                dense = self.dense_token_inputs(batch)
                size = self._result_bytes(dense)
                if reserved + size > self.prefetch_bytes:
                    break
                # Distributed tensors return to CPU on the caller's thread.
                cpu_batch = {
                    "input_ids": dense["input_ids"].cpu(),
                    "attention_mask": dense["attention_mask"].cpu(),
                    "multi_modal_inputs": [
                        {
                            key: value.cpu() if torch.is_tensor(value) else value
                            for key, value in entry.items()
                        }
                        if entry
                        else entry
                        for entry in batch.get("multi_modal_inputs") or []
                    ],
                }
                future = executor.submit(self._prepare, cpu_batch, size)
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
