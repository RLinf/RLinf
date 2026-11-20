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

"""Materialize immutable image references for learner microbatches."""

import asyncio
import hashlib
import io
import json
from collections import OrderedDict, deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock
from typing import ContextManager

import torch

from rlinf.hybrid_engines.sglang.multimodal.replay import (
    REPLAY_FORMAT,
    replay_pixels,
    replay_positions,
)


def store_image_refs(images: list[bytes], directory: Path) -> list[dict[str, str]]:
    """Publish images by content hash and return references in prompt order."""
    directory.mkdir(parents=True, exist_ok=True)
    references = []
    for image in images:
        digest = hashlib.sha256(image).hexdigest()
        path = directory / digest
        if not path.exists():
            # Concurrent writers have identical bytes. Publish only complete files.
            with NamedTemporaryFile(dir=directory, delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    stream.write(image)
                    stream.flush()
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
        references.append({"path": str(path.resolve()), "sha256": digest})
    return references


@lru_cache(maxsize=2)
def _image_processor(model_path: str):
    """Load the policy's image processor only for file-backed training rows."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True
    ).image_processor


class _ProcessedImageCache:
    """A process-local, byte-bounded cache of immutable CPU processor outputs."""

    def __init__(self):
        self.entries = OrderedDict()
        self.bytes = 0
        self.lock = RLock()

    def trim(self, budget):
        with self.lock:
            while self.entries and self.bytes > budget:
                _, (pixels, grid) = self.entries.popitem(last=False)
                self.bytes -= pixels.nbytes + grid.nbytes

    def get(self, key):
        with self.lock:
            if key not in self.entries:
                return None
            self.entries.move_to_end(key)
            return self.entries[key]

    def put(self, key, value, budget):
        size = sum(tensor.nbytes for tensor in value)
        if size > budget:
            return
        with self.lock:
            previous = self.entries.pop(key, None)
            if previous is not None:
                self.bytes -= sum(tensor.nbytes for tensor in previous)
            self.entries[key] = value
            self.bytes += size
            self.trim(budget)


_processed_image_cache = _ProcessedImageCache()


def _materialize_image_refs(entry, model_path, cache_bytes, processor=None):
    """Read immutable media and cache the complete, validated image sequence."""
    grid = entry["image_grid_thw"]
    token_space = processor is not None
    from PIL import Image

    contents = []
    for reference in entry["image_refs"]:
        content = Path(reference["path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != reference["sha256"]:
            raise ValueError(f"Training image checksum mismatch: {reference['path']}")
        contents.append(content)
    if not token_space:
        processor = _image_processor(model_path)
    # Cache the complete image sequence: generic processors can depend
    # on batch composition. Never assume per-image preprocessing commutes.
    cached = None
    if cache_bytes:
        key = (
            model_path,
            id(processor),
            json.dumps(
                {
                    key: value
                    for key, value in entry.items()
                    if key not in ("image_refs", "image_grid_thw")
                },
                sort_keys=True,
            )
            if token_space
            else None,
            tuple(grid.flatten().tolist()),
            json.dumps(processor.to_dict(), sort_keys=True)
            if hasattr(processor, "to_dict")
            else None,
            tuple(reference["sha256"] for reference in entry["image_refs"]),
        )
        cached = _processed_image_cache.get(key)
    if cached is None:
        images = []
        if token_space and processor.cache_directory is not None:
            # Use the same immutable sources as serving, before decoding them.
            images = [reference["path"] for reference in entry["image_refs"]]
        else:
            for content in contents:
                with Image.open(io.BytesIO(content)) as image:
                    # Preserve alpha/mode for SGLang's own conversion rules.
                    images.append(image.copy() if token_space else image.convert("RGB"))
        if token_space:
            cached = asyncio.run(processor.replay(entry, images))
        else:
            processed = processor(images=images, return_tensors="pt")
            cached = (
                processed["pixel_values"].to(torch.bfloat16),
                processed["image_grid_thw"],
            )
        if cache_bytes:
            _processed_image_cache.put(key, cached, cache_bytes)
    if not torch.equal(grid, cached[1]):
        raise ValueError("Training image grid differs from the recorded rollout grid")
    return cached[0]


def materialize_multimodal_inputs(
    payloads: list[dict],
    model_path: str,
    device,
    cache_bytes: int = 0,
    *,
    input_ids=None,
    attention_mask=None,
    token_space: bool = False,
) -> dict:
    """Build one microbatch's pixels in row order from tensors or immutable images.

    File-backed rows carry checksums and the rollout token grid. A missing,
    changed or differently processed image is rejected before model forward.
    """
    pixels, grids = [], []
    positions = (
        torch.zeros((3, *input_ids.shape), dtype=torch.long) if token_space else None
    )
    if token_space and not payloads:
        payloads = [None] * input_ids.shape[0]
    if token_space and len(payloads) != input_ids.shape[0]:
        raise ValueError("Token-space media rows must align with the training batch")
    image_mask = None
    _processed_image_cache.trim(cache_bytes)
    for row, entry in enumerate(payloads):
        if token_space:
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
                mask = torch.zeros(len(row_ids), dtype=torch.bool)
                for start, end in entry["image_spans"]:
                    mask[start:end] = True
                image_mask[row, row_mask] = mask
                pixels.append(replay_pixels(entry))
                grids.append(entry["image_grid_thw"])
                continue
            elif entry.get("format") != "sglang-token-space-v1":
                raise ValueError(
                    "Legacy media cannot enter a token-space batch; generate fresh rollouts"
                )
        elif entry and entry.get("format") in (REPLAY_FORMAT, "sglang-token-space-v1"):
            raise ValueError(
                "Token-space media requires token-space training to be enabled"
            )
        if not entry:
            continue
        if token_space:
            if (
                "image_refs" not in entry
                or row_ids[: len(entry["prompt_token_ids"])]
                != entry["prompt_token_ids"]
            ):
                raise ValueError(
                    "Training tokens or immutable media references differ from rollout"
                )
            from rlinf.hybrid_engines.sglang.multimodal.token_space import (
                token_space_processor,
            )

            if entry["processor_settings"]["model_path"] != model_path:
                raise ValueError("Training and rollout token-space model paths differ")
            processor = token_space_processor(
                json.dumps(entry["processor_settings"], sort_keys=True)
            )
            if processor.fingerprint != entry["processor_fingerprint"]:
                raise ValueError(
                    "Token-space processor changed; generate fresh rollouts"
                )
            positions[:, row, row_mask] = processor.positions(entry, row_ids)
        grid = entry["image_grid_thw"]
        if "image_refs" in entry:
            pixels.append(
                _materialize_image_refs(
                    entry,
                    model_path,
                    cache_bytes,
                    processor if token_space else None,
                )
            )
        else:
            pixels.append(entry["pixel_values"])
        grids.append(grid)
    result = {"multimodal_position_ids": positions.to(device)} if token_space else {}
    if image_mask is not None:
        if any(e and e.get("format") != REPLAY_FORMAT for e in payloads):
            raise ValueError(
                "Cannot mix captured and legacy vision inputs in one microbatch"
            )
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

    One background thread serializes native processor use. Prefetch limits
    include the current microbatch and queued results, not processor scratch
    space or the separately bounded process-local image cache. Futures live
    only inside a forward/backward schedule, after distributed batch exchange.
    """

    def __init__(
        self,
        model_path: str,
        *,
        token_space: bool = False,
        cache_bytes: int = 0,
        prefetch_batches: int = 0,
        prefetch_bytes: int = 0,
    ):
        self.model_path = model_path
        self.token_space = token_space
        self.cache_bytes = cache_bytes
        self.prefetch_batches = prefetch_batches
        self.prefetch_bytes = prefetch_bytes

    @staticmethod
    def dense_token_inputs(batch: dict) -> dict:
        """Pad only the current image microbatch before Bridge embeds its inputs."""
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
        """Move only this microbatch's prepared inputs onto the training device."""
        if self.prefetch_batches:
            prepared = batch.pop("_prepared_multimodal_inputs")
            return {key: value.to(device) for key, value in prepared.items()}
        return self._materialize(batch, device)

    def _materialize(self, batch: dict, device) -> dict[str, torch.Tensor]:
        return materialize_multimodal_inputs(
            batch.get("multi_modal_inputs") or [],
            self.model_path,
            device,
            cache_bytes=self.cache_bytes,
            input_ids=batch.get("input_ids"),
            attention_mask=batch.get("attention_mask"),
            token_space=self.token_space,
        )

    def iterator_context(
        self, iterator: Iterator[dict]
    ) -> ContextManager[Iterator[dict]]:
        """Own and drain the prefetch thread even when model forward fails."""
        if not self.prefetch_batches:
            return nullcontext(iterator)
        return self._prefetch_context(iterator)

    def _result_bytes(self, batch: dict) -> int:
        size = 3 * batch["input_ids"].numel() * 8 if self.token_space else 0
        entries = batch.get("multi_modal_inputs") or []
        if any(entry and entry.get("format") == REPLAY_FORMAT for entry in entries):
            size += batch["input_ids"].numel()
        for entry in entries:
            if not entry:
                continue
            if "image_refs" in entry or "pixel_refs" in entry:
                if "pixel_values_bytes" not in entry:
                    raise ValueError(
                        "Image prefetch needs pixel_values_bytes; generate fresh "
                        "token-space rollouts or disable multimodal.prefetch_batches."
                    )
                size += entry["pixel_values_bytes"]
            elif entry.get("format") == REPLAY_FORMAT:
                size += entry["pixel_values"].numel() * 2
            else:
                size += entry["pixel_values"].nbytes
            size += entry["image_grid_thw"].nbytes
        if size > self.prefetch_bytes:
            raise ValueError(
                f"Image microbatch needs {size} prepared bytes, exceeding "
                f"multimodal.prefetch_bytes={self.prefetch_bytes}; reduce the "
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
            while batch is not None and len(pending) < self.prefetch_batches:
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
