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

"""Read and write the image inputs shared by rollout and learner workers."""

import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from safetensors.torch import load, save

REPLAY_FORMAT = "rlinf-qwen-image-inputs-v1"


def replay_positions(entry: dict, ids: list[int]) -> torch.Tensor:
    """Extend the saved prompt positions through the sampled response."""
    prompt = entry["prompt_token_ids"]
    positions = entry["mrope_positions"]
    if ids[: len(prompt)] != prompt or tuple(positions.shape) != (3, len(prompt)):
        raise ValueError(
            "Training prompt differs from the captured SGLang prompt: "
            f"expected prefix length {len(prompt)} with M-RoPE shape (3, {len(prompt)}), "
            f"got sequence length {len(ids)} and M-RoPE shape {tuple(positions.shape)}."
        )
    extension = torch.arange(len(prompt), len(ids), dtype=positions.dtype) + entry[
        "mrope_position_delta"
    ].to(positions.dtype)
    return torch.cat([positions, extension.expand(3, -1)], dim=1)


def split_image_pixels(
    entry: dict, *, dtype: torch.dtype | None = None
) -> list[torch.Tensor]:
    """Split concatenated ``pixel_values`` by per-image patch counts from ``image_grid_thw``."""
    pixels = entry["pixel_values"] if dtype is None else entry["pixel_values"].to(dtype)
    return list(pixels.split(entry["image_grid_thw"].prod(dim=1).tolist()))


def store_pixels(entry: dict, directory: Path) -> None:
    """Publish immutable BF16 per-image tensors to a caller-managed store directory.

    Callers own cleanup of ``directory`` once all in-flight batches and saved
    checkpoints referencing it have completed.
    """
    directory.mkdir(parents=True, exist_ok=True)
    images = split_image_pixels(entry, dtype=torch.bfloat16)
    references = []
    total_bytes = 0
    for image in images:
        contiguous = image.contiguous()
        total_bytes += contiguous.nbytes
        data = save({"pixels": contiguous})
        digest = hashlib.sha256(data).hexdigest()
        target = directory / (digest + ".safetensors")
        if not target.exists():
            with NamedTemporaryFile(dir=directory, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
            try:
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        references.append({"path": str(target.resolve()), "sha256": digest})
    entry["pixel_refs"] = references
    entry["pixel_values_bytes"] = total_bytes
    del entry["pixel_values"]


def replay_pixels(entry: dict) -> torch.Tensor:
    """Read inline or content-addressed pixels in image order as BF16."""
    if "pixel_refs" not in entry:
        return entry["pixel_values"].to(torch.bfloat16)
    images = []
    for ref in entry["pixel_refs"]:
        data = Path(ref["path"]).read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != ref["sha256"]:
            raise ValueError(
                f"Captured image tensor checksum mismatch for {ref['path']}: "
                f"expected {ref['sha256']}, got {actual}."
            )
        images.append(load(data)["pixels"].to(torch.bfloat16))
    return torch.cat(images)
