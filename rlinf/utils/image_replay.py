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
        raise ValueError("Training prompt differs from the captured SGLang prompt")
    extension = torch.arange(len(prompt), len(ids)) + entry["mrope_position_delta"]
    return torch.cat([positions, extension.expand(3, -1)], dim=1)


def store_pixels(entry: dict, directory: Path) -> None:
    """Publish immutable per-image tensors for pending rows and checkpoint replay."""
    directory.mkdir(parents=True, exist_ok=True)
    pixels = entry["pixel_values"]
    references = []
    for image in pixels.split(entry["image_grid_thw"].prod(dim=1).tolist()):
        data = save({"pixels": image.contiguous()})
        digest = hashlib.sha256(data).hexdigest()
        target = directory / (digest + ".safetensors")
        if not target.exists():
            with NamedTemporaryFile(dir=directory, delete=False) as stream:
                temporary = Path(stream.name)
            try:
                temporary.write_bytes(data)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        references.append({"path": str(target.resolve()), "sha256": digest})
    entry["pixel_refs"] = references
    entry["pixel_values_bytes"] = pixels.numel() * 2
    del entry["pixel_values"]


def replay_pixels(entry: dict) -> torch.Tensor:
    """Read inline or content-addressed pixels in image order as BF16."""
    if "pixel_refs" not in entry:
        return entry["pixel_values"].to(torch.bfloat16)
    images = []
    for ref in entry["pixel_refs"]:
        data = Path(ref["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != ref["sha256"]:
            raise ValueError("Captured image tensor checksum mismatch")
        images.append(load(data)["pixels"].to(torch.bfloat16))
    return torch.cat(images)
